# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Base predictor class shared by SAM3 and SAM3.1 (multiplex) video predictors.

Provides the common handle_request/handle_stream_request API and session management.
Subclasses only need to override methods where their behavior differs.
"""

import gc
import inspect
import os
import time
import uuid
from typing import Dict, List, Optional

import torch
from sam3.logger import get_logger

logger = get_logger(__name__)

# torch.cuda.empty_cache() forces a CUDA synchronization that stalls all
# streams in the process. Calling it on every close_session produces visible
# compute-utilization gaps when many sessions are active concurrently.
# Gate the call on device memory pressure: only fire when usage crosses the
# threshold. The allocator's caching pool already covers the common case
# where freed blocks get reused by the next session — empty_cache is only
# needed to keep memory from growing unbounded.
_CLEAR_CACHE_THRESHOLD = 80


class Sam3BasePredictor:
    """
    Base class for SAM3 video predictors. Provides:
    - Session management (start, reset, close)
    - Request dispatch (handle_request / handle_stream_request)
    - Common add_prompt / propagate_in_video / remove_object / reset_session / close_session

    Subclasses must set `self.model` and `self._all_inference_states` before use.
    """

    def __init__(self):
        # Subclasses must populate these
        self.model = None
        self._all_inference_states: Dict[str, dict] = {}

    # ── Request dispatch ──────────────────────────────────────────────

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        request_type = request["type"]
        if request_type == "start_session":
            long_video_mode = request.get("long_video_mode", False)
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id", None),
                offload_video_to_cpu=request.get(
                    "offload_video_to_cpu", True if long_video_mode else False
                ),
                offload_state_to_cpu=request.get("offload_state_to_cpu", False),
                long_video_mode=long_video_mode,
                long_video_history_frames=request.get(
                    "long_video_history_frames", 32
                ),
                long_video_loader_type=request.get(
                    "long_video_loader_type", "auto"
                ),
                long_video_cache_outputs=request.get(
                    "long_video_cache_outputs", False
                ),
                long_video_postprocess_batch_size=request.get(
                    "long_video_postprocess_batch_size", None
                ),
                long_video_grounding_batch_size=request.get(
                    "long_video_grounding_batch_size", None
                ),
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text", None),
                points=request.get("points", None),
                point_labels=request.get("point_labels", None),
                clear_old_points=request.get("clear_old_points", True),
                bounding_boxes=request.get("bounding_boxes", None),
                bounding_box_labels=request.get("bounding_box_labels", None),
                clear_old_boxes=request.get("clear_old_boxes", True),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
                obj_id=request.get("obj_id", None),
                rel_coordinates=request.get("rel_coordinates", True),
            )
        elif request_type == "remove_object":
            return self.remove_object(
                session_id=request["session_id"],
                frame_idx=request.get("frame_index", 0),
                obj_id=request["obj_id"],
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "cancel_propagation":
            return self.cancel_propagation(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(
                session_id=request["session_id"],
                run_gc_collect=request.get("run_gc_collect", True),
                clear_cache_threshold=int(
                    request.get("clear_cache_threshold", _CLEAR_CACHE_THRESHOLD)
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_index", None),
                max_frame_num_to_track=request.get("max_frame_num_to_track", None),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    # ── Session management ────────────────────────────────────────────

    def start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=None,
        offload_state_to_cpu=False,
        long_video_mode=False,
        long_video_history_frames=32,
        long_video_loader_type="auto",
        long_video_cache_outputs=False,
        long_video_postprocess_batch_size=None,
        long_video_grounding_batch_size=None,
    ):
        """Start a new inference session on a video directory or path."""
        if offload_video_to_cpu is None:
            offload_video_to_cpu = bool(long_video_mode)
        if long_video_mode:
            if long_video_history_frames < 1:
                raise ValueError("long_video_history_frames must be >= 1")
            if long_video_loader_type not in {"auto", "torchcodec", "image_folder"}:
                raise ValueError(
                    "long_video_loader_type must be one of 'auto', 'torchcodec', "
                    f"or 'image_folder'; got {long_video_loader_type!r}"
                )

        init_kwargs = dict(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        if long_video_mode:
            init_kwargs["async_loading_frames"] = True
            init_kwargs["long_video_mode"] = True
            init_kwargs["long_video_history_frames"] = long_video_history_frames
            init_kwargs["long_video_loader_type"] = long_video_loader_type
            init_kwargs["long_video_cache_outputs"] = long_video_cache_outputs
            init_kwargs[
                "long_video_postprocess_batch_size"
            ] = long_video_postprocess_batch_size
            init_kwargs["long_video_grounding_batch_size"] = (
                long_video_grounding_batch_size
            )
            if isinstance(resource_path, str):
                if os.path.isdir(resource_path):
                    if long_video_loader_type == "torchcodec":
                        raise ValueError(
                            "long_video_loader_type='torchcodec' requires a video "
                            f"file, but got image folder {resource_path!r}"
                        )
                elif long_video_loader_type == "image_folder":
                    raise ValueError(
                        "long_video_loader_type='image_folder' requires an image "
                        f"folder, but got {resource_path!r}"
                    )
                else:
                    init_kwargs["video_loader_type"] = "torchcodec"

        sig = inspect.signature(self.model.init_state)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not has_var_kwargs:
            init_kwargs = {
                k: v for k, v in init_kwargs.items() if k in sig.parameters
            }
        inference_state = self.model.init_state(**init_kwargs)
        inference_state.setdefault(
            "long_video",
            {
                "enabled": bool(long_video_mode),
                "history_frames": int(long_video_history_frames),
                "loader_type": long_video_loader_type,
                "cache_outputs": bool(long_video_cache_outputs),
                "postprocess_batch_size": long_video_postprocess_batch_size,
                "grounding_batch_size": long_video_grounding_batch_size,
            },
        )

        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        logger.info(f"started new session {session_id}")
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        points=None,
        point_labels=None,
        clear_old_points: bool = True,
        bounding_boxes=None,
        bounding_box_labels=None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: Optional[int] = None,
        rel_coordinates: bool = True,
    ):
        """Add text, box and/or point prompt on a specific video frame."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        # Convert lists to tensors if needed
        if points is not None and not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if point_labels is not None and not isinstance(point_labels, torch.Tensor):
            point_labels = torch.tensor(point_labels, dtype=torch.int32)
        if bounding_boxes is not None and not isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        if bounding_box_labels is not None and not isinstance(
            bounding_box_labels, torch.Tensor
        ):
            bounding_box_labels = torch.tensor(bounding_box_labels, dtype=torch.int32)

        kwargs = dict(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            points=points,
            point_labels=point_labels,
            clear_old_points=clear_old_points,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
            clear_old_boxes=clear_old_boxes,
            output_prob_thresh=output_prob_thresh,
            rel_coordinates=rel_coordinates,
        )
        if obj_id is not None:
            kwargs["obj_id"] = obj_id

        # Filter kwargs to only pass what the model accepts
        # (SAM3 has a simpler add_prompt than SAM3.1)
        import inspect

        sig = inspect.signature(self.model.add_prompt)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            frame_idx, outputs = self.model.add_prompt(**filtered_kwargs)
        return {"frame_index": frame_idx, "outputs": outputs}

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ):
        """Remove an object from tracking."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        result = self.model.remove_object(
            inference_state, obj_id, frame_idx=frame_idx, is_user_action=is_user_action
        )
        # Handle both return conventions
        if result is None or (isinstance(result, tuple) and result[1] is None):
            import numpy as np

            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_binary_masks = torch.zeros(
                0,
                inference_state["orig_height"],
                inference_state["orig_width"],
                dtype=torch.bool,
            )
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
            outputs = {
                "out_obj_ids": out_obj_ids.cpu().numpy(),
                "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
                "out_binary_masks": out_binary_masks.cpu().numpy(),
            }
        elif isinstance(result, tuple):
            _, outputs = result
        else:
            outputs = result
        return {"frame_index": frame_idx, "outputs": outputs}

    def cancel_propagation(self, session_id):
        """Cancel any ongoing propagation. No-op if not supported by the model."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)
        if hasattr(self.model, "cancel_propagation"):
            self.model.cancel_propagation(inference_state)
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id,
        propagation_direction="both",
        start_frame_idx=None,
        max_frame_num_to_track=None,
        output_prob_thresh=0.5,
        **kwargs,
    ):
        """Propagate the added prompts to get results on all video frames."""
        try:
            session = self._get_session(session_id)
            inference_state = session["state"]
            restore_runtime_overrides = self._apply_long_video_runtime_overrides(
                inference_state
            )
            self._extend_expiration_time(session)
            if propagation_direction not in ["both", "forward", "backward"]:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )

            propagate_kwargs = dict(
                inference_state=inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            )
            # Only pass output_prob_thresh / extra kwargs if the model supports them
            sig = inspect.signature(self.model.propagate_in_video)
            if "output_prob_thresh" in sig.parameters:
                propagate_kwargs["output_prob_thresh"] = output_prob_thresh
            for k, v in kwargs.items():
                if k in sig.parameters:
                    propagate_kwargs[k] = v

            # Forward propagation
            if propagation_direction in ["both", "forward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=False,
                ):
                    self._maintain_long_video_active_objects(
                        inference_state, frame_idx, outputs, False
                    )
                    self._prune_long_video_state(inference_state, frame_idx, False)
                    yield {"frame_index": frame_idx, "outputs": outputs}
            # Backward propagation
            if propagation_direction in ["both", "backward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=True,
                ):
                    self._maintain_long_video_active_objects(
                        inference_state, frame_idx, outputs, True
                    )
                    self._prune_long_video_state(inference_state, frame_idx, True)
                    yield {"frame_index": frame_idx, "outputs": outputs}
        finally:
            if "restore_runtime_overrides" in locals():
                restore_runtime_overrides()
            logger.info(f"propagation ended in session {session_id}")

    def _apply_long_video_runtime_overrides(self, inference_state):
        long_video = inference_state.get("long_video") or {}
        if not long_video.get("enabled", False):
            return lambda: None

        overrides = {
            "postprocess_batch_size": long_video.get("postprocess_batch_size"),
            "batched_grounding_batch_size": long_video.get("grounding_batch_size"),
        }
        original_values = {}
        for attr_name, override_value in overrides.items():
            if override_value is None:
                continue
            override_value = int(override_value)
            if override_value < 1 or not hasattr(self.model, attr_name):
                continue
            original_values[attr_name] = getattr(self.model, attr_name)
            setattr(
                self.model,
                attr_name,
                min(original_values[attr_name], override_value),
            )

        def restore():
            for attr_name, original_value in original_values.items():
                setattr(self.model, attr_name, original_value)

        return restore

    def _maintain_long_video_active_objects(
        self, inference_state, frame_idx, outputs, reverse
    ):
        long_video = inference_state.get("long_video") or {}
        if not long_video.get("enabled", False):
            return
        if not hasattr(self.model, "remove_object"):
            return

        tracker_metadata = inference_state.get("tracker_metadata")
        if not isinstance(tracker_metadata, dict):
            return

        tracked_obj_ids = self._normalize_long_video_obj_ids(
            tracker_metadata.get("obj_ids_all_gpu")
        )
        if not tracked_obj_ids:
            return

        history_frames = int(long_video.get("history_frames", 32))
        first_seen = long_video.setdefault("object_first_seen_frame", {})
        last_output = long_video.setdefault("object_last_output_frame", {})
        removed = long_video.setdefault("removed_inactive_object_ids", set())

        for obj_id in tracked_obj_ids:
            if obj_id not in removed:
                first_seen.setdefault(obj_id, frame_idx)

        for obj_id in self._extract_long_video_output_obj_ids(outputs):
            if obj_id not in removed:
                first_seen.setdefault(obj_id, frame_idx)
                last_output[obj_id] = frame_idx

        stale_obj_ids = []
        for obj_id in tracked_obj_ids:
            if obj_id in removed:
                continue
            reference_frame = last_output.get(obj_id, first_seen.get(obj_id, frame_idx))
            if self._long_video_object_is_stale(
                reference_frame, frame_idx, history_frames, reverse
            ):
                stale_obj_ids.append(obj_id)

        for obj_id in stale_obj_ids:
            self.model.remove_object(
                inference_state,
                obj_id,
                frame_idx=None,
                is_user_action=False,
            )
            removed.add(obj_id)
            first_seen.pop(obj_id, None)
            last_output.pop(obj_id, None)

    def _long_video_object_is_stale(
        self, reference_frame, frame_idx, history_frames, reverse
    ):
        if reverse:
            return reference_frame > frame_idx + history_frames - 1
        return reference_frame < frame_idx - history_frames + 1

    def _extract_long_video_output_obj_ids(self, outputs):
        if not isinstance(outputs, dict):
            return []
        return self._normalize_long_video_obj_ids(outputs.get("out_obj_ids"))

    def _normalize_long_video_obj_ids(self, obj_ids):
        if obj_ids is None:
            return []
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.detach().cpu()
        if hasattr(obj_ids, "tolist"):
            obj_ids = obj_ids.tolist()
        if not isinstance(obj_ids, (list, tuple, set)):
            obj_ids = [obj_ids]

        normalized = []
        for obj_id in obj_ids:
            if isinstance(obj_id, (list, tuple)):
                normalized.extend(self._normalize_long_video_obj_ids(obj_id))
                continue
            if hasattr(obj_id, "item"):
                obj_id = obj_id.item()
            try:
                obj_id = int(obj_id)
            except (TypeError, ValueError):
                continue
            normalized.append(obj_id)
        return normalized

    def _prune_long_video_state(self, inference_state, frame_idx, reverse):
        long_video = inference_state.get("long_video") or {}
        if not long_video.get("enabled", False):
            return
        history_frames = int(long_video.get("history_frames", 32))
        cache_outputs = bool(long_video.get("cache_outputs", False))

        def should_prune(candidate_frame_idx, cond_frames):
            if candidate_frame_idx in cond_frames:
                return False
            if reverse:
                return candidate_frame_idx > frame_idx + history_frames - 1
            return candidate_frame_idx < frame_idx - history_frames + 1

        self._prune_long_video_state_dict(
            inference_state,
            cache_outputs=cache_outputs,
            should_prune=should_prune,
        )
        for key in ("tracker_inference_states", "sam2_inference_states"):
            for tracker_state in inference_state.get(key, []):
                self._prune_long_video_state_dict(
                    tracker_state,
                    cache_outputs=cache_outputs,
                    should_prune=should_prune,
                )

    def _prune_long_video_state_dict(
        self,
        state,
        cache_outputs,
        should_prune,
    ):
        output_dict = state.get("output_dict")
        cond_frames = set()
        pruned_frames = set()
        if isinstance(output_dict, dict):
            cond_frames = set(output_dict.get("cond_frame_outputs", {}).keys())
            non_cond_outputs = output_dict.get("non_cond_frame_outputs", {})
            for old_frame_idx in list(non_cond_outputs.keys()):
                if should_prune(old_frame_idx, cond_frames):
                    non_cond_outputs.pop(old_frame_idx, None)
                    pruned_frames.add(old_frame_idx)

        consolidated = state.get("consolidated_frame_inds")
        if isinstance(consolidated, dict):
            cond_frames.update(consolidated.get("cond_frame_outputs", set()))

        if pruned_frames:
            for per_obj_key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
                for obj_output_dict in state.get(per_obj_key, {}).values():
                    for storage_key in (
                        "non_cond_frame_outputs",
                        "cond_frame_outputs",
                    ):
                        storage = obj_output_dict.get(storage_key, {})
                        for old_frame_idx in pruned_frames:
                            if old_frame_idx not in cond_frames:
                                storage.pop(old_frame_idx, None)

            frames_already_tracked = state.get("frames_already_tracked")
            if isinstance(frames_already_tracked, dict):
                for old_frame_idx in pruned_frames:
                    frames_already_tracked.pop(old_frame_idx, None)

        cached_frame_outputs = state.get("cached_frame_outputs")
        if isinstance(cached_frame_outputs, dict) and not cache_outputs:
            for old_frame_idx in list(cached_frame_outputs.keys()):
                if should_prune(old_frame_idx, cond_frames):
                    cached_frame_outputs.pop(old_frame_idx, None)

        feature_cache = state.get("feature_cache")
        if isinstance(feature_cache, dict):
            for old_frame_idx in list(feature_cache.keys()):
                if isinstance(old_frame_idx, int) and should_prune(
                    old_frame_idx, cond_frames
                ):
                    feature_cache.pop(old_frame_idx, None)

        tracker_metadata = state.get("tracker_metadata")
        if isinstance(tracker_metadata, dict):
            self._prune_long_video_tracker_metadata(
                tracker_metadata, cond_frames, should_prune
            )

    def _prune_long_video_tracker_metadata(
        self,
        tracker_metadata,
        cond_frames,
        should_prune,
    ):
        # These frame-indexed score maps store GPU tensors in SAM3.1 multiplex.
        # Leaving them unbounded defeats the sliding-state memory limit.
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            score_map = tracker_metadata.get(score_key)
            if isinstance(score_map, dict):
                for old_frame_idx in list(score_map.keys()):
                    if should_prune(old_frame_idx, cond_frames):
                        score_map.pop(old_frame_idx, None)

        rank0_metadata = tracker_metadata.get("rank0_metadata")
        if not isinstance(rank0_metadata, dict):
            return

        suppressed_obj_ids = rank0_metadata.get("suppressed_obj_ids")
        if isinstance(suppressed_obj_ids, dict):
            for old_frame_idx in list(suppressed_obj_ids.keys()):
                if should_prune(old_frame_idx, cond_frames):
                    suppressed_obj_ids.pop(old_frame_idx, None)

        for list_key in ("unmatched_frame_inds", "overlap_pair_to_frame_inds"):
            frame_lists = rank0_metadata.get(list_key)
            if not isinstance(frame_lists, dict):
                continue
            for key, frame_indices in list(frame_lists.items()):
                if not isinstance(frame_indices, list):
                    continue
                frame_lists[key] = [
                    idx for idx in frame_indices if not should_prune(idx, cond_frames)
                ]

    def reset_session(self, session_id):
        """Reset the session to its initial state."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)
        self.model.reset_state(inference_state)
        return {"is_success": True}

    def close_session(
        self,
        session_id,
        run_gc_collect=True,
        clear_cache_threshold: int = _CLEAR_CACHE_THRESHOLD,
    ):
        """Close a session. Idempotent.

        ``run_gc_collect=True`` (the default) also returns the session's
        freed CUDA tensors back to the device by calling
        ``torch.cuda.empty_cache()`` after ``gc.collect()``. Without this,
        PyTorch's caching allocator retains the freed allocations in its
        per-process pool, so reserved memory keeps climbing across
        long-running workloads even though the Python-level objects are gone.

        ``empty_cache()`` itself triggers a CUDA sync, so it is gated on
        device memory pressure via the ``gpu_mem`` snapshot. Callers can
        override the threshold per-call via ``clear_cache_threshold``.

        When ``run_gc_collect=True``, the response includes a ``gpu_mem``
        snapshot (free / total / allocated / reserved bytes, plus active
        session count) so clients can decide whether the device has
        headroom for their next session — no separate admission RPC
        needed. See ``_gpu_mem_snapshot`` for the field shape. The
        first snapshot (after ``gc.collect()``) drives the cache-eviction
        decision; if ``empty_cache()`` fires, a second snapshot is taken
        so the response reflects the post-cleanup state and the freed
        bytes are logged. Old callers that only read ``is_success`` work
        unchanged — additional dict keys are ignored at the JSON layer.
        """
        session = self._all_inference_states.pop(session_id, None)
        result = {"is_success": True}
        if session is None:
            logger.warning(f"cannot close session {session_id} as it does not exist")
        else:
            # Explicitly clear the per-session dicts BEFORE ``del session``.
            #
            # ``inference_state`` (i.e., ``session["state"]``) is the dict
            # built by ``init_state`` / ``_construct_initial_input_batch``
            # and grown by ``add_prompt`` / propagation. It holds heavy
            # GPU-resident references — ``input_batch`` (the video frames
            # as a ``BatchedDatapoint`` on device),
            # ``constants["empty_geometric_prompt"]`` (zeroed device
            # tensors), and the per-inference accumulators
            # ``feature_cache``, ``cached_frame_outputs``,
            # ``tracker_inference_states``, ``tracker_metadata``.
            #
            # Empirically, relying on ``del session`` + ``gc.collect()``
            # alone has been insufficient in prod: across long-lived
            # IPNext replicas serving thousands of sessions, PyTorch's
            # *allocated* bucket monotonically climbs to ~93 GiB even
            # while ``empty_cache`` keeps *reserved-but-unallocated* at
            # ~600 MiB, leading to OOMs at concurrency=1 (SAM3 client
            # observation 2026-05-09 / 2026-05-10, jiids
            # 37154706936429064 and 41658306563594281). The fingerprint:
            #
            #     this process has 94.99 GiB memory in use.
            #     93.54 GiB allocated by PyTorch.
            #     577.70 MiB reserved by PyTorch but unallocated.
            #
            # which is the signature of dict-keyed references staying
            # alive (allocated bucket, not the cache). Calling ``clear()``
            # on the nested dict immediately drops all per-session tensor
            # refs the dict was keying, regardless of whether something
            # else (closure, asyncio task, metrics buffer, parent
            # container) is still holding the wrapper dict alive via a
            # cycle. Lists in the inference state hold ``None``s for
            # per-frame slots, so they don't need separate handling.
            state = session.get("state")
            if isinstance(state, dict):
                state.clear()
            session.clear()
            del session
            if run_gc_collect:
                gc.collect()
                gpu_mem = self._gpu_mem_snapshot()
                if (
                    torch.cuda.is_available()
                    and gpu_mem["total_bytes"] > 0
                    and (100.0 - gpu_mem["free_pct"]) >= clear_cache_threshold
                ):
                    torch.cuda.empty_cache()
                    post_gpu_mem = self._gpu_mem_snapshot()
                    logger.info(
                        f"empty_cache freed "
                        f"{post_gpu_mem['free_bytes'] - gpu_mem['free_bytes']} bytes "
                        f"(free_pct {gpu_mem['free_pct']:.1f}% -> "
                        f"{post_gpu_mem['free_pct']:.1f}%, reserved "
                        f"{gpu_mem['reserved_bytes']} -> "
                        f"{post_gpu_mem['reserved_bytes']} bytes)"
                    )
                    gpu_mem = post_gpu_mem
                result["gpu_mem"] = gpu_mem
            logger.info(f"removed session {session_id}")
        return result

    def _gpu_mem_snapshot(self) -> dict:
        """Snapshot of current GPU memory state for inclusion in
        session-close responses.

        Lets clients track free HBM across the fleet without a separate
        admission RPC: every ``close_session`` naturally exposes the
        post-cleanup state (taken AFTER any ``empty_cache`` call), which
        is exactly what the next session will face.

        Fields:
          - ``free_bytes`` / ``total_bytes`` — raw
            ``torch.cuda.mem_get_info()``.
          - ``allocated_bytes`` — ``torch.cuda.memory_allocated()``,
            live tensor footprint (no caching pool overhead).
          - ``reserved_bytes`` — ``torch.cuda.memory_reserved()``,
            caching-allocator pool size.
          - ``free_pct`` — ``free_bytes / total_bytes * 100`` for
            convenient % thresholds.
          - ``active_session_count`` — sessions still resident on this
            predictor instance after the close.

        Fail-open: any error reading the device returns zero stats so
        a broken CUDA context (e.g., CPU-only test env) NEVER breaks
        the session-close response.
        """
        active_count = len(self._all_inference_states)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            # No active CUDA context (e.g., CPU-only test env).
            return {
                "free_bytes": 0,
                "total_bytes": 0,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "free_pct": 0.0,
                "active_session_count": active_count,
            }
        free_pct = (free_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0
        # ``memory_allocated`` / ``memory_reserved`` are cheap host-side
        # bookkeeping reads (no CUDA sync) and complement
        # ``mem_get_info`` by exposing the caching-allocator pool size
        # vs the actual live tensor footprint.
        allocated_bytes = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        reserved_bytes = (
            torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        )
        return {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
            "free_pct": free_pct,
            "active_session_count": active_count,
        }

    def _get_session(self, session_id):
        session = self._all_inference_states.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _extend_expiration_time(self, session):
        """Update last-use time for session expiration tracking."""
        session["last_use_time"] = time.time()

    def shutdown(self):
        """Shutdown the predictor and clear all sessions."""
        self._all_inference_states.clear()
