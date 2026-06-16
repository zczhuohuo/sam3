import numpy as np
import pytest
import torch
from PIL import Image

from sam3.model import io_utils
from sam3.model.io_utils import LazyImageFrameLoader
from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_multiplex_tracking import (
    Sam3MultiplexTracking,
    Sam3MultiplexTrackingWithInteractivity,
)


def _frame_out(frame_idx):
    return {
        "pred_masks": torch.tensor([frame_idx]),
        "object_score_logits": torch.tensor([frame_idx]),
    }


def _state_with_frames(frame_count=6, offload_lookback_frames=2):
    non_cond = {idx: _frame_out(idx) for idx in range(1, frame_count)}
    long_video = {
        "enabled": True,
        "history_frames": 2,
        "cache_outputs": False,
    }
    # Collapse the offload tier onto the history window so these bookkeeping
    # tests exercise the deletion path directly (see _long_video_lossless_lookback).
    if offload_lookback_frames is not None:
        long_video["offload_lookback_frames"] = offload_lookback_frames
    return {
        "long_video": long_video,
        "output_dict": {
            "cond_frame_outputs": {0: _frame_out(0)},
            "non_cond_frame_outputs": dict(non_cond),
        },
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: _frame_out(0)},
                "non_cond_frame_outputs": dict(non_cond),
            }
        },
        "temp_output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": dict(non_cond),
            }
        },
        "frames_already_tracked": {
            idx: {"reverse": False} for idx in range(frame_count)
        },
        "cached_frame_outputs": {
            idx: {1: torch.tensor([idx])} for idx in range(frame_count)
        },
        "feature_cache": {
            **{idx: (torch.tensor([idx]), {}) for idx in range(frame_count)},
            "text": "keep",
        },
        "tracker_metadata": {
            "obj_id_to_sam2_score_frame_wise": {
                idx: {1: torch.tensor(float(idx))} for idx in range(frame_count)
            },
            "obj_id_to_tracker_score_frame_wise": {
                idx: {1: torch.tensor(float(idx))} for idx in range(frame_count)
            },
            "rank0_metadata": {
                "suppressed_obj_ids": {idx: set() for idx in range(frame_count)},
                "unmatched_frame_inds": {1: list(range(frame_count))},
                "overlap_pair_to_frame_inds": {(1, 2): list(range(frame_count))},
            },
        },
    }


def test_long_video_pruning_keeps_cond_and_recent_non_cond_frames():
    predictor = Sam3BasePredictor()
    state = _state_with_frames()

    predictor._prune_long_video_state(state, frame_idx=5, reverse=False)

    assert set(state["output_dict"]["cond_frame_outputs"]) == {0}
    assert set(state["output_dict"]["non_cond_frame_outputs"]) == {4, 5}
    assert set(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == {4, 5}
    assert set(state["temp_output_dict_per_obj"][0]["non_cond_frame_outputs"]) == {
        4,
        5,
    }
    assert set(state["frames_already_tracked"]) == {0, 4, 5}
    assert set(state["cached_frame_outputs"]) == {0, 4, 5}
    assert set(k for k in state["feature_cache"] if isinstance(k, int)) == {0, 4, 5}
    assert state["feature_cache"]["text"] == "keep"
    metadata = state["tracker_metadata"]
    assert set(metadata["obj_id_to_sam2_score_frame_wise"]) == {0, 4, 5}
    assert set(metadata["obj_id_to_tracker_score_frame_wise"]) == {0, 4, 5}
    assert set(metadata["rank0_metadata"]["suppressed_obj_ids"]) == {0, 4, 5}
    assert metadata["rank0_metadata"]["unmatched_frame_inds"][1] == [0, 4, 5]
    assert metadata["rank0_metadata"]["overlap_pair_to_frame_inds"][(1, 2)] == [
        0,
        4,
        5,
    ]


class FakeModel:
    def __init__(self):
        self.init_kwargs = None

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        self.init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        return {
            "long_video": {"enabled": False},
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        }

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        yield 0, {"ok": True}


def test_start_session_compatibility_without_long_video_fields():
    predictor = Sam3BasePredictor()
    predictor.model = FakeModel()

    result = predictor.start_session("video.mp4", session_id="s")
    outputs = list(
        predictor.propagate_in_video("s", propagation_direction="forward")
    )

    assert result == {"session_id": "s"}
    assert predictor.model.init_kwargs == {
        "resource_path": "video.mp4",
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
    }
    assert outputs == [{"frame_index": 0, "outputs": {"ok": True}}]


class StreamingFakeModel:
    def __init__(self):
        self.postprocess_batch_size = 16
        self.batched_grounding_batch_size = 16
        self.seen_runtime_values = []

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        long_video_mode=False,
        long_video_history_frames=32,
        long_video_loader_type="auto",
        long_video_cache_outputs=False,
        long_video_postprocess_batch_size=None,
        long_video_grounding_batch_size=None,
        async_loading_frames=False,
        video_loader_type="cv2",
    ):
        return {
            "long_video": {
                "enabled": long_video_mode,
                "history_frames": long_video_history_frames,
                "loader_type": long_video_loader_type,
                "cache_outputs": long_video_cache_outputs,
                "postprocess_batch_size": long_video_postprocess_batch_size,
                "grounding_batch_size": long_video_grounding_batch_size,
            },
            "output_dict": {
                "cond_frame_outputs": {0: _frame_out(0)},
                "non_cond_frame_outputs": {},
            },
            "frames_already_tracked": {},
            "cached_frame_outputs": {},
            "feature_cache": {},
        }

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        for frame_idx in range(6):
            self.seen_runtime_values.append(
                (self.postprocess_batch_size, self.batched_grounding_batch_size)
            )
            inference_state["output_dict"]["non_cond_frame_outputs"][
                frame_idx
            ] = _frame_out(frame_idx)
            inference_state["cached_frame_outputs"][frame_idx] = {
                1: torch.tensor([frame_idx])
            }
            inference_state["feature_cache"][frame_idx] = (torch.tensor([frame_idx]), {})
            yield frame_idx, {"frame": frame_idx}


def test_streaming_long_video_state_stays_bounded():
    predictor = Sam3BasePredictor()
    predictor.model = StreamingFakeModel()
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
        long_video_offload_lookback_frames=2,
        long_video_postprocess_batch_size=1,
        long_video_grounding_batch_size=4,
    )

    outputs = list(
        predictor.propagate_in_video("s", propagation_direction="forward")
    )
    state = predictor._all_inference_states["s"]["state"]

    assert [out["frame_index"] for out in outputs] == list(range(6))
    assert set(state["output_dict"]["cond_frame_outputs"]) == {0}
    assert set(state["output_dict"]["non_cond_frame_outputs"]) == {0, 4, 5}
    assert set(state["cached_frame_outputs"]) == {0, 4, 5}
    assert set(k for k in state["feature_cache"] if isinstance(k, int)) == {0, 4, 5}
    assert predictor.model.seen_runtime_values == [(1, 4)] * 6
    assert predictor.model.postprocess_batch_size == 16
    assert predictor.model.batched_grounding_batch_size == 16


class ActiveObjectWindowFakeModel:
    def __init__(self, frames):
        self.frames = frames
        self.removed_objects = []

    def init_state(
        self,
        resource_path,
        long_video_mode=False,
        long_video_history_frames=32,
        **kwargs,
    ):
        return {
            "long_video": {
                "enabled": long_video_mode,
                "history_frames": long_video_history_frames,
                "cache_outputs": False,
            },
            "tracker_metadata": {
                "obj_ids_all_gpu": np.array([], dtype=np.int64),
                "obj_id_to_sam2_score_frame_wise": {},
            },
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        }

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        for frame in self.frames:
            if len(frame) == 3:
                frame_idx, out_obj_ids, tracked_obj_ids = frame
                sam2_scores = {}
            else:
                frame_idx, out_obj_ids, tracked_obj_ids, sam2_scores = frame
            inference_state["tracker_metadata"]["obj_ids_all_gpu"] = np.array(
                tracked_obj_ids, dtype=np.int64
            )
            inference_state["tracker_metadata"]["obj_id_to_sam2_score_frame_wise"][
                frame_idx
            ] = {
                obj_id: torch.tensor(score, dtype=torch.float32)
                for obj_id, score in sam2_scores.items()
            }
            yield frame_idx, {
                "out_obj_ids": np.array(out_obj_ids, dtype=np.int64),
                "out_probs": np.array(
                    [sam2_scores.get(obj_id, 1.0) for obj_id in out_obj_ids],
                    dtype=np.float32,
                ),
                "out_binary_masks": np.zeros((len(out_obj_ids), 1, 1), dtype=bool),
            }

    def remove_object(
        self,
        inference_state,
        obj_id,
        frame_idx,
        is_user_action=False,
    ):
        self.removed_objects.append((obj_id, frame_idx, is_user_action))
        obj_ids = inference_state["tracker_metadata"]["obj_ids_all_gpu"]
        inference_state["tracker_metadata"]["obj_ids_all_gpu"] = obj_ids[
            obj_ids != obj_id
        ]
        return None


def test_long_video_active_window_removes_disappeared_objects():
    predictor = Sam3BasePredictor()
    predictor.model = ActiveObjectWindowFakeModel(
        [
            (0, [1], [1], {1: 0.9}),
            (1, [1], [1], {1: 0.9}),
            (2, [], [1]),
            (3, [], [1]),
        ]
    )
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
    )

    outputs = list(
        predictor.propagate_in_video("s", propagation_direction="forward")
    )

    assert [out["frame_index"] for out in outputs] == [0, 1, 2, 3]
    assert outputs[0]["outputs"]["out_obj_ids"].tolist() == [1]
    assert outputs[1]["outputs"]["out_obj_ids"].tolist() == [1]
    assert predictor.model.removed_objects == [(1, None, False)]


def test_long_video_active_window_keeps_recently_output_objects():
    predictor = Sam3BasePredictor()
    predictor.model = ActiveObjectWindowFakeModel(
        [
            (0, [2], [2], {2: 0.9}),
            (1, [2], [2], {2: 0.9}),
            (2, [2], [2], {2: 0.9}),
            (3, [2], [2], {2: 0.9}),
        ]
    )
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
    )

    list(predictor.propagate_in_video("s", propagation_direction="forward"))

    assert predictor.model.removed_objects == []


def test_long_video_active_window_ignores_low_score_ghost_masks():
    predictor = Sam3BasePredictor()
    predictor.model = ActiveObjectWindowFakeModel(
        [
            (0, [5], [5], {5: 0.9}),
            (1, [5], [5], {5: 0.2}),
            (2, [5], [5], {5: 0.2}),
        ]
    )
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
    )

    outputs = list(
        predictor.propagate_in_video(
            "s", propagation_direction="forward", output_prob_thresh=0.5
        )
    )

    assert [out["outputs"]["out_obj_ids"].tolist() for out in outputs] == [
        [5],
        [5],
        [5],
    ]
    assert predictor.model.removed_objects == [(5, None, False)]


def test_long_video_active_window_graces_objects_without_outputs():
    predictor = Sam3BasePredictor()
    predictor.model = ActiveObjectWindowFakeModel(
        [
            (0, [], [3]),
            (1, [], [3]),
            (2, [], [3]),
        ]
    )
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
    )

    list(predictor.propagate_in_video("s", propagation_direction="forward"))

    assert predictor.model.removed_objects == [(3, None, False)]


def test_long_video_active_window_is_disabled_outside_long_video_mode():
    predictor = Sam3BasePredictor()
    predictor.model = ActiveObjectWindowFakeModel(
        [
            (0, [4], [4]),
            (1, [], [4]),
            (2, [], [4]),
        ]
    )
    predictor.start_session("video.mp4", session_id="s", long_video_mode=False)

    list(predictor.propagate_in_video("s", propagation_direction="forward"))

    assert predictor.model.removed_objects == []


def test_base_predictor_skips_active_window_when_model_manages_it():
    predictor = Sam3BasePredictor()
    predictor.model = ActiveObjectWindowFakeModel(
        [
            (0, [6], [6], {6: 0.9}),
            (1, [6], [6], {6: 0.2}),
            (2, [6], [6], {6: 0.2}),
        ]
    )
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
    )
    state = predictor._all_inference_states["s"]["state"]
    state["long_video"]["active_window_managed_in_model"] = True

    list(
        predictor.propagate_in_video(
            "s", propagation_direction="forward", output_prob_thresh=0.5
        )
    )

    assert predictor.model.removed_objects == []


class InternalActiveWindowFakeMultiplex(Sam3MultiplexTracking):
    def __init__(self, frame_scores):
        self.frame_scores = frame_scores
        self.capacity_prev_counts = []
        self.removed_objects = []
        self.rank = 0
        self.hotstart_delay = 0
        self.postprocess_batch_size = 16
        self.masklet_confirmation_consecutive_det_thresh = 1
        self.is_multiplex = False

    def _compile_model(self):
        return None

    def _get_processing_order(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        return range(len(self.frame_scores)), len(self.frame_scores) - 1

    def _run_single_frame_inference(
        self,
        inference_state,
        frame_idx,
        reverse,
        is_instance_processing=False,
    ):
        tracker_metadata = inference_state["tracker_metadata"]
        tracked_obj_ids = tracker_metadata["obj_ids_all_gpu"]
        self.capacity_prev_counts.append((frame_idx, len(tracked_obj_ids)))
        scores = {
            obj_id: torch.tensor(score, dtype=torch.float32)
            for obj_id, score in self.frame_scores[frame_idx].items()
        }
        tracker_metadata["obj_id_to_sam2_score_frame_wise"][frame_idx] = scores
        return {
            "obj_id_to_mask": {
                obj_id: torch.ones(1, 1, 1, dtype=torch.bool)
                for obj_id in tracked_obj_ids.tolist()
            },
            "obj_id_to_score": {},
            "obj_id_to_sam2_score": scores,
            "removed_obj_ids": set(),
            "suppressed_obj_ids": set(),
            "frame_stats": {},
        }

    def _cache_frame_outputs(
        self,
        inference_state,
        frame_idx,
        obj_id_to_mask,
        suppressed_obj_ids=None,
        removed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        return None

    def _postprocess_output_batched(self, H_video, W_video, batched_outs):
        outputs = []
        for out, _, _, _ in batched_outs:
            outputs.append(
                {
                    "out_obj_ids": np.array(
                        sorted(out["obj_id_to_mask"].keys()), dtype=np.int64
                    ),
                    "out_probs": np.zeros(
                        len(out["obj_id_to_mask"]), dtype=np.float32
                    ),
                    "out_boxes_xywh": np.zeros(
                        (len(out["obj_id_to_mask"]), 4), dtype=np.float32
                    ),
                    "out_binary_masks": np.zeros(
                        (len(out["obj_id_to_mask"]), H_video, W_video), dtype=bool
                    ),
                    "frame_stats": out["frame_stats"],
                }
            )
        return outputs

    def remove_object(
        self,
        inference_state,
        obj_id,
        frame_idx,
        is_user_action=False,
    ):
        self.removed_objects.append((obj_id, frame_idx, is_user_action))
        tracker_metadata = inference_state["tracker_metadata"]
        remaining_obj_ids = tracker_metadata["obj_ids_all_gpu"][
            tracker_metadata["obj_ids_all_gpu"] != obj_id
        ]
        tracker_metadata["obj_ids_all_gpu"] = remaining_obj_ids
        tracker_metadata["obj_ids_per_gpu"] = [remaining_obj_ids]
        tracker_metadata["num_obj_per_gpu"] = [len(remaining_obj_ids)]
        tracker_metadata["obj_id_to_score"].pop(obj_id, None)
        return None


def _internal_active_window_state():
    return {
        "long_video": {
            "enabled": True,
            "history_frames": 2,
            "active_window_managed_in_model": True,
            "object_last_output_frame": {},
            "object_first_seen_frame": {},
            "removed_inactive_object_ids": set(),
        },
        "tracker_metadata": {
            "obj_ids_all_gpu": np.array([5], dtype=np.int64),
            "obj_ids_per_gpu": [np.array([5], dtype=np.int64)],
            "num_obj_per_gpu": [1],
            "obj_id_to_score": {5: 1.0},
            "obj_id_to_sam2_score_frame_wise": {},
        },
        "feature_cache": {},
        "num_frames": 3,
        "orig_height": 1,
        "orig_width": 1,
    }


def test_multiplex_active_window_prunes_before_next_frame_capacity_check():
    model = InternalActiveWindowFakeMultiplex(
        [
            {5: 0.9},
            {5: 0.2},
            {},
        ]
    )
    state = _internal_active_window_state()

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=3,
            reverse=False,
            output_prob_thresh=0.5,
        )
    )

    assert model.capacity_prev_counts == [(0, 1), (1, 1), (2, 0)]
    assert model.removed_objects == [(5, None, False)]
    assert [frame_idx for frame_idx, _ in outputs] == [0, 1, 2]


def test_long_video_runtime_overrides_are_opt_in():
    predictor = Sam3BasePredictor()
    predictor.model = StreamingFakeModel()
    predictor.start_session(
        "video.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
    )

    list(predictor.propagate_in_video("s", propagation_direction="forward"))

    assert predictor.model.seen_runtime_values == [(16, 16)] * 6
    assert predictor.model.postprocess_batch_size == 16
    assert predictor.model.batched_grounding_batch_size == 16


def test_long_video_image_folder_uses_lazy_loader(tmp_path):
    for idx in range(2):
        Image.new("RGB", (4, 4), color=(idx, idx, idx)).save(tmp_path / f"{idx}.jpg")

    images, height, width = io_utils.load_video_frames(
        str(tmp_path),
        image_size=4,
        offload_video_to_cpu=True,
        async_loading_frames=False,
        long_video_mode=True,
    )

    assert isinstance(images, LazyImageFrameLoader)
    assert len(images) == 2
    assert (height, width) == (4, 4)


def test_long_video_video_file_uses_torchcodec_lazy_loader(monkeypatch):
    constructed = {}

    class FakeLazyLoader:
        video_height = 10
        video_width = 20

        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def __len__(self):
            return 3

    monkeypatch.setattr(io_utils, "LazyVideoFileLoaderWithTorchCodec", FakeLazyLoader)

    images, height, width = io_utils.load_video_frames(
        "clip.mp4",
        image_size=8,
        offload_video_to_cpu=True,
        long_video_mode=True,
    )

    assert isinstance(images, FakeLazyLoader)
    assert constructed["video_path"] == "clip.mp4"
    assert (height, width) == (10, 20)


def test_long_video_video_file_requires_torchcodec(monkeypatch):
    class MissingTorchCodecLoader:
        def __init__(self, **kwargs):
            raise RuntimeError(
                "long_video_mode for video files requires TorchCodec. "
                "Install torchcodec or use an image-folder input."
            )

    monkeypatch.setattr(
        io_utils, "LazyVideoFileLoaderWithTorchCodec", MissingTorchCodecLoader
    )

    with pytest.raises(RuntimeError, match="requires TorchCodec"):
        io_utils.load_video_frames(
            "clip.mp4",
            image_size=8,
            offload_video_to_cpu=True,
            long_video_mode=True,
        )


def test_multiplex_interactivity_init_forwards_long_video_loader_options(
    monkeypatch,
):
    captured = {}

    class FakeFrames:
        def __len__(self):
            return 3

    class FakeTracker:
        per_obj_inference = False

    def fake_load_resource_as_video_frames(**kwargs):
        captured.update(kwargs)
        return FakeFrames(), 10, 20

    monkeypatch.setattr(
        "sam3.model.sam3_multiplex_tracking.load_resource_as_video_frames",
        fake_load_resource_as_video_frames,
    )

    model = object.__new__(Sam3MultiplexTrackingWithInteractivity)
    model.image_size = 8
    model.image_mean = (0.5, 0.5, 0.5)
    model.image_std = (0.5, 0.5, 0.5)
    model.tracker = FakeTracker()
    model._construct_initial_input_batch = (
        lambda inference_state, images: inference_state.update(
            {"loaded_images": images}
        )
    )

    state = model.init_state(
        resource_path="clip.mp4",
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
        long_video_mode=True,
        long_video_history_frames=7,
        long_video_loader_type="torchcodec",
        long_video_cache_outputs=True,
        long_video_postprocess_batch_size=2,
        long_video_grounding_batch_size=5,
    )

    assert captured["long_video_mode"] is True
    assert captured["long_video_loader_type"] == "torchcodec"
    assert captured["offload_video_to_cpu"] is True
    assert captured["async_loading_frames"] is True
    assert state["num_frames"] == 3
    assert state["offload_state_to_cpu"] is True
    assert state["long_video"] == {
        "enabled": True,
        "history_frames": 7,
        "loader_type": "torchcodec",
        "cache_outputs": True,
        "postprocess_batch_size": 2,
        "grounding_batch_size": 5,
        "active_window_managed_in_model": True,
        "object_last_output_frame": {},
        "object_first_seen_frame": {},
        "removed_inactive_object_ids": set(),
    }


def test_multiplex_inner_tracker_receives_offload_state_to_cpu():
    captured = {}

    class FakeTracker:
        def init_state(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True}

    model = object.__new__(Sam3MultiplexTrackingWithInteractivity)
    model.tracker = FakeTracker()

    result = model._init_new_sam2_state(
        {
            "feature_cache": {},
            "orig_height": 10,
            "orig_width": 20,
            "num_frames": 3,
            "offload_state_to_cpu": True,
        }
    )

    assert result == {"ok": True}
    assert captured["offload_state_to_cpu"] is True


# ── A1 / B4: two-tier pruning (offload mid-range, delete beyond lookback) ──


def test_long_video_offload_horizon_keeps_midrange_frames():
    predictor = Sam3BasePredictor()
    state = _state_with_frames(frame_count=12, offload_lookback_frames=5)

    predictor._prune_long_video_state(state, frame_idx=11, reverse=False)

    # Frame outputs: only hard-deleted beyond lookback (frame < 11 - 5 + 1 = 7);
    # frames in (history, lookback] are kept (offloaded), cond frame 0 preserved.
    assert set(state["output_dict"]["non_cond_frame_outputs"]) == {7, 8, 9, 10, 11}
    assert set(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == {
        7,
        8,
        9,
        10,
        11,
    }
    # frames_already_tracked aligns with frame-output deletion (lookback horizon).
    assert set(state["frames_already_tracked"]) == {0, 7, 8, 9, 10, 11}
    # Provably-dead structures are dropped at the (smaller) history window.
    assert set(state["cached_frame_outputs"]) == {0, 10, 11}
    assert set(k for k in state["feature_cache"] if isinstance(k, int)) == {0, 10, 11}
    metadata = state["tracker_metadata"]
    assert set(metadata["obj_id_to_sam2_score_frame_wise"]) == {0, 10, 11}


def test_long_video_offload_tier_selection():
    # frame_idx=8, history=2, lookback=4 partitions past frames into three tiers:
    #   frame 2  -> older than lookback        -> hard-deleted
    #   frame 6  -> between history & lookback  -> kept, spatial memory offloaded
    #   frame 7  -> within history window        -> kept untouched
    offloaded = []

    class _RecordingPredictor(Sam3BasePredictor):
        def _offload_long_video_frame_output(self, out):
            offloaded.append(out["_frame"])
            super()._offload_long_video_frame_output(out)

    predictor = _RecordingPredictor()
    state = {
        "long_video": {
            "enabled": True,
            "history_frames": 2,
            "offload_lookback_frames": 4,
        },
        "output_dict": {
            "cond_frame_outputs": {0: {"_frame": 0}},
            "non_cond_frame_outputs": {
                2: {"_frame": 2},
                6: {"_frame": 6},
                7: {"_frame": 7},
            },
        },
    }

    predictor._prune_long_video_state(state, frame_idx=8, reverse=False)

    assert set(state["output_dict"]["non_cond_frame_outputs"]) == {6, 7}
    assert offloaded == [6]


def test_long_video_offload_only_targets_spatial_memory_tensors():
    # The offload helper moves only maskmem_* tensors (both reloaded via .cuda()
    # on read); obj_ptr / pred_masks are consumed on-device and left in place.
    # Tensors are already on CPU here, so this is a structural / no-error check.
    predictor = Sam3BasePredictor()
    maskmem = torch.zeros(2, 2)
    pos = torch.zeros(2, 2)
    obj_ptr = torch.zeros(2, 2)
    out = {
        "maskmem_features": maskmem,
        "maskmem_pos_enc": [pos],
        "obj_ptr": obj_ptr,
    }
    predictor._offload_long_video_frame_output(out)
    assert out["maskmem_features"].equal(maskmem)
    assert out["maskmem_pos_enc"][0].equal(pos)
    assert out["obj_ptr"] is obj_ptr


# ── A1 / C7: lookback horizon derives from the model's real attention reach ──


def test_lossless_lookback_uses_model_horizon():
    predictor = Sam3BasePredictor()

    class _Model:
        use_memory_selection = True
        max_obj_ptrs_in_encoder = 16
        num_maskmem = 7
        memory_temporal_stride_for_eval = 1

    predictor.model = _Model()
    # With memory selection on, mirror the model's far-old trim horizon.
    assert predictor._long_video_lossless_lookback({}) == 320
    predictor.model.use_memory_selection = False
    assert predictor._long_video_lossless_lookback({}) == 16
    # Explicit override wins.
    assert predictor._long_video_lossless_lookback(
        {"offload_lookback_frames": 50}
    ) == 50


class _SelectionFakeModel:
    def __init__(self):
        self.use_memory_selection = True
        self.num_maskmem = 7
        self.max_obj_ptrs_in_encoder = 16
        self.postprocess_batch_size = 16
        self.batched_grounding_batch_size = 16
        self.seen = []

    def init_state(self, resource_path, **kwargs):
        return {
            "long_video": {"enabled": True, "history_frames": 2},
            "output_dict": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        }

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        self.seen.append(
            (
                self.use_memory_selection,
                self.num_maskmem,
                self.max_obj_ptrs_in_encoder,
            )
        )
        yield 0, {"ok": True}


def test_disable_temporal_disambiguation_and_shrink_overrides():
    predictor = Sam3BasePredictor()
    predictor.model = _SelectionFakeModel()
    predictor.start_session(
        "v.mp4",
        session_id="s",
        long_video_mode=True,
        long_video_history_frames=2,
        long_video_disable_temporal_disambiguation=True,
        long_video_num_maskmem=4,
        long_video_max_obj_ptrs=8,
    )

    list(predictor.propagate_in_video("s", propagation_direction="forward"))

    # Overrides applied during propagation...
    assert predictor.model.seen == [(False, 4, 8)]
    # ...and restored afterwards.
    assert predictor.model.use_memory_selection is True
    assert predictor.model.num_maskmem == 7
    assert predictor.model.max_obj_ptrs_in_encoder == 16


# ── A3: bounded sliding-window prefetch loader ──


def test_lazy_image_loader_prefetch_returns_correct_frames(tmp_path):
    paths = []
    for idx in range(10):
        arr = np.full((4, 4, 3), idx * 20, dtype=np.uint8)
        path = tmp_path / f"{idx}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))

    mean = torch.zeros(3, 1, 1, dtype=torch.float16)
    std = torch.ones(3, 1, 1, dtype=torch.float16)
    loader = LazyImageFrameLoader(
        paths,
        image_size=4,
        offload_video_to_cpu=True,
        img_mean=mean,
        img_std=std,
        prefetch_window=3,
        keep_behind=1,
    )
    try:
        assert len(loader) == 10
        assert loader[0].shape == (3, 4, 4)
        # list / slice indexing returns a stacked tensor
        assert loader[[1, 2, 3]].shape == (3, 3, 4, 4)
        assert loader[2:5].shape == (3, 3, 4, 4)
        # sequential access keeps the cache bounded by capacity
        for idx in range(10):
            _ = loader[idx]
        assert len(loader.cache) <= loader.capacity
    finally:
        loader.close()
