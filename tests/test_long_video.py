import numpy as np
import pytest
import torch
from PIL import Image

from sam3.model import io_utils
from sam3.model.io_utils import LazyImageFrameLoader
from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity


def _frame_out(frame_idx):
    return {
        "pred_masks": torch.tensor([frame_idx]),
        "object_score_logits": torch.tensor([frame_idx]),
    }


def _state_with_frames(frame_count=6):
    non_cond = {idx: _frame_out(idx) for idx in range(1, frame_count)}
    return {
        "long_video": {
            "enabled": True,
            "history_frames": 2,
            "cache_outputs": False,
        },
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
        "frames_already_tracked": {idx: {"reverse": False} for idx in range(6)},
        "cached_frame_outputs": {idx: {1: torch.tensor([idx])} for idx in range(6)},
        "feature_cache": {
            **{idx: (torch.tensor([idx]), {}) for idx in range(6)},
            "text": "keep",
        },
        "tracker_metadata": {
            "obj_id_to_sam2_score_frame_wise": {
                idx: {1: torch.tensor(float(idx))} for idx in range(6)
            },
            "obj_id_to_tracker_score_frame_wise": {
                idx: {1: torch.tensor(float(idx))} for idx in range(6)
            },
            "rank0_metadata": {
                "suppressed_obj_ids": {idx: set() for idx in range(6)},
                "unmatched_frame_inds": {1: list(range(6))},
                "overlap_pair_to_frame_inds": {(1, 2): list(range(6))},
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
