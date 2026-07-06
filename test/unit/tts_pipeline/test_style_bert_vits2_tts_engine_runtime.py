"""StyleBertVITS2TTSEngine のランタイム状態管理テスト。"""

import threading
from typing import Any, Literal, cast

import pytest
from fastapi import HTTPException

from voicevox_engine.model import (
    AivmInfo,
    AivmModelRuntimePolicy,
    AivmModelRuntimeState,
)
from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
    StyleBertVITS2TTSEngine,
)


def _create_runtime_state(
    model_uuid: str,
    *,
    is_loaded: bool,
    is_pinned: bool = False,
    residency: Literal["unloaded", "ram", "vram"] | None = None,
    load_count: int,
    last_loaded_at: float | None,
    last_used_at: float | None,
) -> AivmModelRuntimeState:
    return AivmModelRuntimeState(
        model_uuid=model_uuid,
        is_loaded=is_loaded,
        is_cached_in_ram=is_loaded,
        is_loaded_in_vram=(residency == "vram"),
        is_pinned=is_pinned,
        residency=(
            residency
            if residency is not None
            else ("ram" if is_loaded is True else "unloaded")
        ),
        load_count=load_count,
        inference_device="cpu",
        onnx_providers=["CPUExecutionProvider"] if is_loaded is True else [],
        last_loaded_at=last_loaded_at,
        last_used_at=last_used_at,
        last_unloaded_at=None,
    )


def _create_engine_with_runtime_states(
    runtime_states: dict[str, AivmModelRuntimeState],
) -> StyleBertVITS2TTSEngine:
    engine = StyleBertVITS2TTSEngine.__new__(StyleBertVITS2TTSEngine)
    engine.tts_models = {}
    engine._tts_models_lock = threading.Lock()
    engine._runtime_states = runtime_states
    engine._runtime_states_lock = threading.Lock()
    engine._prefetched_model_artifacts = {}
    engine._prefetched_model_artifacts_lock = threading.Lock()
    engine._runtime_policy = AivmModelRuntimePolicy(max_loaded_models=None)
    engine.onnx_providers = ["CPUExecutionProvider"]
    engine.aivm_manager = cast(
        Any,
        type(
            "Manager",
            (),
            {
                "get_aivm_info": lambda self, aivm_model_uuid: AivmInfo.model_construct(  # type: ignore[call-arg]
                    file_size=1024**3,
                ),
                "get_installed_aivm_infos": lambda self: {},
            },
        )(),
    )
    return engine


def test_get_lru_unload_candidates_returns_oldest_loaded_models_first() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "old-prefetched": _create_runtime_state(
                "old-prefetched",
                is_loaded=True,
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=None,
            ),
            "recently-used": _create_runtime_state(
                "recently-used",
                is_loaded=True,
                load_count=2,
                last_loaded_at=20.0,
                last_used_at=50.0,
            ),
            "older-used": _create_runtime_state(
                "older-used",
                is_loaded=True,
                load_count=1,
                last_loaded_at=15.0,
                last_used_at=30.0,
            ),
            "already-unloaded": _create_runtime_state(
                "already-unloaded",
                is_loaded=False,
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=5.0,
            ),
        }
    )

    candidates = engine.get_lru_unload_candidates(limit=3)

    assert [candidate.model_uuid for candidate in candidates] == [
        "old-prefetched",
        "older-used",
        "recently-used",
    ]


def test_get_lru_unload_candidates_respects_exclude_list() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "candidate-a": _create_runtime_state(
                "candidate-a",
                is_loaded=True,
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=20.0,
            ),
            "candidate-b": _create_runtime_state(
                "candidate-b",
                is_loaded=True,
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=15.0,
            ),
        }
    )

    candidates = engine.get_lru_unload_candidates(
        limit=2,
        exclude_aivm_model_uuids={"candidate-b"},
    )

    assert [candidate.model_uuid for candidate in candidates] == ["candidate-a"]


def test_evict_lru_models_unloads_only_excess_loaded_models() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "old-prefetched": _create_runtime_state(
                "old-prefetched",
                is_loaded=True,
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=None,
            ),
            "older-used": _create_runtime_state(
                "older-used",
                is_loaded=True,
                load_count=1,
                last_loaded_at=15.0,
                last_used_at=30.0,
            ),
            "recently-used": _create_runtime_state(
                "recently-used",
                is_loaded=True,
                load_count=2,
                last_loaded_at=20.0,
                last_used_at=50.0,
            ),
        }
    )

    def _fake_unload_model(aivm_model_uuid: str) -> None:
        runtime_state = engine._runtime_states[aivm_model_uuid]
        runtime_state.is_loaded = False
        runtime_state.residency = "unloaded"
        runtime_state.last_unloaded_at = 100.0

    engine.unload_model = _fake_unload_model  # type: ignore[method-assign]

    evicted_states = engine.evict_lru_models(max_loaded_models=1)

    assert [state.model_uuid for state in evicted_states] == [
        "old-prefetched",
        "older-used",
    ]
    assert all(state.is_loaded is False for state in evicted_states)
    assert engine._runtime_states["recently-used"].is_loaded is True


def test_evict_lru_models_respects_exclude_list() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "candidate-a": _create_runtime_state(
                "candidate-a",
                is_loaded=True,
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=20.0,
            ),
            "candidate-b": _create_runtime_state(
                "candidate-b",
                is_loaded=True,
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=15.0,
            ),
        }
    )

    def _fake_unload_model(aivm_model_uuid: str) -> None:
        runtime_state = engine._runtime_states[aivm_model_uuid]
        runtime_state.is_loaded = False
        runtime_state.residency = "unloaded"
        runtime_state.last_unloaded_at = 100.0

    engine.unload_model = _fake_unload_model  # type: ignore[method-assign]

    evicted_states = engine.evict_lru_models(
        max_loaded_models=0,
        exclude_aivm_model_uuids={"candidate-b"},
    )

    assert [state.model_uuid for state in evicted_states] == ["candidate-a"]
    assert engine._runtime_states["candidate-b"].is_loaded is True


def test_get_lru_unload_candidates_skips_pinned_models() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "pinned-model": _create_runtime_state(
                "pinned-model",
                is_loaded=True,
                is_pinned=True,
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=None,
            ),
            "unpinned-model": _create_runtime_state(
                "unpinned-model",
                is_loaded=True,
                load_count=1,
                last_loaded_at=15.0,
                last_used_at=None,
            ),
        }
    )

    candidates = engine.get_lru_unload_candidates(limit=2)

    assert [candidate.model_uuid for candidate in candidates] == ["unpinned-model"]


def test_pin_model_marks_runtime_state_as_pinned() -> None:
    engine = _create_engine_with_runtime_states({})

    runtime_state = engine.pin_model("model-a")

    assert runtime_state.is_pinned is True
    assert engine.get_model_runtime_state("model-a").is_pinned is True  # type: ignore[union-attr]


def test_unpin_model_marks_runtime_state_as_unpinned() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.pin_model("model-a")

    runtime_state = engine.unpin_model("model-a")

    assert runtime_state.is_pinned is False
    assert engine.get_model_runtime_state("model-a").is_pinned is False  # type: ignore[union-attr]


def test_set_runtime_policy_updates_max_loaded_models() -> None:
    engine = _create_engine_with_runtime_states({})

    runtime_policy = engine.set_runtime_policy(2, 1, 3.5, 1.25)

    assert runtime_policy.max_loaded_models == 2
    assert runtime_policy.max_vram_loaded_models == 1
    assert runtime_policy.min_available_ram_gb == 3.5
    assert runtime_policy.min_available_vram_gb == 1.25
    assert engine.get_runtime_policy().max_loaded_models == 2
    assert engine.get_runtime_policy().max_vram_loaded_models == 1
    assert engine.get_runtime_policy().min_available_ram_gb == 3.5
    assert engine.get_runtime_policy().min_available_vram_gb == 1.25


def test_apply_runtime_policy_uses_configured_limit() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.set_runtime_policy(1)
    captured: dict[str, object] = {}

    def _fake_evict_lru_models(
        max_loaded_models: int,
        exclude_aivm_model_uuids: set[str] | None = None,
    ) -> list[AivmModelRuntimeState]:
        captured["max_loaded_models"] = max_loaded_models
        captured["exclude_aivm_model_uuids"] = exclude_aivm_model_uuids
        return []

    engine.evict_lru_models = _fake_evict_lru_models  # type: ignore[method-assign]

    engine.apply_runtime_policy(exclude_aivm_model_uuids={"keep-me"})

    assert captured == {
        "max_loaded_models": 1,
        "exclude_aivm_model_uuids": {"keep-me"},
    }


def test_get_lru_demote_candidates_skips_ram_only_models() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "ram-only": _create_runtime_state(
                "ram-only",
                is_loaded=True,
                residency="ram",
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=20.0,
            ),
            "vram-model": _create_runtime_state(
                "vram-model",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=15.0,
            ),
        }
    )

    candidates = engine.get_lru_demote_candidates(limit=2)

    assert [candidate.model_uuid for candidate in candidates] == ["vram-model"]


def test_demote_lru_models_only_counts_vram_loaded_models() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "ram-only": _create_runtime_state(
                "ram-only",
                is_loaded=True,
                residency="ram",
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=20.0,
            ),
            "vram-a": _create_runtime_state(
                "vram-a",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=15.0,
            ),
            "vram-b": _create_runtime_state(
                "vram-b",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=6.0,
                last_used_at=16.0,
            ),
        }
    )
    demoted: list[str] = []

    def _fake_demote_model(aivm_model_uuid: str) -> AivmModelRuntimeState:
        demoted.append(aivm_model_uuid)
        runtime_state = engine._runtime_states[aivm_model_uuid]
        runtime_state.is_loaded_in_vram = False
        runtime_state.is_cached_in_ram = True
        runtime_state.is_loaded = True
        runtime_state.residency = "ram"
        return runtime_state

    engine.demote_model = _fake_demote_model  # type: ignore[method-assign]

    runtime_states = engine.demote_lru_models(max_vram_loaded_models=1)

    assert demoted == ["vram-a"]
    assert [runtime_state.model_uuid for runtime_state in runtime_states] == ["vram-a"]


def test_apply_runtime_policy_prefers_demote_policy_over_unload_policy() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.set_runtime_policy(1, 0)
    captured: dict[str, object] = {}

    def _fake_demote_lru_models(
        max_vram_loaded_models: int,
        exclude_aivm_model_uuids: set[str] | None = None,
    ) -> list[AivmModelRuntimeState]:
        captured["max_vram_loaded_models"] = max_vram_loaded_models
        captured["exclude_aivm_model_uuids"] = exclude_aivm_model_uuids
        return []

    engine.demote_lru_models = _fake_demote_lru_models  # type: ignore[method-assign]

    engine.apply_runtime_policy(exclude_aivm_model_uuids={"keep-me"})

    assert captured == {
        "max_vram_loaded_models": 0,
        "exclude_aivm_model_uuids": {"keep-me"},
    }


def test_demote_models_to_free_vram_uses_resource_threshold() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "vram-a": _create_runtime_state(
                "vram-a",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=10.0,
            )
        }
    )
    available_vram_gb = iter([0.5, 2.0])

    engine._get_available_vram_gb = lambda: next(available_vram_gb)  # type: ignore[method-assign]

    def _fake_demote_model(aivm_model_uuid: str) -> AivmModelRuntimeState:
        runtime_state = engine._runtime_states[aivm_model_uuid]
        runtime_state.is_loaded_in_vram = False
        runtime_state.is_cached_in_ram = True
        runtime_state.is_loaded = True
        runtime_state.residency = "ram"
        return runtime_state

    engine.demote_model = _fake_demote_model  # type: ignore[method-assign]

    runtime_states = engine.demote_models_to_free_vram(1.0)

    assert [runtime_state.model_uuid for runtime_state in runtime_states] == ["vram-a"]
    assert runtime_states[0].residency == "ram"


def test_evict_models_to_free_ram_uses_resource_threshold() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "ram-a": _create_runtime_state(
                "ram-a",
                is_loaded=True,
                residency="ram",
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=10.0,
            )
        }
    )
    available_ram_gb = iter([0.5, 2.0])

    engine._get_available_ram_gb = lambda: next(available_ram_gb)  # type: ignore[method-assign]

    def _fake_unload_model(aivm_model_uuid: str) -> None:
        runtime_state = engine._runtime_states[aivm_model_uuid]
        runtime_state.is_loaded = False
        runtime_state.is_cached_in_ram = False
        runtime_state.is_loaded_in_vram = False
        runtime_state.residency = "unloaded"
        runtime_state.last_unloaded_at = 100.0

    engine.unload_model = _fake_unload_model  # type: ignore[method-assign]

    runtime_states = engine.evict_models_to_free_ram(1.0)

    assert [runtime_state.model_uuid for runtime_state in runtime_states] == ["ram-a"]
    assert runtime_states[0].residency == "unloaded"


def test_apply_runtime_policy_combines_resource_thresholds() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.set_runtime_policy(None, None, 2.0, 1.0)
    captured: dict[str, object] = {}

    def _fake_demote_models_to_free_vram(
        min_available_vram_gb: float,
        exclude_aivm_model_uuids: set[str] | None = None,
    ) -> list[AivmModelRuntimeState]:
        captured["min_available_vram_gb"] = min_available_vram_gb
        captured["demote_exclude"] = exclude_aivm_model_uuids
        return []

    def _fake_evict_models_to_free_ram(
        min_available_ram_gb: float,
        exclude_aivm_model_uuids: set[str] | None = None,
    ) -> list[AivmModelRuntimeState]:
        captured["min_available_ram_gb"] = min_available_ram_gb
        captured["evict_exclude"] = exclude_aivm_model_uuids
        return []

    engine.demote_models_to_free_vram = _fake_demote_models_to_free_vram  # type: ignore[method-assign]
    engine.evict_models_to_free_ram = _fake_evict_models_to_free_ram  # type: ignore[method-assign]

    engine.apply_runtime_policy(exclude_aivm_model_uuids={"keep-me"})

    assert captured == {
        "min_available_vram_gb": 1.0,
        "demote_exclude": {"keep-me"},
        "min_available_ram_gb": 2.0,
        "evict_exclude": {"keep-me"},
    }


def test_get_runtime_resource_snapshot_returns_current_resources_and_estimates() -> (
    None
):
    engine = _create_engine_with_runtime_states(
        {
            "ram-a": _create_runtime_state(
                "ram-a",
                is_loaded=True,
                residency="ram",
                load_count=1,
                last_loaded_at=5.0,
                last_used_at=10.0,
            ),
            "vram-a": _create_runtime_state(
                "vram-a",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=6.0,
                last_used_at=11.0,
            ),
        }
    )
    engine._runtime_policy = AivmModelRuntimePolicy(min_available_ram_gb=2.0)
    engine._get_total_ram_gb = lambda: 16.0  # type: ignore[method-assign]
    engine._get_available_ram_gb = lambda: 8.0  # type: ignore[method-assign]
    engine._get_total_vram_gb = lambda: 12.0  # type: ignore[method-assign]
    engine._get_available_vram_gb = lambda: 6.0  # type: ignore[method-assign]
    engine.aivm_manager = cast(
        Any,
        type(
            "Manager",
            (),
            {
                "get_installed_aivm_infos": lambda self: {
                    "ram-a": object(),
                    "vram-a": object(),
                },
                "get_aivm_info": lambda self, aivm_model_uuid: AivmInfo.model_construct(  # type: ignore[call-arg]
                    file_size=1610612736 if aivm_model_uuid == "ram-a" else 2147483648,
                ),
            },
        )(),
    )

    snapshot = engine.get_runtime_resource_snapshot()

    assert snapshot.inference_device == "cpu"
    assert snapshot.total_ram_gb == 16.0
    assert snapshot.available_ram_gb == 8.0
    assert snapshot.total_vram_gb == 12.0
    assert snapshot.available_vram_gb == 6.0
    assert snapshot.loaded_model_count == 2
    assert snapshot.vram_loaded_model_count == 1
    assert snapshot.runtime_policy.min_available_ram_gb == 2.0
    assert [estimate.model_uuid for estimate in snapshot.model_resource_estimates] == [
        "ram-a",
        "vram-a",
    ]
    assert snapshot.model_resource_estimates[0].estimated_ram_cache_size_gb == 1.5
    assert snapshot.model_resource_estimates[1].estimated_vram_load_size_gb == 2.0


def test_inspect_model_admission_returns_ram_shortage_for_prefetch() -> None:
    engine = _create_engine_with_runtime_states({})
    engine._runtime_policy = AivmModelRuntimePolicy(min_available_ram_gb=2.0)
    engine._get_total_ram_gb = lambda: 16.0  # type: ignore[method-assign]
    engine._get_available_ram_gb = lambda: 2.5  # type: ignore[method-assign]
    engine._get_total_vram_gb = lambda: None  # type: ignore[method-assign]
    engine._get_available_vram_gb = lambda: None  # type: ignore[method-assign]

    admission = engine.inspect_model_admission("model-a", "prefetch")

    assert admission.operation == "prefetch"
    assert admission.can_admit is False
    assert admission.predicted_available_ram_gb == 1.5
    assert admission.ram_shortage_gb == 0.5
    assert admission.vram_shortage_gb is None


def test_inspect_model_admission_returns_vram_shortage_for_promote_on_gpu() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.onnx_providers = [("CUDAExecutionProvider", {"device_id": 0})]
    engine._runtime_policy = AivmModelRuntimePolicy(min_available_vram_gb=2.0)
    engine._get_total_ram_gb = lambda: 16.0  # type: ignore[method-assign]
    engine._get_available_ram_gb = lambda: 16.0  # type: ignore[method-assign]
    engine._get_total_vram_gb = lambda: 12.0  # type: ignore[method-assign]
    engine._get_available_vram_gb = lambda: 2.5  # type: ignore[method-assign]

    admission = engine.inspect_model_admission("model-a", "promote")

    assert admission.operation == "promote"
    assert admission.can_admit is False
    assert admission.predicted_available_vram_gb == 1.5
    assert admission.vram_shortage_gb == 0.5


def test_mark_model_loaded_uses_ram_residency_on_cpu() -> None:
    engine = _create_engine_with_runtime_states({})

    engine._mark_model_loaded("model-a")

    runtime_state = engine.get_model_runtime_state("model-a")
    assert runtime_state is not None
    assert runtime_state.is_cached_in_ram is True
    assert runtime_state.is_loaded_in_vram is False
    assert runtime_state.residency == "ram"
    assert runtime_state.inference_device == "cpu"


def test_mark_model_loaded_uses_vram_residency_on_gpu() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.onnx_providers = [("CUDAExecutionProvider", {"device_id": 0})]

    engine._mark_model_loaded("model-a")

    runtime_state = engine.get_model_runtime_state("model-a")
    assert runtime_state is not None
    assert runtime_state.is_cached_in_ram is True
    assert runtime_state.is_loaded_in_vram is True
    assert runtime_state.residency == "vram"
    assert runtime_state.inference_device == "gpu"


def test_prefetch_model_marks_ram_cache_without_vram_on_cpu() -> None:
    engine = _create_engine_with_runtime_states({})

    engine._get_or_prefetch_model_artifacts = lambda aivm_model_uuid: cast(
        Any, object()
    )  # type: ignore[method-assign]

    runtime_state = engine.prefetch_model("model-a")

    assert runtime_state.is_loaded is True
    assert runtime_state.is_cached_in_ram is True
    assert runtime_state.is_loaded_in_vram is False
    assert runtime_state.residency == "ram"


def test_prefetch_model_marks_ram_cache_without_vram_on_gpu() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.onnx_providers = [("CUDAExecutionProvider", {"device_id": 0})]

    engine._get_or_prefetch_model_artifacts = lambda aivm_model_uuid: cast(
        Any, object()
    )  # type: ignore[method-assign]

    runtime_state = engine.prefetch_model("model-a")

    assert runtime_state.is_loaded is True
    assert runtime_state.is_cached_in_ram is True
    assert runtime_state.is_loaded_in_vram is False
    assert runtime_state.residency == "ram"
    assert runtime_state.inference_device == "gpu"


def test_promote_model_returns_loaded_runtime_state() -> None:
    engine = _create_engine_with_runtime_states({})

    def _fake_load_model(aivm_model_uuid: str) -> object:
        engine._mark_model_loaded(aivm_model_uuid)
        return object()

    engine.load_model = _fake_load_model  # type: ignore[method-assign]

    runtime_state = engine.promote_model("model-a")

    assert runtime_state.model_uuid == "model-a"
    assert runtime_state.is_loaded is True
    assert runtime_state.is_cached_in_ram is True
    assert runtime_state.load_count == 1
    assert runtime_state.last_loaded_at is not None


def test_prefetch_model_raises_when_ram_policy_cannot_be_satisfied() -> None:
    engine = _create_engine_with_runtime_states({})
    engine._runtime_policy = AivmModelRuntimePolicy(min_available_ram_gb=2.0)
    engine._estimate_model_cache_size_gb = lambda aivm_model_uuid: 1.0  # type: ignore[method-assign]
    engine._get_available_ram_gb = lambda: 2.5  # type: ignore[method-assign]
    engine.apply_runtime_policy = lambda exclude_aivm_model_uuids=None: []  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc_info:
        engine.prefetch_model("model-a")
    assert exc_info.value.status_code == 507
    assert "Insufficient RAM capacity" in str(exc_info.value.detail)


def test_promote_model_raises_when_vram_policy_cannot_be_satisfied() -> None:
    engine = _create_engine_with_runtime_states({})
    engine.onnx_providers = [("CUDAExecutionProvider", {"device_id": 0})]
    engine._runtime_policy = AivmModelRuntimePolicy(min_available_vram_gb=1.5)
    engine._estimate_model_cache_size_gb = lambda aivm_model_uuid: 1.0  # type: ignore[method-assign]
    engine._get_available_ram_gb = lambda: 100.0  # type: ignore[method-assign]
    engine._get_available_vram_gb = lambda: 2.0  # type: ignore[method-assign]
    engine.apply_runtime_policy = lambda exclude_aivm_model_uuids=None: []  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc_info:
        engine.promote_model("model-a")
    assert exc_info.value.status_code == 507
    assert "Insufficient VRAM capacity" in str(exc_info.value.detail)


def test_demote_model_keeps_ram_cache_after_gpu_load() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "model-a": _create_runtime_state(
                "model-a",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=20.0,
            )
        }
    )
    engine.onnx_providers = [("CUDAExecutionProvider", {"device_id": 0})]
    engine.tts_models = cast(Any, {"model-a": object()})
    engine._prefetched_model_artifacts = cast(Any, {"model-a": object()})

    class _Unloadable:
        def __init__(self) -> None:
            self.unloaded = False

        def unload(self) -> None:
            self.unloaded = True

    unloadable = _Unloadable()
    engine.tts_models = {"model-a": cast(object, unloadable)}  # type: ignore[assignment]
    engine.aivm_manager = cast(
        Any,
        type(
            "Manager",
            (),
            {"update_model_load_state": lambda self, aivm_model_uuid, is_loaded: None},
        )(),
    )

    runtime_state = engine.demote_model("model-a")

    assert unloadable.unloaded is True
    assert runtime_state.is_loaded is True
    assert runtime_state.is_cached_in_ram is True
    assert runtime_state.is_loaded_in_vram is False
    assert runtime_state.residency == "ram"


def test_demote_model_without_ram_cache_becomes_unloaded() -> None:
    engine = _create_engine_with_runtime_states(
        {
            "model-a": _create_runtime_state(
                "model-a",
                is_loaded=True,
                residency="vram",
                load_count=1,
                last_loaded_at=10.0,
                last_used_at=20.0,
            )
        }
    )

    class _Unloadable:
        def unload(self) -> None:
            return None

    engine.tts_models = {"model-a": cast(object, _Unloadable())}  # type: ignore[assignment]
    engine.aivm_manager = cast(
        Any,
        type(
            "Manager",
            (),
            {"update_model_load_state": lambda self, aivm_model_uuid, is_loaded: None},
        )(),
    )

    runtime_state = engine.demote_model("model-a")

    assert runtime_state.is_loaded is False
    assert runtime_state.is_cached_in_ram is False
    assert runtime_state.is_loaded_in_vram is False
    assert runtime_state.residency == "unloaded"
