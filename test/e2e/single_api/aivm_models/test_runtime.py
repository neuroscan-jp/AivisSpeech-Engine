"""/aivm_models runtime API のテスト。"""

from fastapi.testclient import TestClient

from test.e2e.single_api.utils import gen_mora


def _get_first_aivm_model_uuid(client: TestClient) -> str:
    response = client.get("/aivm_models")
    assert response.status_code == 200
    installed_models = response.json()
    assert len(installed_models) > 0
    return next(iter(installed_models.keys()))


def test_get_model_runtime_state_200_for_unloaded_model(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    response = client.get(f"/aivm_models/{aivm_model_uuid}/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_uuid"] == aivm_model_uuid
    assert payload["is_loaded"] is False
    assert payload["is_cached_in_ram"] is False
    assert payload["is_loaded_in_vram"] is False
    assert payload["is_pinned"] is False
    assert payload["residency"] == "unloaded"
    assert payload["load_count"] == 0
    assert payload["inference_device"] == "cpu"
    assert payload["onnx_providers"] == []
    assert payload["last_loaded_at"] is None
    assert payload["last_used_at"] is None
    assert payload["last_unloaded_at"] is None


def test_post_prefetch_model_200_updates_runtime_state(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")

    assert prefetch_response.status_code == 200
    payload = prefetch_response.json()
    assert payload["model_uuid"] == aivm_model_uuid
    assert payload["is_loaded"] is True
    assert payload["is_cached_in_ram"] is True
    assert payload["is_loaded_in_vram"] is False
    assert payload["is_pinned"] is False
    assert payload["residency"] == "ram"
    assert payload["load_count"] == 0
    assert payload["inference_device"] == "cpu"
    assert payload["onnx_providers"]
    assert payload["last_loaded_at"] is None

    runtime_list_response = client.get("/aivm_models/runtime")
    assert runtime_list_response.status_code == 200
    runtime_list = runtime_list_response.json()
    assert len(runtime_list) >= 1
    assert runtime_list[0]["model_uuid"] == aivm_model_uuid
    assert runtime_list[0]["is_loaded"] is True


def test_get_runtime_resources_200_returns_current_snapshot(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    response = client.get("/aivm_models/runtime/resources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_device"] == "cpu"
    assert payload["total_ram_gb"] >= payload["available_ram_gb"]
    assert payload["total_vram_gb"] is None
    assert payload["available_vram_gb"] is None
    assert payload["loaded_model_count"] >= 0
    assert payload["vram_loaded_model_count"] == 0
    assert "runtime_policy" in payload
    estimates = payload["model_resource_estimates"]
    assert len(estimates) >= 1
    target_estimate = next(
        estimate for estimate in estimates if estimate["model_uuid"] == aivm_model_uuid
    )
    assert target_estimate["estimated_ram_cache_size_gb"] > 0
    assert target_estimate["estimated_vram_load_size_gb"] > 0


def test_get_model_admission_200_returns_dry_run_decision(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    response = client.get(
        f"/aivm_models/{aivm_model_uuid}/admission",
        params={"operation": "prefetch"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_uuid"] == aivm_model_uuid
    assert payload["operation"] == "prefetch"
    assert isinstance(payload["can_admit"], bool)
    assert payload["estimated_ram_cache_size_gb"] > 0
    assert payload["predicted_available_ram_gb"] >= 0
    assert payload["runtime_resources"]["inference_device"] == "cpu"


def test_get_model_admission_200_reports_ram_shortage(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    update_response = client.put(
        "/aivm_models/runtime/policy",
        json={
            "max_loaded_models": None,
            "max_vram_loaded_models": None,
            "min_available_ram_gb": 1000000.0,
            "min_available_vram_gb": None,
        },
    )
    assert update_response.status_code == 200

    response = client.get(
        f"/aivm_models/{aivm_model_uuid}/admission",
        params={"operation": "prefetch"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["can_admit"] is False
    assert payload["ram_shortage_gb"] > 0
    assert payload["required_min_available_ram_gb"] == 1000000.0


def test_post_promote_model_200_loads_prefetched_model(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200
    assert prefetch_response.json()["load_count"] == 0

    promote_response = client.post(f"/aivm_models/{aivm_model_uuid}/promote")

    assert promote_response.status_code == 200
    payload = promote_response.json()
    assert payload["model_uuid"] == aivm_model_uuid
    assert payload["is_loaded"] is True
    assert payload["is_cached_in_ram"] is True
    assert payload["is_loaded_in_vram"] is False
    assert payload["residency"] == "ram"
    assert payload["load_count"] == 1
    assert payload["last_loaded_at"] is not None


def test_post_unload_model_204_updates_runtime_state(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    unload_response = client.post(f"/aivm_models/{aivm_model_uuid}/unload")
    assert unload_response.status_code == 204

    runtime_response = client.get(f"/aivm_models/{aivm_model_uuid}/runtime")
    assert runtime_response.status_code == 200
    payload = runtime_response.json()
    assert payload["model_uuid"] == aivm_model_uuid
    assert payload["is_loaded"] is False
    assert payload["is_cached_in_ram"] is False
    assert payload["is_loaded_in_vram"] is False
    assert payload["is_pinned"] is False
    assert payload["residency"] == "unloaded"
    assert payload["load_count"] == 0
    assert payload["last_loaded_at"] is None
    assert payload["last_unloaded_at"] is not None


def test_get_runtime_unload_candidates_200(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    response = client.get("/aivm_models/runtime/unload_candidates", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["model_uuid"] == aivm_model_uuid
    assert payload[0]["is_loaded"] is True


def test_post_pin_model_200_and_candidates_skip_pinned_model(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    pin_response = client.post(f"/aivm_models/{aivm_model_uuid}/pin")
    assert pin_response.status_code == 200
    assert pin_response.json()["is_pinned"] is True

    candidates_response = client.get("/aivm_models/runtime/unload_candidates", params={"limit": 1})
    assert candidates_response.status_code == 200
    assert candidates_response.json() == []


def test_post_unpin_model_200_restores_candidate_selection(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    pin_response = client.post(f"/aivm_models/{aivm_model_uuid}/pin")
    assert pin_response.status_code == 200

    unpin_response = client.post(f"/aivm_models/{aivm_model_uuid}/unpin")
    assert unpin_response.status_code == 200
    assert unpin_response.json()["is_pinned"] is False

    candidates_response = client.get("/aivm_models/runtime/unload_candidates", params={"limit": 1})
    assert candidates_response.status_code == 200
    payload = candidates_response.json()
    assert len(payload) == 1
    assert payload[0]["model_uuid"] == aivm_model_uuid


def test_post_runtime_evict_200_skips_pinned_model(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    pin_response = client.post(f"/aivm_models/{aivm_model_uuid}/pin")
    assert pin_response.status_code == 200

    evict_response = client.post("/aivm_models/runtime/evict", params={"max_loaded_models": 0})
    assert evict_response.status_code == 200
    assert evict_response.json() == []

    runtime_response = client.get(f"/aivm_models/{aivm_model_uuid}/runtime")
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["is_loaded"] is True
    assert runtime_payload["is_pinned"] is True


def test_get_runtime_demote_candidates_200_returns_empty_on_cpu(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    response = client.get("/aivm_models/runtime/demote_candidates", params={"limit": 1})

    assert response.status_code == 200
    assert response.json() == []


def test_post_runtime_evict_200_unloads_excess_models(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    evict_response = client.post("/aivm_models/runtime/evict", params={"max_loaded_models": 0})

    assert evict_response.status_code == 200
    payload = evict_response.json()
    assert len(payload) == 1
    assert payload[0]["model_uuid"] == aivm_model_uuid
    assert payload[0]["is_loaded"] is False
    assert payload[0]["residency"] == "unloaded"

    runtime_response = client.get(f"/aivm_models/{aivm_model_uuid}/runtime")
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["is_loaded"] is False


def test_post_demote_model_200_keeps_ram_cache(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    prefetch_response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")
    assert prefetch_response.status_code == 200

    demote_response = client.post(f"/aivm_models/{aivm_model_uuid}/demote")

    assert demote_response.status_code == 200
    payload = demote_response.json()
    assert payload["model_uuid"] == aivm_model_uuid
    assert payload["is_loaded"] is True
    assert payload["is_cached_in_ram"] is True
    assert payload["is_loaded_in_vram"] is False
    assert payload["residency"] == "ram"
    assert payload["last_unloaded_at"] is not None


def test_put_runtime_policy_200_and_auto_evict_after_synthesis(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    update_response = client.put(
        "/aivm_models/runtime/policy",
        json={
            "max_loaded_models": 0,
            "max_vram_loaded_models": None,
            "min_available_ram_gb": None,
            "min_available_vram_gb": None,
        },
    )
    assert update_response.status_code == 200
    assert update_response.json() == {
        "max_loaded_models": 0,
        "max_vram_loaded_models": None,
        "min_available_ram_gb": None,
        "min_available_vram_gb": None,
    }

    query = {
        "accent_phrases": [
            {
                "moras": [
                    gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                    gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                    gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                ],
                "accent": 1,
                "pause_mora": None,
                "is_interrogative": False,
            }
        ],
        "speedScale": 1.0,
        "pitchScale": 1.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "pauseLength": None,
        "pauseLengthScale": 1.0,
        "outputSamplingRate": 44100,
        "outputStereo": False,
        "kana": "テスト",
    }
    synthesis_response = client.post(
        "/synthesis",
        params={"speaker": 888753760},
        json=query,
    )
    assert synthesis_response.status_code == 200

    runtime_policy_response = client.get("/aivm_models/runtime/policy")
    assert runtime_policy_response.status_code == 200
    assert runtime_policy_response.json() == {
        "max_loaded_models": 0,
        "max_vram_loaded_models": None,
        "min_available_ram_gb": None,
        "min_available_vram_gb": None,
    }

    runtime_response = client.get(f"/aivm_models/{aivm_model_uuid}/runtime")
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    assert runtime_payload["is_loaded"] is False
    assert runtime_payload["residency"] == "unloaded"


def test_put_runtime_policy_200_and_auto_demote_after_synthesis_keeps_ram_on_cpu(client: TestClient) -> None:
    update_response = client.put(
        "/aivm_models/runtime/policy",
        json={
            "max_loaded_models": None,
            "max_vram_loaded_models": 0,
            "min_available_ram_gb": None,
            "min_available_vram_gb": None,
        },
    )
    assert update_response.status_code == 200
    assert update_response.json() == {
        "max_loaded_models": None,
        "max_vram_loaded_models": 0,
        "min_available_ram_gb": None,
        "min_available_vram_gb": None,
    }

    query = {
        "accent_phrases": [
            {
                "moras": [
                    gen_mora("テ", "t", 0.0, "e", 0.0, 0.0),
                    gen_mora("ス", "s", 0.0, "U", 0.0, 0.0),
                    gen_mora("ト", "t", 0.0, "o", 0.0, 0.0),
                ],
                "accent": 1,
                "pause_mora": None,
                "is_interrogative": False,
            }
        ],
        "speedScale": 1.0,
        "pitchScale": 1.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "pauseLength": None,
        "pauseLengthScale": 1.0,
        "outputSamplingRate": 44100,
        "outputStereo": False,
        "kana": "テスト",
    }
    synthesis_response = client.post(
        "/synthesis",
        params={"speaker": 888753760},
        json=query,
    )
    assert synthesis_response.status_code == 200

    runtime_response = client.get("/aivm_models/runtime")
    assert runtime_response.status_code == 200
    runtime_payload = runtime_response.json()
    loaded_models = [state for state in runtime_payload if state["is_loaded"] is True]
    assert len(loaded_models) == 1
    assert loaded_models[0]["is_cached_in_ram"] is True
    assert loaded_models[0]["is_loaded_in_vram"] is False
    assert loaded_models[0]["residency"] == "ram"


def test_put_runtime_policy_200_accepts_resource_thresholds(client: TestClient) -> None:
    update_response = client.put(
        "/aivm_models/runtime/policy",
        json={
            "max_loaded_models": None,
            "max_vram_loaded_models": None,
            "min_available_ram_gb": 2.5,
            "min_available_vram_gb": 1.0,
        },
    )

    assert update_response.status_code == 200
    assert update_response.json() == {
        "max_loaded_models": None,
        "max_vram_loaded_models": None,
        "min_available_ram_gb": 2.5,
        "min_available_vram_gb": 1.0,
    }


def test_post_prefetch_model_507_when_ram_admission_control_rejects(client: TestClient) -> None:
    aivm_model_uuid = _get_first_aivm_model_uuid(client)

    update_response = client.put(
        "/aivm_models/runtime/policy",
        json={
            "max_loaded_models": None,
            "max_vram_loaded_models": None,
            "min_available_ram_gb": 1000000.0,
            "min_available_vram_gb": None,
        },
    )
    assert update_response.status_code == 200

    response = client.post(f"/aivm_models/{aivm_model_uuid}/prefetch")

    assert response.status_code == 507
    assert response.json()["detail"] == (
        "Insufficient RAM capacity to cache this model while satisfying the current runtime policy."
    )
