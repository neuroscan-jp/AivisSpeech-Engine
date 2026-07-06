"""音声合成モデル管理機能を提供する API Router"""

from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
)

from voicevox_engine.aivm_manager import AivmManager
from voicevox_engine.model import (
    AivmInfo,
    AivmModelAdmissionDecision,
    AivmModelRuntimePolicy,
    AivmModelRuntimeResourceSnapshot,
    AivmModelRuntimeState,
)
from voicevox_engine.tts_pipeline.tts_engine import LATEST_VERSION, TTSEngineManager

from ..dependencies import VerifyMutabilityAllowed


def generate_aivm_models_router(
    aivm_manager: AivmManager,
    tts_engines: TTSEngineManager,
    verify_mutability: VerifyMutabilityAllowed,
) -> APIRouter:
    """音声合成モデル管理 API Router を生成する"""

    # ごく稀に style_bert_vits2_tts_engine.py (が依存する onnxruntime) のインポート自体に失敗し
    # 例外が発生する環境があるようなので、例外をキャッチしてエラーログに出力できるよう、敢えてルーター初期化時にインポートする
    from voicevox_engine.tts_pipeline.style_bert_vits2_tts_engine import (
        StyleBertVITS2TTSEngine,
    )

    router = APIRouter(
        prefix="/aivm_models",
        tags=["音声合成モデル管理"],
    )

    @router.get(
        "/runtime",
        summary="音声合成モデルのランタイム状態を取得する",
        response_description="音声合成モデルのランタイム状態一覧",
    )
    def get_model_runtime_states() -> list[AivmModelRuntimeState]:
        """音声合成モデルのランタイム状態一覧を返します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.get_all_model_runtime_states()

    @router.get(
        "/runtime/demote_candidates",
        summary="LRU に基づく降格候補の音声合成モデル一覧を取得する",
        response_description="降格候補の音声合成モデル一覧",
    )
    def get_model_runtime_demote_candidates(
        limit: Annotated[
            int,
            Query(description="返却する候補数", ge=1, le=100),
        ] = 1,
        exclude_aivm_model_uuids: Annotated[
            list[str] | None,
            Query(description="候補から除外する音声合成モデルの UUID 一覧"),
        ] = None,
    ) -> list[AivmModelRuntimeState]:
        """LRU に基づく VRAM 降格候補の音声合成モデル一覧を返します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.get_lru_demote_candidates(
            limit=limit,
            exclude_aivm_model_uuids=set(exclude_aivm_model_uuids or []),
        )

    @router.get(
        "/runtime/unload_candidates",
        summary="LRU に基づくアンロード候補の音声合成モデル一覧を取得する",
        response_description="アンロード候補の音声合成モデル一覧",
    )
    def get_model_runtime_unload_candidates(
        limit: Annotated[
            int,
            Query(description="返却する候補数", ge=1, le=100),
        ] = 1,
        exclude_aivm_model_uuids: Annotated[
            list[str] | None,
            Query(description="候補から除外する音声合成モデルの UUID 一覧"),
        ] = None,
    ) -> list[AivmModelRuntimeState]:
        """LRU に基づくアンロード候補の音声合成モデル一覧を返します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.get_lru_unload_candidates(
            limit=limit,
            exclude_aivm_model_uuids=set(exclude_aivm_model_uuids or []),
        )

    @router.get(
        "/runtime/policy",
        summary="音声合成モデルのランタイム運用ポリシーを取得する",
        response_description="音声合成モデルのランタイム運用ポリシー",
    )
    def get_model_runtime_policy() -> AivmModelRuntimePolicy:
        """音声合成モデルのランタイム運用ポリシーを返します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.get_runtime_policy()

    @router.get(
        "/runtime/resources",
        summary="音声合成モデル運用の現在のリソース状況を取得する",
        response_description="音声合成モデル運用の現在のリソース状況",
    )
    def get_model_runtime_resources() -> AivmModelRuntimeResourceSnapshot:
        """音声合成モデル運用の現在のリソース状況を返します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.get_runtime_resource_snapshot()

    @router.get(
        "/{aivm_model_uuid}/admission",
        summary="指定された音声合成モデル操作の dry-run admission 判定を取得する",
        response_description="指定された音声合成モデル操作の dry-run admission 判定結果",
    )
    def get_model_admission_decision(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
        operation: Annotated[
            Literal["prefetch", "promote"],
            Query(description="dry-run 判定対象の操作"),
        ] = "prefetch",
    ) -> AivmModelAdmissionDecision:
        """指定された音声合成モデル操作の dry-run admission 判定結果を返します。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.inspect_model_admission(str(aivm_info.manifest.uuid), operation)

    @router.put(
        "/runtime/policy",
        summary="音声合成モデルのランタイム運用ポリシーを更新する",
        response_description="更新後の音声合成モデルのランタイム運用ポリシー",
    )
    def update_model_runtime_policy(
        runtime_policy: AivmModelRuntimePolicy,
    ) -> AivmModelRuntimePolicy:
        """音声合成モデルのランタイム運用ポリシーを更新します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.set_runtime_policy(
            runtime_policy.max_loaded_models,
            runtime_policy.max_vram_loaded_models,
            runtime_policy.min_available_ram_gb,
            runtime_policy.min_available_vram_gb,
        )

    @router.post(
        "/runtime/demote",
        summary="LRU に基づいて音声合成モデルを RAM キャッシュへ降格する",
        response_description="降格された音声合成モデル一覧",
    )
    def demote_model_runtime_states(
        max_vram_loaded_models: Annotated[
            int,
            Query(
                description="降格後に維持したい最大 VRAM ロード済みモデル数",
                ge=0,
                le=100,
            ),
        ],
        exclude_aivm_model_uuids: Annotated[
            list[str] | None,
            Query(description="降格対象から除外する音声合成モデルの UUID 一覧"),
        ] = None,
    ) -> list[AivmModelRuntimeState]:
        """LRU に基づいて、指定上限を超えた VRAM ロード済みモデルを RAM キャッシュへ降格します。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.demote_lru_models(
            max_vram_loaded_models=max_vram_loaded_models,
            exclude_aivm_model_uuids=set(exclude_aivm_model_uuids or []),
        )

    @router.post(
        "/runtime/evict",
        summary="LRU に基づいて音声合成モデルをアンロードする",
        response_description="アンロードされた音声合成モデル一覧",
    )
    def evict_model_runtime_states(
        max_loaded_models: Annotated[
            int,
            Query(
                description="アンロード後に維持したい最大ロード済みモデル数",
                ge=0,
                le=100,
            ),
        ],
        exclude_aivm_model_uuids: Annotated[
            list[str] | None,
            Query(description="アンロード対象から除外する音声合成モデルの UUID 一覧"),
        ] = None,
    ) -> list[AivmModelRuntimeState]:
        """LRU に基づいて、指定上限を超えたロード済みモデルをアンロードします。"""

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.evict_lru_models(
            max_loaded_models=max_loaded_models,
            exclude_aivm_model_uuids=set(exclude_aivm_model_uuids or []),
        )

    @router.get(
        "",
        summary="インストール済みのすべての音声合成モデルの情報を取得する",
        response_description="インストール済みのすべての音声合成モデルの情報",
    )
    def get_installed_aivm_infos() -> dict[str, AivmInfo]:
        """
        インストール済みのすべての音声合成モデルの情報を返します。
        """

        return aivm_manager.get_installed_aivm_infos()

    @router.post(
        "/install",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="音声合成モデルをインストールする",
    )
    def install_model(
        file: Annotated[
            UploadFile | None,
            File(description="AIVMX ファイル (`.aivmx`)"),
        ] = None,
        url: Annotated[
            str | None,
            Form(description="AIVMX ファイルの URL"),
        ] = None,
    ) -> None:
        """
        音声合成モデルをインストールします。<br>
        ファイルからインストールする場合は `file` を指定してください。<br>
        URL からインストールする場合は `url` を指定してください。
        """

        if file is not None:
            # ファイルから音声合成モデルをインストール
            aivm_manager.install_model(file.file)
        elif url is not None:
            # URL から音声合成モデルをダウンロードしてインストール
            aivm_manager.install_model_from_url(url)
        else:
            raise HTTPException(
                status_code=422,
                detail="Either file or url must be provided.",
            )

    @router.get(
        "/{aivm_model_uuid}/runtime",
        summary="指定された音声合成モデルのランタイム状態を取得する",
        response_description="指定された音声合成モデルのランタイム状態",
    )
    def get_model_runtime_state(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmModelRuntimeState:
        """指定された音声合成モデルのランタイム状態を返します。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        runtime_state = engine.get_model_runtime_state(str(aivm_info.manifest.uuid))
        if runtime_state is None:
            return AivmModelRuntimeState(
                model_uuid=str(aivm_info.manifest.uuid),
                is_loaded=False,
                is_cached_in_ram=False,
                is_loaded_in_vram=False,
                is_pinned=False,
                residency="unloaded",
                load_count=0,
                inference_device="gpu" if engine.use_gpu else "cpu",
                onnx_providers=[],
                last_loaded_at=None,
                last_used_at=None,
                last_unloaded_at=None,
            )
        return runtime_state

    @router.get(
        "/{aivm_model_uuid}",
        summary="指定された音声合成モデルの情報を取得する",
        response_description="指定された音声合成モデルの情報",
    )
    def get_aivm_info(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmInfo:
        """
        指定された音声合成モデルの情報を取得します。
        """

        return aivm_manager.get_aivm_info(aivm_model_uuid)

    @router.post(
        "/{aivm_model_uuid}/load",
        status_code=204,
        summary="指定された音声合成モデルをロードする",
    )
    def load_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        指定された音声合成モデルをロードします。すでにロード済みの場合は何も行われません。<br>
        実行しなくても他の API は利用できますが、音声合成の初回実行時に時間がかかることがあります。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # StyleBertVITS2TTSEngine を取得し、音声合成モデルをロード
        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        engine.load_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/promote",
        summary="指定された音声合成モデルを推論ロード状態へ昇格する",
        response_description="昇格後の音声合成モデルのランタイム状態",
    )
    def promote_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmModelRuntimeState:
        """指定された音声合成モデルを RAM キャッシュ状態から推論ロード状態へ昇格します。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.promote_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/prefetch",
        summary="指定された音声合成モデルを先読みする",
        response_description="先読み後の音声合成モデルのランタイム状態",
    )
    def prefetch_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmModelRuntimeState:
        """指定された音声合成モデルを RAM キャッシュへ先読みします。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.prefetch_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/pin",
        summary="指定された音声合成モデルを eviction 対象から保護する",
        response_description="更新後の音声合成モデルのランタイム状態",
    )
    def pin_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmModelRuntimeState:
        """指定された音声合成モデルを pin し、eviction 対象から保護します。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.pin_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/unpin",
        summary="指定された音声合成モデルの pin を解除する",
        response_description="更新後の音声合成モデルのランタイム状態",
    )
    def unpin_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmModelRuntimeState:
        """指定された音声合成モデルの pin を解除します。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.unpin_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/unload",
        status_code=204,
        summary="指定された音声合成モデルをアンロードする",
    )
    def unload_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        指定された音声合成モデルをアンロードします。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # StyleBertVITS2TTSEngine を取得し、音声合成モデルをアンロード
        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        engine.unload_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/demote",
        summary="指定された音声合成モデルを RAM キャッシュへ降格する",
        response_description="降格後の音声合成モデルのランタイム状態",
    )
    def demote_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> AivmModelRuntimeState:
        """指定された音声合成モデルを推論ロード状態から RAM キャッシュ状態へ降格します。"""

        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        return engine.demote_model(str(aivm_info.manifest.uuid))

    @router.post(
        "/{aivm_model_uuid}/update",
        status_code=204,
        summary="指定された音声合成モデルを更新する",
    )
    def update_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        AivisHub から指定された音声合成モデルの一番新しいバージョンをダウンロードし、
        インストール済みの音声合成モデルへ上書き更新します。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # AivisHub からダウンロードした新しいバージョンの音声合成モデルを上書きインストール
        aivm_manager.update_model(str(aivm_info.manifest.uuid))

    @router.delete(
        "/{aivm_model_uuid}/uninstall",
        status_code=204,
        dependencies=[Depends(verify_mutability)],
        summary="指定された音声合成モデルをアンインストールする",
    )
    def uninstall_model(
        aivm_model_uuid: Annotated[
            str, Path(description="AIVM マニフェスト記載の音声合成モデルの UUID")
        ],
    ) -> None:
        """
        指定された音声合成モデルをアンインストールします。
        """

        # まず対応する音声合成モデルがインストールされているかを確認
        # 存在しない場合は内部で HTTPException が送出される
        aivm_info = aivm_manager.get_aivm_info(aivm_model_uuid)

        # StyleBertVITS2TTSEngine を取得し、音声合成モデルをアンロード
        # アンインストール前にアンロードしておかないと、既にロードされている場合に
        # プロセス終了までモデルがアンロードされなくなってしまう
        engine = tts_engines.get_tts_engine(LATEST_VERSION)
        assert isinstance(engine, StyleBertVITS2TTSEngine)
        engine.unload_model(str(aivm_info.manifest.uuid))

        # インストール済みの音声合成モデルをアンインストール
        aivm_manager.uninstall_model(str(aivm_info.manifest.uuid))

    return router
