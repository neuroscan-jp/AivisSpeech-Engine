"""
API と ENGINE 内部実装が共有するモデル

このモジュールで定義されるモデル（データ構造）は API と ENGINE の 2 箇所から使われる。そのため
- モデルの変更は API 変更となるため慎重に検討する。
- モデルの docstring や Field は API スキーマとして使われるため、ユーザー向けに丁寧に書く。
- モデルクラスは FastAPI の制約から `BaseModel` を継承しなければならない。
"""

from pathlib import Path
from typing import Literal

from aivmlib.schemas.aivm_manifest import AivmManifest
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from voicevox_engine.library.model import LibrarySpeaker
from voicevox_engine.tts_pipeline.model import AccentPhrase


class AudioQuery(BaseModel):
    """音声合成用のクエリ。"""

    accent_phrases: list[AccentPhrase] = Field(description="アクセント句のリスト。")
    speedScale: float = Field(
        description=(
            "全体の話速を 0.5 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "2.0 で 2 倍速、0.5 で 0.5 倍速になる。"
        ),
    )
    intonationScale: float = Field(
        description=(
            "選択した話者スタイルの感情表現の強弱を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "「全体の抑揚」ではない点で VOICEVOX ENGINE と仕様が異なる。\n"
            "数値が大きいほど、選択した話者スタイルに近い感情表現が込められた声になる。\n"
            "例えば話者スタイルが「上機嫌」なら、数値が大きいほどより嬉しそうな明るい話し方になる。\n"
            "一方で、話者やスタイルによっては、数値を上げすぎると発声がおかしくなったり、棒読みで不自然な声になる場合もある。\n"
            "正しく発声できる上限値は話者やスタイルごとに異なる。必要に応じて最適な値を見つけて調整すること。\n"
            "全スタイルの平均であるノーマルスタイルでは自動で適切な感情表現が選択されるため、この値を指定しても無視される。"
        ),
    )
    tempoDynamicsScale: float = Field(
        default=1.0,
        description=(
            "話す速さ（テンポ）の緩急の強弱を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "AivisSpeech Engine 固有のフィールドで、VOICEVOX ENGINE には存在しない。\n"
            "数値が大きいほどより早口で生っぽい抑揚がついた声になる。\n"
            "VOICEVOX ENGINE との互換性のため、未指定時はデフォルト値が適用される。"
        ),
    )
    pitchScale: float = Field(
        description=(
            "全体の音高を -0.15 ~ 0.15 の範囲で指定する (デフォルト: 0.0) 。\n"
            "数値が大きいほど高い声になる。\n"
            "VOICEVOX ENGINE と異なり、この値を 0.0 から変更すると音質が劣化するため注意が必要。"
        ),
    )
    volumeScale: float = Field(
        description=(
            "全体の音量を 0.0 ~ 2.0 の範囲で指定する (デフォルト: 1.0) 。\n"
            "数値が大きいほど大きな声になる。"
        ),
    )
    prePhonemeLength: float = Field(description="音声の前の無音時間 (秒)。")
    postPhonemeLength: float = Field(description="音声の後の無音時間 (秒)。")
    pauseLength: float | None = Field(
        default=None,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間。null のときは無視される。デフォルト値は null 。",
    )
    pauseLengthScale: float = Field(
        default=1,
        title="AivisSpeech Engine ではサポートされていないフィールドです (常に無視されます)",
        description="句読点などの無音時間（倍率）。デフォルト値は 1 。",
    )
    outputSamplingRate: int = Field(description="音声データの出力サンプリングレート。")
    outputStereo: bool = Field(description="音声データをステレオ出力するか否か。")
    kana: str | SkipJsonSchema[None] = Field(
        default=None,
        description=(
            "読み上げるテキストを指定する。「読みの AquesTalk 風記法テキスト」ではない点で VOICEVOX ENGINE と仕様が異なる。\n"
            "VOICEVOX ENGINE では AquesTalk 風記法テキストが入る読み取り専用フィールドだが (音声合成時には無視される) 、"
            "AivisSpeech Engine では音声合成時に漢字や記号が含まれた通常の読み上げテキストも必要なため、"
            "苦肉の策で読み上げテキスト指定用のフィールドとして転用した。\n"
            "VOICEVOX ENGINE との互換性のため None や空文字列が指定された場合も動作するが、"
            "その場合はアクセント句から自動生成されたひらがな文字列が読み上げテキストになるため、不自然なイントネーションになってしまう。\n"
            "可能な限り kana に通常の読み上げテキストを指定した上で音声合成 API に渡すことを推奨する。"
        ),
    )

    def __hash__(self) -> int:
        """内容に対して一意なハッシュ値を返す。"""
        # NOTE: lru_cache がユースケースのひとつ
        items = [
            (k, tuple(v)) if isinstance(v, list) else (k, v)
            for k, v in self.__dict__.items()
        ]
        return hash(tuple(sorted(items)))


class AivmInfo(BaseModel):
    """
    AIVM (Aivis Voice Model) 仕様に準拠した音声合成モデルのメタデータ情報。

    AIVM マニフェストには、音声合成モデルに関連する全てのメタデータが含まれる。
    speakers フィールド内の話者情報は、VOICEVOX ENGINE との API 互換性のために、
    AIVM マニフェストを基に Speaker / SpeakerStyle / SpeakerInfo / StyleInfo モデルに変換したもの。
    """

    is_loaded: bool = Field(description="この音声合成モデルがロードされているかどうか")
    is_update_available: bool = Field(
        description="この音声合成モデルの新しいバージョンが AivisHub で公開されているかどうか"
    )
    is_private_model: bool = Field(
        description="AivisHub で公開されておらず、ユーザーがローカルからインストールしたモデルの場合は True (ネットワークエラーなどで AivisHub から情報を取得できなかった場合も True を返す)",
    )
    is_default_model: bool = Field(
        description="AivisHub がデフォルトインストール対象として指定した音声合成モデルかどうか",
        default=False,  # 旧バージョンで生成されたモデル情報キャッシュの読み込みエラー回避のためデフォルト値を指定
    )
    latest_version: str = Field(
        description="この音声合成モデルの AivisHub で公開されている最新バージョン (AivisHub で公開されていない場合は AIVM マニフェスト記載のバージョン)"
    )
    file_path: Path = Field(description="AIVMX ファイルのインストール先パス")
    file_size: int = Field(
        description="AIVMX ファイルのインストールサイズ (バイト単位)"
    )
    manifest: AivmManifest = Field(description="AIVM マニフェスト")
    speakers: list[LibrarySpeaker] = Field(
        description="話者情報のリスト (VOICEVOX ENGINE 互換)"
    )


class AivmModelRuntimeState(BaseModel):
    """音声合成モデルのプロセス内ランタイム状態。"""

    model_uuid: str = Field(description="AIVM マニフェスト記載の音声合成モデルの UUID")
    is_loaded: bool = Field(description="この音声合成モデルが現在ロード済みかどうか")
    is_cached_in_ram: bool = Field(
        description="この音声合成モデルが現在 RAM にキャッシュされているかどうか"
    )
    is_loaded_in_vram: bool = Field(
        description="この音声合成モデルが現在 VRAM にロードされているかどうか"
    )
    is_pinned: bool = Field(description="この音声合成モデルが eviction 対象から保護されているかどうか")
    residency: Literal["unloaded", "ram", "vram"] = Field(
        description="この音声合成モデルの現在の常駐状態"
    )
    load_count: int = Field(description="このプロセスでモデルをロードした回数")
    inference_device: Literal["cpu", "gpu"] = Field(
        description="この音声合成モデルがロードされた推論デバイス"
    )
    onnx_providers: list[str] = Field(
        description="この音声合成モデルのロード時に使用した ONNX Runtime Provider 一覧"
    )
    last_loaded_at: float | None = Field(
        description="最後にロードされた時刻 (Unix time)。未ロードの場合は null"
    )
    last_used_at: float | None = Field(
        description="最後に推論で使用された時刻 (Unix time)。未使用の場合は null"
    )
    last_unloaded_at: float | None = Field(
        description="最後にアンロードされた時刻 (Unix time)。未アンロードの場合は null"
    )


class AivmModelRuntimePolicy(BaseModel):
    """音声合成モデルのプロセス内ランタイム運用ポリシー。"""

    max_loaded_models: int | None = Field(
        default=None,
        description="自動 eviction 後に維持したい最大ロード済みモデル数。null の場合は自動 eviction を無効化する"
    )
    max_vram_loaded_models: int | None = Field(
        default=None,
        description="自動 demote 後に維持したい最大 VRAM ロード済みモデル数。null の場合は自動 demote を無効化する"
    )
    min_available_ram_gb: float | None = Field(
        default=None,
        description="自動 eviction 後に確保したい最小空き RAM 容量 (GB)。null の場合は RAM 残量ベースの自動 eviction を無効化する"
    )
    min_available_vram_gb: float | None = Field(
        default=None,
        description="自動 demote 後に確保したい最小空き VRAM 容量 (GB)。null の場合は VRAM 残量ベースの自動 demote を無効化する"
    )


class AivmModelResourceEstimate(BaseModel):
    """音声合成モデルごとの推定リソース使用量。"""

    model_uuid: str = Field(description="AIVM マニフェスト記載の音声合成モデルの UUID")
    estimated_ram_cache_size_gb: float = Field(
        description="この音声合成モデルを RAM キャッシュした場合の推定使用量 (GB)"
    )
    estimated_vram_load_size_gb: float = Field(
        description="この音声合成モデルを VRAM にロードした場合の推定使用量 (GB)"
    )


class AivmModelRuntimeResourceSnapshot(BaseModel):
    """音声合成モデル運用に関する現在のリソース状況。"""

    inference_device: Literal["cpu", "gpu"] = Field(description="現在の推論デバイス")
    total_ram_gb: float = Field(description="ホストで利用可能な総 RAM 容量 (GB)")
    available_ram_gb: float = Field(description="現在の空き RAM 容量 (GB)")
    total_vram_gb: float | None = Field(
        description="推論に使用する GPU の総 VRAM 容量 (GB)。取得できない場合は null"
    )
    available_vram_gb: float | None = Field(
        description="推論に使用する GPU の空き VRAM 容量 (GB)。取得できない場合は null"
    )
    loaded_model_count: int = Field(description="現在ロード済みの音声合成モデル数")
    vram_loaded_model_count: int = Field(description="現在 VRAM にロード済みの音声合成モデル数")
    runtime_policy: AivmModelRuntimePolicy = Field(description="現在のランタイム運用ポリシー")
    model_resource_estimates: list[AivmModelResourceEstimate] = Field(
        description="音声合成モデルごとの推定リソース使用量一覧"
    )


class AivmModelAdmissionDecision(BaseModel):
    """音声合成モデル操作の dry-run admission 判定結果。"""

    model_uuid: str = Field(description="対象の音声合成モデル UUID")
    operation: Literal["prefetch", "promote"] = Field(description="判定対象の操作")
    can_admit: bool = Field(description="現在の資源状況でこの操作を受け入れ可能かどうか")
    estimated_ram_cache_size_gb: float = Field(
        description="この操作で必要となる推定 RAM キャッシュ使用量 (GB)"
    )
    estimated_vram_load_size_gb: float = Field(
        description="この操作で必要となる推定 VRAM 使用量 (GB)"
    )
    predicted_available_ram_gb: float = Field(
        description="この操作を行った直後に見込まれる空き RAM 容量 (GB)"
    )
    predicted_available_vram_gb: float | None = Field(
        description="この操作を行った直後に見込まれる空き VRAM 容量 (GB)。取得できない場合は null"
    )
    required_min_available_ram_gb: float | None = Field(
        description="policy で要求される最小空き RAM 容量 (GB)。未設定の場合は null"
    )
    required_min_available_vram_gb: float | None = Field(
        description="policy で要求される最小空き VRAM 容量 (GB)。未設定の場合は null"
    )
    ram_shortage_gb: float = Field(
        description="受け入れのために不足している RAM 容量 (GB)。不足がない場合は 0"
    )
    vram_shortage_gb: float | None = Field(
        description="受け入れのために不足している VRAM 容量 (GB)。判定不能または不足がない場合は null または 0"
    )
    runtime_resources: AivmModelRuntimeResourceSnapshot = Field(
        description="判定時点のリソース状況スナップショット"
    )
