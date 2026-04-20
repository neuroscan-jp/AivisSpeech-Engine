"""音声波形を加工する。"""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from soxr import resample

from ..model import AudioQuery
from .model import (
    FrameAudioQuery,
)


def raw_wave_to_output_wave(
    query: AudioQuery | FrameAudioQuery, wave: NDArray[np.float32], sr_wave: int
) -> NDArray[np.float32]:
    """生音声波形に音声合成用のクエリを適用して出力音声波形を生成する"""
    wave = _apply_volume_scale(wave, query)
    wave = _apply_output_sampling_rate(wave, sr_wave, query)
    wave = _apply_output_stereo(wave, query)
    return wave


def output_wave_to_pcm_chunks(
    wave: NDArray[np.float32], *, frames_per_chunk: int = 4096
) -> Iterator[bytes]:
    """出力音声波形を little-endian signed 16-bit PCM chunk に変換する。"""
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be greater than 0")

    total_frames = wave.shape[0]
    for start in range(0, total_frames, frames_per_chunk):
        wave_chunk = np.clip(wave[start : start + frames_per_chunk], -1.0, 1.0)
        pcm_chunk = np.rint(wave_chunk * 32767.0).astype("<i2", copy=False).tobytes()
        if pcm_chunk:
            yield pcm_chunk


def _apply_volume_scale(
    wave: NDArray[np.float32], query: AudioQuery | FrameAudioQuery
) -> NDArray[np.float32]:
    """音声波形へ音声合成用のクエリがもつ音量スケール（`volumeScale`）を適用する"""
    return wave * query.volumeScale


def _apply_output_sampling_rate(
    wave: NDArray[np.float32], sr_wave: float, query: AudioQuery | FrameAudioQuery
) -> NDArray[np.float32]:
    """音声波形へ音声合成用のクエリがもつ出力サンプリングレート（`outputSamplingRate`）を適用する"""
    # サンプリングレート一致のときはスルー
    if sr_wave == query.outputSamplingRate:
        return wave
    wave = resample(wave, sr_wave, query.outputSamplingRate)
    return wave


def _apply_output_stereo(
    wave: NDArray[np.float32], query: AudioQuery | FrameAudioQuery
) -> NDArray[np.float32]:
    """音声波形へ音声合成用のクエリがもつステレオ出力設定（`outputStereo`）を適用する"""
    if query.outputStereo:
        wave = np.array([wave, wave]).T
    return wave
