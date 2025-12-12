"""Utilities for handling multimodal inputs before they are written to Memcontext."""

from .converter import (
    MultimodalConverter,
    ConversionChunk,
    ConversionOutput,
    ProgressCallback,
)
from .factory import ConverterFactory

# Import lightweight converters so their registrations run on package import.
from .converters import audio_converter, image_converter, file_converter  # noqa: F401

# Video converter依赖更多第三方组件（Ark/ffmpeg等），放在可选导入，避免阻塞无关场景。
try:  # noqa: F401
    from .converters import video_converter
except Exception:
    video_converter = None

__all__ = [
    "MultimodalConverter",
    "ConversionChunk",
    "ConversionOutput",
    "ProgressCallback",
    "ConverterFactory",
]

