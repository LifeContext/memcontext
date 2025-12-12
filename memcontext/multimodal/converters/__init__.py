"""Concrete converter implementations."""

from .audio_converter import AudioConverter
from .file_converter import DocumentConverter
from .image_converter import ImageConverter
from .video_converter import VideoConverter

# VideoRAG 依赖 hnswlib 等重组件，放在可选导入，避免普通视频转换时强制加载
try:
    from .videorag_converter import VideoConverter as VideoRAGConverter
except Exception:
    VideoRAGConverter = None

__all__ = [
    "AudioConverter",
    "DocumentConverter",
    "ImageConverter",
    "VideoConverter",
]

# 仅在可用时导出 VideoRAGConverter，防止导入失败
if VideoRAGConverter is not None:
    __all__.append("VideoRAGConverter")

