"""
文件存储管理模块

提供统一的文件存储、定位和访问功能。
支持视频、图片、文档等多种文件类型。
"""

from .storage_manager import FileStorageManager
from .file_types import FileRecord, FileType, BaseFileHandler
from .video_handler import VideoHandler
from .image_handler import ImageHandler
from .document_handler import DocumentHandler

__all__ = [
    'FileStorageManager',
    'FileRecord',
    'FileType',
    'BaseFileHandler',
    'VideoHandler',
    'ImageHandler',
    'DocumentHandler',
]
