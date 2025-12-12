"""
文件类型定义和基类
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class FileType(Enum):
    """文件类型枚举"""
    VIDEO = "video"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class FileRecord:
    """文件记录数据结构"""
    file_id: str
    file_type: FileType
    original_filename: str
    stored_path: str
    upload_time: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_id': self.file_id,
            'file_type': self.file_type.value,
            'original_filename': self.original_filename,
            'stored_path': self.stored_path,
            'upload_time': self.upload_time,
            'user_id': self.user_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileRecord':
        """从字典创建"""
        return cls(
            file_id=data['file_id'],
            file_type=FileType(data['file_type']),
            original_filename=data['original_filename'],
            stored_path=data['stored_path'],
            upload_time=data['upload_time'],
            user_id=data['user_id'],
            metadata=data.get('metadata', {})
        )


@dataclass
class VideoSegmentInfo:
    """视频片段信息"""
    index: int
    start_time: float
    end_time: float
    path: str
    duration: Optional[float] = None


@dataclass
class ImageRegionInfo:
    """图片区域信息（预留）"""
    x: int
    y: int
    width: int
    height: int
    path: Optional[str] = None


@dataclass
class DocumentPageInfo:
    """文档页面信息（预留）"""
    page_number: int
    path: Optional[str] = None


class BaseFileHandler(ABC):
    """文件处理器基类"""
    
    def __init__(self, storage_base_path: str):
        """
        初始化文件处理器
        
        Args:
            storage_base_path: 存储根路径
        """
        self.storage_base_path = storage_base_path
    
    @abstractmethod
    def get_file_path(self, file_id: str) -> str:
        """
        获取文件完整路径
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件路径
        """
        pass
    
    @abstractmethod
    def get_segment_path(self, file_id: str, location_info: Dict[str, Any]) -> Optional[str]:
        """
        根据定位信息获取文件片段路径
        
        Args:
            file_id: 文件ID
            location_info: 定位信息（不同类型文件有不同的格式）
        
        Returns:
            片段路径，如果不存在则返回None
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取文件元数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            元数据字典
        """
        pass
    
    def get_storage_dir(self, file_id: str) -> str:
        """
        获取文件存储目录
        
        Args:
            file_id: 文件ID
        
        Returns:
            存储目录路径
        """
        file_type_dir = self.get_file_type_dir()
        return os.path.join(self.storage_base_path, 'files', file_type_dir, file_id)
    
    @abstractmethod
    def get_file_type_dir(self) -> str:
        """
        获取文件类型目录名
        
        Returns:
            目录名（如：videos, images, documents）
        """
        pass
