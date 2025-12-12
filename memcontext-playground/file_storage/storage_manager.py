"""
文件存储管理器核心类
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from .file_types import FileRecord, FileType
from .utils import (
    generate_file_id,
    ensure_directory_exists,
    get_file_extension,
    sanitize_filename,
    copy_file_to_storage,
    get_timestamp
)


class FileStorageManager:
    """统一文件存储管理器"""
    
    def __init__(self, storage_base_path: str, user_id: str = "default"):
        """
        初始化文件存储管理器
        
        Args:
            storage_base_path: 存储根路径
            user_id: 用户ID
        """
        self.storage_base_path = os.path.abspath(storage_base_path)
        self.user_id = user_id
        
        # 创建存储目录结构
        self.files_dir = os.path.join(self.storage_base_path, 'files')
        self.metadata_dir = os.path.join(self.storage_base_path, 'files', 'metadata')
        self.metadata_file = os.path.join(self.metadata_dir, 'files_index.json')
        
        ensure_directory_exists(self.files_dir)
        ensure_directory_exists(self.metadata_dir)
        
        # 创建各类型文件目录
        for file_type in ['videos', 'images', 'documents', 'audio']:
            ensure_directory_exists(os.path.join(self.files_dir, file_type))
        
        # 加载元数据索引
        self.metadata_index = self._load_metadata_index()
        
        # 文件处理器映射（延迟加载）
        self._handlers: Dict[FileType, Any] = {}
    
    def _load_metadata_index(self) -> Dict[str, Dict[str, Any]]:
        """
        加载文件元数据索引
        
        Returns:
            元数据索引字典
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading metadata index: {e}")
                return {}
        return {}
    
    def _save_metadata_index(self) -> None:
        """保存文件元数据索引"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_index, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving metadata index: {e}")
    
    def _get_file_type_from_extension(self, filename: str) -> FileType:
        """
        根据文件扩展名判断文件类型
        
        Args:
            filename: 文件名
        
        Returns:
            文件类型
        """
        ext = get_file_extension(filename)
        
        video_exts = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'm4v']
        image_exts = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
        document_exts = ['pdf', 'doc', 'docx', 'txt', 'md', 'rtf']
        audio_exts = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a']
        
        if ext in video_exts:
            return FileType.VIDEO
        elif ext in image_exts:
            return FileType.IMAGE
        elif ext in document_exts:
            return FileType.DOCUMENT
        elif ext in audio_exts:
            return FileType.AUDIO
        else:
            return FileType.UNKNOWN
    
    def upload_file(
        self,
        file_path: str,
        file_type: Optional[FileType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_id: Optional[str] = None
    ) -> FileRecord:
        """
        上传文件到存储系统
        
        Args:
            file_path: 源文件路径
            file_type: 文件类型（如果为None则自动判断）
            metadata: 额外元数据
            file_id: 指定文件ID（可选，不指定则自动生成）
        
        Returns:
            文件记录对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        original_filename = os.path.basename(file_path)
        
        # 判断文件类型
        if file_type is None:
            file_type = self._get_file_type_from_extension(original_filename)
        
        # 生成文件ID
        if file_id is None:
            file_id = generate_file_id(self.user_id, original_filename)
        
        # 检查文件是否已存在
        if file_id in self.metadata_index:
            existing_record = FileRecord.from_dict(self.metadata_index[file_id])
            print(f"File {file_id} already exists, returning existing record")
            return existing_record
        
        # 确定存储目录
        file_type_dir = self._get_file_type_dir(file_type)
        file_storage_dir = os.path.join(self.files_dir, file_type_dir, file_id)
        ensure_directory_exists(file_storage_dir)
        
        # 确定目标文件名
        ext = get_file_extension(original_filename)
        target_filename = f"original.{ext}"
        target_path = os.path.join(file_storage_dir, target_filename)
        
        # 复制文件
        if not copy_file_to_storage(file_path, target_path):
            raise IOError(f"Failed to copy file to storage: {file_path}")
        
        # 提取文件元数据
        file_metadata = self._extract_file_metadata(target_path, file_type)
        if metadata:
            file_metadata.update(metadata)
        
        # 创建文件记录
        file_record = FileRecord(
            file_id=file_id,
            file_type=file_type,
            original_filename=original_filename,
            stored_path=target_path,
            upload_time=get_timestamp(),
            user_id=self.user_id,
            metadata=file_metadata
        )
        
        # 保存到索引
        self.metadata_index[file_id] = file_record.to_dict()
        self._save_metadata_index()
        
        print(f"File uploaded successfully: {file_id} -> {target_path}")
        return file_record
    
    def _get_file_type_dir(self, file_type: FileType) -> str:
        """
        获取文件类型对应的目录名
        
        Args:
            file_type: 文件类型
        
        Returns:
            目录名
        """
        type_dir_map = {
            FileType.VIDEO: 'videos',
            FileType.IMAGE: 'images',
            FileType.DOCUMENT: 'documents',
            FileType.AUDIO: 'audio',
        }
        return type_dir_map.get(file_type, 'unknown')
    
    def _extract_file_metadata(self, file_path: str, file_type: FileType) -> Dict[str, Any]:
        """
        提取文件元数据
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
        
        Returns:
            元数据字典
        """
        metadata = {
            'file_size': os.path.getsize(file_path),
        }
        
        # 根据文件类型提取特定元数据
        if file_type == FileType.VIDEO:
            metadata.update(self._extract_video_metadata(file_path))
        elif file_type == FileType.IMAGE:
            metadata.update(self._extract_image_metadata(file_path))
        elif file_type == FileType.DOCUMENT:
            metadata.update(self._extract_document_metadata(file_path))
        
        return metadata
    
    def _extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取视频元数据"""
        metadata = {}
        
        # 尝试使用ffprobe获取视频信息
        try:
            import subprocess
            import shutil
            
            if shutil.which("ffprobe"):
                cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration,size",
                    "-show_entries", "stream=width,height,codec_name",
                    "-of", "json",
                    file_path
                ]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    format_info = info.get('format', {})
                    stream_info = info.get('streams', [{}])[0]
                    
                    if 'duration' in format_info:
                        metadata['duration'] = float(format_info['duration'])
                    if 'width' in stream_info:
                        metadata['width'] = int(stream_info['width'])
                    if 'height' in stream_info:
                        metadata['height'] = int(stream_info['height'])
                    if 'codec_name' in stream_info:
                        metadata['codec'] = stream_info['codec_name']
        except Exception as e:
            print(f"Error extracting video metadata: {e}")
        
        return metadata
    
    def _extract_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取图片元数据（预留）"""
        metadata = {}
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['format'] = img.format
        except Exception as e:
            print(f"Error extracting image metadata: {e}")
        return metadata
    
    def _extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取文档元数据（预留）"""
        metadata = {}
        # 后续可以实现PDF页数提取等
        return metadata
    
    def get_file_record(self, file_id: str) -> Optional[FileRecord]:
        """
        获取文件记录
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件记录对象，不存在返回None
        """
        if file_id in self.metadata_index:
            return FileRecord.from_dict(self.metadata_index[file_id])
        return None
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """
        获取文件完整路径
        
        Args:
            file_id: 文件ID
        
        Returns:
            文件路径，不存在返回None
        """
        record = self.get_file_record(file_id)
        if record and os.path.exists(record.stored_path):
            return record.stored_path
        return None
    
    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文件元数据
        
        Args:
            file_id: 文件ID
        
        Returns:
            元数据字典，不存在返回None
        """
        record = self.get_file_record(file_id)
        if record:
            return record.metadata
        return None
    
    def list_files(self, file_type: Optional[FileType] = None) -> List[FileRecord]:
        """
        列出所有文件或指定类型的文件
        
        Args:
            file_type: 文件类型过滤（可选）
        
        Returns:
            文件记录列表
        """
        records = []
        for file_id, data in self.metadata_index.items():
            record = FileRecord.from_dict(data)
            if file_type is None or record.file_type == file_type:
                records.append(record)
        return records
    
    def delete_file(self, file_id: str) -> bool:
        """
        删除文件
        
        Args:
            file_id: 文件ID
        
        Returns:
            是否成功删除
        """
        record = self.get_file_record(file_id)
        if not record:
            return False
        
        try:
            # 删除文件目录
            file_dir = os.path.dirname(record.stored_path)
            if os.path.exists(file_dir):
                shutil.rmtree(file_dir)
            
            # 从索引中删除
            del self.metadata_index[file_id]
            self._save_metadata_index()
            
            print(f"File deleted: {file_id}")
            return True
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def get_handler(self, file_type: FileType):
        """
        获取文件类型处理器（延迟加载）
        
        Args:
            file_type: 文件类型
        
        Returns:
            文件处理器实例
        """
        if file_type not in self._handlers:
            if file_type == FileType.VIDEO:
                from .video_handler import VideoHandler
                self._handlers[file_type] = VideoHandler(self.storage_base_path)
            elif file_type == FileType.IMAGE:
                from .image_handler import ImageHandler
                self._handlers[file_type] = ImageHandler(self.storage_base_path)
            elif file_type == FileType.DOCUMENT:
                from .document_handler import DocumentHandler
                self._handlers[file_type] = DocumentHandler(self.storage_base_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        
        return self._handlers[file_type]
