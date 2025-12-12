"""
视频文件处理器
支持视频片段时间戳定位和动态生成
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from .file_types import BaseFileHandler, FileType, VideoSegmentInfo
from .utils import ensure_directory_exists, format_time_for_filename, parse_time_from_filename


class VideoHandler(BaseFileHandler):
    """视频文件处理器"""
    
    def __init__(self, storage_base_path: str):
        """
        初始化视频处理器
        
        Args:
            storage_base_path: 存储根路径
        """
        super().__init__(storage_base_path)
        self.storage_base_path = storage_base_path
    
    def get_file_type_dir(self) -> str:
        """返回视频文件目录名"""
        return "videos"
    
    def get_file_path(self, file_id: str) -> str:
        """
        获取视频文件完整路径
        
        Args:
            file_id: 文件ID
        
        Returns:
            视频文件路径
        """
        file_dir = self.get_storage_dir(file_id)
        # 查找original文件
        for ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'm4v']:
            original_path = os.path.join(file_dir, f"original.{ext}")
            if os.path.exists(original_path):
                return original_path
        
        # 如果找不到，返回目录下的第一个文件
        if os.path.exists(file_dir):
            files = [f for f in os.listdir(file_dir) if f.startswith('original.')]
            if files:
                return os.path.join(file_dir, files[0])
        
        raise FileNotFoundError(f"Video file not found for file_id: {file_id}")
    
    def get_segment_path(
        self,
        file_id: str,
        location_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        根据时间戳获取视频片段路径
        
        Args:
            file_id: 文件ID
            location_info: 定位信息，包含：
                - start_time: 开始时间（秒）
                - end_time: 结束时间（秒）
                - duration: 可选，片段时长（秒），如果不提供则计算
        
        Returns:
            片段文件路径，如果生成失败返回None
        """
        start_time = location_info.get('start_time')
        end_time = location_info.get('end_time')
        
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time are required in location_info")
        
        start_time = float(start_time)
        end_time = float(end_time)
        
        if start_time >= end_time:
            raise ValueError("start_time must be less than end_time")
        
        # 获取原始视频路径
        original_path = self.get_file_path(file_id)
        
        # 片段存储目录
        segments_dir = os.path.join(self.get_storage_dir(file_id), 'segments')
        ensure_directory_exists(segments_dir)
        
        # 生成片段文件名
        segment_filename = f"segment_{format_time_for_filename(start_time)}_{format_time_for_filename(end_time)}.mp4"
        segment_path = os.path.join(segments_dir, segment_filename)
        
        # 如果片段已存在，直接返回
        if os.path.exists(segment_path):
            return segment_path
        
        # 动态生成片段
        return self._generate_segment(original_path, segment_path, start_time, end_time)
    
    def _generate_segment(
        self,
        source_path: str,
        target_path: str,
        start_time: float,
        end_time: float
    ) -> Optional[str]:
        """
        使用ffmpeg生成视频片段
        
        Args:
            source_path: 源视频路径
            target_path: 目标片段路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
        
        Returns:
            生成的片段路径，失败返回None
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is not installed or not in PATH")
        
        duration = end_time - start_time
        
        try:
            cmd = [
                "ffmpeg",
                "-i", source_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c", "copy",  # 使用copy模式，不重新编码，速度快
                "-avoid_negative_ts", "make_zero",
                "-y",  # 覆盖输出文件
                target_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                print(f"Video segment generated: {target_path} ({start_time:.2f}s - {end_time:.2f}s)")
                return target_path
            else:
                print(f"Error generating segment: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception while generating segment: {e}")
            return None
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取视频元数据
        
        Args:
            file_path: 视频文件路径
        
        Returns:
            元数据字典
        """
        metadata = {}
        
        if not shutil.which("ffprobe"):
            return metadata
        
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration,size",
                "-show_entries", "stream=width,height,codec_name,bit_rate",
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
                streams = info.get('streams', [])
                
                # 查找视频流
                video_stream = None
                for stream in streams:
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if 'duration' in format_info:
                    metadata['duration'] = float(format_info['duration'])
                
                if video_stream:
                    if 'width' in video_stream:
                        metadata['width'] = int(video_stream['width'])
                    if 'height' in video_stream:
                        metadata['height'] = int(video_stream['height'])
                    if 'codec_name' in video_stream:
                        metadata['codec'] = video_stream['codec_name']
                    if 'bit_rate' in video_stream:
                        metadata['bit_rate'] = int(video_stream['bit_rate'])
        except Exception as e:
            print(f"Error extracting video metadata: {e}")
        
        return metadata
    
    def list_segments(self, file_id: str) -> List[VideoSegmentInfo]:
        """
        列出所有已生成的视频片段
        
        Args:
            file_id: 文件ID
        
        Returns:
            片段信息列表
        """
        segments = []
        segments_dir = os.path.join(self.get_storage_dir(file_id), 'segments')
        
        if not os.path.exists(segments_dir):
            return segments
        
        for filename in os.listdir(segments_dir):
            if filename.startswith('segment_') and filename.endswith('.mp4'):
                time_range = parse_time_from_filename(filename)
                if time_range:
                    start_time, end_time = time_range
                    segment_path = os.path.join(segments_dir, filename)
                    segments.append(VideoSegmentInfo(
                        index=len(segments),
                        start_time=start_time,
                        end_time=end_time,
                        path=segment_path,
                        duration=end_time - start_time
                    ))
        
        # 按开始时间排序
        segments.sort(key=lambda x: x.start_time)
        return segments
    
    def get_segment_by_time(
        self,
        file_id: str,
        start_time: float,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> Optional[str]:
        """
        根据时间获取视频片段（便捷方法）
        
        Args:
            file_id: 文件ID
            start_time: 开始时间（秒）
            end_time: 结束时间（秒），如果提供则使用
            duration: 片段时长（秒），如果end_time未提供则使用
        
        Returns:
            片段路径
        """
        if end_time is None:
            if duration is None:
                raise ValueError("Either end_time or duration must be provided")
            end_time = start_time + duration
        
        location_info = {
            'start_time': start_time,
            'end_time': end_time
        }
        
        return self.get_segment_path(file_id, location_info)
