"""
文件存储工具函数
"""

import os
import uuid
import hashlib
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime


def generate_file_id(user_id: str, original_filename: Optional[str] = None) -> str:
    """
    生成唯一文件ID
    
    Args:
        user_id: 用户ID
        original_filename: 原始文件名（可选）
    
    Returns:
        唯一文件ID字符串
    """
    timestamp = datetime.now().isoformat()
    unique_str = f"{user_id}_{timestamp}_{uuid.uuid4().hex[:8]}"
    
    if original_filename:
        # 添加文件名哈希以增加唯一性
        filename_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        unique_str = f"{user_id}_{filename_hash}_{uuid.uuid4().hex[:8]}"
    
    return hashlib.sha256(unique_str.encode()).hexdigest()[:32]


def ensure_directory_exists(path: str) -> None:
    """
    确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名（不含点）
    
    Args:
        filename: 文件名
    
    Returns:
        扩展名字符串
    """
    return Path(filename).suffix.lstrip('.').lower()


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不安全字符
    
    Args:
        filename: 原始文件名
    
    Returns:
        清理后的文件名
    """
    # 移除路径分隔符和其他不安全字符
    unsafe_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    sanitized = filename
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized


def copy_file_to_storage(source_path: str, target_path: str) -> bool:
    """
    复制文件到存储目录
    
    Args:
        source_path: 源文件路径
        target_path: 目标文件路径
    
    Returns:
        是否成功
    """
    try:
        ensure_directory_exists(os.path.dirname(target_path))
        shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False


def format_time_for_filename(seconds: float) -> str:
    """
    将秒数格式化为文件名友好的格式
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串（如：05.30）
    """
    return f"{seconds:.2f}".replace('.', '_')


def parse_time_from_filename(filename: str) -> Optional[tuple[float, float]]:
    """
    从文件名解析时间范围
    
    Args:
        filename: 文件名（如：segment_5_30_8_00.mp4）
    
    Returns:
        (start_time, end_time) 元组，解析失败返回None
    """
    try:
        # 假设格式为 segment_{start}_{end}.mp4
        parts = Path(filename).stem.split('_')
        if len(parts) >= 3 and parts[0] == 'segment':
            start = float(parts[1].replace('_', '.'))
            end = float(parts[2].replace('_', '.'))
            return (start, end)
    except (ValueError, IndexError):
        pass
    return None


def get_timestamp() -> str:
    """
    获取当前时间戳字符串
    
    Returns:
        ISO格式时间戳字符串
    """
    return datetime.now().isoformat()
