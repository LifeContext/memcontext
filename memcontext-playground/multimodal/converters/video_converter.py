"""Real video converter implementation using Volcengine SDK for video analysis."""
from __future__ import annotations
import base64
import os
import re
import subprocess
import sys
import tempfile
import shutil
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    Ark = None
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from multimodal.converter import ConversionChunk, ConversionOutput, MultimodalConverter
from multimodal.factory import ConverterFactory
def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    加载 .env 文件到环境变量
    如果 env_path 为 None，会在当前文件目录和父目录中查找 .env 文件
    """
    if env_path is None:
        current_file_dir = Path(__file__).parent
        env_path = current_file_dir / ".env"
        if not env_path.exists():
            parent_dir = current_file_dir.parent.parent
            env_path = parent_dir / ".env"
    
    if env_path and env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                loaded_count = 0
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue
                    # 解析 KEY=VALUE 格式
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 移除引号
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # 设置环境变量（如果不存在则设置，避免覆盖已存在的环境变量）
                        if key and key not in os.environ:
                            os.environ[key] = value
                            loaded_count += 1
                if loaded_count > 0:
                    print(f"[INFO] 从 {env_path} 加载了 {loaded_count} 个环境变量")
        except Exception as e:
            print(f"[WARNING] 读取 .env 文件失败: {e}")
    else:
        # 调试信息：显示查找的路径
        if env_path:
            print(f"[DEBUG] 未找到 .env 文件，查找路径: {env_path}")


# 在模块加载时尝试加载 .env 文件
load_env_file()


class VideoConverter(MultimodalConverter):
    SUPPORTED_EXTENSIONS: List[str] = ["mp4", "mov", "avi", "mkv", "webm", "flv"]
    def __init__(
        self,
        *,
        max_chunk_tokens: int = 4000,
        progress_callback: Optional[Any] = None,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        **config: Any,
    ) -> None:
        super().__init__(
            max_chunk_tokens=max_chunk_tokens,
            progress_callback=progress_callback,
            retry_count=retry_count,
            retry_delay=retry_delay,
            **config,
        )
        # 从 config 或环境变量获取配置
        # 优先使用传入参数，其次使用环境变量 LLM_API_KEY
        self.api_key =  os.environ.get("LLM_API_KEY")
        self.base_url =  os.environ.get("LLM_BASE_URL")
        self.model =  os.environ.get("LLM_MODEL")
        
        # 初始化 Volcengine SDK 客户端
        if Ark is None:
            raise ImportError(
                "volcenginesdkarkruntime 未安装。请运行: pip install 'volcengine-python-sdk[ark]'"
            )
        
        if not self.api_key:
            raise ValueError("API key 必须配置。可通过 config 参数或环境变量 LLM_API_KEY 设置。")
        
        self.client = Ark(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # 音频转录配置（可选）
        self.enable_audio_transcription = os.environ.get("ENABLE_AUDIO_TRANSCRIPTION", "false").lower() == "true"
        # SiliconFlow API配置（用于音频转录）
        # 从环境变量读取，如果没有设置则使用默认值
        self.siliconflow_api_key = os.environ.get("SILICONFLOW_API_KEY") or "your api key"
        self.siliconflow_api_url = os.environ.get("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/audio/transcriptions")
        self.siliconflow_model = os.environ.get("SILICONFLOW_MODEL", "TeleAI/TeleSpeechASR")
        
        # 输出音频转录配置状态
        if self.enable_audio_transcription:
            if not self.siliconflow_api_key:
                self._report_progress(0.0, "⚠️  警告: 音频转录功能已启用，但 SILICONFLOW_API_KEY 未配置")
        else:
            self._report_progress(0.0, "音频转录功能未启用（设置 ENABLE_AUDIO_TRANSCRIPTION=true 启用）")

    def _split_video_by_time(self, video_path: str, segment_duration: int = 60) -> List[Tuple[str, float, float]]:
        """
        使用 ffmpeg 将视频按时间切分成多个片段（每个片段 segment_duration 秒）
        返回: [(片段路径, 开始时间, 结束时间), ...]
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg 未安装或不在 PATH 中。请先安装 ffmpeg。")
        
        # 获取视频总时长
        video_duration = self._get_video_duration(video_path)
        if video_duration is None:
            raise ValueError("无法获取视频时长")
        
        # 创建临时目录存储片段
        temp_dir = tempfile.mkdtemp(prefix="video_chunks_")
        segments = []
        
        try:
            segment_index = 0
            start_time = 0.0
            
            while start_time < video_duration:
                end_time = min(start_time + segment_duration, video_duration)
                segment_path = os.path.join(temp_dir, f"segment_{segment_index:04d}.mp4")
                
                self._report_progress(
                    0.1 + (start_time / video_duration) * 0.1,
                    f"正在切分视频片段 {segment_index + 1} ({start_time:.1f}s - {end_time:.1f}s)..."
                )
                
                # 使用 ffmpeg 切分视频
                cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-c", "copy",  # 使用 copy 模式，不重新编码，速度快
                    "-avoid_negative_ts", "make_zero",
                    "-y",  # 覆盖输出文件
                    segment_path
                ]
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg 切分失败: {result.stderr}")
                
                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    segments.append((segment_path, start_time, end_time))
                    self._report_progress(
                        0.1 + (end_time / video_duration) * 0.1,
                        f"片段 {segment_index + 1} 切分完成 ({start_time:.1f}s - {end_time:.1f}s)"
                    )
                    segment_index += 1
                else:
                    break
                
                start_time = end_time
            
            return segments
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    def _cleanup_temp_segments(self, segments: List[Tuple[str, float, float]]) -> None:
        """清理临时视频片段文件"""
        for segment_path, _, _ in segments:
            if os.path.exists(segment_path):
                try:
                    os.remove(segment_path)
                except Exception:
                    pass
        # 删除临时目录
        if segments:
            temp_dir = os.path.dirname(segments[0][0])
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

    def _adjust_timestamps(self, text: str, time_offset: float) -> str:
        """
        调整时间戳，加上时间偏移量
        time_offset: 时间偏移（秒）
        """
        def replace_timestamp(match):
            prefix = match.group(1) or ""
            hours = int(match.group(2))
            minutes = int(match.group(3))
            seconds = int(match.group(4)) if match.group(4) else 0
            suffix = match.group(5) or ""
            
            # 计算总秒数并加上偏移
            total_seconds = hours * 3600 + minutes * 60 + seconds + time_offset
            
            # 转换回 HH:MM:SS 格式
            new_hours = int(total_seconds // 3600)
            new_minutes = int((total_seconds % 3600) // 60)
            new_seconds = int(total_seconds % 60)
            
            return f"{prefix}{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d}{suffix}"
        
        # 匹配 [HH:MM:SS] 或 [MM:SS] 格式
        pattern = r'(\[?)(\d{1,2}):(\d{2})(?::(\d{2}))?(\]?)'
        return re.sub(pattern, replace_timestamp, text)

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        self._report_progress(0.1, f"正在获取视频时长: {os.path.basename(video_path)}...")
        # 仅依赖 ffprobe 获取时长，避免 OpenCV 依赖
        if not shutil.which("ffprobe"):
            self._report_progress(0.15, "未找到 ffprobe，无法获取视频时长")
            return None

        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                if duration > 0:
                    self._report_progress(0.15, f"视频时长: {duration:.2f} 秒")
                    return duration
        except Exception:
            pass

        return None
    
    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """获取音频时长（秒）"""
        if not shutil.which("ffprobe"):
            return None
        
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                   "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            return None
        except Exception:
            return None
    
    def _extract_audio_from_video(self, video_path: str, audio_output_path: str, start_time: float = None, duration: float = None) -> bool:
        """从视频文件提取音频"""
        if not shutil.which("ffmpeg"):
            return False
        
        try:
            # 从视频提取音频，使用copy模式（不重新编码，速度快）
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", "-y"]
            
            # 如果指定了时间范围，添加相应参数
            if start_time is not None:
                cmd.extend(["-ss", str(start_time)])
            if duration is not None:
                cmd.extend(["-t", str(duration)])
            
            cmd.append(audio_output_path)
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 如果copy模式失败（可能视频没有音频流或格式不支持），尝试重新编码
            if result.returncode != 0 or not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
                cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-b:a", "128k", "-y"]
                
                if start_time is not None:
                    cmd.extend(["-ss", str(start_time)])
                if duration is not None:
                    cmd.extend(["-t", str(duration)])
                
                cmd.append(audio_output_path)
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            if result.returncode == 0 and os.path.exists(audio_output_path) and os.path.getsize(audio_output_path) > 0:
                return True
            else:
                # 检查是否是视频没有音频流
                stderr_lower = result.stderr.lower()
                if "no audio stream" in stderr_lower or "does not contain any stream" in stderr_lower:
                    return False
                return False
        except Exception:
            return False
    
    def _split_audio_by_time(self, audio_path: str, segment_duration: int = 60, output_dir: str = None) -> List[Tuple[str, float, float]]:
        """
        将音频文件按时间切分成多个片段
        
        Args:
            audio_path: 音频文件路径
            segment_duration: 每个片段的时长（秒），默认60秒
            output_dir: 输出目录，如果为None则使用临时目录
        
        Returns:
            list: [(片段路径, 开始时间, 结束时间), ...]
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg 未安装或不在 PATH 中")
        
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="audio_segments_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取音频总时长
        total_duration = self._get_audio_duration(audio_path)
        if total_duration is None:
            raise ValueError("无法获取音频时长")
        
        segments = []
        audio_name = Path(audio_path).stem
        audio_ext = Path(audio_path).suffix
        
        segment_index = 0
        start_time = 0.0
        
        while start_time < total_duration:
            end_time = min(start_time + segment_duration, total_duration)
            segment_path = os.path.join(output_dir, f"{audio_name}_segment_{segment_index:04d}{audio_ext}")
            
            # 使用ffmpeg切分音频
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-acodec", "copy",  # 使用copy模式，不重新编码
                "-y",
                segment_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0 and os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                segments.append((segment_path, start_time, end_time))
                segment_index += 1
            else:
                # 如果copy模式失败，尝试重新编码
                cmd = [
                    "ffmpeg",
                    "-i", audio_path,
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-acodec", "libmp3lame" if audio_ext == ".mp3" else "aac",
                    "-b:a", "128k",
                    "-y",
                    segment_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if result.returncode == 0 and os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                    segments.append((segment_path, start_time, end_time))
                    segment_index += 1
            
            start_time = end_time
        
        return segments
    
    def _transcribe_audio_segment_with_siliconflow(self, segment_path: str, segment_start: float) -> Optional[str]:
        """使用SiliconFlow API转录单个音频片段，返回转录文本"""
        if not self.siliconflow_api_key:
            return None
        
        file_ext = os.path.splitext(segment_path)[1].lower()
        
        try:
            with open(segment_path, 'rb') as audio_file:
                files = { 
                    "file": (os.path.basename(segment_path), audio_file, f"audio/{file_ext[1:] if file_ext else 'mp3'}")
                }
                payload = { 
                    "model": self.siliconflow_model
                }
                headers = {
                    "Authorization": f"Bearer {self.siliconflow_api_key}"
                }
                
                response = requests.post(self.siliconflow_api_url, data=payload, files=files, headers=headers, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "")
                else:
                    # 输出详细的错误信息
                    error_msg = f"⚠️  SiliconFlow API调用失败: {response.status_code}"
                    try:
                        error_detail = response.json()
                        if isinstance(error_detail, dict):
                            error_msg += f" - {error_detail.get('message', error_detail.get('error', str(error_detail)))}"
                        else:
                            error_msg += f" - {str(error_detail)}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                    
                    # 如果是401错误，提示检查API key
                    if response.status_code == 401:
                        error_msg += " (认证失败，请检查 SILICONFLOW_API_KEY 是否正确)"
                    
                    self._report_progress(0.0, error_msg)
                    return None
        except Exception as e:
            self._report_progress(0.0, f"⚠️  音频转录出错: {str(e)[:100]}")
            return None
    
    def _transcribe_audio_segments_list(self, audio_path: str, segment_start_time: float = 0.0) -> List[str]:
        """
        将音频文件按1分钟切分并转录，返回文本列表
        
        Args:
            audio_path: 音频文件路径
            segment_start_time: 音频在整个视频中的开始时间偏移
        
        Returns:
            List[str]: 每个片段的转录文本列表
        """
        if not self.enable_audio_transcription:
            return []
        
        if not self.siliconflow_api_key:
            return []
        
        try:
            # 切分音频（每60秒一个片段）
            segments = self._split_audio_by_time(audio_path, segment_duration=60)
            
            transcription_list = []
            
            for i, (segment_path, seg_start, seg_end) in enumerate(segments):
                # 调整时间偏移（加上音频在整个视频中的开始时间）
                adjusted_start = seg_start + segment_start_time
                adjusted_end = seg_end + segment_start_time
                
                self._report_progress(
                    0.0,
                    f"正在转录音频片段 {i+1}/{len(segments)} [{adjusted_start:.1f}s - {adjusted_end:.1f}s]..."
                )
                
                # 转录片段
                text = self._transcribe_audio_segment_with_siliconflow(segment_path, adjusted_start)
                
                if text:
                    transcription_list.append(text)
                    self._report_progress(0.0, f"✅ 音频片段 {i+1} 转录成功: {len(text)} 字符")
                else:
                    transcription_list.append("")  # 失败时添加空字符串
                    self._report_progress(0.0, f"⚠️  音频片段 {i+1} 转录失败")
                
                # 清理临时片段文件
                try:
                    os.remove(segment_path)
                except Exception:
                    pass
            
            # 清理临时目录
            temp_dir = os.path.dirname(segments[0][0]) if segments else None
            if temp_dir and os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
            
            return transcription_list
            
        except Exception as e:
            self._report_progress(0.0, f"⚠️  音频转录过程出错: {str(e)[:100]}")
            return []
    
    def _merge_video_and_audio_analysis(self, video_description: str, audio_transcription: Optional[str]) -> str:
        """合并视频分析和音频转录结果"""
        if not audio_transcription:
            return video_description
        
        merged = f"""【视频分析】
{video_description}

【音频转录】
{audio_transcription}"""
        return merged

    def _filter_timestamps_by_duration(self, text: str, max_duration: Optional[float] = None) -> str:
        """
        过滤掉超过视频长度的时间戳
        """
        if max_duration is None:
            return text
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            timestamp_pattern = r'(\[?)(\d{1,2}):(\d{2})(?::(\d{2}))?(\]?)'
            matches = list(re.finditer(timestamp_pattern, line))
            
            if matches:
                line_valid = True
                for match in matches:
                    groups = match.groups()
                    if groups[3] is not None:  # HH:MM:SS 格式
                        hours, minutes, seconds = map(int, [groups[1], groups[2], groups[3]])
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                    else:  # MM:SS 格式
                        minutes, seconds = map(int, [groups[1], groups[2]])
                        total_seconds = minutes * 60 + seconds
                    
                    if total_seconds > max_duration:
                        line_valid = False
                        break
                
                if line_valid:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

    def _encode_video(self, video_path: str) -> str:
        """
        将视频文件编码为 Base64 字符串
        """
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    def _get_video_format(self, video_path: str) -> str:
        """
        从文件扩展名获取视频格式
        """
        ext = Path(video_path).suffix.lower().lstrip('.')
        # 映射常见扩展名到 MIME 类型
        format_map = {
            'mp4': 'mp4',
            'mov': 'quicktime',
            'avi': 'x-msvideo',
            'mkv': 'x-matroska',
            'webm': 'webm',
            'flv': 'x-flv',
        }
        return format_map.get(ext, 'mp4')

    def _analyze_video_segment(self, video_path: str, segment_duration: Optional[float] = None, segment_start: float = 0.0) -> str:
        """
        分析本地视频文件并返回带时间戳的描述
        使用 Volcengine SDK 和 Base64 编码
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 构建 prompt
        if segment_start > 0:
            # 这是视频片段，需要说明这是片段
            base_prompt = f"""这是完整视频的一个片段（从 {segment_start:.1f} 秒开始）。请详细分析这个视频片段的内容，并按照以下格式生成带时间戳的描述：

**重要说明**：
- 这是视频片段，时间戳应该从 [00:00:00] 开始（相对于片段本身）
- 片段的实际长度会在下面明确告知
- **绝对不要**生成超过片段实际长度的时间戳

输出格式要求：
1. 使用时间戳格式 [HH:MM:SS] 或 [MM:SS]（例如：[00:00:05] 或 [05:30]）
2. 每个时间点描述 1-2 句话
3. 包含以下信息：
   - 场景描述（地点、环境）
   - 主要人物或对象
   - 关键动作或事件
   - 视觉元素（颜色、位置、变化等）
4. **必须确保所有时间戳都在片段的实际长度范围内**

输出格式示例（假设片段长度为12秒）：
[00:00:00] - 片段开始，显示一个室内场景，有一个人在桌子前工作
[00:00:05] - 人物站起来，走向窗户
[00:00:10] - 窗外可以看到城市风景，阳光透过窗户

请按照时间顺序，详细描述这个视频片段的内容，确保每个重要时刻都有对应的时间戳，并且**所有时间戳都不超过片段的实际长度**。"""
        else:
            # 这是完整视频
            base_prompt = """请详细分析这个视频的内容，并按照以下格式生成带时间戳的描述：
        **严格限制**：
        - 视频的实际长度会在下面明确告知
        - **绝对不要**生成超过视频实际长度的时间戳
        - 如果视频只有12秒，最大时间戳只能是 [00:00:12] 或更早
        - 如果视频只有5秒，最大时间戳只能是 [00:00:05] 或更早

        输出格式要求：
        1. 使用时间戳格式 [HH:MM:SS] 或 [MM:SS]（例如：[00:00:05] 或 [05:30]）
        2. 每个时间点描述 1-2 句话
        3. 包含以下信息：
        - 场景描述（地点、环境）
        - 主要人物或对象
        - 关键动作或事件
        - 视觉元素（颜色、位置、变化等）
        4. **必须确保所有时间戳都在视频的实际长度范围内**

        输出格式示例（假设视频长度为12秒）：
        [00:00:00] - 视频开始，显示一个室内场景，有一个人在桌子前工作
        [00:00:05] - 人物站起来，走向窗户
        [00:00:10] - 窗外可以看到城市风景，阳光透过窗户

        请按照时间顺序，详细描述整个视频的内容，确保每个重要时刻都有对应的时间戳，并且**所有时间戳都不超过视频的实际长度**。"""
        
        # 如果提供了视频时长，在 prompt 中添加时长限制
        if segment_duration is not None:
            duration_minutes = int(segment_duration // 60)
            duration_seconds = int(segment_duration % 60)
            max_timestamp = f"{duration_minutes:02d}:{duration_seconds:02d}"
            duration_text = f"\n\n**片段实际长度：{segment_duration}秒（最大时间戳：[00:00:{duration_seconds}] 或 [{max_timestamp}]）**\n\n**重要**：请严格遵守以下规则：\n- 片段只有 {segment_duration} 秒长\n- 最大允许的时间戳是 [00:00:{duration_seconds}] 或 [{max_timestamp}]\n- **绝对不要**生成超过这个时间的时间戳（如 [00:00:{duration_seconds+1}] 或更晚）\n- 如果片段在 {duration_seconds} 秒结束，最后一个时间戳应该是 [00:00:{duration_seconds}] 或更早\n\n请确保所有生成的时间戳都在 00:00:00 到 [00:00:{duration_seconds}] 之间。"
            base_prompt += duration_text
        
        # 将视频编码为 Base64
        self._report_progress(0.25, "正在编码视频文件...")
        base64_video = self._encode_video(video_path)
        video_format = self._get_video_format(video_path)
        
        # 使用 SDK 调用 API
        self._report_progress(0.3, f"正在分析视频: {os.path.basename(video_path)}...")
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/{video_format};base64,{base64_video}",
                                "fps":4,
                            },
                        },
                        {
                            "type": "text",
                            "text": base_prompt,
                        },
                    ],
                }
            ],
        )
        
        # 提取视频描述
        if completion.choices and len(completion.choices) > 0:
            video_description = completion.choices[0].message.content
            
            # 如果提供了视频时长，过滤掉超过时长的时间戳
            if segment_duration is not None:
                self._report_progress(0.9, "过滤时间戳...")
                video_description = self._filter_timestamps_by_duration(video_description, segment_duration)
            
            return video_description
        else:
            raise ValueError("无法从 API 响应中提取视频描述")
    def convert(self, source, *, source_type: str = "file_path", **kwargs: Any) -> ConversionOutput:
        """
        真实视频识别实现：使用 API 进行本地视频文件分析
        如果视频文件超过 50MB，会自动切分成多个片段分别分析
        """
        segments = []
        try:
            # 只支持本地文件路径
            if source_type != "file_path":
                raise ValueError(f"VideoConverter 只支持本地文件路径 (source_type='file_path')，当前为: {source_type}")
            video_path = str(source)
            # 验证文件是否存在
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            # 获取视频信息
            self._report_progress(0.05, "获取视频信息...")
            video_duration = self._get_video_duration(video_path)
            
            # 不论视频大小，都按1分钟（60秒）切分
            chunks = []
            self._report_progress(0.1, f"开始按1分钟切分视频...")
            segments = self._split_video_by_time(video_path, segment_duration=60)
            self._report_progress(0.2, f"视频已切分成 {len(segments)} 个片段")
            
            # 提取整个视频的音频（如果启用音频转录）
            audio_transcription_list = []
            if self.enable_audio_transcription and self.siliconflow_api_key:
                try:
                    self._report_progress(0.15, "正在提取视频音频...")
                    temp_audio_path = os.path.join(tempfile.gettempdir(), f"video_audio_{uuid.uuid4().hex[:8]}.mp3")
                    
                    if self._extract_audio_from_video(video_path, temp_audio_path):
                        audio_size = os.path.getsize(temp_audio_path) if os.path.exists(temp_audio_path) else 0
                        self._report_progress(0.16, f"✅ 音频提取成功: {audio_size / 1024:.2f}KB")
                        
                        # 对音频进行1分钟切片并转录
                        self._report_progress(0.17, "正在对音频进行1分钟切片并转录...")
                        audio_transcription_list = self._transcribe_audio_segments_list(temp_audio_path, segment_start_time=0.0)
                        
                        if audio_transcription_list:
                            self._report_progress(0.18, f"✅ 音频转录完成: {len(audio_transcription_list)} 个片段")
                        else:
                            self._report_progress(0.18, "⚠️  音频转录失败或无结果")
                        
                        # 清理临时音频文件
                        if os.path.exists(temp_audio_path):
                            try:
                                os.remove(temp_audio_path)
                            except Exception:
                                pass
                    else:
                        self._report_progress(0.16, "⚠️  音频提取失败（可能视频没有音频流）")
                except Exception as e:
                    self._report_progress(0.16, f"⚠️  音频处理过程出错: {str(e)[:100]}")
            
            # 分析每个片段，每个片段生成一个独立的 chunk
            for i, (segment_path, start_time, end_time) in enumerate(segments):
                segment_duration = end_time - start_time
                progress_start = 0.2 + (i / len(segments)) * 0.7
                progress_end = 0.2 + ((i + 1) / len(segments)) * 0.7
                
                self._report_progress(
                    progress_start,
                    f"正在分析片段 {i+1}/{len(segments)} ({start_time:.1f}s - {end_time:.1f}s)..."
                )
                
                # 分析片段
                segment_description = self._analyze_video_segment(
                    segment_path,
                    segment_duration=segment_duration,
                    segment_start=start_time
                )
                
                # 调整时间戳，加上片段开始时间偏移
                if start_time > 0:
                    segment_description = self._adjust_timestamps(segment_description, start_time)
                
                # 获取对应的音频转录文本（如果启用）
                audio_text = ""
                if audio_transcription_list and i < len(audio_transcription_list):
                    audio_text = audio_transcription_list[i]
                
                # 合并视频分析和音频转录结果
                if audio_text:
                    segment_description = self._merge_video_and_audio_analysis(segment_description, audio_text)
                
                # 为每个片段创建独立的 chunk
                start_minutes = int(start_time // 60)
                start_seconds = int(start_time % 60)
                end_minutes = int(end_time // 60)
                end_seconds = int(end_time % 60)
                
                chunk_metadata = {
                    "source_type": "video",
                    "chunk_index": i,
                    "chunk_count_estimate": len(segments),
                    "duration_seconds": int(segment_duration),
                    "time_range": f"{start_minutes:02d}:{start_seconds:02d}-{end_minutes:02d}:{end_seconds:02d}",
                    "scene_label": "video_analysis",
                    "objects_detected": [],
                    "confidence": 0.92,
                    "chunk_summary": segment_description[:100] + "..." if len(segment_description) > 100 else segment_description,
                    "transcription_model": self.model,
                    "notes": segment_description,
                    "segment_start_time": round(start_time, 2),
                    "segment_end_time": round(end_time, 2),
                    "has_audio": bool(audio_text),
                    "audio_transcription": audio_text if audio_text else None,
                    "audio_transcription_list": audio_transcription_list if i == 0 else None,  # 只在第一个chunk保存完整列表
                }
                
                chunk = ConversionChunk(
                    text=segment_description,
                    chunk_index=i,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)
                
                self._report_progress(progress_end, f"片段 {i+1}/{len(segments)} 分析完成")
            
            self._report_progress(1.0, "视频分析完成")
            
            return ConversionOutput(
                status="success",
                chunks=chunks,
                metadata={
                    "converter_provider": "video_api_converter",
                    "converter_version": "1.0.0",
                    "conversion_time": datetime.utcnow().isoformat() + "Z",
                    "video_duration": video_duration,
                    "segments_count": len(segments) if segments else 0,
                    "chunks_count": len(chunks),
                    "model": self.model,
                },
            )
        except Exception as e:
            return ConversionOutput(
                status="failed",
                chunks=[],
                metadata={
                    "converter_provider": "video_api_converter",
                    "converter_version": "1.0.0",
                },
                error=str(e),
            )
        finally:
            # 清理临时片段文件
            if segments:
                self._cleanup_temp_segments(segments)
    def supports(self, *, file_type: str, mime_type: str = None) -> bool:
        return file_type.lower() in self.SUPPORTED_EXTENSIONS


# Register the converter
ConverterFactory.register("video", VideoConverter, priority=0)


def main():
    """测试 VideoConverter 功能"""
    import sys
    # 再次尝试加载 .env 文件（确保能找到）
    load_env_file()
    # 显示当前环境变量状态（用于调试）
    print("当前环境变量状态:")
    print(f"  LLM_API_KEY: {'已设置' if os.environ.get('LLM_API_KEY') else '未设置'}")
    print(f"  LLM_BASE_URL: {os.environ.get('LLM_BASE_URL', '未设置')}")
    print(f"  LLM_MODEL: {os.environ.get('LLM_MODEL', '未设置')}")
    print()
    
    # 检查是否提供了视频文件路径
    if len(sys.argv) < 2:
        print("用法: python video_converter.py <视频文件路径>")
        print("示例: python video_converter.py test_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("VideoConverter 测试")
    print("=" * 60)
    print(f"视频文件: {video_path}")
    print()
    
    # 进度回调函数
    def progress_callback(progress: float, message: str):
        print(f"[进度 {progress*100:.1f}%] {message}")
    
    try:
        # 创建转换器实例
        # 注意: 需要设置 API key 和 base_url
        # 可以通过环境变量 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL 设置，或在这里直接传入
        converter = VideoConverter(
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL"),
            model=os.environ.get("LLM_MODEL"),
            progress_callback=progress_callback,
        )
        
        print("开始转换视频...")
        print()
        # 执行转换
        result = converter.convert(video_path, source_type="file_path")
        print()
        print("=" * 60)
        print("转换结果")
        print("=" * 60)
        print(f"状态: {result.status}")
        if result.status == "success":
            print(f"Chunk 数量: {len(result.chunks)}")
            print()  
            for i, chunk in enumerate(result.chunks):
                print(f"--- Chunk {i+1} ---")
                print(f"文本长度: {len(chunk.text)} 字符")
                print(f"元数据: {chunk.metadata}")
                print()
                print("视频描述内容:")
                print("-" * 60)
                print(chunk.text)
                print("-" * 60)
                print()
            print("输出元数据:")
            print(result.metadata)
        else:
            print(f"错误: {result.error}")
            if result.error:
                print(f"错误详情: {result.error}")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请安装依赖: pip install 'volcengine-python-sdk[ark]'")
        sys.exit(1)
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置环境变量 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL")
        print("或创建 .env 文件并配置这些变量")
        sys.exit(1)
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()