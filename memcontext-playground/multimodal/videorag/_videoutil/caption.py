import json
import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

# time parsing helper to avoid eval on strings like "00:30"
def _to_seconds(t):
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if ":" in s:
        parts = [p for p in s.split(":") if p != ""]
        try:
            parts = [float(p) for p in parts]
        except Exception:
            return 0.0
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, m, sec = parts[-3], parts[-2], parts[-1]
        return h * 3600 + m * 60 + sec
    try:
        return float(s)
    except Exception:
        return 0.0

# Import prompt from centralized prompts file
try:
    from prompts import VIDEO_STRUCTURED_CAPTION_PROMPT
    STRUCTURED_PROMPT_TEMPLATE = VIDEO_STRUCTURED_CAPTION_PROMPT
except ImportError:
    # Fallback if import fails; keep placeholders for intervals/transcript/focus
    STRUCTURED_PROMPT_TEMPLATE = (
        "你是视频逐帧描述助手，请将视觉内容与字幕融合，按帧时间段输出中文时间轴，"
        "每行格式为`[start -> end] 描述`，描述须包含画面关键信息并结合对应时间的字幕内容。"
        "{focus_clause}"
        "\n帧时间段：\n{intervals}\n"
        "字幕：\n{transcript}\n"
        "请直接输出时间轴列表，不要额外说明。"
    )

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames

def _format_time_intervals(frame_times):
    """Return list of '[start -> end]' strings from frame sampling points."""
    times = list(frame_times)
    intervals = []
    for i in range(len(times) - 1):
        intervals.append(f"[{times[i]:.2f}s -> {times[i+1]:.2f}s]")
    return intervals


def _extract_json_from_response(raw_text: str) -> tuple[str, dict]:
    clean_text = raw_text.replace("<|endoftext|>", "").strip()
    fenced = re.search(r"```(?:json)?(.*?)```", clean_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        candidate = clean_text
    candidate = candidate.replace("\u200b", "").strip()
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    else:
        candidate = ""

    metadata = {}
    if candidate:
        try:
            metadata = json.loads(candidate)
        except json.JSONDecodeError:
            metadata = {}

    return clean_text, metadata


def _normalize_actions_field(actions_value) -> str:
    if isinstance(actions_value, list):
        # 如果是数组，转换为逗号分隔的字符串
        return ", ".join(str(item).strip() for item in actions_value if item)
    
    if isinstance(actions_value, str):
        return actions_value.strip()
    
    return str(actions_value).strip()


def _merge_adjacent_identical_lines(text: str) -> str:
    """Merge consecutive timestamped lines with identical descriptions.

    Input expected lines like: "[0.00s -> 2.50s] 描述"
    If adjacent lines have identical 描述, merge their time ranges.
    """
    if not text:
        return text
    lines = text.splitlines()
    line_re = re.compile(r"^\s*\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]\s*(.*)\s*$")
    out_lines = []
    cur = None  # (start, end, desc)
    for L in lines:
        m = line_re.match(L)
        if not m:
            if cur is not None:
                s, e, d = cur
                out_lines.append(f"[{s:.2f}s -> {e:.2f}s] {d}")
                cur = None
            out_lines.append(L)
            continue
        s = float(m.group(1))
        e = float(m.group(2))
        d = m.group(3).strip()
        if cur is None:
            cur = (s, e, d)
        else:
            cs, ce, cd = cur
            # if descriptions identical and intervals are consecutive (start == previous end)
            if d == cd and abs(s - ce) <= 1e-6:
                cur = (cs, e, cd)
            else:
                out_lines.append(f"[{cs:.2f}s -> {ce:.2f}s] {cd}")
                cur = (s, e, d)
    if cur is not None:
        cs, ce, cd = cur
        out_lines.append(f"[{cs:.2f}s -> {ce:.2f}s] {cd}")
    return "\n".join(out_lines)


def _coarsen_frame_times(frame_times, max_samples=15):
    """Reduce frame_times to at most max_samples evenly spaced samples.

    If frame_times is an array-like of times, return a shorter list with the same start and end.
    """
    try:
        times = list(frame_times)
    except Exception:
        return frame_times
    n = len(times)
    if n <= max_samples or max_samples <= 0:
        return times
    # choose indices evenly spaced including first and last
    import math
    indices = [int(round(i * (n - 1) / (max_samples - 1))) for i in range(max_samples)]
    # ensure unique and sorted
    indices = sorted(list(dict.fromkeys(indices)))
    return [times[i] for i in indices]


def _integrate_transcript_into_captions(captions_text: str, transcript_text: str, max_chars=120) -> str:
    """For each caption line with a time range, find overlapping transcript snippets and append them.

    captions_text: multiple lines like "[start -> end] 描述"
    transcript_text: multiple lines like "[start -> end] text"
    Returns captions with appended ` 字幕:"..."` when transcript overlaps.
    """
    if not captions_text or not transcript_text:
        return captions_text
    line_re = re.compile(r"^\s*\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]\s*(.*)\s*$")
    t_re = re.compile(r"^\s*\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]\s*(.*)\s*$")
    # parse transcript lines
    t_lines = []
    for ln in transcript_text.splitlines():
        m = t_re.match(ln)
        if not m:
            continue
        ts = float(m.group(1))
        te = float(m.group(2))
        txt = m.group(3).strip()
        # Strip common speaker prefixes like 'speaker:' that may appear in ASR outputs
        txt = re.sub(r'^\s*(?:speaker\s*:|Speaker\s*:|SPEAKER\s*:)', '', txt).strip()
        if txt:
            t_lines.append((ts, te, txt))

    out_lines = []
    for ln in captions_text.splitlines():
        m = line_re.match(ln)
        if not m:
            out_lines.append(ln)
            continue
        cs = float(m.group(1))
        ce = float(m.group(2))
        desc = m.group(3).strip()
        # collect overlapping transcript snippets
        pieces = []
        for ts, te, txt in t_lines:
            if ts < ce and te > cs:
                pieces.append(txt)
        if pieces:
            joined = " ".join(pieces)
            joined = joined.replace('\n', ' ').strip()
            if len(joined) > max_chars:
                joined = joined[:max_chars].rsplit(' ', 1)[0] + '...'
            desc = f"{desc} 字幕：\"{joined}\""
        out_lines.append(f"[{cs:.2f}s -> {ce:.2f}s] {desc}")
    return "\n".join(out_lines)


def _ensure_metadata_defaults(metadata: dict, fallback_summary: str) -> dict:
    """
    确保 metadata 有必需的字段，但不设置 language（会在 merge_segment_information 中从 ASR 获取）。
    """
    metadata = metadata or {}
    metadata.setdefault("chunk_summary", fallback_summary)
    metadata.setdefault("scene_label", "unknown_scene")
    metadata.setdefault("objects_detected", [])
    
    # 标准化 actions 字段为字符串格式
    if "actions" in metadata:
        metadata["actions"] = _normalize_actions_field(metadata["actions"])
    else:
        metadata.setdefault("actions", "")
    
    metadata.setdefault("emotions", "")
    # language 不在这里设置，会在 merge_segment_information 中从 Whisper ASR 结果获取
    metadata.setdefault("confidence", 0.75)
    metadata.setdefault("notes", "")
    # 移除 model 可能错误输出的 language 字段
    if "language" in metadata:
        del metadata["language"]
    return metadata


def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MiniCPM-V captioning but no GPU is available.")
        model = AutoModel.from_pretrained(
            '/root/models/MiniCPM-V-2_6-int4',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            '/root/models/MiniCPM-V-2_6-int4', 
            trust_remote_code=True,
        )
        model.eval()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                try:
                    frame_times = segment_times_info[index]["frame_times"]
                    # 允许更多帧采样，但通过分批处理避免张量大小不匹配
                    # 如果 rough_num_frames_per_segment 较大，可以采样更多帧
                    max_samples = min(50, len(frame_times))  # 最多50帧，但不超过实际帧数
                    frame_times = _coarsen_frame_times(frame_times, max_samples=max_samples)
                    video_frames = encode_video(video, frame_times)
                    segment_transcript = transcripts.get(index, "")
                    start_time, end_time = segment_times_info[index]["timestamp"]
                    
                    # 分批处理图像，避免张量大小不匹配
                    num_frames = len(video_frames)
                    batch_size = 8  # 每批最多8帧，这是模型能稳定处理的帧数
                    num_batches = (num_frames + batch_size - 1) // batch_size if num_frames > batch_size else 1
                    all_captions = []  # 用于存储分批处理的结果
                    
                    if num_frames <= batch_size:
                        # 帧数较少，直接处理
                        intervals = "\n".join(_format_time_intervals(frame_times))
                        query = STRUCTURED_PROMPT_TEMPLATE.format(
                            intervals=intervals,
                            transcript=segment_transcript or "",
                            focus_clause="",
                        )
                        msgs = [{'role': 'user', 'content': video_frames + [query]}]
                        params = {}
                        params["use_image_id"] = False
                        params["max_slice_nums"] = min(10, max(2, num_frames // 2))
                        
                        segment_caption = model.chat(
                            image=None,
                            msgs=msgs,
                            tokenizer=tokenizer,
                            **params
                        )
                    else:
                        # 帧数较多，分批处理并合并结果
                        
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = min((batch_idx + 1) * batch_size, num_frames)
                            batch_frames = video_frames[start_idx:end_idx]
                            batch_frame_times = frame_times[start_idx:end_idx]
                            
                            intervals = "\n".join(_format_time_intervals(batch_frame_times))
                            # 对于非第一批，添加上下文提示
                            if batch_idx > 0:
                                focus_clause = f" 这是视频片段的一部分（第{batch_idx+1}/{num_batches}批），请结合之前的上下文。"
                            else:
                                focus_clause = ""
                            
                            query = STRUCTURED_PROMPT_TEMPLATE.format(
                                intervals=intervals,
                                transcript=segment_transcript or "",
                                focus_clause=focus_clause,
                            )
                            msgs = [{'role': 'user', 'content': batch_frames + [query]}]
                            params = {}
                            params["use_image_id"] = False
                            params["max_slice_nums"] = min(10, max(2, len(batch_frames) // 2))
                            
                            try:
                                batch_caption = model.chat(
                                    image=None,
                                    msgs=msgs,
                                    tokenizer=tokenizer,
                                    **params
                                )
                                all_captions.append(batch_caption)
                            except RuntimeError as e:
                                if "Sizes of tensors must match" in str(e):
                                    # 即使分批也失败，使用更小的批次或 transcript
                                    error_queue.put(f"Warning: Segment {index} batch {batch_idx+1} failed, using transcript fallback")
                                    batch_intervals = "\n".join(_format_time_intervals(batch_frame_times))
                                    all_captions.append(f"[{batch_frame_times[0]:.2f}s -> {batch_frame_times[-1]:.2f}s] {segment_transcript}")
                                else:
                                    raise
                            
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # 合并所有批次的 caption
                        segment_caption = "\n\n".join(all_captions)
                    
                    # Debug: 打印实际发送的 prompt 和 LLM 的原始响应
                    if index == 0:  # 只打印第一个 segment 的调试信息
                        print("=" * 80)
                        print(f"DEBUG: Processed {num_frames} frames in {num_batches if num_frames > batch_size else 1} batch(es)")
                        print("DEBUG: LLM raw response (first 1000 chars):")
                        print(segment_caption[:1000] if isinstance(segment_caption, str) else str(segment_caption)[:1000])
                        print("=" * 80)
                    
                    # 对于分批处理的结果，需要特殊处理 JSON 提取
                    if num_frames > batch_size:
                        # 分批处理的结果是多个 caption 的合并，可能不是标准 JSON 格式
                        # 尝试提取所有批次的文本内容
                        raw_text = segment_caption
                        # 尝试从最后一个批次提取 metadata（如果有）
                        try:
                            if all_captions and len(all_captions) > 0:
                                _, parsed_metadata = _extract_json_from_response(all_captions[-1])
                            else:
                                parsed_metadata = {}
                        except:
                            parsed_metadata = {}
                    else:
                        raw_text, parsed_metadata = _extract_json_from_response(segment_caption)
                    
                    # Perform inline dedupe/merge of adjacent identical timestamped lines
                    try:
                        raw_text = _merge_adjacent_identical_lines(raw_text)
                        if isinstance(parsed_metadata, dict) and isinstance(parsed_metadata.get("chunk_summary"), str):
                            parsed_metadata["chunk_summary"] = _merge_adjacent_identical_lines(parsed_metadata["chunk_summary"])
                    except Exception:
                        # In case merging fails, keep the original raw_text
                        pass
                    
                    # Debug: 打印解析后的 metadata
                    if index == 0:
                        print("DEBUG: Parsed metadata chunk_summary type:", type(parsed_metadata.get("chunk_summary")))
                        if "chunk_summary" in parsed_metadata:
                            chunk_summary_value = parsed_metadata["chunk_summary"]
                            print("DEBUG: chunk_summary value (first 200 chars):", str(chunk_summary_value)[:200])
                            if isinstance(chunk_summary_value, str) and chunk_summary_value.strip().startswith("{"):
                                print("DEBUG: WARNING! chunk_summary is a JSON string!")
                        print("=" * 80)
                    
                    normalized_metadata = _ensure_metadata_defaults(parsed_metadata, raw_text)
                    caption_result[index] = {
                        "raw": normalized_metadata["chunk_summary"],
                        "metadata": normalized_metadata,
                    }
                except RuntimeError as e:
                    # 处理张量大小不匹配等运行时错误
                    error_msg = str(e)
                    if "Sizes of tensors must match" in error_msg or "tensor" in error_msg.lower():
                        # 使用 transcript 作为 fallback
                        segment_transcript = transcripts.get(index, "")
                        start_time, end_time = segment_times_info[index]["timestamp"]
                        fallback_text = f"[{start_time:.2f}s -> {end_time:.2f}s] {segment_transcript}" if segment_transcript else f"[{start_time:.2f}s -> {end_time:.2f}s] 视频片段内容"
                        error_queue.put(f"Warning: Segment {index} caption failed (tensor size mismatch), using transcript fallback: {error_msg[:200]}")
                        caption_result[index] = {
                            "raw": fallback_text,
                            "metadata": _ensure_metadata_defaults({}, fallback_text),
                        }
                    else:
                        # 其他运行时错误，重新抛出
                        raise
                except Exception as e:
                    # 其他错误，使用 transcript 作为 fallback
                    segment_transcript = transcripts.get(index, "")
                    start_time, end_time = segment_times_info[index]["timestamp"]
                    fallback_text = f"[{start_time:.2f}s -> {end_time:.2f}s] {segment_transcript}" if segment_transcript else f"[{start_time:.2f}s -> {end_time:.2f}s] 视频片段内容"
                    error_queue.put(f"Warning: Segment {index} caption failed, using transcript fallback: {str(e)[:200]}")
                    caption_result[index] = {
                        "raw": fallback_text,
                        "metadata": _ensure_metadata_defaults({}, fallback_text),
                    }
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions, languages):
    inserting_segments = {}
    segment_total = len(segment_index2name)
    for index in segment_index2name:
        caption_entry = captions[index]
        if isinstance(caption_entry, dict):
            caption_text = caption_entry.get("raw", "")
            caption_metadata = caption_entry.get("metadata", {}).copy()
        else:
            caption_text = str(caption_entry)
            caption_metadata = {}

        # 清理模型可能插入的标记，例如中文的 [开始] 和 [结束]
        # Pull transcript early so integration can use it during cleaning
        transcript_text = transcripts.get(index, "")

        try:
            # 移除标记并整理多余空格
            caption_text = re.sub(r"\[开始\]|\[结束\]", "", caption_text)
            # 将多个连续空白替换为单个空格，保留换行用于时间轴分行
            caption_text = re.sub(r"[ \t]+", " ", caption_text)
            # 去掉行首尾多余空白
            caption_text = "\n".join([ln.strip() for ln in caption_text.splitlines() if ln.strip()])
            # merge strictly identical adjacent lines (by text) to compact output
            caption_text = _merge_adjacent_identical_lines(caption_text)
            # integrate ASR transcript snippets into caption lines for clarity
            try:
                if transcript_text:
                    caption_text = _integrate_transcript_into_captions(caption_text, transcript_text)
            except Exception:
                # fallback: keep merged caption_text unchanged
                pass
        except Exception:
            pass

        start_time, end_time = segment_times_info[index]["timestamp"]
        duration_seconds = float(max(end_time - start_time, 0.0))
        # Format time range as absolute seconds with two decimals (consistent across pipeline)
        # Use plain numeric values without unit suffix so downstream parsing (_to_seconds) works.
        time_range = f"{float(start_time):.2f}-{float(end_time):.2f}"

        # 使用从 Whisper ASR 检测到的语言
        detected_language = languages.get(index, "unknown")
        
        # 设置必需的 metadata 字段
        caption_metadata.setdefault("chunk_summary", caption_text)
        caption_metadata.setdefault("scene_label", "unknown_scene")
        if "objects_detected" not in caption_metadata:
            caption_metadata["objects_detected"] = []
        # 标准化 actions 字段为字符串格式
        caption_metadata["actions"] = _normalize_actions_field(caption_metadata.get("actions", ""))
        if "emotions" not in caption_metadata:
            caption_metadata["emotions"] = ""
        caption_metadata.setdefault("confidence", 0.75)
        caption_metadata.setdefault("notes", "")
        
        # 强制使用从 ASR 检测到的语言，覆盖 model 可能错误输出的值
        caption_metadata["language"] = detected_language
        
        caption_metadata["source_type"] = "video"
        caption_metadata["chunk_index"] = int(index)
        caption_metadata["chunk_count_estimate"] = segment_total
        caption_metadata["duration_seconds"] = duration_seconds
        caption_metadata["time_range"] = time_range
        caption_metadata["transcription_model"] = "faster-whisper-large-v3-turbo"

        inserting_segments[index] = {
            "content": f"Caption:\n{caption_text}\nTranscript:\n{transcript_text}\n\n",
            "time": time_range,
            "transcript": transcript_text,
            "frame_times": segment_times_info[index]["frame_times"].tolist(),
            "duration_seconds": duration_seconds,
            "metadata": caption_metadata,
        }
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MiniCPM-V captioning but no GPU is available.")
    if caption_model is None:
        raise RuntimeError("caption_model is not initialized for retrieved_segment_caption.")
    caption_result = {}
    
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = _to_seconds(timestamp[0]), _to_seconds(timestamp[1])
        video = VideoFileClip(video_path)
        # Clamp requested samples and coarsen to a small number to reduce repetitive captions
        try:
            num_sampled_frames = max(1, min(int(num_sampled_frames), 3))
        except Exception:
            num_sampled_frames = 3
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False).tolist()
        frame_times = _coarsen_frame_times(frame_times, max_samples=30)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index].get("transcript", "")
        intervals = "\n".join(_format_time_intervals(frame_times))
        focus_clause = f" 并重点提取：{refine_knowledge}。" if refine_knowledge else ""
        query = STRUCTURED_PROMPT_TEMPLATE.format(
            intervals=intervals,
            transcript=segment_transcript or "",
            focus_clause=focus_clause,
        )
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return caption_result
