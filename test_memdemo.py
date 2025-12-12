#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：给 memdemo Flask 项目提交视频并触发 VideoRAG 记忆同步
"""

import json
import pathlib
import requests
import os
SERVER = "http://127.0.0.1:5019"
SESSION_INIT_ENDPOINT = f"{SERVER}/init_memory"
ADD_MM_ENDPOINT = f"{SERVER}/add_multimodal_memory"

def init_memory(session):
    payload = {
        "user_id": "video_user5",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
        "siliconflow_key": os.environ.get("SILICONFLOW_API_KEY", ""),
    }
    resp = session.post(SESSION_INIT_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"初始化失败: {data}")
    return data["session_id"]

def add_video(session, video_path):
    payload = {
        "converter_type": "videorag",
        # 若想自定义 VideoRAG 参数，可在此 JSON 中补充
        "converter_kwargs": json.dumps({
            "working_dir": "./videorag-workdir",
            "auto_summary": False,
            "deepseek_key": "",
            "siliconflow_key": "",
        })
    }
    files = {
        "file": (pathlib.Path(video_path).name, open(video_path, "rb"), "video/mp4")
    }
    resp = session.post(
        ADD_MM_ENDPOINT,
        data=payload,
        files=files,
        timeout=24000,  # VideoRAG 处理较耗时
    )
    print(resp.text)
    resp.raise_for_status()
    return resp.json()

def main():
    session = requests.Session()
    session_id = init_memory(session)
    print(f"Session ready: {session_id}")

    # result = add_video(session=session, video_path="/root/repo/uni-mem/files/test_video.mp4")
    result = add_video(session=session, video_path="/root/repo/uni-mem/files/BigBuckBunny_320x180.mp4")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()