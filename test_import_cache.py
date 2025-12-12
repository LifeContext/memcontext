#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试从 temp_memory 缓存导入
"""

import json
import requests
import os

SERVER = "http://127.0.0.1:5019"
SESSION_INIT_ENDPOINT = f"{SERVER}/init_memory"
IMPORT_CACHE_ENDPOINT = f"{SERVER}/import_from_cache"

def init_memory(session):
    payload = {
        "user_id": "video_user3",
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
        "siliconflow_key": os.environ.get("SILICONFLOW_API_KEY"),
    }
    resp = session.post(SESSION_INIT_ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"初始化失败: {data}")
    return data["session_id"]

def import_from_cache(session, file_id):
    """从缓存导入"""
    payload = {
        "file_id": file_id
    }
    resp = session.post(IMPORT_CACHE_ENDPOINT, json=payload, timeout=600)
    print(resp.text)
    resp.raise_for_status()
    return resp.json()

def main():
    session = requests.Session()
    session_id = init_memory(session)
    print(f"Session ready: {session_id}")

    # 从 temp_memory 文件获取 file_id
    # file_id 就是文件名（不含 .json 扩展名）
    file_id = "0b5b1bf25a408121ddfd3c44b9b1ec3a79964185b4a31502ad6505a51180ef71"
    
    print(f"Importing from cache: {file_id}")
    result = import_from_cache(session=session, file_id=file_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

