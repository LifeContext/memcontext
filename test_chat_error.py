#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 chat 接口，复现 video_path 错误
"""

import json
import requests
import os
import sys

SERVER = "http://127.0.0.1:5019"
SESSION_INIT_ENDPOINT = f"{SERVER}/init_memory"
CHAT_ENDPOINT = f"{SERVER}/chat"

def init_memory(session):
    payload = {
        "user_id": "test_user",
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

def test_chat(session, query):
    """测试 chat 接口"""
    payload = {
        "query": query
    }
    try:
        resp = session.post(CHAT_ENDPOINT, json=payload, timeout=60)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")
        if resp.status_code != 200:
            print(f"Error response: {resp.text}")
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response text: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        raise
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    session = requests.Session()
    
    print("1. 初始化 memory system...")
    session_id = init_memory(session)
    print(f"Session ID: {session_id}")
    
    print("\n2. 测试 chat 接口，询问视频内容...")
    query = "f42906.mp4这个视频的内容主要是什么"
    print(f"Query: {query}")
    
    try:
        result = test_chat(session, query)
        print(f"\nSuccess! Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"\nFailed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

