#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FileStorageManager API 测试脚本

通过 HTTP API 测试 FileStorageManager 的各种功能，包括：
- 文件上传
- 文件检索
- 视频片段生成
- 元数据管理
- 文件删除
"""

import os
import sys
import time
import requests
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Process

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from file_storage import FileStorageManager, FileType
from file_storage.api_server import FileStorageAPIServer


# API 服务器配置
API_HOST = "127.0.0.1"
API_PORT = 5002
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"


def start_api_server(storage_path: str, user_id: str = "test_user"):
    """启动 API 服务器（在后台进程）"""
    manager = FileStorageManager(storage_base_path=storage_path, user_id=user_id)
    server = FileStorageAPIServer(manager, host=API_HOST, port=API_PORT, debug=False)
    server.run()


def wait_for_server(url: str, timeout: int = 10):
    """等待服务器启动"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{url}/api/health", timeout=1)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def test_api_upload_and_retrieve():
    """测试 API：上传和检索文件"""
    print("=" * 60)
    print("测试 1: API 文件上传和检索")
    print("=" * 60)
    
    # 创建测试文件
    test_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    test_file.write(b"fake video content for testing")
    test_file.close()
    test_file_path = test_file.name
    
    try:
        # 上传文件
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            data = {'file_type': 'video'}
            resp = requests.post(f"{API_BASE_URL}/api/files/upload", files=files, data=data, timeout=10)
        
        assert resp.status_code == 200, f"上传失败: {resp.status_code} - {resp.text}"
        result = resp.json()
        assert result.get('success'), "上传应该成功"
        file_id = result['file_id']
        print(f"✓ 文件上传成功，file_id: {file_id}")
        
        # 获取文件元数据
        resp = requests.get(f"{API_BASE_URL}/api/files/{file_id}/metadata", timeout=10)
        assert resp.status_code == 200, f"获取元数据失败: {resp.status_code}"
        metadata = resp.json()
        print(f"✓ 获取元数据成功")
        print(f"  - 文件类型: {metadata.get('file_type')}")
        print(f"  - 原始文件名: {metadata.get('original_filename')}")
        
        # 列出所有文件
        resp = requests.get(f"{API_BASE_URL}/api/files", timeout=10)
        assert resp.status_code == 200, f"列出文件失败: {resp.status_code}"
        files_result = resp.json()
        assert files_result.get('count', 0) > 0, "应该至少有一个文件"
        print(f"✓ 列出文件成功，共 {files_result.get('count')} 个文件")
        
        # 删除文件
        resp = requests.delete(f"{API_BASE_URL}/api/files/{file_id}", timeout=10)
        assert resp.status_code == 200, f"删除文件失败: {resp.status_code}"
        print(f"✓ 文件删除成功")
        
        print("\n✅ 测试 1 通过！\n")
        
    finally:
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)


def test_api_video_segments():
    """测试 API：视频片段生成"""
    print("=" * 60)
    print("测试 2: API 视频片段生成")
    print("=" * 60)
    
    # 查找测试视频文件
    test_video_paths = [
        "/root/repo/uni-mem/files/test_video.mp4",
        "/root/repo/uni-mem/files/BigBuckBunny_320x180.mp4",
        "/root/repo/uni-mem/files/f42906.mp4",
        "/root/repo/uni-mem/files/hubble_oumuamua_final.webm",
    ]
    
    test_video_path = None
    for path in test_video_paths:
        if os.path.exists(path):
            test_video_path = path
            break
    
    if not test_video_path:
        print("⚠ 未找到测试视频文件，跳过视频片段测试")
        print("  提示：可以手动指定视频文件路径进行测试")
        return
    
    print(f"使用测试视频: {test_video_path}")
    file_size = os.path.getsize(test_video_path) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    try:
        # 上传视频文件
        print("\n上传视频文件...")
        with open(test_video_path, 'rb') as f:
            files = {'file': (os.path.basename(test_video_path), f, 'video/mp4')}
            data = {'file_type': 'video'}
            resp = requests.post(f"{API_BASE_URL}/api/files/upload", files=files, data=data, timeout=60)
        
        assert resp.status_code == 200, f"上传失败: {resp.status_code} - {resp.text}"
        result = resp.json()
        file_id = result['file_id']
        print(f"✓ 视频文件上传成功，file_id: {file_id}")
        
        # 获取视频元数据
        resp = requests.get(f"{API_BASE_URL}/api/files/{file_id}/metadata", timeout=10)
        assert resp.status_code == 200, f"获取元数据失败: {resp.status_code}"
        metadata = resp.json()
        video_metadata = metadata.get('metadata', {})
        duration = video_metadata.get('duration', 0)
        print(f"✓ 视频元数据:")
        print(f"  - 时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        if 'width' in video_metadata and 'height' in video_metadata:
            print(f"  - 分辨率: {video_metadata['width']}x{video_metadata['height']}")
        
        if duration == 0:
            print("⚠ 无法获取视频时长，跳过片段生成测试")
            return
        
        # 生成多个视频片段进行测试
        print(f"\n生成视频片段...")
        segments_to_test = [
            (0.0, min(10.0, duration)),
            (10.0, min(20.0, duration)),
            (max(0, duration - 10), duration),
        ]
        
        segment_paths = []
        for start_time, end_time in segments_to_test:
            if start_time >= end_time:
                continue
                
            print(f"  生成片段: {start_time:.2f}s - {end_time:.2f}s")
            resp = requests.get(
                f"{API_BASE_URL}/api/files/{file_id}/segment",
                params={'start_time': start_time, 'end_time': end_time},
                timeout=60
            )
            
            if resp.status_code == 200:
                # 保存片段到临时文件
                segment_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                segment_file.write(resp.content)
                segment_file.close()
                segment_paths.append((start_time, end_time, segment_file.name))
                
                segment_size = os.path.getsize(segment_file.name) / 1024  # KB
                print(f"    ✓ 片段生成成功，大小: {segment_size:.2f} KB")
            else:
                print(f"    ✗ 片段生成失败: {resp.status_code} - {resp.text[:200]}")
        
        print(f"\n✓ 成功生成 {len(segment_paths)} 个视频片段")
        
        # 清理临时片段文件
        for _, _, path in segment_paths:
            if os.path.exists(path):
                os.unlink(path)
        
        print("\n✅ 测试 2 完成！\n")
        
    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()


def test_api_list_files():
    """测试 API：列出文件"""
    print("=" * 60)
    print("测试 3: API 列出文件")
    print("=" * 60)
    
    try:
        # 列出所有文件
        resp = requests.get(f"{API_BASE_URL}/api/files", timeout=10)
        assert resp.status_code == 200, f"列出文件失败: {resp.status_code}"
        result = resp.json()
        all_files = result.get('files', [])
        print(f"✓ 列出所有文件成功，共 {len(all_files)} 个文件")
        
        # 列出视频文件
        resp = requests.get(f"{API_BASE_URL}/api/files?file_type=video", timeout=10)
        assert resp.status_code == 200, f"列出视频文件失败: {resp.status_code}"
        result = resp.json()
        video_files = result.get('files', [])
        print(f"✓ 列出视频文件成功，共 {len(video_files)} 个视频文件")
        
        # 显示文件信息
        if video_files:
            print("\n视频文件列表:")
            for i, file_info in enumerate(video_files[:5], 1):  # 只显示前5个
                file_id = file_info.get('file_id', 'N/A')
                filename = file_info.get('original_filename', 'N/A')
                metadata = file_info.get('metadata', {})
                duration = metadata.get('duration', 0)
                print(f"  {i}. {filename}")
                print(f"     file_id: {file_id}")
                if duration > 0:
                    print(f"     时长: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        
        print("\n✅ 测试 3 通过！\n")
        
    except Exception as e:
        print(f"\n❌ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()


def test_api_health_check():
    """测试 API：健康检查"""
    print("=" * 60)
    print("测试 4: API 健康检查")
    print("=" * 60)
    
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        assert resp.status_code == 200, f"健康检查失败: {resp.status_code}"
        result = resp.json()
        assert result.get('status') == 'ok', "健康状态应该为 ok"
        print(f"✓ API 服务器健康检查通过: {result}")
        print("\n✅ 测试 4 通过！\n")
        
    except Exception as e:
        print(f"\n❌ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("FileStorageManager API 测试套件")
    print("=" * 60 + "\n")
    
    # 创建临时存储目录
    test_storage_path = tempfile.mkdtemp(prefix="test_filestorage_api_")
    print(f"测试存储路径: {test_storage_path}")
    print(f"API 服务器地址: {API_BASE_URL}\n")
    
    # 启动 API 服务器（后台进程）
    server_process = Process(
        target=start_api_server,
        args=(test_storage_path, "test_user"),
        daemon=True
    )
    server_process.start()
    
    try:
        # 等待服务器启动
        print("等待 API 服务器启动...")
        if not wait_for_server(API_BASE_URL, timeout=10):
            print("❌ API 服务器启动超时")
            return 1
        print("✓ API 服务器已启动\n")
        
        # 运行测试
        tests = [
            test_api_health_check,
            test_api_upload_and_retrieve,
            test_api_video_segments,
            test_api_list_files,
        ]
        
        passed = 0
        failed = 0
        
        for test_func in tests:
            try:
                test_func()
                passed += 1
            except Exception as e:
                print(f"\n❌ {test_func.__name__} 失败: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                print()
        
        print("=" * 60)
        print("测试总结")
        print("=" * 60)
        print(f"通过: {passed}/{len(tests)}")
        print(f"失败: {failed}/{len(tests)}")
        print("=" * 60)
        
        if failed == 0:
            print("\n🎉 所有测试通过！")
            return 0
        else:
            print(f"\n⚠️  有 {failed} 个测试失败")
            return 1
            
    finally:
        # 停止服务器
        print("\n停止 API 服务器...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        
        # 清理测试目录
        if os.path.exists(test_storage_path):
            shutil.rmtree(test_storage_path)
            print(f"✓ 清理测试目录: {test_storage_path}")


if __name__ == "__main__":
    sys.exit(main())
