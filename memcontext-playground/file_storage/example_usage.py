"""
文件存储管理模块使用示例
"""

import os
from .storage_manager import FileStorageManager
from .file_types import FileType
from .api_server import create_api_server


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 初始化存储管理器
    storage_path = "./test_storage"
    user_id = "test_user"
    manager = FileStorageManager(storage_path, user_id)
    
    # 上传文件（假设有一个测试视频文件）
    test_video_path = "test_video.mp4"  # 替换为实际文件路径
    
    if os.path.exists(test_video_path):
        # 上传文件
        file_record = manager.upload_file(test_video_path)
        print(f"文件上传成功: {file_record.file_id}")
        print(f"文件类型: {file_record.file_type.value}")
        print(f"存储路径: {file_record.stored_path}")
        print(f"元数据: {file_record.metadata}")
        
        # 获取文件路径
        file_path = manager.get_file_path(file_record.file_id)
        print(f"文件路径: {file_path}")
        
        # 获取视频片段（如果文件是视频）
        if file_record.file_type == FileType.VIDEO:
            handler = manager.get_handler(FileType.VIDEO)
            
            # 获取5-10秒的片段
            segment_path = handler.get_segment_by_time(
                file_record.file_id,
                start_time=5.0,
                end_time=10.0
            )
            print(f"视频片段路径: {segment_path}")
            
            # 列出所有片段
            segments = handler.list_segments(file_record.file_id)
            print(f"已生成片段数: {len(segments)}")
            for seg in segments:
                print(f"  片段: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
    else:
        print(f"测试文件不存在: {test_video_path}")


def example_api_server():
    """API服务器使用示例"""
    print("\n=== API服务器示例 ===")
    
    storage_path = "./test_storage"
    user_id = "test_user"
    
    # 创建API服务器
    server = create_api_server(
        storage_base_path=storage_path,
        user_id=user_id,
        host="0.0.0.0",
        port=5001,
        debug=True
    )
    
    print("API服务器已创建")
    print("启动服务器请运行: server.run()")
    print("\n可用端点:")
    print("  POST /api/files/upload - 上传文件")
    print("  GET  /api/files/<file_id> - 获取文件")
    print("  GET  /api/files/<file_id>/segment?start_time=X&end_time=Y - 获取视频片段")
    print("  GET  /api/files/<file_id>/metadata - 获取文件元数据")
    print("  GET  /api/files - 列出所有文件")
    print("  DELETE /api/files/<file_id> - 删除文件")


def example_file_management():
    """文件管理示例"""
    print("\n=== 文件管理示例 ===")
    
    storage_path = "./test_storage"
    user_id = "test_user"
    manager = FileStorageManager(storage_path, user_id)
    
    # 列出所有文件
    all_files = manager.list_files()
    print(f"所有文件数: {len(all_files)}")
    
    # 列出特定类型的文件
    video_files = manager.list_files(FileType.VIDEO)
    print(f"视频文件数: {len(video_files)}")
    
    # 获取文件元数据
    if all_files:
        file_id = all_files[0].file_id
        metadata = manager.get_file_metadata(file_id)
        print(f"文件 {file_id} 的元数据: {metadata}")


if __name__ == '__main__':
    print("文件存储管理模块使用示例\n")
    
    # 运行示例
    example_basic_usage()
    example_file_management()
    example_api_server()
    
    print("\n示例完成！")
