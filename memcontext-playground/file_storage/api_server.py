"""
文件存储HTTP API服务
提供文件上传、访问和定位接口
"""

import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from typing import Optional
from .storage_manager import FileStorageManager
from .file_types import FileType


class FileStorageAPIServer:
    """文件存储API服务器"""
    
    def __init__(
        self,
        storage_manager: FileStorageManager,
        host: str = "0.0.0.0",
        port: int = 5001,
        debug: bool = False
    ):
        """
        初始化API服务器
        
        Args:
            storage_manager: 文件存储管理器实例
            host: 服务器主机地址
            port: 服务器端口
            debug: 是否开启调试模式
        """
        self.storage_manager = storage_manager
        self.host = host
        self.port = port
        self.debug = debug
        
        self.app = Flask(__name__)
        self._register_routes()
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/api/files/upload', methods=['POST'])
        def upload_file():
            """文件上传接口"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # 获取可选参数
                file_type_str = request.form.get('file_type')
                file_type = None
                if file_type_str:
                    try:
                        file_type = FileType(file_type_str)
                    except ValueError:
                        return jsonify({'error': f'Invalid file_type: {file_type_str}'}), 400
                
                # 保存临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    file.save(tmp_file.name)
                    tmp_path = tmp_file.name
                
                try:
                    # 上传到存储系统
                    file_record = self.storage_manager.upload_file(
                        file_path=tmp_path,
                        file_type=file_type,
                        metadata={
                            'original_filename': file.filename,
                            'content_type': file.content_type
                        }
                    )
                    
                    return jsonify({
                        'success': True,
                        'file_id': file_record.file_id,
                        'file_type': file_record.file_type.value,
                        'metadata': file_record.metadata
                    }), 200
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/files/<file_id>', methods=['GET'])
        def get_file(file_id: str):
            """获取完整文件"""
            try:
                file_path = self.storage_manager.get_file_path(file_id)
                if file_path is None:
                    return jsonify({'error': 'File not found'}), 404
                
                return send_file(file_path, as_attachment=True)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/files/<file_id>/segment', methods=['GET'])
        def get_video_segment(file_id: str):
            """获取视频片段"""
            try:
                start_time = request.args.get('start_time', type=float)
                end_time = request.args.get('end_time', type=float)
                duration = request.args.get('duration', type=float)
                
                if start_time is None:
                    return jsonify({'error': 'start_time is required'}), 400
                
                if end_time is None and duration is None:
                    return jsonify({'error': 'Either end_time or duration is required'}), 400
                
                if end_time is None:
                    end_time = start_time + duration
                
                # 获取文件记录
                file_record = self.storage_manager.get_file_record(file_id)
                if file_record is None:
                    return jsonify({'error': 'File not found'}), 404
                
                if file_record.file_type != FileType.VIDEO:
                    return jsonify({'error': 'File is not a video'}), 400
                
                # 获取视频处理器
                handler = self.storage_manager.get_handler(FileType.VIDEO)
                segment_path = handler.get_segment_by_time(
                    file_id=file_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if segment_path is None:
                    return jsonify({'error': 'Failed to generate segment'}), 500
                
                return send_file(segment_path, as_attachment=True)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/files/<file_id>/metadata', methods=['GET'])
        def get_file_metadata(file_id: str):
            """获取文件元数据"""
            try:
                file_record = self.storage_manager.get_file_record(file_id)
                if file_record is None:
                    return jsonify({'error': 'File not found'}), 404
                
                return jsonify({
                    'file_id': file_record.file_id,
                    'file_type': file_record.file_type.value,
                    'original_filename': file_record.original_filename,
                    'upload_time': file_record.upload_time,
                    'user_id': file_record.user_id,
                    'metadata': file_record.metadata
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/files', methods=['GET'])
        def list_files():
            """列出所有文件"""
            try:
                file_type_str = request.args.get('file_type')
                file_type = None
                if file_type_str:
                    try:
                        file_type = FileType(file_type_str)
                    except ValueError:
                        return jsonify({'error': f'Invalid file_type: {file_type_str}'}), 400
                
                files = self.storage_manager.list_files(file_type=file_type)
                
                return jsonify({
                    'count': len(files),
                    'files': [f.to_dict() for f in files]
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/files/<file_id>', methods=['DELETE'])
        def delete_file(file_id: str):
            """删除文件"""
            try:
                success = self.storage_manager.delete_file(file_id)
                if success:
                    return jsonify({'success': True, 'message': 'File deleted'}), 200
                else:
                    return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({'status': 'ok', 'service': 'file_storage_api'}), 200
    
    def run(self):
        """启动API服务器"""
        self.app.run(host=self.host, port=self.port, debug=self.debug)
    
    def get_app(self):
        """获取Flask应用实例（用于集成到其他应用）"""
        return self.app


def create_api_server(
    storage_base_path: str,
    user_id: str = "default",
    host: str = "0.0.0.0",
    port: int = 5001,
    debug: bool = False
) -> FileStorageAPIServer:
    """
    创建并返回API服务器实例
    
    Args:
        storage_base_path: 存储根路径
        user_id: 用户ID
        host: 服务器主机地址
        port: 服务器端口
        debug: 是否开启调试模式
    
    Returns:
        API服务器实例
    """
    storage_manager = FileStorageManager(storage_base_path, user_id)
    return FileStorageAPIServer(storage_manager, host, port, debug)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python api_server.py <storage_base_path> [port]")
        sys.exit(1)
    
    storage_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5001
    
    server = create_api_server(storage_path, port=port, debug=True)
    print(f"Starting file storage API server on port {port}...")
    server.run()
