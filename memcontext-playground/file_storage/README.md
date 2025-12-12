# 文件存储管理模块

统一的文件存储、定位和访问模块，支持视频、图片、文档等多种文件类型。

## 功能特性

- ✅ **统一文件存储**：所有上传文件统一管理，按类型和文件ID组织
- ✅ **视频片段定位**：支持通过时间戳精确定位视频片段（0.1秒级精度）
- ✅ **动态片段生成**：按需生成视频片段，自动缓存避免重复生成
- ✅ **元数据索引**：快速查找文件信息和元数据
- ✅ **HTTP API服务**：提供完整的RESTful API接口
- 🔄 **图片区域提取**：预留接口（待实现）
- 🔄 **文档页面提取**：预留接口（待实现）

## 目录结构

```
file_storage/
├── __init__.py              # 模块导出
├── storage_manager.py       # 核心存储管理类
├── file_types.py            # 文件类型定义和基类
├── video_handler.py         # 视频文件处理（已实现）
├── image_handler.py        # 图片文件处理（预留接口）
├── document_handler.py      # 文档文件处理（预留接口）
├── api_server.py           # HTTP API服务
├── utils.py                # 工具函数
├── example_usage.py        # 使用示例
└── README.md               # 本文档
```

## 存储目录结构

```
storage_base_path/
├── files/
│   ├── videos/             # 视频文件
│   │   └── {file_id}/
│   │       ├── original.mp4
│   │       └── segments/    # 视频片段
│   │           └── segment_{start}_{end}.mp4
│   ├── images/             # 图片文件
│   ├── documents/          # 文档文件
│   └── metadata/           # 元数据索引
│       └── files_index.json
```

## 快速开始

### 1. 基本使用

```python
from file_storage import FileStorageManager, FileType

# 初始化存储管理器
manager = FileStorageManager(
    storage_base_path="./storage",
    user_id="user123"
)

# 上传文件
file_record = manager.upload_file("video.mp4")
print(f"文件ID: {file_record.file_id}")

# 获取文件路径
file_path = manager.get_file_path(file_record.file_id)

# 获取视频片段（5-10秒）
if file_record.file_type == FileType.VIDEO:
    handler = manager.get_handler(FileType.VIDEO)
    segment_path = handler.get_segment_by_time(
        file_record.file_id,
        start_time=5.0,
        end_time=10.0
    )
    print(f"片段路径: {segment_path}")
```

### 2. HTTP API服务

```python
from file_storage.api_server import create_api_server

# 创建API服务器
server = create_api_server(
    storage_base_path="./storage",
    user_id="user123",
    host="0.0.0.0",
    port=5001
)

# 启动服务器
server.run()
```

### 3. API端点

#### 上传文件
```bash
POST /api/files/upload
Content-Type: multipart/form-data

file: <文件>
file_type: video (可选)
```

#### 获取文件
```bash
GET /api/files/{file_id}
```

#### 获取视频片段
```bash
GET /api/files/{file_id}/segment?start_time=5.0&end_time=10.0
# 或
GET /api/files/{file_id}/segment?start_time=5.0&duration=5.0
```

#### 获取文件元数据
```bash
GET /api/files/{file_id}/metadata
```

#### 列出所有文件
```bash
GET /api/files?file_type=video (可选)
```

#### 删除文件
```bash
DELETE /api/files/{file_id}
```

## 核心类说明

### FileStorageManager

文件存储管理器，提供文件上传、检索、删除等功能。

**主要方法：**
- `upload_file(file_path, file_type=None, metadata=None)` - 上传文件
- `get_file_path(file_id)` - 获取文件路径
- `get_file_record(file_id)` - 获取文件记录
- `get_file_metadata(file_id)` - 获取文件元数据
- `list_files(file_type=None)` - 列出文件
- `delete_file(file_id)` - 删除文件
- `get_handler(file_type)` - 获取文件类型处理器

### VideoHandler

视频文件处理器，支持视频片段定位和生成。

**主要方法：**
- `get_file_path(file_id)` - 获取视频文件路径
- `get_segment_path(file_id, location_info)` - 获取视频片段路径
- `get_segment_by_time(file_id, start_time, end_time=None, duration=None)` - 根据时间获取片段
- `list_segments(file_id)` - 列出所有已生成的片段
- `extract_metadata(file_path)` - 提取视频元数据

## 文件元数据结构

```python
FileRecord {
    file_id: str              # 唯一标识
    file_type: FileType       # 文件类型
    original_filename: str    # 原始文件名
    stored_path: str         # 存储路径
    upload_time: str         # 上传时间
    user_id: str            # 用户ID
    metadata: {
        # 视频特有
        duration: float      # 时长（秒）
        width: int          # 宽度
        height: int         # 高度
        codec: str          # 编码格式
        
        # 图片特有（预留）
        width: int
        height: int
        format: str
        
        # 文档特有（预留）
        page_count: int
    }
}
```

## 依赖要求

- Python 3.9+
- Flask (用于API服务)
- ffmpeg (用于视频处理)
- Pillow (可选，用于图片处理)

## 扩展开发

### 实现图片区域提取

在 `image_handler.py` 中实现 `get_segment_path` 方法：

```python
def get_segment_path(self, file_id: str, location_info: Dict[str, Any]) -> Optional[str]:
    x = location_info['x']
    y = location_info['y']
    width = location_info['width']
    height = location_info['height']
    
    # 使用PIL裁剪图片
    from PIL import Image
    img = Image.open(self.get_file_path(file_id))
    region = img.crop((x, y, x + width, y + height))
    
    # 保存裁剪后的图片
    # ...
```

### 实现文档页面提取

在 `document_handler.py` 中实现 `get_segment_path` 方法：

```python
def get_segment_path(self, file_id: str, location_info: Dict[str, Any]) -> Optional[str]:
    page_number = location_info['page_number']
    
    # 使用PyPDF2或pdf2image提取页面
    # ...
```

## 注意事项

1. **ffmpeg要求**：视频片段生成需要系统安装ffmpeg
2. **存储空间**：生成的视频片段会占用额外存储空间
3. **文件ID唯一性**：基于用户ID和时间戳生成，确保唯一性
4. **线程安全**：当前实现未考虑多线程并发，生产环境需要添加锁机制

## 后续迁移

本模块设计为独立模块，可轻松迁移到 `memcontext-chromadb` 或其他模块：

1. 复制 `file_storage/` 目录到目标位置
2. 更新导入路径
3. 根据需要调整配置

## 许可证

Apache 2.0
