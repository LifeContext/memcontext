import asyncio
import os
import torch
from dataclasses import dataclass
import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from imagebind.models import imagebind_model

from .._utils import logger
from ..base import BaseVectorStorage
from .._videoutil import encode_video_segments, encode_string_query


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    
    def __post_init__(self):

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["llm"]["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NanoVectorDBVideoSegmentStorage(BaseVectorStorage):
    embedding_func = None
    segment_retrieval_top_k: float = 2
    
    def __post_init__(self):
        
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["video_embedding_batch_num"]
        self._client = NanoVectorDB(
            self.global_config["video_embedding_dim"], storage_file=self._client_file_name
        )
        self.top_k = self.global_config.get(
            "segment_retrieval_top_k", self.segment_retrieval_top_k
        )
    
    async def upsert(self, video_name, segment_index2name, video_output_format):
        # 使用本地模型文件，不重新下载
        model_path = "/root/models/imagebind_huge.pth"
        device = torch.device("cuda")
        embedder = imagebind_model.imagebind_huge(pretrained=False)
        state_dict = torch.load(model_path, map_location=device)
        embedder.load_state_dict(state_dict)
        embedder.to(device)
        embedder.eval()
        
        logger.info(f"Inserting {len(segment_index2name)} segments to {self.namespace}")
        if not len(segment_index2name):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data, video_paths = [], []
        
        # 使用 FileStorageManager 的 segments/ 目录
        file_storage_manager = self.global_config.get("file_storage_manager")
        file_storage_id = self.global_config.get("file_storage_id")
        
        if not file_storage_manager or not file_storage_id:
            raise ValueError("file_storage_manager and file_storage_id are required in global_config")
        
        from file_storage import FileType
        segments_dir = os.path.join(
            file_storage_manager.storage_base_path,
            'files', 'videos', file_storage_id, 'segments'
        )
        cache_path = segments_dir
        logger.info(f"Using FileStorageManager segments directory: {segments_dir}")
        
        def _format_time_for_filename(seconds: float) -> str:
            return f"{seconds:.2f}".replace('.', '_')
        
        index_list = list(segment_index2name.keys())
        for index in index_list:
            list_data.append({
                "__id__": f"{video_name}_{index}",
                "__video_name__": video_name,
                "__index__": index,
            })
            segment_name = segment_index2name[index]
            
            # 从 segment_index2name 中提取时间信息，格式：{timestamp}-{index}-{start}-{end}
            parts = segment_name.split('-')
            if len(parts) >= 4:
                start_time = parts[-2]
                end_time = parts[-1]
                video_file = os.path.join(cache_path, f"segment_{_format_time_for_filename(float(start_time))}_{_format_time_for_filename(float(end_time))}.{video_output_format}")
            else:
                raise ValueError(f"Invalid segment_name format: {segment_name}")
            
            video_paths.append(video_file)
        batches = [
            video_paths[i: i + self._max_batch_size]
            for i in range(0, len(video_paths), self._max_batch_size)
        ]
        embeddings = []
        for _batch in tqdm(batches, desc=f"Encoding Video Segments {video_name}"):
            batch_embeddings = encode_video_segments(_batch, embedder)
            embeddings.append(batch_embeddings)
        embeddings = torch.concat(embeddings, dim=0)
        embeddings = embeddings.numpy()
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results
    
    async def query(self, query: str):
        # 使用本地模型文件，不重新下载
        model_path = "/root/models/imagebind_huge.pth"
        embedder = imagebind_model.imagebind_huge(pretrained=False).cuda()
        state_dict = torch.load(model_path, map_location="cuda")
        embedder.load_state_dict(state_dict)
        embedder.eval()
        
        embedding = encode_string_query(query, embedder)
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=self.top_k,
            better_than_threshold=-1,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    async def index_done_callback(self):
        self._client.save()
