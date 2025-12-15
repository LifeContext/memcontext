import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# 修改为绝对导入
try:
    # 尝试相对导入（当作为包使用时）
    from .utils import (
        OpenAIClient,
        get_timestamp,
        generate_id,
        gpt_user_profile_analysis,
        gpt_knowledge_extraction,
        ensure_directory_exists,
    )
    from . import prompts
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory, compute_segment_heat  # For H_THRESHOLD logic
    from .long_term import LongTermMemory
    from .updater import Updater
    from .retriever import Retriever
    from .multimodal import ConverterFactory
    from .multimodal.utils import guess_file_extension, guess_mime_type, compute_file_hash
except ImportError:
    # 回退到绝对导入（当作为独立模块使用时）
    from utils import (
        OpenAIClient,
        get_timestamp,
        generate_id,
        gpt_user_profile_analysis,
        gpt_knowledge_extraction,
        ensure_directory_exists,
    )
    import prompts
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory, compute_segment_heat  # For H_THRESHOLD logic
    from long_term import LongTermMemory
    from updater import Updater
    from retriever import Retriever
    from multimodal import ConverterFactory
    from multimodal.utils import guess_file_extension, guess_mime_type, compute_file_hash

# Heat threshold for triggering profile/knowledge update from mid-term memory
H_PROFILE_UPDATE_THRESHOLD = 5.0 
DEFAULT_ASSISTANT_ID = "default_assistant_profile"

class Memcontext:
    def __init__(self, user_id: str, 
                 openai_api_key: str, 
                 data_storage_path: str,
                 openai_base_url: str = None, 
                 assistant_id: str = DEFAULT_ASSISTANT_ID, 
                 short_term_capacity=10,
                 mid_term_capacity=2000,
                 long_term_knowledge_capacity=100,
                 retrieval_queue_capacity=7,
                 mid_term_heat_threshold=H_PROFILE_UPDATE_THRESHOLD,
                 mid_term_similarity_threshold=0.6,
                 llm_model="gpt-4o-mini",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_model_kwargs: dict = None,
                 multimodal_config: dict = None,
                 ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.embedding_model_name = embedding_model_name
        self.multimodal_config = multimodal_config or {}
        
        # Smart defaults for embedding_model_kwargs
        if embedding_model_kwargs is None:
            if 'bge-m3' in self.embedding_model_name.lower():
                print("INFO: Detected bge-m3 model, defaulting embedding_model_kwargs to {'use_fp16': True}")
                self.embedding_model_kwargs = {'use_fp16': True}
            else:
                self.embedding_model_kwargs = {}
        else:
            self.embedding_model_kwargs = embedding_model_kwargs


        print(f"Initializing Memcontext for user '{self.user_id}' and assistant '{self.assistant_id}'. Data path: {self.data_storage_path}")
        print(f"Using unified LLM model: {self.llm_model}")
        print(f"Using embedding model: {self.embedding_model_name} with kwargs: {self.embedding_model_kwargs}")
        if self.multimodal_config:
            print(f"Configuring multimodal converters: {list(self.multimodal_config.keys())}")
            for converter_type, config in self.multimodal_config.items():
                ConverterFactory.configure(converter_type, **config)

        # Initialize OpenAI Client
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)

        # Define file paths for user-specific data
        self.user_data_dir = os.path.join(self.data_storage_path, "users", self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_mid_term_path = os.path.join(self.user_data_dir, "mid_term.json")
        user_long_term_path = os.path.join(self.user_data_dir, "long_term_user.json") # User profile and their knowledge

        # Define file paths for assistant-specific data (knowledge)
        self.assistant_data_dir = os.path.join(self.data_storage_path, "assistants", self.assistant_id)
        assistant_long_term_path = os.path.join(self.assistant_data_dir, "long_term_assistant.json")

        # Ensure directories exist
        ensure_directory_exists(user_short_term_path) # ensure_directory_exists operates on the file path, creating parent dirs
        ensure_directory_exists(user_mid_term_path)
        ensure_directory_exists(user_long_term_path)
        ensure_directory_exists(assistant_long_term_path)

        # Initialize Memory Modules for User
        self.short_term_memory = ShortTermMemory(file_path=user_short_term_path, max_capacity=short_term_capacity)
        self.mid_term_memory = MidTermMemory(
            file_path=user_mid_term_path, 
            client=self.client, 
            max_capacity=mid_term_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )
        self.user_long_term_memory = LongTermMemory(
            file_path=user_long_term_path, 
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )

        # Initialize Memory Module for Assistant Knowledge
        self.assistant_long_term_memory = LongTermMemory(
            file_path=assistant_long_term_path, 
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )

        # Initialize Orchestration Modules
        self.updater = Updater(short_term_memory=self.short_term_memory, 
                               mid_term_memory=self.mid_term_memory, 
                               long_term_memory=self.user_long_term_memory, # Updater primarily updates user's LTM profile/knowledge
                               client=self.client,
                               topic_similarity_threshold=mid_term_similarity_threshold,  # 传递中期记忆相似度阈值
                               llm_model=self.llm_model)
        self.retriever = Retriever(
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            assistant_long_term_memory=self.assistant_long_term_memory, # Pass assistant LTM
            queue_capacity=retrieval_queue_capacity
        )
        
        self.mid_term_heat_threshold = mid_term_heat_threshold

    def _trigger_profile_and_knowledge_update_if_needed(self):
        """
        Checks mid-term memory for hot segments and triggers profile/knowledge update if threshold is met.
        Adapted from main_memoybank.py's update_user_profile_from_top_segment.
        Enhanced with parallel LLM processing for better performance.
        """
        if not self.mid_term_memory.heap:
            return

        # Peek at the top of the heap (hottest segment)
        # MidTermMemory heap stores (-H_segment, sid)
        neg_heat, sid = self.mid_term_memory.heap[0] 
        current_heat = -neg_heat

        if current_heat >= self.mid_term_heat_threshold:
            session = self.mid_term_memory.sessions.get(sid)
            if not session:
                self.mid_term_memory.rebuild_heap() # Clean up if session is gone
                return

            # Get unanalyzed pages from this hot session
            # A page is a dict: {"user_input": ..., "agent_response": ..., "timestamp": ..., "analyzed": False, ...}
            unanalyzed_pages = [p for p in session.get("details", []) if not p.get("analyzed", False)]

            if unanalyzed_pages:
                print(f"Memcontext: Mid-term session {sid} heat ({current_heat:.2f}) exceeded threshold. Analyzing {len(unanalyzed_pages)} pages for profile/knowledge update.")
                
                # 并行执行两个LLM任务：用户画像分析（已包含更新）、知识提取
                def task_user_profile_analysis():
                    print("Memcontext: Starting parallel user profile analysis and update...")
                    # 获取现有用户画像
                    existing_profile = self.user_long_term_memory.get_raw_user_profile(self.user_id)
                    if not existing_profile or existing_profile.lower() == "none":
                        existing_profile = "No existing profile data."
                    
                    # 直接输出更新后的完整画像
                    return gpt_user_profile_analysis(unanalyzed_pages, self.client, model=self.llm_model, existing_user_profile=existing_profile)
                
                def task_knowledge_extraction():
                    print("Memcontext: Starting parallel knowledge extraction...")
                    return gpt_knowledge_extraction(unanalyzed_pages, self.client, model=self.llm_model)
                
                # 使用并行任务执行                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # 提交两个主要任务
                    future_profile = executor.submit(task_user_profile_analysis)
                    future_knowledge = executor.submit(task_knowledge_extraction)
                    
                    # 等待结果
                    try:
                        updated_user_profile = future_profile.result()  # 直接是更新后的完整画像
                        knowledge_result = future_knowledge.result()
                    except Exception as e:
                        print(f"Error in parallel LLM processing: {e}")
                        return
                
                new_user_private_knowledge = knowledge_result.get("private")
                new_assistant_knowledge = knowledge_result.get("assistant_knowledge")

                # 直接使用更新后的完整用户画像
                if updated_user_profile and updated_user_profile.lower() != "none":
                    print("Memcontext: Updating user profile with integrated analysis...")
                    self.user_long_term_memory.update_user_profile(self.user_id, updated_user_profile, merge=False)  # 直接替换为新的完整画像
                
                # Add User Private Knowledge to user's LTM
                if new_user_private_knowledge and new_user_private_knowledge.lower() != "none":
                    for line in new_user_private_knowledge.split('\n'):
                         if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                            self.user_long_term_memory.add_user_knowledge(line.strip())

                # Add Assistant Knowledge to assistant's LTM
                if new_assistant_knowledge and new_assistant_knowledge.lower() != "none":
                    for line in new_assistant_knowledge.split('\n'):
                        if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                           self.assistant_long_term_memory.add_assistant_knowledge(line.strip()) # Save to dedicated assistant LTM

                # Mark pages as analyzed and reset session heat contributors
                for p in session["details"]:
                    p["analyzed"] = True # Mark all pages in session, or just unanalyzed_pages?
                                          # Original code marked all pages in session
                
                session["N_visit"] = 0 # Reset visits after analysis
                session["L_interaction"] = 0 # Reset interaction length contribution
                # session["R_recency"] = 1.0 # Recency will re-calculate naturally
                session["H_segment"] = compute_segment_heat(session) # Recompute heat with reset factors
                session["last_visit_time"] = get_timestamp() # Update last visit time
                
                self.mid_term_memory.rebuild_heap() # Heap needs rebuild due to H_segment change
                self.mid_term_memory.save()
                print(f"Memcontext: Profile/Knowledge update for session {sid} complete. Heat reset.")
            else:
                print(f"Memcontext: Hot session {sid} has no unanalyzed pages. Skipping profile update.")
        else:
            # print(f"Memcontext: Top session {sid} heat ({current_heat:.2f}) below threshold. No profile update.")
            pass # No action if below threshold

    def add_memory(self, user_input: str, agent_response: str, timestamp: str = None, meta_data: dict = None):
        """ 
        Adds a new QA pair (memory) to the system.
        meta_data is not used in the current refactoring but kept for future use.
        """
        if not timestamp:
            timestamp = get_timestamp()
        
        qa_pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
            "meta_data": meta_data or {}
        }
        self.short_term_memory.add_qa_pair(qa_pair)
        print(f"Memorycontext: Added QA to short-term. User: {user_input[:30]}...")

        if self.short_term_memory.is_full():
            print("Memorycontext: Short-term memory full. Processing to mid-term.")
            self.updater.process_short_term_to_mid_term()
        
        # After any memory addition that might impact mid-term, check for profile updates
        self._trigger_profile_and_knowledge_update_if_needed()

    def _needs_metadata(self, query: str) -> list:
        """
        方案 4：查询类型识别 - 判断查询是否需要 metadata，返回需要的字段列表
        """
        metadata_keywords = {
            'video_name': ['文件名', 'filename', '视频名', '视频文件', 'video name'],
            'name': ['片段名', '段名', 'name'],
            'time_range': ['时间范围', 'time range', '时间段', '时间', 'time', 'range'],
            'chunk_index': ['片段', 'chunk', 'segment', '第.*个片段', '片段编号', '片段数量'],
            'objects_detected': ['对象', 'object', '物体', '检测', 'detect', '识别'],
            'scene_label': ['场景', 'scene', '场景分类', '场景类型'],
            'duration_seconds': ['时长', 'duration', '秒', 'seconds', '多长时间']
        }
        
        query_lower = query.lower()
        needed_fields = []
        for field, keywords in metadata_keywords.items():
            if any(kw in query_lower for kw in keywords):
                needed_fields.append(field)
        
        return needed_fields
    
    def _filter_and_rank_by_metadata(self, retrieved_pages: list, needed_fields: list) -> list:
        """
        根据 metadata 字段过滤和排序检索结果，优先返回包含所需字段的页面
        """
        if not retrieved_pages:
            return retrieved_pages
        
        # 如果没有指定需要的字段，仍然优先返回有 metadata 的页面
        if not needed_fields:
            pages_with_metadata = []
            pages_without_metadata = []
            for page in retrieved_pages:
                meta_data = page.get('meta_data', {}) or {}
                if meta_data and len(meta_data) > 0:
                    pages_with_metadata.append(page)
                else:
                    pages_without_metadata.append(page)
            return pages_with_metadata + pages_without_metadata
        
        # 分离有 metadata 和没有 metadata 的页面
        pages_with_needed_metadata = []  # 包含所需字段的页面
        pages_with_other_metadata = []   # 有其他metadata但没有所需字段的页面
        pages_without_metadata = []      # 完全没有metadata的页面
        
        for page in retrieved_pages:
            meta_data = page.get('meta_data', {}) or {}
            if meta_data and len(meta_data) > 0:
                # 检查是否包含需要的字段
                has_needed_fields = any(
                    meta_data.get(field) not in [None, '', [], {}] 
                    for field in needed_fields
                )
                if has_needed_fields:
                    pages_with_needed_metadata.append(page)
                else:
                    pages_with_other_metadata.append(page)
            else:
                pages_without_metadata.append(page)
        
        # 优先返回包含所需 metadata 的页面
        return pages_with_needed_metadata + pages_with_other_metadata + pages_without_metadata
    

    def get_response(self, query: str, relationship_with_user="friend", style_hint="", user_conversation_meta_data: dict = None) -> str:
        """
        Generates a response to the user's query, incorporating memory and context.
        
        Args:
            query: 用户查询
            relationship_with_user: 与用户的关系
            style_hint: 风格提示
            user_conversation_meta_data: 当前对话的 metadata
        """
        print(f"Memorycontext: Generating response for query: '{query[:50]}...'")

        # 1. Retrieve context
        retrieval_results = self.retriever.retrieve_context(
            user_query=query,
            user_id=self.user_id
        )
        retrieved_pages = retrieval_results["retrieved_pages"]
        retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]
        retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]
        
        # 1.1 识别需要的 metadata 字段并重新排序检索结果
        needed_metadata_fields = self._needs_metadata(query)
        retrieved_pages = self._filter_and_rank_by_metadata(retrieved_pages, needed_metadata_fields)
        if needed_metadata_fields:
            print(f"Memorycontext: Query needs metadata fields: {needed_metadata_fields}, re-ranked {len(retrieved_pages)} pages")

        # 2. Get short-term history
        short_term_history = self.short_term_memory.get_all()
        history_text = "\n".join([
            f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
            for qa in short_term_history
        ])

        # 3. Format retrieved mid-term pages (retrieval_queue equivalent)
        def _format_meta(meta_obj):
            if not meta_obj or not isinstance(meta_obj, dict) or len(meta_obj) == 0:
                return "None"
            try:
                # 格式化 metadata，突出关键字段
                formatted = []
                key_fields = ['name', 'video_name', 'time_range', 'chunk_index', 'objects_detected', 'scene_label', 'duration_seconds']
                for key in key_fields:
                    if key in meta_obj and meta_obj[key]:
                        formatted.append(f"  {key}: {meta_obj[key]}")
                # 添加其他字段
                for key, value in meta_obj.items():
                    if key not in key_fields and value:
                        formatted.append(f"  {key}: {value}")
                if formatted:
                    return "\n" + "\n".join(formatted)
                return "None"
            except TypeError:
                return str(meta_obj)

        retrieval_text = "\n".join([
            f"【Historical Memory】\n"
            f"User: {page.get('user_input', '')}\n"
            f"Assistant: {page.get('agent_response', '')}\n"
            f"Time: {page.get('timestamp', '')}\n"
            f"Conversation chain overview: {page.get('meta_info','N/A')}\n"
            f"Metadata:{_format_meta(page.get('meta_data', {}) or {})}"
            for page in retrieved_pages
        ])

        # 4. Get user profile
        user_profile_text = self.user_long_term_memory.get_raw_user_profile(self.user_id)
        if not user_profile_text or user_profile_text.lower() == "none": 
            user_profile_text = "No detailed profile available yet."

        # 5. Format retrieved user knowledge for background
        user_knowledge_background = ""
        if retrieved_user_knowledge:
            user_knowledge_background = "\n【Relevant User Knowledge Entries】\n"
            for kn_entry in retrieved_user_knowledge:
                user_knowledge_background += f"- {kn_entry['knowledge']} (Recorded: {kn_entry['timestamp']})\n"
        
        background_context = f"【User Profile】\n{user_profile_text}\n{user_knowledge_background}"

        # 6. Format retrieved Assistant Knowledge (from assistant's LTM)
        # Use retrieved assistant knowledge instead of all assistant knowledge
        assistant_knowledge_text_for_prompt = "【Assistant Knowledge Base】\n"
        if retrieved_assistant_knowledge:
            for ak_entry in retrieved_assistant_knowledge:
                assistant_knowledge_text_for_prompt += f"- {ak_entry['knowledge']} (Recorded: {ak_entry['timestamp']})\n"
        else:
            assistant_knowledge_text_for_prompt += "- No relevant assistant knowledge found for this query.\n"

        # 7. Format user_conversation_meta_data (if provided)
        meta_data_text_for_prompt = "【Current Conversation Metadata】\n"
        if user_conversation_meta_data:
            try:
                meta_data_text_for_prompt += json.dumps(user_conversation_meta_data, ensure_ascii=False, indent=2)
            except TypeError:
                meta_data_text_for_prompt += str(user_conversation_meta_data)
        else:
            meta_data_text_for_prompt += "None provided for this turn."

        # 8. Construct Prompts
        system_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT.format(
            relationship=relationship_with_user,
            assistant_knowledge_text=assistant_knowledge_text_for_prompt,
            meta_data_text=meta_data_text_for_prompt # Using meta_data_text placeholder for user_conversation_meta_data
        )
        
        # 8. Construct Prompts
        user_prompt_text = prompts.GENERATE_SYSTEM_RESPONSE_USER_PROMPT.format(
            history_text=history_text,
            retrieval_text=retrieval_text,
            background=background_context,
            relationship=relationship_with_user,
            query=query
        )
        
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text}
        ]

        # 9. Call LLM for response
        print("Memorycontext: Calling LLM for final response generation...")
        # print("System Prompt:\n", system_prompt_text)
        # print("User Prompt:\n", user_prompt_text)
        response_content = self.client.chat_completion(
            model=self.llm_model, 
            messages=messages, 
            temperature=0.7, 
            max_tokens=1500 # As in original main
        )
        self.add_memory(user_input=query, agent_response=response_content, timestamp=get_timestamp())
        
        return response_content

    # --- Multimodal ingestion ---
    def add_multimodal_memory(
        self,
        source: Union[str, Path, bytes, Sequence[Union[str, Path, bytes]]],
        *,
        source_type: str = "file_path",
        converter_type: Optional[str] = None,
        agent_response: Optional[str] = None,
        converter_kwargs: Optional[Dict[str, Any]] = None,
        progress_callback=None,
    ):
        """
        Convert multimodal inputs into textual memories and store them.
        """

        converter_kwargs = converter_kwargs or {}
        sources = source if isinstance(source, (list, tuple)) else [source]
        ingestion_results = []

        for item in sources:
            result = self._ingest_single_multimodal(
                item,
                source_type=source_type,
                converter_type=converter_type,
                agent_response=agent_response,
                converter_kwargs=converter_kwargs,
                progress_callback=progress_callback,
            )
            ingestion_results.append(result)

        return ingestion_results[0] if len(ingestion_results) == 1 else ingestion_results

    def _ingest_single_multimodal(
        self,
        item: Union[str, Path, bytes],
        *,
        source_type: str,
        converter_type: Optional[str],
        agent_response: Optional[str],
        converter_kwargs: Dict[str, Any],
        progress_callback,
    ):
        file_path = Path(item) if source_type == "file_path" else None
        file_extension = (
            guess_file_extension(file_path.name) if file_path else converter_kwargs.get("file_extension")
        )
        mime_type = guess_mime_type(str(file_path)) if file_path else converter_kwargs.get("mime_type")

        converter = ConverterFactory.create(
            converter_type=converter_type,
            file_extension=file_extension,
            mime_type=mime_type,
            progress_callback=progress_callback,
            **converter_kwargs,
        )

        if not converter:
            return {
                "status": "failed",
                "error": f"No converter registered for type={converter_type} extension={file_extension}",
            }

        base_metadata = self._build_multimodal_metadata(item, source_type, file_path, file_extension, mime_type)
        output = converter.convert(
            item,
            source_type=source_type,
        )
        output.ensure_chunks()

        timestamps = []
        for chunk in output.chunks:
            chunk_meta = {
                **base_metadata,
                **output.metadata,
                **chunk.metadata,
                "source_type": "multimodal",
            }
            chunk_agent_response = agent_response or chunk_meta.get(
                "chunk_summary",
                f"[Multimodal] Stored content from {chunk_meta.get('original_filename', 'file')}",
            )
            self.add_memory(
                user_input=chunk.text,
                agent_response=chunk_agent_response,
                meta_data=chunk_meta,
            )
            timestamps.append(get_timestamp())

        return {
            "status": output.status,
            "file_id": base_metadata.get("source_file_id"),
            "chunks_written": len(output.chunks),
            "error": output.error,
            "timestamps": timestamps,
        }

    def _build_multimodal_metadata(
        self,
        source: Union[str, Path, bytes],
        source_type: str,
        file_path: Optional[Path],
        file_extension: Optional[str],
        mime_type: Optional[str],
    ) -> Dict[str, Any]:
        file_size = file_path.stat().st_size if file_path and file_path.exists() else None
        if file_path and file_path.exists():
            algorithm, digest = compute_file_hash(file_path=file_path)
        elif isinstance(source, bytes):
            algorithm, digest = compute_file_hash(data=source)
        else:
            algorithm, digest = "sha256", None
        file_id = digest or generate_id("file")

        return {
            "source_file_id": file_id,
            "file_extension": file_extension,
            "mime_type": mime_type,
            "file_size": file_size,
            "original_filename": file_path.name if file_path else None,
            "hash_algorithm": algorithm,
            "hash_value": digest,
            "ingest_source_type": source_type,
        }

    # --- Helper/Maintenance methods (optional additions) ---
    def get_user_profile_summary(self) -> str:
        return self.user_long_term_memory.get_raw_user_profile(self.user_id)

    def get_assistant_knowledge_summary(self) -> list:
        return self.assistant_long_term_memory.get_assistant_knowledge()

    def force_mid_term_analysis(self):
        """Forces analysis of all unanalyzed pages in the hottest mid-term segment if heat is above 0.
           Useful for testing or manual triggering.
        """
        original_threshold = self.mid_term_heat_threshold
        self.mid_term_heat_threshold = 0.0 # Temporarily lower threshold
        print("Memorycontext: Force-triggering mid-term analysis...")
        self._trigger_profile_and_knowledge_update_if_needed()
        self.mid_term_heat_threshold = original_threshold # Restore original threshold

    def __repr__(self):
            return f"<Memcontext user_id='{self.user_id}' assistant_id='{self.assistant_id}' data_path='{self.data_storage_path}'>" 