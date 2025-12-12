import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import prompts
from .utils import (
    OpenAIClient,
    get_timestamp,
    generate_id,
    gpt_user_profile_analysis,
    gpt_knowledge_extraction,
    ensure_directory_exists,
)
from .short_term import ShortTermMemory
from .mid_term import MidTermMemory, compute_segment_heat  # For H_THRESHOLD logic
from .long_term import LongTermMemory
from .updater import Updater
from .retriever import Retriever
from .multimodal import ConverterFactory
from .multimodal.converter import ConversionChunk, ConversionOutput
from .multimodal.utils import guess_file_extension, guess_mime_type, compute_file_hash

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
                 retrieval_queue_capacity=4,
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

    def _extract_knowledge_from_recent_mid_term(self, pages_to_extract=None):
        """
        从最近的 mid_term 页面中提取知识，不依赖 heat 阈值。
        用于批量添加记忆后立即提取知识。
        """
        if pages_to_extract is None:
            # 如果没有提供页面，从最新的 session 中获取未分析的页面
            if not self.mid_term_memory.heap:
                return
            
            # 获取最新的 session（heat 最高的）
            neg_heat, sid = self.mid_term_memory.heap[0]
            session = self.mid_term_memory.sessions.get(sid)
            if not session:
                return
            
            pages_to_extract = [p for p in session.get("details", []) if not p.get("analyzed", False)]
        
        if not pages_to_extract:
            print("Memorycontext: No unanalyzed pages to extract knowledge from.")
            return
        
        print(f"Memorycontext: Extracting knowledge from {len(pages_to_extract)} pages...")
        
        try:
            # 提取知识
            knowledge_result = gpt_knowledge_extraction(pages_to_extract, self.client, model=self.llm_model)
            
            new_user_private_knowledge = knowledge_result.get("private")
            new_assistant_knowledge = knowledge_result.get("assistant_knowledge")
            
            # 存储用户私有知识
            if new_user_private_knowledge and new_user_private_knowledge.lower() != "none":
                for line in new_user_private_knowledge.split('\n'):
                    if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                        self.user_long_term_memory.add_user_knowledge(line.strip())
                        print(f"Memorycontext: Added user knowledge: {line.strip()[:50]}...")
            
            # 存储 Assistant Knowledge
            if new_assistant_knowledge and new_assistant_knowledge.lower() != "none":
                for line in new_assistant_knowledge.split('\n'):
                    if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                        self.assistant_long_term_memory.add_assistant_knowledge(line.strip())
                        print(f"Memorycontext: Added assistant knowledge: {line.strip()[:50]}...")
            
            print("Memorycontext: Knowledge extraction completed.")
        except Exception as e:
            print(f"Memorycontext: Error extracting knowledge: {e}")
            import traceback
            traceback.print_exc()

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

    def add_memories_batch(self, memories: List[Dict[str, Any]], skip_short_term: bool = False):
        """
        批量添加多条记忆，优化大量内容的存储效率。
        
        Args:
            memories: 记忆列表，每个元素包含 user_input, agent_response, timestamp, meta_data
            skip_short_term: 如果为 True，直接批量添加到 mid_term，跳过 short_term
        """
        if not memories:
            return
        
        print(f"Memorycontext: Batch adding {len(memories)} memories (skip_short_term={skip_short_term})")
        
        if skip_short_term:
            # 直接批量添加到 mid_term，跳过 short_term
            # 这样可以避免频繁的 short_term -> mid_term 转换
            from .utils import check_conversation_continuity, generate_page_meta_info, gpt_generate_multi_summary
            
            # get_timestamp 和 generate_id 已经在文件顶部导入了，直接使用
            
            # 准备页面数据
            pages_to_insert = []
            temp_last_page = None
            
            for mem in memories:
                if not mem.get("user_input") or not mem.get("agent_response"):
                    continue
                
                page_obj = {
                    "page_id": generate_id("page"),
                    "user_input": mem.get("user_input", ""),
                    "agent_response": mem.get("agent_response", ""),
                    "timestamp": mem.get("timestamp", get_timestamp()),
                    "preloaded": False,
                    "analyzed": False,
                    "pre_page": None,
                    "next_page": None,
                    "meta_info": None
                }
                
                # 检查连续性
                is_continuous = check_conversation_continuity(
                    temp_last_page, page_obj, self.client, model=self.llm_model
                )
                
                if is_continuous and temp_last_page:
                    page_obj["pre_page"] = temp_last_page["page_id"]
                    last_meta = temp_last_page.get("meta_info")
                    new_meta = generate_page_meta_info(last_meta, page_obj, self.client, model=self.llm_model)
                    page_obj["meta_info"] = new_meta
                else:
                    page_obj["meta_info"] = generate_page_meta_info(None, page_obj, self.client, model=self.llm_model)
                
                pages_to_insert.append(page_obj)
                temp_last_page = page_obj
            
            if not pages_to_insert:
                return
            
            # 生成批量摘要
            input_text_for_summary = "\n".join([
                f"User: {p.get('user_input','')}\nAssistant: {p.get('agent_response','')}" 
                for p in pages_to_insert
            ])
            
            print(f"Memorycontext: Generating multi-topic summary for {len(pages_to_insert)} pages...")
            multi_summary_result = gpt_generate_multi_summary(
                input_text_for_summary, self.client, model=self.llm_model
            )
            
            # 插入到 mid_term
            # 对于批量添加，将所有页面作为一个整体插入到一个 session
            # 如果 multi_summary 返回多个主题，合并所有主题的摘要和关键词
            if multi_summary_result and multi_summary_result.get("summaries"):
                # 合并所有主题的摘要和关键词
                all_summaries = []
                all_keywords = set()
                for summary_item in multi_summary_result["summaries"]:
                    theme = summary_item.get("theme", "")
                    content = summary_item.get("content", "")
                    if theme and content:
                        all_summaries.append(f"{theme}: {content}")
                    keywords = summary_item.get("keywords", [])
                    if isinstance(keywords, list):
                        all_keywords.update(keywords)
                    elif isinstance(keywords, str):
                        all_keywords.update([k.strip() for k in keywords.split(",") if k.strip()])
                
                combined_summary = " | ".join(all_summaries) if all_summaries else "Batch of memories from multimodal content ingestion."
                combined_keywords = list(all_keywords)
                
                # 只插入一次，使用合并后的摘要
                self.mid_term_memory.insert_pages_into_session(
                    summary_for_new_pages=combined_summary,
                    keywords_for_new_pages=combined_keywords,
                    pages_to_insert=pages_to_insert,
                    similarity_threshold=self.mid_term_similarity_threshold
                )
            else:
                # Fallback
                fallback_summary = "Batch of memories from multimodal content ingestion."
                self.mid_term_memory.insert_pages_into_session(
                    summary_for_new_pages=fallback_summary,
                    keywords_for_new_pages=[],
                    pages_to_insert=pages_to_insert,
                    similarity_threshold=self.mid_term_similarity_threshold
                )
            
            # 更新页面连接
            for page in pages_to_insert:
                if page.get("pre_page"):
                    self.mid_term_memory.update_page_connections(page["pre_page"], page["page_id"])
            
            self.mid_term_memory.save()
            print(f"Memorycontext: Successfully batch added {len(pages_to_insert)} memories to mid-term.")
            
            # 批量添加后，不立即提取知识，等待 heat 达到阈值后再提取
            # 这样可以避免过度提取细粒度的知识，让系统自然触发知识提取
            # 只检查是否需要触发 profile 更新（基于 heat）
            self._trigger_profile_and_knowledge_update_if_needed()
        else:
            # 使用原来的方式，逐个添加到 short_term
            for mem in memories:
                self.add_memory(
                    user_input=mem.get("user_input", ""),
                    agent_response=mem.get("agent_response", ""),
                    timestamp=mem.get("timestamp"),
                    meta_data=mem.get("meta_data")
                )

    def get_response(self, query: str, relationship_with_user="friend", style_hint="", user_conversation_meta_data: dict = None) -> str:
        """
        Generates a response to the user's query, incorporating memory and context.
        """
        print(f"Memorycontext: Generating response for query: '{query[:50]}...'")

        # 1. Retrieve context
        retrieval_results = self.retriever.retrieve_context(
            user_query=query,
            user_id=self.user_id
            # Using default thresholds from Retriever class for now
        )
        retrieved_pages = retrieval_results["retrieved_pages"]
        retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]
        retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]

        # 2. Get short-term history
        short_term_history = self.short_term_memory.get_all()
        history_text = "\n".join([
            f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
            for qa in short_term_history
        ])

        # 3. Format retrieved mid-term pages (retrieval_queue equivalent)
        # 提取查询中提到的视频信息（如果有），用于过滤结果
        query_video_path = None
        if '视频' in query:
            import re
            # 尝试从查询中提取视频路径或名称
            video_match = re.search(r'["\']?([^"\']+\.(mp4|avi|mov|mkv))["\']?', query)
            if video_match:
                query_video_path = video_match.group(1)
            else:
                # 尝试提取"xxx视频"格式
                video_match = re.search(r'([^\s]+)视频', query)
                if video_match:
                    query_video_path = video_match.group(1)
        
        retrieval_text_parts = []
        for page in retrieved_pages:
            # 安全获取 meta_data，确保是字典类型
            try:
                page_meta = page.get('meta_data', {})
                if not isinstance(page_meta, dict):
                    page_meta = {}
            except (AttributeError, TypeError):
                page_meta = {}
            
            page_video_path = None
            
            # 从 user_input 中提取视频路径（格式：描述{video_path}视频的{time_range}的内容）
            user_input = page.get('user_input', '')
            if '视频' in user_input and '描述' in user_input:
                import re
                try:
                    match = re.search(r'描述(.+?)视频的', user_input)
                    if match:
                        page_video_path = match.group(1)
                except Exception as e:
                    print(f"Memorycontext: Error extracting video_path from user_input: {e}")
            
            # 如果从 user_input 中提取失败，尝试从 meta_data 中获取（使用 .get() 安全访问）
            if not page_video_path:
                page_video_path = page_meta.get('video_path') or page_meta.get('video_name')
            
            # 如果查询中指定了视频，只返回匹配的视频内容
            if query_video_path and page_video_path:
                try:
                    if query_video_path not in page_video_path and page_video_path not in query_video_path:
                        continue  # 跳过不匹配的视频
                except TypeError:
                    # 如果 page_video_path 不是字符串，跳过比较
                    pass
            
            page_text = f"【Historical Memory】\nUser: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nTime: {page.get('timestamp', '')}\nConversation chain overview: {page.get('meta_info','N/A')}"
            if page_video_path:
                page_text += f"\n[Video Source: {page_video_path}]"
            retrieval_text_parts.append(page_text)
        
        retrieval_text = "\n\n".join(retrieval_text_parts)

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
        
        # 10. Add this interaction to memory
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

        # --- Temp cache: avoid重复解析同一文件 ---
        cache_dir = Path(self.data_storage_path) / "temp_memory"
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_id = base_metadata.get("source_file_id")
        cache_file = cache_dir / f"{file_id}.json" if file_id else None

        output = None
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_chunks = []
                for idx, ch in enumerate(cached.get("chunks", [])):
                    meta = ch.get("metadata", {}) or {}
                    chunk_idx = meta.get("chunk_index", idx)
                    cached_chunks.append(
                        ConversionChunk(
                            text=ch.get("text", ""),
                            chunk_index=chunk_idx,
                            metadata=meta,
                        )
                    )
                output = ConversionOutput(
                    status="success",
                    text="\n\n".join(c.text for c in cached_chunks),
                    chunks=cached_chunks,
                    metadata=cached.get("metadata", {}),
                )
                output.ensure_chunks()
                print(f"Memorycontext: Video already processed (file_id={file_id}), using cached result. Skipping re-processing.")
                # 如果缓存存在，直接使用缓存，不重新处理
            except Exception as e:
                print(f"Memorycontext: Failed to load cache for file_id={file_id}, will re-run conversion. Error: {e}")
                output = None

        if output is None:
            output = converter.convert(
                item,
                source_type=source_type,
            )
            output.ensure_chunks()
            # 写入缓存，便于同一文件再次导入时直接复用
            if cache_file:
                try:
                    cache_payload = {
                        "metadata": output.metadata,
                        "chunks": [
                            {"text": ch.text, "metadata": ch.metadata}
                            for ch in output.chunks
                        ],
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Memorycontext: Failed to write cache for file_id={file_id}, skip caching. Error: {e}")

        timestamps = []
        
        # 准备所有记忆数据
        memories_to_add = []
        for chunk in output.chunks:
            # 安全地合并元数据，确保所有值都是字典
            try:
                base_meta = base_metadata if isinstance(base_metadata, dict) else {}
                output_meta = output.metadata if isinstance(output.metadata, dict) else {}
                chunk_meta_dict = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                
                chunk_meta = {
                    **base_meta,
                    **output_meta,
                    **chunk_meta_dict,
                    "source_type": "multimodal",
                }
            except (TypeError, AttributeError) as e:
                print(f"Memorycontext: Error merging metadata: {e}")
                chunk_meta = {"source_type": "multimodal"}
            
            # 对于视频内容，使用完整的文本描述（chunk.text）作为 agent_response
            # 这样检索时能获得详细的视频内容，而不是只有摘要
            if chunk_meta.get("video_path") or chunk_meta.get("video_name"):
                chunk_agent_response = chunk.text  # 使用完整的视频描述文本
            else:
                chunk_agent_response = agent_response or chunk_meta.get(
                    "chunk_summary",
                    f"[Multimodal] Stored content from {chunk_meta.get('original_filename', 'file')}",
                )
            
            # 对于视频内容，构建格式化的 user_input
            # 安全获取 video_path，确保不会为 None 或抛出 KeyError
            try:
                video_path = chunk_meta.get("video_path") or chunk_meta.get("original_filename") or chunk_meta.get("video_name") or "视频"
                time_range = chunk_meta.get("time_range", "")
                
                # 如果是视频内容（通过 video_path 或 time_range 判断）且有 time_range，使用新格式
                if (chunk_meta.get("video_path") or chunk_meta.get("video_name")) and time_range:
                    # 确保 video_path 是字符串且不为空
                    if not isinstance(video_path, str) or not video_path or video_path == "视频":
                        video_path = chunk_meta.get("video_name") or chunk_meta.get("original_filename") or "视频"
                    if not isinstance(video_path, str):
                        video_path = str(video_path) if video_path else "视频"
                    user_input = f"描述{video_path}视频的{time_range}的内容"
                else:
                    # 非视频内容或没有 time_range，使用原来的格式
                    user_input = chunk.text
            except Exception as e:
                print(f"Memorycontext: Error building user_input: {e}, using chunk.text as fallback")
                user_input = chunk.text
                video_path = "视频"  # 设置默认值
            
            memories_to_add.append({
                "user_input": user_input,
                "agent_response": chunk_agent_response,
                "timestamp": get_timestamp(),
                "meta_data": chunk_meta,
            })
            timestamps.append(get_timestamp())
        
        # 使用正常流程：逐个添加到 short_term，超出容量后自动转到 mid_term
        # 这样更简单，也更符合原来的设计逻辑
        for mem in memories_to_add:
            self.add_memory(
                user_input=mem["user_input"],
                agent_response=mem["agent_response"],
                timestamp=mem["timestamp"],
                meta_data=mem["meta_data"]
            )

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