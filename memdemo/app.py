from flask import Flask, render_template, request, jsonify, session
import sys
import os
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import secrets
from werkzeug.utils import secure_filename

# Add parent directory to path to import memcontext
# Ensure the path is /root/autodl-tmp for consistent imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memcontext modules directly
from memcontext import Memcontext
# Import utils directly from the playground directory
from memcontext.utils import get_timestamp

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global memcontext instance (in production, you'd use proper session management)
memory_systems = {}

# 删除了固定的API_KEY, BASE_URL, MODEL

# 有效邀请码列表 - 在实际部署中应该存储在数据库或加密文件中
# VALID_INVITE_CODES = [
#     'DEMO2024',
#     'MEMORY001',
#     'TESTUSER',
#     'BETA2024',
#     'INVITE123'
# ]

# def load_invite_codes():
#     """从文件加载邀请码列表"""
#     invite_codes_file = os.path.join(os.path.dirname(__file__), 'invite_codes.json')
#     try:
#         if os.path.exists(invite_codes_file):
#             with open(invite_codes_file, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         else:
#             # 如果文件不存在，创建默认邀请码文件
#             with open(invite_codes_file, 'w', encoding='utf-8') as f:
#                 json.dump(VALID_INVITE_CODES, f, ensure_ascii=False, indent=2)
#             return VALID_INVITE_CODES
#     except Exception as e:
#         print(f"Error loading invite codes: {e}")
#         return VALID_INVITE_CODES

# def save_invite_codes(codes):
#     """保存邀请码列表到文件"""
#     invite_codes_file = os.path.join(os.path.dirname(__file__), 'invite_codes.json')
#     try:
#         with open(invite_codes_file, 'w', encoding='utf-8') as f:
#             json.dump(codes, f, ensure_ascii=False, indent=2)
#     except Exception as e:
#         print(f"Error saving invite codes: {e}")

# 启动时加载邀请码
# VALID_INVITE_CODES = load_invite_codes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_memory', methods=['POST'])
def init_memory():
    data = request.json
    user_id = data.get('user_id', '').strip()
    api_key = data.get('api_key', '').strip()
    base_url = data.get('base_url', '').strip()
    model = data.get('model_name', '').strip()
    siliconflow_key = data.get('siliconflow_key', '').strip()

    # 默认使用 DeepSeek API
    if not base_url:
        base_url = "https://api.deepseek.com/v1"
    if not model:
        model = "deepseek-chat"

    if not user_id or not api_key:
        return jsonify({'error': 'User ID 和 API Key 是必需的。Base URL 和 Model Name 如果不提供，将默认使用 DeepSeek API。'}), 400
    
    assistant_id = f"assistant_{user_id}"
    embedding_kwargs = {}
    if siliconflow_key:
        os.environ['SILICONFLOW_API_KEY'] = siliconflow_key
        embedding_kwargs = {
            'use_siliconflow': True,
            'siliconflow_model': "BAAI/bge-m3"
        }
    elif os.environ.get('SILICONFLOW_API_KEY'):
        embedding_kwargs = {
            'use_siliconflow': True,
            'siliconflow_model': "BAAI/bge-m3"
        }
    
    try:
        # Initialize memcontext for this session
        data_path = './data'
        os.makedirs(data_path, exist_ok=True)
        
        memory_system = Memcontext(
            user_id=user_id,
            openai_api_key=api_key,
            openai_base_url=base_url,
            data_storage_path=data_path,
            assistant_id=assistant_id,  # 使用邀请码作为assistant_id
            short_term_capacity=7,  # Smaller for demo
            mid_term_capacity=200,   # Smaller for demo
            long_term_knowledge_capacity=1000,  # Smaller for demo
            mid_term_heat_threshold=10.0,
            embedding_model_name="BAAI/bge-m3" if embedding_kwargs.get('use_siliconflow') else "all-MiniLM-L6-v2",
            embedding_model_kwargs=embedding_kwargs,
            llm_model=model
        )
        
        session_id = secrets.token_hex(8)
        memory_systems[session_id] = memory_system
        session['memory_session_id'] = session_id
        # 将配置存入session
        session['memory_config'] = {
            'api_key': api_key,
            'base_url': base_url,
            'model': model,
            'embedding_provider': 'siliconflow' if embedding_kwargs.get('use_siliconflow') else 'local'
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'user_id': user_id,
            'assistant_id': assistant_id,
            'model': model,
            'base_url': base_url,
            'embedding_provider': session['memory_config']['embedding_provider']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400
    
    memory_system = memory_systems[session_id]
    
    try:
        # Get response from memcontext (this already adds the memory internally)
        response = memory_system.get_response(user_input)
        
        # Do NOT add memory again here - it's already done in get_response()
        
        return jsonify({
            'response': response,
            'timestamp': get_timestamp()
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Chat error: {error_trace}")
        return jsonify({
            'error': str(e), 
            'traceback': error_trace
        }), 500

@app.route('/import_from_cache', methods=['POST'])
def import_from_cache_endpoint():
    """从 temp_memory 缓存直接导入，跳过 videorag 解析"""
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400

    memory_system = memory_systems[session_id]
    
    try:
        data = request.get_json(silent=True) or {}
        file_id = data.get('file_id', '').strip()
        
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # 检查缓存文件
        cache_dir = Path(memory_system.data_storage_path) / "temp_memory"
        cache_file = cache_dir / f"{file_id}.json"
        
        if not cache_file.exists():
            return jsonify({'error': f'Cache file not found for file_id: {file_id}'}), 404
        
        # 加载缓存
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        
        # 构建 ConversionOutput
        from memcontext.multimodal.converter import ConversionChunk, ConversionOutput
        
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
        
        # 直接使用 _ingest_single_multimodal 的逻辑来添加记忆
        base_metadata = cached.get("metadata", {})
        timestamps = []
        memories_to_add = []
        
        for chunk in output.chunks:
            # 安全地合并元数据，确保所有值都是字典
            try:
                base_meta = base_metadata if isinstance(base_metadata, dict) else {}
                chunk_meta_dict = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                
                chunk_meta = {
                    **base_meta,
                    **chunk_meta_dict,
                    "source_type": "multimodal",
                }
            except (TypeError, AttributeError) as e:
                print(f"Import cache: Error merging metadata: {e}")
                chunk_meta = {"source_type": "multimodal"}
            
            # 对于视频内容，使用完整的文本描述（chunk.text）作为 agent_response
            # 这样检索时能获得详细的视频内容，而不是只有摘要
            if chunk_meta.get("video_path") or chunk_meta.get("video_name"):
                chunk_agent_response = chunk.text  # 使用完整的视频描述文本
            else:
                chunk_agent_response = chunk_meta.get(
                    "chunk_summary",
                    f"[Multimodal] Stored content from {chunk_meta.get('original_filename', 'file')}",
                )
            
            # 对于视频内容，构建格式化的 user_input
            # 安全获取 video_path，确保不会为 None 或抛出 KeyError
            try:
                video_path = chunk_meta.get("video_path") or chunk_meta.get("original_filename") or chunk_meta.get("video_name") or "视频"
                time_range = chunk_meta.get("time_range", "")
                
                if (chunk_meta.get("video_path") or chunk_meta.get("video_name")) and time_range:
                    # 确保 video_path 是字符串且不为空
                    if not isinstance(video_path, str) or not video_path or video_path == "视频":
                        video_path = chunk_meta.get("video_name") or chunk_meta.get("original_filename") or "视频"
                    if not isinstance(video_path, str):
                        video_path = str(video_path) if video_path else "视频"
                    user_input = f"描述{video_path}视频的{time_range}的内容"
                else:
                    user_input = chunk.text
            except Exception as e:
                print(f"Import cache: Error building user_input: {e}, using chunk.text as fallback")
                user_input = chunk.text
                video_path = "视频"  # 设置默认值
            
            memories_to_add.append({
                "user_input": user_input,
                "agent_response": chunk_agent_response,
                "timestamp": get_timestamp(),
                "meta_data": chunk_meta,
            })
            timestamps.append(get_timestamp())
        
        # 使用正常流程：逐个添加到 short_term
        for mem in memories_to_add:
            memory_system.add_memory(
                user_input=mem["user_input"],
                agent_response=mem["agent_response"],
                timestamp=mem["timestamp"],
                meta_data=mem["meta_data"]
            )
        
        return jsonify({
            'success': True,
            'ingested_rounds': len(memories_to_add),
            'file_id': file_id,
            'timestamps': timestamps,
            'message': f'Successfully imported {len(memories_to_add)} chunks from cache'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'导入缓存失败: {str(e)}',
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/add_multimodal_memory', methods=['POST'])
def add_multimodal_memory_endpoint():
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400

    memory_system = memory_systems[session_id]
    cleanup_paths = []

    try:
        converter_type = None
        agent_response = None
        converter_kwargs = {}

        if request.content_type and 'multipart/form-data' in request.content_type:
            uploaded_file = request.files.get('file')
            if not uploaded_file or not uploaded_file.filename:
                return jsonify({'error': 'File upload is required.'}), 400

            safe_name = secure_filename(uploaded_file.filename)
            temp_dir = tempfile.mkdtemp(prefix="memcontext_upload_")
            temp_path = os.path.join(temp_dir, safe_name or "upload.bin")
            uploaded_file.save(temp_path)
            cleanup_paths.append(temp_dir)

            source = temp_path
            source_type = 'file_path'
            agent_response = request.form.get('agent_response')
            converter_type = request.form.get('converter_type')
            if request.form.get('converter_kwargs'):
                try:
                    converter_kwargs = json.loads(request.form['converter_kwargs'])
                except json.JSONDecodeError:
                    return jsonify({'error': 'converter_kwargs must be valid JSON'}), 400
        else:
            data = request.get_json(silent=True) or {}
            if data.get('file_path'):
                source = data['file_path']
                source_type = 'file_path'
            elif data.get('url'):
                source = data['url']
                source_type = 'url'
            else:
                return jsonify({'error': 'file_path or url must be provided'}), 400

            converter_type = data.get('converter_type')
            agent_response = data.get('agent_response')
            converter_kwargs = data.get('converter_kwargs', {})

        if source_type != 'file_path':
            return jsonify({'error': '当前仅支持本地文件路径(file_path)的视频源'}), 400

        converter_type = (converter_type or 'videorag').lower()
        if converter_type not in ('video', 'videorag'):
            return jsonify({'error': f'不支持的 converter_type: {converter_type}，可选 video | videorag'}), 400

        converter_settings = dict(converter_kwargs or {})
        deepseek_key = converter_settings.pop('deepseek_key', None)
        silicon_key = converter_settings.pop('siliconflow_key', None)
        if deepseek_key:
            os.environ['DEEPSEEK_API_KEY'] = deepseek_key
        if silicon_key:
            os.environ['SILICONFLOW_API_KEY'] = silicon_key

        converter_settings.setdefault('working_dir', './videorag-workdir')
        progress_events = []

        def progress_callback(progress: float, message: str) -> None:
            progress_events.append({
                'progress': round(float(progress), 4),
                'message': message
            })

        # 使用 memory_system.add_multimodal_memory() 方法，它会自动处理所有chunks并存储到记忆中
        try:
            result = memory_system.add_multimodal_memory(
                source=source,
                source_type=source_type,
                converter_type=converter_type,
                agent_response=agent_response,
                converter_kwargs=converter_settings,
                progress_callback=progress_callback,
            )
            
            if result.get('status') != 'success':
                return jsonify({
                    'error': result.get('error', '处理失败'),
                    'metadata': result,
                    'progress': progress_events
                }), 500

            return jsonify({
                'success': True,
                'ingested_rounds': result.get('chunks_written', 0),
                'file_id': result.get('file_id'),
                'timestamps': result.get('timestamps', []),
                'progress': progress_events
            })
        except Exception as e:
            return jsonify({
                'error': f'调用 add_multimodal_memory 失败: {str(e)}',
                'progress': progress_events
            }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for path in cleanup_paths:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass

@app.route('/memory_state', methods=['GET'])
def get_memory_state():
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400
    
    memory_system = memory_systems[session_id]
    
    try:
        # Get short-term memory
        short_term = memory_system.short_term_memory.get_all()
        mid_term_sessions = []
        # Get mid-term memory sessions (top 5)
        for sid, session_data in list(memory_system.mid_term_memory.sessions.items())[:5]:
            mid_term_sessions.append({
                'id': sid,
                'summary': session_data.get('summary', ''),
                'keywords': session_data.get('summary_keywords', []),
                'heat': session_data.get('H_segment', 0),
                'visit_count': session_data.get('N_visit', 0),
                'last_visit': session_data.get('last_visit_time', ''),
                'page_count': len(session_data.get('details', []))
            })
        
        # Sort by heat
        mid_term_sessions.sort(key=lambda x: x['heat'], reverse=True)
        
        # Get long-term memory - separate user profile, user knowledge, and assistant knowledge
        user_profile = memory_system.user_long_term_memory.get_raw_user_profile(memory_system.user_id)
        user_knowledge = memory_system.user_long_term_memory.get_user_knowledge()
        assistant_knowledge = memory_system.assistant_long_term_memory.get_assistant_knowledge()
        return jsonify({
            'short_term': {
                'capacity': memory_system.short_term_memory.max_capacity,
                'current_count': len(short_term),
                'memories': short_term
            },
            'mid_term': {
                'capacity': memory_system.mid_term_memory.max_capacity,
                'current_count': len(memory_system.mid_term_memory.sessions),
                'sessions': mid_term_sessions,
                'heat_threshold': memory_system.mid_term_heat_threshold
            },
            'long_term': {
                'user_profile': user_profile,
                'user_knowledge': [k.get('knowledge', '') for k in user_knowledge],
                'assistant_knowledge': [k.get('knowledge', '') for k in assistant_knowledge]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trigger_analysis', methods=['POST'])
def trigger_analysis():
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400
    
    memory_system = memory_systems[session_id]
    
    try:
        # Check if there are any mid-term memory sessions to analyze
        if not memory_system.mid_term_memory.sessions:
            return jsonify({'error': 'No Mid-term memory, but at least keep short-term memory for seven rounds.'}), 400
        
        # Check if there are any unanalyzed pages in mid-term memory
        has_unanalyzed_pages = False
        for session_data in memory_system.mid_term_memory.sessions.values():
            unanalyzed_pages = [p for p in session_data.get('details', []) if not p.get('analyzed', False)]
            if unanalyzed_pages:
                has_unanalyzed_pages = True
                break
        
        if not has_unanalyzed_pages:
            return jsonify({'error': 'No Mid-term memory, but at least keep short-term memory for seven rounds.'}), 400
        
        # Force mid-term analysis
        memory_system.force_mid_term_analysis()
        return jsonify({'success': True, 'message': 'Analysis triggered successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/personality_analysis', methods=['POST'])
def personality_analysis():
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400
    
    memory_system = memory_systems[session_id]
    
    try:
        # Get user profile
        user_profile = memory_system.user_long_term_memory.get_raw_user_profile(memory_system.user_id)
        
        if not user_profile or user_profile.lower() in ['none', 'no profile data yet']:
            return jsonify({'error': 'No user profile available for analysis'}), 400
        
        # Parse personality traits from the user profile
        personality_analysis = parse_personality_traits(user_profile)
        
        return jsonify({
            'success': True,
            'personality_analysis': personality_analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def parse_personality_traits(user_profile):
    """
    Parse personality traits from user profile text.
    Extract traits in format: Dimension ( Level(High/Medium/Low) )
    """
    # Define the three main categories
    categories = {
        'Psychological Model': [
            'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness', 'Neuroticism',
            'Physiological Needs', 'Need for Security', 'Need for Belonging', 'Need for Self-Esteem',
            'Cognitive Needs', 'Aesthetic Appreciation', 'Self-Actualization', 'Need for Order',
            'Need for Autonomy', 'Need for Power', 'Need for Achievement'
        ],
        'AI Alignment Dimensions': [
            'Helpfulness', 'Honesty', 'Safety', 'Instruction Compliance', 'Truthfulness',
            'Coherence', 'Complexity', 'Conciseness'
        ],
        'Content Platform Interest Tags': [
            'Science Interest', 'Education Interest', 'Psychology Interest', 'Family Concern',
            'Fashion Interest', 'Art Interest', 'Health Concern', 'Financial Management Interest',
            'Sports Interest', 'Food Interest', 'Travel Interest', 'Music Interest',
            'Literature Interest', 'Film Interest', 'Social Media Activity', 'Tech Interest',
            'Environmental Concern', 'History Interest', 'Political Concern', 'Religious Interest',
            'Gaming Interest', 'Animal Concern', 'Emotional Expression', 'Sense of Humor',
            'Information Density', 'Language Style', 'Practicality'
        ]
    }
    
    # Extract traits from user profile
    extracted_traits = {}
    
    import re
    
    # Look for patterns like "Dimension ( Level(High/Medium/Low) )"
    pattern = r'([A-Za-z\s]+)\s*\(\s*([A-Za-z]+)\s*\)'
    matches = re.findall(pattern, user_profile)
    
    for match in matches:
        dimension = match[0].strip()
        level = match[1].strip()
        
        # Find which category this dimension belongs to
        for category, dimensions in categories.items():
            for cat_dimension in dimensions:
                if dimension.lower() in cat_dimension.lower() or cat_dimension.lower() in dimension.lower():
                    if category not in extracted_traits:
                        extracted_traits[category] = []
                    extracted_traits[category].append({
                        'dimension': dimension,
                        'level': level
                    })
                    break
    
    # Alternative pattern: look for lines containing trait descriptions
    lines = user_profile.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for mentions of High/Medium/Low levels
        for level in ['High', 'Medium', 'Low']:
            if level.lower() in line.lower():
                # Try to extract the dimension name
                for category, dimensions in categories.items():
                    for dimension in dimensions:
                        if dimension.lower() in line.lower():
                            if category not in extracted_traits:
                                extracted_traits[category] = []
                            
                            # Check if this trait is already added
                            existing = [t for t in extracted_traits[category] if t['dimension'] == dimension]
                            if not existing:
                                extracted_traits[category].append({
                                    'dimension': dimension,
                                    'level': level
                                })
                            break
    
    return extracted_traits

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400
    
    memory_system = memory_systems[session_id]
    
    try:
        # Clear all memory files
        user_data_dir = memory_system.user_data_dir
        assistant_data_dir = memory_system.assistant_data_dir
        
        # Remove the entire user data directory
        if os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir)
        
        # Remove the entire assistant data directory  
        if os.path.exists(assistant_data_dir):
            shutil.rmtree(assistant_data_dir)
        
        # 从session中获取配置来重新初始化
        config = session.get('memory_config')
        if not config:
            return jsonify({'error': 'Configuration not found in session. Please re-initialize.'}), 400

        api_key = config['api_key']
        base_url = config['base_url']
        model = config['model']
        user_id = memory_system.user_id
        assistant_id = memory_system.assistant_id
        data_path = memory_system.data_storage_path
        
        # Create new memory system
        new_memory_system = Memcontext(
            user_id=user_id,
            openai_api_key=api_key,
            openai_base_url=base_url,
            data_storage_path=data_path,
            assistant_id=assistant_id,
            short_term_capacity=7,
            mid_term_capacity=200,
            long_term_knowledge_capacity=100,
            mid_term_heat_threshold=5.0,
            llm_model=model
        )
        
        # Replace the old memory system
        memory_systems[session_id] = new_memory_system
        
        return jsonify({'success': True, 'message': 'All memories cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/import_conversations', methods=['POST'])
def import_conversations():
    session_id = session.get('memory_session_id')
    if not session_id or session_id not in memory_systems:
        return jsonify({'error': 'Memory system not initialized'}), 400
    
    memory_system = memory_systems[session_id]
    data = request.json
    conversations = data.get('conversations', [])
    
    if not conversations:
        return jsonify({'error': 'No conversations provided'}), 400
    
    try:
        imported_count = 0
        for conv in conversations:
            user_input = conv.get('user_input', '')
            agent_response = conv.get('agent_response', '')
            timestamp = conv.get('timestamp', get_timestamp())
            
            if user_input and agent_response:
                # Add each conversation to memory system
                memory_system.add_memory(
                    user_input=user_input,
                    agent_response=agent_response,
                    timestamp=timestamp
                )
                imported_count += 1
            else:
                print(f"Skipping invalid conversation: {conv}")
        
        return jsonify({
            'success': True,
            'imported_count': imported_count,
            'message': f'Successfully imported {imported_count} conversations'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5019) 