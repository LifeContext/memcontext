"""
This file stores all the prompts used by the Memoryos system.
"""

# Prompt for generating system response (from main_memoybank.py, generate_system_response_with_meta)
GENERATE_SYSTEM_RESPONSE_SYSTEM_PROMPT = (
    "As a communication expert with outstanding communication habits, you embody the role of {relationship} throughout the following dialogues.\n"
    "Here are some of your distinctive personal traits and knowledge:\n{assistant_knowledge_text}\n"
    "User's profile:\n"
    "{meta_data_text}\n"
    "Your task is to generate responses that align with these traits and maintain the tone.\n"
)

GENERATE_SYSTEM_RESPONSE_USER_PROMPT = (
    "<CONTEXT>\n"
    "Drawing from your recent conversation with the user:\n"
    "{history_text}\n\n"
    "<MEMORY>\n"
    "The memories linked to the ongoing conversation are:\n"
    "{retrieval_text}\n\n"
    "<USER TRAITS>\n"
    "During the conversation process between you and the user in the past, you found that the user has the following characteristics:\n"
    "{background}\n\n"
    "Now, please role-play as {relationship} to continue the dialogue between you and the user.\n"
    "The user just said: {query}\n"
    "Please respond to the user's statement using the following format (maximum 30 words, must be in English):\n "
    "When answering questions, be sure to check whether the timestamp of the referenced information matches the timeframe of the question"
)

# Prompt for assistant knowledge extraction (from utils.py, analyze_assistant_knowledge)
ASSISTANT_KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None"."""

ASSISTANT_KNOWLEDGE_EXTRACTION_USER_PROMPT = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
- [Fact 1]
- [Fact 2]
- (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That's interesting. I didn't know you could do that.
   AI: I'm glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
{conversation}
"""

# Prompt for summarizing dialogs (from utils.py, gpt_summarize)
SUMMARIZE_DIALOGS_SYSTEM_PROMPT = "You are an expert in summarizing dialogue topics. Generate extremely concise and precise summaries. Be as brief as possible while capturing the essence."
SUMMARIZE_DIALOGS_USER_PROMPT = "Please generate an concise topic summary based on the following conversation. Keep it to 2-3 short sentences maximum:\n{dialog_text}\nConcise Summary："

# Prompt for multi-summary generation (from utils.py, gpt_generate_multi_summary)
MULTI_SUMMARY_SYSTEM_PROMPT = "You are an expert in analyzing dialogue topics. Generate  concise summaries. No more than two topics. Be as brief as possible."
MULTI_SUMMARY_USER_PROMPT = ("Please analyze the following dialogue and generate extremely concise subtopic summaries (if applicable), with a maximum of two themes.\n"
                           "Each summary should be very brief - just a few words for the theme and content. Format as JSON array:\n"
                           "[\n  {{\"theme\": \"Brief theme\", \"keywords\": [\"key1\", \"key2\"], \"content\": \"summary\"}}\n]\n"
                           "\nConversation content:\n{text}")

# Prompt for personality analysis (NEW TEMPLATE)
PERSONALITY_ANALYSIS_SYSTEM_PROMPT = """You are a professional user preference analysis assistant. Your task is to analyze the user's personality preferences from the given dialogue based on the provided dimensions.

For each dimension:
1. Carefully read the conversation and determine if the dimension is reflected.
2. If reflected, determine the user's preference level: High / Medium / Low, and briefly explain the reasoning, including time, people, and context if possible.
3. If the dimension is not reflected, do not extract or list it.

Focus only on the user's preferences and traits for the personality analysis section.
Output only the user profile section.
"""

PERSONALITY_ANALYSIS_USER_PROMPT = """Please analyze the latest user-AI conversation below and update the user profile based on the 90 personality preference dimensions.

Here are the 90 dimensions and their explanations:

[Psychological Model (Basic Needs & Personality)]
Extraversion: Preference for social activities.
Openness: Willingness to embrace new ideas and experiences.
Agreeableness: Tendency to be friendly and cooperative.
Conscientiousness: Responsibility and organizational ability.
Neuroticism: Emotional stability and sensitivity.
Physiological Needs: Concern for comfort and basic needs.
Need for Security: Emphasis on safety and stability.
Need for Belonging: Desire for group affiliation.
Need for Self-Esteem: Need for respect and recognition.
Cognitive Needs: Desire for knowledge and understanding.
Aesthetic Appreciation: Appreciation for beauty and art.
Self-Actualization: Pursuit of one's full potential.
Need for Order: Preference for cleanliness and organization.
Need for Autonomy: Preference for independent decision-making and action.
Need for Power: Desire to influence or control others.
Need for Achievement: Value placed on accomplishments.

[AI Alignment Dimensions]
Helpfulness: Whether the AI's response is practically useful to the user. (This reflects user's expectation of AI)
Honesty: Whether the AI's response is truthful. (This reflects user's expectation of AI)
Safety: Avoidance of sensitive or harmful content. (This reflects user's expectation of AI)
Instruction Compliance: Strict adherence to user instructions. (This reflects user's expectation of AI)
Truthfulness: Accuracy and authenticity of content. (This reflects user's expectation of AI)
Coherence: Clarity and logical consistency of expression. (This reflects user's expectation of AI)
Complexity: Preference for detailed and complex information.
Conciseness: Preference for brief and clear responses.

[Content Platform Interest Tags]
Science Interest: Interest in science topics.
Education Interest: Concern with education and learning.
Psychology Interest: Interest in psychology topics.
Family Concern: Interest in family and parenting.
Fashion Interest: Interest in fashion topics.
Art Interest: Engagement with or interest in art.
Health Concern: Concern with physical health and lifestyle.
Financial Management Interest: Interest in finance and budgeting.
Sports Interest: Interest in sports and physical activity.
Food Interest: Passion for cooking and cuisine.
Travel Interest: Interest in traveling and exploring new places.
Music Interest: Interest in music appreciation or creation.
Literature Interest: Interest in literature and reading.
Film Interest: Interest in movies and cinema.
Social Media Activity: Frequency and engagement with social media.
Tech Interest: Interest in technology and innovation.
Environmental Concern: Attention to environmental and sustainability issues.
History Interest: Interest in historical knowledge and topics.
Political Concern: Interest in political and social issues.
Religious Interest: Interest in religion and spirituality.
Gaming Interest: Enjoyment of video games or board games.
Animal Concern: Concern for animals or pets.
Emotional Expression: Preference for direct vs. restrained emotional expression.
Sense of Humor: Preference for humorous or serious communication style.
Information Density: Preference for detailed vs. concise information.
Language Style: Preference for formal vs. casual tone.
Practicality: Preference for practical advice vs. theoretical discussion.

**Task Instructions:**
1. Review the existing user profile below
2. Analyze the new conversation for evidence of the 90 dimensions above
3. Update and integrate the findings into a comprehensive user profile
4. For each dimension that can be identified, use the format: Dimension ( Level(High/Medium/Low) )
5. Include brief reasoning for each dimension when possible
6. Maintain existing insights from the old profile while incorporating new observations
7. If a dimension cannot be inferred from either the old profile or new conversation, do not include it

**Existing User Profile:**
{existing_user_profile}

**Latest User-AI Conversation:**
{conversation}

**Updated User Profile:**
Please provide the comprehensive updated user profile below, combining insights from both the existing profile and new conversation:"""

# Prompt for knowledge extraction (NEW)
KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction assistant. Your task is to extract user private data and assistant knowledge from conversations.

Focus on:
1. User private data: personal information, preferences, or private facts about the user
2. Assistant knowledge: explicit statements about what the assistant did, provided, or demonstrated

Be extremely concise and factual in your extractions. Use the shortest possible phrases.
"""

KNOWLEDGE_EXTRACTION_USER_PROMPT = """Please extract user private data and assistant knowledge from the latest user-AI conversation below.

Latest User-AI Conversation:
{conversation}

【User Private Data】
Extract personal information about the user. Be extremely concise - use shortest possible phrases:
- [Brief fact]: [Minimal context(Including entities and time)]
- [Brief fact]: [Minimal context(Including entities and time)]
- (If no private data found, write "None")

【Assistant Knowledge】
Extract what the assistant demonstrated. Use format "Assistant [action] at [time]". Be extremely brief:
- Assistant [brief action] at [time/context]
- Assistant [brief capability] during [brief context]
- (If no assistant knowledge found, write "None")
"""

# Prompt for updating user profile (from utils.py, gpt_update_profile)
UPDATE_PROFILE_SYSTEM_PROMPT = "You are an expert in merging and updating user profiles. Integrate the new information into the old profile, maintaining consistency and improving the overall understanding of the user. Avoid redundancy. The new analysis is based on specific dimensions, try to incorporate these insights meaningfully."
UPDATE_PROFILE_USER_PROMPT = "Please update the following user profile based on the new analysis. If the old profile is empty or \"None\", create a new one based on the new analysis.\n\nOld User Profile:\n{old_profile}\n\nNew Analysis Data:\n{new_analysis}\n\nUpdated User Profile:"

# Prompt for extracting theme (from utils.py, gpt_extract_theme)
EXTRACT_THEME_SYSTEM_PROMPT = "You are an expert in extracting the main theme from a text. Provide a concise theme."
EXTRACT_THEME_USER_PROMPT = "Please extract the main theme from the following text:\n{answer_text}\n\nTheme:"



# Prompt for conversation continuity check (from dynamic_update.py, _is_conversation_continuing)
CONTINUITY_CHECK_SYSTEM_PROMPT = "You are a conversation continuity detector. Return ONLY 'true' or 'false'."
CONTINUITY_CHECK_USER_PROMPT = ("Determine if these two conversation pages are continuous (true continuation without topic shift).\n"
                                "Return ONLY \"true\" or \"false\".\n\n"
                                "Previous Page:\nUser: {prev_user}\nAssistant: {prev_agent}\n\n"
                                "Current Page:\nUser: {curr_user}\nAssistant: {curr_agent}\n\n"
                                "Continuous?")

# Prompt for generating meta info (from dynamic_update.py, _generate_meta_info)
META_INFO_SYSTEM_PROMPT = ("""You are a conversation meta-summary updater. Your task is to:
1. Preserve relevant context from previous meta-summary
2. Integrate new information from current dialogue
3. Output ONLY the updated summary (no explanations)""" )
META_INFO_USER_PROMPT = ("""Update the conversation meta-summary by incorporating the new dialogue while maintaining continuity.
        
    Guidelines:
    1. Start from the previous meta-summary (if exists)
    2. Add/update information based on the new dialogue
    3. Keep it concise (1-2 sentences max)
    4. Maintain context coherence

    Previous Meta-summary: {last_meta}
    New Dialogue:
    {new_dialogue}

    Updated Meta-summary:""") 

# Prompt for video structured caption generation (used by videorag/_videoutil/caption.py)
VIDEO_STRUCTURED_CAPTION_PROMPT = """
你是视频逐帧视觉描述助手（不要生成额外标记如[开始]/[结束]）。
目标：为每个给定的帧时间段输出一行可被人理解的中文描述，结合画面视觉信息与该时间段的字幕（如果有）。

格式要求（严格遵守）：
- 每行仅包含：`[start -> end] 描述`，例如：`[420.00s -> 422.00s] 两名人物在昏暗房间内对话，左侧人物穿红色衣服`。
- 时间必须保留秒为单位，保留两位小数（例如 420.00s、422.50s）。
- 时间必须为相对于整部视频的绝对时间（以秒为单位，保留两位小数）。禁止使用相对于片段起点的相对时间（例如 `[0.00s -> 2.00s]`）、MM:SS 格式或混合格式。
- 描述必须基于画面可见内容（人物、物体、动作、颜色、场景变化、显著情绪等）并结合该段字幕；不要加入未在帧中直接可见或无法确认的信息（禁止臆测）。
- 如果画面信息不明确，只描述你能确定的视觉事实，比如“若干人影”或“模糊的血迹”；不要输出“不确定”或“可能”的长句。
- 不要输出任何 JSON、注释、多余空行或额外解释；每个时间段一行。
- 如果提供的字幕块为空或缺失，**禁止**在输出中生成任何形式的字幕内容或占位符（例如不要输出 `字幕: "..."`、`字幕继续`、`(无)` 等）。
- 不要插入或保留任何方括号标签（例如不要输出 [开始] 或 [结束]）。


引导说明：
1) 优先描述视觉事实（谁、在哪、做了什么、有什么明显物体/颜色/动作）。
2) 在同一行中简洁地把该段字幕融入描述（如果字幕提供了语义信息），并且**确保时间为整部视频的绝对时间（秒，保留两位小数）**，例如：`[420.00s -> 422.00s] 画面为昏暗房间，一人躺着，字幕:"If we're living in together"`。
3) 每行长度控制在 10-30 个汉字（简短、信息密集）。

示例输出（仅作格式示范，实际内容请基于提供的帧与字幕）：
[420.00s -> 422.00s] 昏暗房间内一人躺着，穿红色传统服饰；屏幕下方出现英语字幕。
[422.00s -> 424.50s] 两人近身交谈，右侧人物表情紧张，背景可见一扇木门。

{focus_clause}
帧时间段：
{intervals}
字幕：
{transcript}

请严格按照上述规范返回中文时间轴列表。

附加规则（必须遵守）：
1) 如果若干相邻时间段的描述完全相同，请合并为一行，时间区间为这些相邻段的起始时间到结束时间，例如：`[13.18s -> 50.18s] 插图展示了彗星的轨迹。`。
2) 禁止在不同时间段重复输出字面相同的描述（若视觉未发生变化，请合并而非复述）。
3) 描述要尽量精炼且具区分性，避免泛化冗词和模糊表述。
4) 如果画面无明显变化或为静态插图，优先输出简洁总结并合并连续区间，而不是多次重复。
5) 严格禁止输出任何占位字幕词（例如“字幕继续”、“字幕：继续”、“字幕……”等）。只有在确实存在真实 ASR/字幕文本时，才在描述中以 `字幕:"..."` 形式包含字幕；如果该时间段没有真实 ASR，请不要写占位符或臆造字幕，直接省略字幕部分或只输出视觉描述。

示例（Few-shot）：
# Example 1: 静态插图应被合并
Example Input Intervals:
[10.00s -> 12.50s]
[12.50s -> 15.00s]
Transcript: (无)
Expected Output:
[10.00s -> 15.00s] 白色插图显示彗星轨迹与轨道标注。

# Example 2: 将视觉与 ASR 合并为一句简洁描述（示例使用绝对视频时间）
Example Input Intervals:
[270.00s -> 272.50s]
[272.50s -> 275.00s]
Transcript:
[270.00s -> 275.00s] "the outgassing can be seen because it ejects large amounts of dust"
Expected Output:
[270.00s -> 275.00s] 近景显示彗星外逸物质吹出细小尘埃，字幕:"the outgassing can be seen..."

# Example 3: 简洁的动作描述并合并时间段（避免臆造不存在的角色）
Example Input Intervals:
[90.00s -> 92.50s]
[92.50s -> 95.00s]
Transcript:
[90.00s -> 95.00s] "around the object called a coma and a tail."
Expected Output:
[90.00s -> 95.00s] 画面为宇宙圖像，背景出现彗星轨迹，字幕:"around the object called a coma and a tail."

# Example 4: 未提供 ASR 时不要写占位字幕，合并相邻视觉相同段（示例使用绝对视频时间）
Example Input Intervals:
[105.00s -> 117.50s]
[117.50s -> 124.00s]
Transcript: (无)
Expected Output:
[105.00s -> 124.00s] 画面为同一静态宇宙背景，主体未变化，镜头无明显移动。
"""

