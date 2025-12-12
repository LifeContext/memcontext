# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ContextBase (MemContext) is a multimodal AI agent memory architecture that serves as a "Second Brain" for AI agents. It implements a three-tier memory system (Short-Term, Medium-Term, Long-Term) with heat-based memory consolidation, supporting text, video, audio, and image inputs.

## Development Commands

### Quick Start
```bash
# Install dependencies for main playground
cd memcontext-playground
pip install -r requirements.txt

# Set required environment variable
export OPENAI_API_KEY="your-api-key"

# Run basic test
python test.py

# Start web demo
cd memdemo
python app.py
```

### Running Tests
```bash
# Basic functionality test
cd memcontext-playground
python test.py

# MCP server test
cd memcontext-mcp
python test_simple.py

# Video processing test
cd memcontext-playground
python test_video_converter.py
```

### Starting Services
```bash
# Web demo (http://localhost:5000)
cd memcontext-playground/memdemo
python app.py

# MCP Server
cd memcontext-mcp
python server_new.py --config config.json
```

## Architecture Overview

### Three-Tier Memory System

1. **Short-Term Memory (STM)** - `short_term.py`
   - Circular buffer holding recent conversation turns
   - Capacity: 10 items (configurable)
   - FIFO eviction when full

2. **Medium-Term Memory (MTM)** - `mid_term.py`
   - Session-based organization with heat calculation
   - Heat formula: H = α×N_visit + β×L_interaction + γ×R_recency
   - Sessions with heat ≥ 5.0 trigger LTM updates
   - Uses FAISS for similarity search

3. **Long-Term Memory (LTM)** - `long_term.py`
   - Persistent storage for user profiles and knowledge
   - Separate knowledge bases for user and assistant
   - Embedding-based retrieval

### Key Components

- **Memcontext** (`memcontext.py`): Main orchestrator managing all memory tiers
- **Retriever** (`retriever.py`): Parallel retrieval from all memory tiers
- **Multimodal Processing**: VideoRAG converter splits videos into segments, extracts frames, performs ASR
- **Updater**: Background process that moves data between memory tiers based on heat

### Data Flow
```
User Input → STM → (when full) → MTM Session → (if hot) → LTM
                 ↓
            Retrieval ← All Tiers (parallel)
```

### Key Design Patterns
- Heat-based memory consolidation (biologically inspired)
- Parallel processing for retrieval operations
- Circular buffers for automatic capacity management
- Embedding-based semantic similarity search
- Factory pattern for multimodal converter selection

## Important Configuration

- Memory capacities are configurable in `config.json`
- Default embedding model: "all-MiniLM-L6-v2"
- Mid-term similarity threshold: 0.6
- Heat calculation uses 24-hour half-life for recency
- Supports multiple embedding models including BAAI/bge-m3

## Module Structure

- `memcontext-playground/`: Main implementation with demos
- `memcontext-mcp/`: MCP server integration
- `memcontext-chromadb/`: ChromaDB alternative implementation
- `memcontext-pypi/`: PyPI package version