# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BigBearSFX is an audio classification and organization tool that uses the Audio Spectrogram Transformer (AST) model to automatically categorize sound effects. It processes audio files from source directories, classifies them into 14 custom categories, generates semantic tags, and organizes them into a structured output directory.

## Commands

### Running the Main Application
```bash
cd ast/python
python main.py
```

### Setting Up the Environment
```bash
cd ast
python -m venv venvast
source venvast/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### AST Model Recipes (Original AST Repository)
The `ast/egs/` directory contains training/evaluation recipes for the original AST model:
- **AudioSet**: `cd ast/egs/audioset && ./run.sh`
- **ESC-50**: `cd ast/egs/esc50 && ./run_esc.sh`
- **SpeechCommands**: `cd ast/egs/speechcommands && ./run_sc.sh`

## Architecture

### Core Application (`ast/python/main.py`)
The main application is a single-file Python script (~1770 lines) with these key components:

1. **Configuration Section** (lines 33-312):
   - `SOURCE_DIRS`: List of source directories to scan for audio files
   - `TARGET_DIR`: Output directory for organized files
   - `CATEGORY_LIST`: 14 custom sound effect categories (自然环境, 城市环境, 机械设备, 生活家居, 人声, 动物声音, 冷兵器, 热兵器, UI交互, 抽象音效, 转场音效, 电影氛围, 特殊效果, 未分类素材)
   - `PROFESSIONAL_KEYWORDS`: Bilingual keyword dictionary for filename analysis

2. **AIEngine Class** (line 758+):
   - Loads AST model via HuggingFace transformers (`MIT/ast-finetuned-audioset-10-10-0.4593`)
   - `classify_audio()`: Fuses AST predictions with filename keyword analysis
   - `get_semantic_tags()`: Generates Chinese/English tags via Ollama API (Qwen model)
   - Short audio enhancement for files <1.5 seconds

3. **Audio Processing Pipeline**:
   - `preprocess_audio()`: Resamples to 16kHz, converts to mono, handles short audio
   - `extract_filename_keywords()`: Extracts professional terms from filenames
   - `generate_readable_filename()`: Creates descriptive filenames from analysis

4. **Main Workflow** (`main()` function):
   - Collects audio files from source directories
   - Computes MD5 for deduplication
   - Classifies and tags each file
   - Copies to categorized output directories
   - Maintains JSON database (`audio_library_v2.json`)

### AST Submodule Structure
The `ast/` directory contains the original AST repository:
- `egs/audioset/`, `egs/esc50/`, `egs/speechcommands/`: Training recipes
- `pretrained_models/`: Model weight storage
- `colab/`: Google Colab inference demos

## Key Dependencies

From `ast/requirements.txt`:
- `torch==1.8.1`, `torchaudio==0.8.1`, `torchvision==0.10.0`
- `timm==0.4.5` (Vision transformer models)
- `transformers` (HuggingFace, for AST model loading)
- `librosa`, `soundfile` (Audio processing)
- `numpy`, `scipy`, `scikit-learn`

## Important Notes

- **Audio Format**: All audio is resampled to 16kHz (AST model requirement)
- **Short Audio Handling**: Files <1.5s receive special enhancement (time stretching, spectral enhancement)
- **Classification Confidence**: Files with confidence <0.5 are redirected to "未分类素材"
- **Bilingual Support**: Keywords and tags support both Chinese and English
- **Model Cache**: HuggingFace models cached at `L:\Models\huggingface` (configurable via `HF_HOME`)
- **Ollama Integration**: Requires local Ollama server running with Qwen model for semantic tagging