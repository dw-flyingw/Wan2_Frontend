# Wan2.2 Frontend

Streamlit frontend for Wan2.2 Animate pipeline. Allows uploading source video, reference image, and audio to generate animated videos.

## Prerequisites
# https://github.com/Wan-Video/Wan2.2

1. Clone Wan2.2 repo (clean, unmodified):
```bash
gh repo clone Wan-Video/Wan2.2 ../Wan2.2
```

T2V-A14B	Text-to-Video MoE model, supports 480P & 720P
I2V-A14B	Image-to-Video MoE model, supports 480P & 720P
TI2V-5B	    High-compression VAE, T2V+I2V, supports 720P
S2V-14B	    Speech-to-Video model, supports 480P & 720P
Animate-14B	Character animation and replacement
SAM2-hiera-large Video predictor for masking


2. Download models:
```bash
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir /opt/huggingface/Wan2.2-T2V-A14B
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir /opt/huggingface/Wan2.2-I2V-A14B
hf download Wan-AI/Wan2.2-S2V-14B --local-dir /opt/huggingface/Wan2.2-S2V-14B
hf download Wan-AI/Want2.2-TI2V-5B --local-dir /opt/huggingface/Wan2.2-TITV-5B	
hf download Wan-AI/Wan2.2-Animate-14B --local-dir /opt/huggingface/Wan2.2-Animate-14B
hf download facebook/sam2-hiera-large --local-dir /opt/huggingface/sam2-hiera-large
```

## Installation

```bash
# Install all dependencies (includes Animate, S2V, and dev tools)
uv sync

# Or with pip
pip install -e .

# Install flash_attn (requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# Install SAM-2 (required for animate preprocessing)
pip install git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266
```

**Notes:**
- `flash_attn` requires a CUDA-compatible GPU and the CUDA toolkit. Falls back to standard attention if unavailable.
- The project includes dependencies for all Wan2.2 features: T2V, I2V, Animate, and S2V.

## Configuration

Copy `.env.example` to `.env` (or edit the existing `.env`):

```bash
# Wan2.2 Repo Location (relative to this project or absolute path)
WAN2_2_REPO=../Wan2.2

# Output folder for generated projects
OUTPUT_PATH=./output

# Path to model weights
MODELS_PATH=/opt/huggingface

# Prompt extension settings
PROMPT_EXTEND_METHOD=openai
PROMPT_EXTEND_LANG=en
PROMPT_EXTEND_MODEL=http://your-llm-endpoint/v1
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
Wan2_Frontend/
├── app.py                 # Main Streamlit application
├── .env                   # Environment configuration
├── pyproject.toml         # Project dependencies
├── wan_overrides/         # Modified Wan2.2 files (override originals)
│   ├── generate.py        # Modified generation script
│   └── wan/utils/
│       ├── prompt_extend.py   # Added OpenAI prompt expander
│       └── system_prompt.py   # Added animate prompts
└── output/                # Generated projects (gitignored)
```

## How Overrides Work

This project keeps the Wan2.2 codebase separate and unmodified. The `wan_overrides/` directory contains modified versions of specific files that are loaded via Python path manipulation before the original Wan2.2 files.

Modified files:
- `generate.py` - Adds OpenAI prompt extension support, fixes `--refert_num` default
- `wan/utils/prompt_extend.py` - Adds `OpenAIPromptExpander` class for external LLM endpoints
- `wan/utils/system_prompt.py` - Adds `ANIMATE_EN_SYS_PROMPT` and `ANIMATE_ZH_SYS_PROMPT`

## Output Structure

Each generation creates a project folder in `output/<project_name>/`:

```
output/animate_20250102_120000/
├── input/
│   ├── video.mp4          # Source driving video
│   ├── image.jpg          # Reference character image
│   └── extracted_audio.mp3 # Audio from source video (if extracted)
├── processed/             # Preprocessed files (pose, face, mask)
├── output_*.mp4           # Generated video(s)
└── prompt.txt             # The prompt used for generation
```
