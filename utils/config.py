"""
Configuration for Wan2.2 models and default settings.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
FRONTEND_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(FRONTEND_ROOT / ".env")

# Environment configuration
WAN2_2_REPO = os.getenv("WAN2_2_REPO", "./Wan2.2")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./output")
MODELS_PATH = os.getenv("MODELS_PATH", "/opt/huggingface")
PROMPT_EXTEND_METHOD = os.getenv("PROMPT_EXTEND_METHOD", "openai")
PROMPT_EXTEND_LANG = os.getenv("PROMPT_EXTEND_LANG", "en")
PROMPT_EXTEND_MODEL = os.getenv("PROMPT_EXTEND_MODEL", "")

# Resolve paths
if not Path(WAN2_2_REPO).is_absolute():
    WAN2_2_ROOT = FRONTEND_ROOT / WAN2_2_REPO
else:
    WAN2_2_ROOT = Path(WAN2_2_REPO)

if not Path(OUTPUT_PATH).is_absolute():
    OUTPUT_ROOT = FRONTEND_ROOT / OUTPUT_PATH
else:
    OUTPUT_ROOT = Path(OUTPUT_PATH)

PREPROCESS_SCRIPT = WAN2_2_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = WAN2_2_ROOT / "generate.py"

# Ensure output directory exists
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "t2v-A14B": {
        "task": "t2v-A14B",
        "name": "Text to Video",
        "description": "Generate video from text prompts only",
        "checkpoint": f"{MODELS_PATH}/Wan2.2-T2V-A14B",
        "sizes": ["1280*720", "720*1280", "832*480", "480*832"],
        "default_size": "1280*720",
        "default_steps": 40,
        "default_shift": 12.0,
        "default_guide_scale": 3.5,
        "frame_num": 81,
        "sample_fps": 16,
        "requires_image": False,
        "requires_video": False,
        "requires_audio": False,
        "requires_preprocessing": False,
    },
    "i2v-A14B": {
        "task": "i2v-A14B",
        "name": "Image to Video",
        "description": "Generate video from an image with text guidance",
        "checkpoint": f"{MODELS_PATH}/Wan2.2-I2V-A14B",
        "sizes": ["1280*720", "720*1280", "832*480", "480*832"],
        "default_size": "1280*720",
        "default_steps": 40,
        "default_shift": 5.0,
        "default_guide_scale": 3.5,
        "frame_num": 81,
        "sample_fps": 16,
        "requires_image": True,
        "requires_video": False,
        "requires_audio": False,
        "requires_preprocessing": False,
    },
    "s2v-14B": {
        "task": "s2v-14B",
        "name": "Speech to Video",
        "description": "Generate talking head video from audio and reference image",
        "checkpoint": f"{MODELS_PATH}/Wan2.2-S2V-14B",
        "sizes": ["1024*704", "704*1024", "1280*720", "720*1280", "832*480", "480*832"],
        "default_size": "1024*704",
        "default_steps": 40,
        "default_shift": 3.0,
        "default_guide_scale": 4.5,
        "frame_num": None,  # Determined by audio length
        "infer_frames": 80,
        "sample_fps": 16,
        "requires_image": True,
        "requires_video": False,
        "requires_audio": True,
        "requires_preprocessing": False,
        "supports_tts": True,
        "supports_pose_video": True,
    },
    "animate-14B": {
        "task": "animate-14B",
        "name": "Animate",
        "description": "Animate a character from reference image using motion from source video",
        "checkpoint": f"{MODELS_PATH}/Wan2.2-Animate-14B",
        "process_checkpoint": f"{MODELS_PATH}/Wan2.2-Animate-14B/process_checkpoint",
        "sizes": ["1280*720", "720*1280"],
        "default_size": "1280*720",
        "default_steps": 20,
        "default_shift": 5.0,
        "default_guide_scale": 1.0,
        "frame_num": 77,
        "sample_fps": 30,
        "requires_image": True,
        "requires_video": True,
        "requires_audio": False,  # Optional
        "requires_preprocessing": True,
    },
}

# Default sample prompts for each model
DEFAULT_PROMPTS = {
    "t2v-A14B": (
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
        "intensely on a spotlighted stage. The camera captures dynamic action "
        "with cinematic lighting and dramatic poses."
    ),
    "i2v-A14B": (
        "The subject in the image comes to life with natural, fluid movement. "
        "Camera slowly pushes in as the scene unfolds with subtle environmental "
        "details like wind and light changes."
    ),
    "s2v-14B": (
        "The person speaks expressively with natural lip movements and subtle "
        "facial expressions that match the audio. Eyes blink naturally and head "
        "moves slightly with speech emphasis."
    ),
    "animate-14B": (
        "The person from the reference image performs the same movements as shown "
        "in the source video, seamlessly matching the original motion and expression."
    ),
}

def init_session_state():
    """Initialize session state for all tasks."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # Per-task state
        for task in MODEL_CONFIGS.keys():
            task_key = task.replace("-", "_")
            st.session_state[f"{task_key}_prompt"] = DEFAULT_PROMPTS.get(task, "")
            st.session_state[f"{task_key}_extended_prompt"] = None
            st.session_state[f"{task_key}_generation_running"] = False
            st.session_state[f"{task_key}_last_output"] = None
            st.session_state[f"{task_key}_last_metadata"] = None


def get_task_session_key(task: str) -> str:
    """Convert task name to session state key format."""
    return task.replace("-", "_")
