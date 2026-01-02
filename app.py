#!/usr/bin/env python3
"""
Streamlit app for Wan2.2 Animate pipeline.
Allows uploading source video, reference image, and audio to generate animated videos.

All output is saved in ./output/<project_name>/ with the following structure:
    input/          - Source video, reference image, extracted audio
    processed/      - Preprocessed files (pose, face, mask, background)
    output_*.mp4    - Generated video(s)
    prompt.txt      - The prompt used for generation

Usage:
    streamlit run app.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
FRONTEND_ROOT = Path(__file__).parent.resolve()
load_dotenv(FRONTEND_ROOT / ".env")

# Configuration from environment
WAN2_2_REPO = os.getenv("WAN2_2_REPO", "./Wan2.2")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./output")
MODELS_PATH = os.getenv("MODELS_PATH", "/opt/huggingface")
PROMPT_EXTEND_METHOD = os.getenv("PROMPT_EXTEND_METHOD", "openai")
PROMPT_EXTEND_LANG = os.getenv("PROMPT_EXTEND_LANG", "en")
PROMPT_EXTEND_MODEL = os.getenv("PROMPT_EXTEND_MODEL", "")

# Resolve Wan2.2 repo path
if not Path(WAN2_2_REPO).is_absolute():
    WAN2_2_ROOT = FRONTEND_ROOT / WAN2_2_REPO
else:
    WAN2_2_ROOT = Path(WAN2_2_REPO)

# Add Wan2.2 repo to Python path (for imports like wan.configs, etc.)
# But first, add our overrides directory so our modified files take precedence
WAN_OVERRIDES = FRONTEND_ROOT / "wan_overrides"
sys.path.insert(0, str(WAN_OVERRIDES))
sys.path.insert(1, str(WAN2_2_ROOT))

# Resolve paths
if not Path(OUTPUT_PATH).is_absolute():
    OUTPUT_ROOT = FRONTEND_ROOT / OUTPUT_PATH
else:
    OUTPUT_ROOT = Path(OUTPUT_PATH)

PREPROCESS_SCRIPT = WAN2_2_ROOT / "wan" / "modules" / "animate" / "preprocess" / "preprocess_data.py"
GENERATE_SCRIPT = WAN_OVERRIDES / "generate.py"  # Use our modified generate.py
DEFAULT_CKPT_DIR = f"{MODELS_PATH}/Wan2.2-Animate-14B"
DEFAULT_PROCESS_CKPT = f"{MODELS_PATH}/Wan2.2-Animate-14B/process_checkpoint"

# Ensure output directory exists
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="Wan2.2 Animate",
    page_icon="",
    layout="wide"
)

st.title("Wan2.2 Animate")
st.markdown("Character animation and replacement in video using Wan2.2")


def get_available_gpus():
    """Get number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        return len(result.stdout.strip().split('\n'))
    except Exception:
        return 1


def save_uploaded_file(uploaded_file, dest_path: Path) -> Path:
    """Save uploaded file to destination path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def extract_audio_from_video(video_path: Path, output_path: Path) -> tuple[bool, str]:
    """Extract audio track from video using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",  # no video
                "-acodec", "libmp3lame",
                "-q:a", "2",
                str(output_path)
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            # Check if video has no audio
            if "does not contain any stream" in result.stderr or "Output file is empty" in result.stderr:
                return False, "Source video has no audio track"
            return False, f"Failed to extract audio: {result.stderr}"
        if output_path.exists() and output_path.stat().st_size > 0:
            return True, str(output_path)
        return False, "Source video has no audio track"
    except subprocess.TimeoutExpired:
        return False, "Audio extraction timed out"
    except Exception as e:
        return False, f"Audio extraction error: {str(e)}"


def run_preprocessing(
    video_path: Path,
    image_path: Path,
    output_path: Path,
    mode: str,
    resolution: tuple[int, int],
    fps: int = 30,
    use_retarget: bool = False,
    use_flux: bool = False,
    iterations: int = 3,
    k: int = 7,
    w_len: int = 1,
    h_len: int = 1,
) -> tuple[bool, str]:
    """Run the preprocessing pipeline."""
    cmd = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--ckpt_path", DEFAULT_PROCESS_CKPT,
        "--video_path", str(video_path),
        "--refer_path", str(image_path),
        "--save_path", str(output_path),
        "--resolution_area", str(resolution[0]), str(resolution[1]),
        "--fps", str(fps),
    ]

    if mode == "replacement":
        cmd.extend([
            "--replace_flag",
            "--iterations", str(iterations),
            "--k", str(k),
            "--w_len", str(w_len),
            "--h_len", str(h_len),
        ])
    else:  # animation mode
        if use_retarget:
            cmd.append("--retarget_flag")
            if use_flux:
                cmd.append("--use_flux")

    # Run from preprocess directory for imports to work
    preprocess_dir = PREPROCESS_SCRIPT.parent

    # Set up environment with our overrides
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WAN_OVERRIDES}:{WAN2_2_ROOT}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(preprocess_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        if result.returncode != 0:
            return False, f"Preprocessing failed:\n{result.stderr}\n{result.stdout}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Preprocessing timed out after 30 minutes"
    except Exception as e:
        return False, f"Preprocessing error: {str(e)}"


def run_generation(
    src_root_path: Path,
    output_file: Path,
    mode: str,
    num_gpus: int,
    prompt: str,
    audio_path: Path | None = None,
    resolution: str = "1280*720",
    refert_num: int = 5,
    sample_steps: int = 20,
    sample_solver: str = "unipc",
    use_relighting_lora: bool = False,
    use_prompt_extend: bool = False,
    prompt_extend_method: str = "local_qwen",
    prompt_extend_model: str | None = None,
    prompt_extend_lang: str = "en",
) -> tuple[bool, str]:
    """Run the generation pipeline."""

    # Set up environment with our overrides
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WAN_OVERRIDES}:{WAN2_2_ROOT}:{env.get('PYTHONPATH', '')}"

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(GENERATE_SCRIPT),
        ]
    else:
        cmd = [sys.executable, str(GENERATE_SCRIPT)]

    cmd.extend([
        "--task", "animate-14B",
        "--size", resolution,
        "--ckpt_dir", DEFAULT_CKPT_DIR,
        "--src_root_path", str(src_root_path),
        "--refert_num", str(refert_num),
        "--sample_steps", str(sample_steps),
        "--sample_solver", sample_solver,
        "--save_file", str(output_file),
        "--prompt", prompt,
    ])

    if mode == "replacement":
        cmd.append("--replace_flag")
        if use_relighting_lora:
            cmd.append("--use_relighting_lora")

    if num_gpus > 1:
        cmd.extend(["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(num_gpus)])

    if audio_path and audio_path.exists():
        cmd.extend(["--audio", str(audio_path)])

    if use_prompt_extend:
        cmd.append("--use_prompt_extend")
        cmd.extend(["--prompt_extend_method", prompt_extend_method])
        cmd.extend(["--prompt_extend_target_lang", prompt_extend_lang])
        if prompt_extend_model:
            cmd.extend(["--prompt_extend_model", prompt_extend_model])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WAN2_2_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        if result.returncode != 0:
            return False, f"Generation failed:\n{result.stderr}\n{result.stdout}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Generation timed out after 2 hours"
    except Exception as e:
        return False, f"Generation error: {str(e)}"


# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Mode selection
    mode = st.radio(
        "Mode",
        ["animation", "replacement"],
        help="Animation: Mimics motion from video. Replacement: Replaces person in video."
    )

    # GPU configuration
    available_gpus = get_available_gpus()
    num_gpus = st.slider(
        "Number of GPUs",
        min_value=1,
        max_value=available_gpus,
        value=min(2, available_gpus),
        help=f"Available GPUs: {available_gpus}"
    )

    # Resolution
    resolution = st.selectbox(
        "Resolution",
        ["1280*720", "720*1280", "832*480", "480*832"],
        index=0
    )
    res_parts = resolution.split("*")
    resolution_tuple = (int(res_parts[0]), int(res_parts[1]))

    st.divider()

    # Preprocessing options
    st.subheader("Preprocessing")
    fps = st.number_input("Target FPS", min_value=1, max_value=60, value=30)

    if mode == "animation":
        use_retarget = st.checkbox(
            "Use pose retargeting",
            help="Recommended when character poses differ significantly"
        )
        use_flux = st.checkbox(
            "Use FLUX image editing",
            disabled=not use_retarget,
            help="Recommended for non-standard poses"
        )
    else:
        use_retarget = False
        use_flux = False
        st.markdown("**Mask parameters:**")
        iterations = st.number_input("Dilation iterations", min_value=1, max_value=10, value=3)
        k = st.number_input("Kernel size", min_value=3, max_value=15, value=7, step=2)
        w_len = st.number_input("W subdivisions", min_value=1, max_value=5, value=1)
        h_len = st.number_input("H subdivisions", min_value=1, max_value=5, value=1)

    st.divider()

    # Generation options
    st.subheader("Generation")
    refert_num = st.selectbox(
        "Reference frames",
        [1, 5],
        index=1,
        help="Number of frames for temporal guidance"
    )
    sample_steps = st.slider("Sampling steps", min_value=10, max_value=50, value=20)
    sample_solver = st.selectbox("Solver", ["unipc", "dpm++"], index=0)

    if mode == "replacement":
        use_relighting_lora = st.checkbox("Use relighting LoRA")
    else:
        use_relighting_lora = False

    st.divider()

    # Prompt extension option
    st.subheader("Prompt Extension")
    use_prompt_extend = st.checkbox(
        "Enable prompt extension",
        value=True,
        help=f"Method: {PROMPT_EXTEND_METHOD}, Lang: {PROMPT_EXTEND_LANG}"
    )
    if use_prompt_extend and PROMPT_EXTEND_MODEL:
        st.caption(f"Using `{PROMPT_EXTEND_METHOD}` method")
    elif use_prompt_extend and not PROMPT_EXTEND_MODEL:
        st.warning("PROMPT_EXTEND_MODEL not set in .env")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Video")
    uploaded_video = st.file_uploader(
        "Upload driving video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Video containing the motion to transfer"
    )
    if uploaded_video:
        st.video(uploaded_video)

with col2:
    st.subheader("Reference Image")
    uploaded_image = st.file_uploader(
        "Upload reference image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Image of the character to animate"
    )
    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)

# Audio options
st.subheader("Audio")
audio_source = st.radio(
    "Audio source",
    ["From source video", "Upload custom audio", "No audio"],
    index=0,
    horizontal=True,
    help="Choose where to get audio for the final video"
)

uploaded_audio = None
if audio_source == "Upload custom audio":
    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "m4a", "aac"],
        help="Audio to merge with the final generated video"
    )
    if uploaded_audio:
        st.audio(uploaded_audio)

# Project name
st.subheader("Project")
default_project_name = datetime.now().strftime("animate_%Y%m%d_%H%M%S")
project_name = st.text_input(
    "Project name",
    value=default_project_name,
    help="All files will be saved in ./output/<project_name>/"
)
# Sanitize project name
project_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
if not project_name:
    project_name = default_project_name

project_dir = OUTPUT_ROOT / project_name
st.caption(f"Output folder: `{project_dir.relative_to(FRONTEND_ROOT)}`")

# Check if project exists
if project_dir.exists():
    existing_files = list(project_dir.glob("*"))
    if existing_files:
        st.warning(f"Project folder already exists with {len(existing_files)} files. Files may be overwritten.")

# Prompt input
st.subheader("Prompt")
prompt = st.text_area(
    "Describe the animation",
    value="The person from the reference image performs the same movements as shown in the source video, seamlessly matching the original motion and expression.",
    height=100,
    help="Describe what you want the animated character to do"
)

# Generate button
st.divider()

if st.button("Generate Animation", type="primary", use_container_width=True):
    if not uploaded_video:
        st.error("Please upload a source video")
    elif not uploaded_image:
        st.error("Please upload a reference image")
    elif not prompt.strip():
        st.error("Please enter a prompt")
    else:
        # Create project directory structure
        project_dir.mkdir(parents=True, exist_ok=True)
        input_dir = project_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = project_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        video_path = save_uploaded_file(uploaded_video, input_dir / "video.mp4")
        image_path = save_uploaded_file(uploaded_image, input_dir / "image.jpg")

        # Handle audio based on user selection
        audio_path = None
        if audio_source == "From source video":
            extracted_audio = input_dir / "extracted_audio.mp3"
            with st.status("Extracting audio from source video...", expanded=False) as audio_status:
                success, result = extract_audio_from_video(video_path, extracted_audio)
                if success:
                    audio_path = extracted_audio
                    audio_status.update(label="Audio extracted", state="complete")
                else:
                    audio_status.update(label=f"Warning: {result}", state="complete")
                    st.warning(f"Could not extract audio: {result}. Continuing without audio.")
        elif audio_source == "Upload custom audio" and uploaded_audio:
            audio_ext = Path(uploaded_audio.name).suffix
            audio_path = save_uploaded_file(uploaded_audio, input_dir / f"audio{audio_ext}")

        # Step 1: Preprocessing
        with st.status("Preprocessing...", expanded=True) as status:
            st.write("Extracting pose, face, and mask from source video...")

            preprocess_kwargs = {
                "video_path": video_path,
                "image_path": image_path,
                "output_path": processed_dir,
                "mode": mode,
                "resolution": resolution_tuple,
                "fps": fps,
                "use_retarget": use_retarget,
                "use_flux": use_flux,
            }

            if mode == "replacement":
                preprocess_kwargs.update({
                    "iterations": iterations,
                    "k": k,
                    "w_len": w_len,
                    "h_len": h_len,
                })

            success, output = run_preprocessing(**preprocess_kwargs)

            if not success:
                status.update(label="Preprocessing failed", state="error")
                st.error(output)
                st.stop()

            status.update(label="Preprocessing complete", state="complete")
            st.write("Generated files:")
            for f in processed_dir.glob("*"):
                st.write(f"  - {f.name}")

        # Step 2: Generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = project_dir / f"output_{timestamp}.mp4"

        with st.status("Generating animation...", expanded=True) as status:
            st.write(f"Running on {num_gpus} GPU(s)...")
            st.write(f"Prompt: {prompt[:100]}...")

            success, output = run_generation(
                src_root_path=processed_dir,
                output_file=final_output,
                mode=mode,
                num_gpus=num_gpus,
                prompt=prompt,
                audio_path=audio_path,
                resolution=resolution,
                refert_num=refert_num,
                sample_steps=sample_steps,
                sample_solver=sample_solver,
                use_relighting_lora=use_relighting_lora,
                use_prompt_extend=use_prompt_extend,
                prompt_extend_method=PROMPT_EXTEND_METHOD,
                prompt_extend_model=PROMPT_EXTEND_MODEL if PROMPT_EXTEND_MODEL else None,
                prompt_extend_lang=PROMPT_EXTEND_LANG,
            )

            if not success:
                status.update(label="Generation failed", state="error")
                st.error(output)
                st.stop()

            status.update(label="Generation complete", state="complete")

        # Save prompt to file for reference
        prompt_file = project_dir / "prompt.txt"
        prompt_file.write_text(prompt)

        # Display result
        if final_output.exists():
            st.success(f"Animation generated successfully! Saved to `{project_dir.relative_to(FRONTEND_ROOT)}`")
            st.video(str(final_output))

            # Show project contents
            with st.expander("Project files"):
                for subdir in [input_dir, processed_dir, project_dir]:
                    if subdir == project_dir:
                        files = [f for f in subdir.glob("*") if f.is_file()]
                    else:
                        files = list(subdir.glob("*"))
                    if files:
                        st.write(f"**{subdir.relative_to(project_dir)}/**" if subdir != project_dir else "**Root:**")
                        for f in files:
                            st.write(f"  - {f.name}")
        else:
            # Check for output in default location
            st.warning("Output file not found at expected location. Check logs for details.")
            st.code(output)

# Footer
st.divider()
st.markdown("""
**Tips:**
- **Animation mode**: Mimics motion from the source video onto the reference character
- **Replacement mode**: Replaces the person in the video with the reference character
- **Audio options**: Keep original audio from source video, upload custom audio, or generate silent video
- Use 2+ GPUs for faster generation with FSDP parallelism
- Enable prompt extension for more detailed descriptions
""")
