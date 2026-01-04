"""
S2V-14B page for Speech-to-Video generation.
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.common import (
    format_duration,
    format_file_size,
    get_available_gpus,
    get_video_info,
    sanitize_project_name,
    save_uploaded_file,
)
from utils.config import (
    DEFAULT_PROMPTS,
    MODEL_CONFIGS,
    OUTPUT_ROOT,
    PROMPT_EXTEND_METHOD,
    PROMPT_EXTEND_MODEL,
    get_task_session_key,
)
from utils.generation import run_generation
from utils.metadata import create_metadata
from utils.prompt_utils import extend_prompt

TASK = "s2v-14B"
TASK_KEY = get_task_session_key(TASK)
CONFIG = MODEL_CONFIGS[TASK]

st.title("Speech to Video")
st.markdown("Generate talking head video from audio and reference image using the S2V-14B model")

# Initialize session state for this page
if f"{TASK_KEY}_extended_prompt" not in st.session_state:
    st.session_state[f"{TASK_KEY}_extended_prompt"] = None

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # GPU configuration
    available_gpus = get_available_gpus()
    num_gpus = st.slider(
        "Number of GPUs",
        min_value=1,
        max_value=available_gpus,
        value=min(2, available_gpus),
        help=f"Available GPUs: {available_gpus}",
    )

    # Resolution
    resolution = st.selectbox(
        "Resolution",
        CONFIG["sizes"],
        index=0,
    )

    st.divider()

    # Generation options
    st.subheader("Generation Settings")

    sample_steps = st.slider(
        "Sampling steps",
        min_value=10,
        max_value=50,
        value=CONFIG["default_steps"],
    )

    sample_shift = st.slider(
        "Sample shift",
        min_value=1.0,
        max_value=10.0,
        value=CONFIG["default_shift"],
        step=0.5,
    )

    sample_guide_scale = st.slider(
        "Guidance scale",
        min_value=1.0,
        max_value=10.0,
        value=CONFIG["default_guide_scale"],
        step=0.5,
    )

    sample_solver = st.selectbox("Solver", ["unipc", "dpm++"], index=0)

    infer_frames = st.selectbox(
        "Frames per clip",
        [48, 64, 80, 96],
        index=2,
        help="Number of frames per video clip (must be multiple of 4)",
    )

    num_clip = st.number_input(
        "Number of clips",
        min_value=1,
        max_value=10,
        value=1,
        help="Number of video clips to generate. Total length won't exceed audio length.",
    )

    start_from_ref = st.checkbox(
        "Start from reference image",
        value=False,
        help="Use reference image as the starting point for generation",
    )

    seed = st.number_input("Seed", min_value=-1, max_value=2147483647, value=-1, help="-1 for random")

    st.divider()

    # Prompt extension option
    st.subheader("Prompt Extension")
    use_prompt_extend = st.checkbox(
        "Enable prompt extension",
        value=True,
        help=f"Method: {PROMPT_EXTEND_METHOD}",
    )
    if use_prompt_extend and PROMPT_EXTEND_MODEL:
        st.caption(f"Using `{PROMPT_EXTEND_METHOD}` method")
    elif use_prompt_extend and not PROMPT_EXTEND_MODEL:
        st.warning("PROMPT_EXTEND_MODEL not set in .env")

# Main content - Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Image")
    uploaded_image = st.file_uploader(
        "Upload reference image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Image of the person who will be speaking",
    )
    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)

with col2:
    st.subheader("Audio Source")
    audio_mode = st.radio(
        "Audio source",
        ["Upload audio file", "Text-to-Speech (TTS)"],
        help="Choose how to provide audio",
    )

    if audio_mode == "Upload audio file":
        uploaded_audio = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "aac"],
            help="Audio file for lip sync",
        )
        if uploaded_audio:
            st.audio(uploaded_audio)
        enable_tts = False
    else:
        enable_tts = True
        st.info("TTS requires a reference audio sample (5-15s, 16kHz+)")

        tts_prompt_audio = st.file_uploader(
            "TTS reference audio",
            type=["wav", "mp3"],
            help="5-15 second audio sample for voice cloning",
        )
        if tts_prompt_audio:
            st.audio(tts_prompt_audio)

        tts_prompt_text = st.text_input(
            "Reference audio text",
            help="Text that exactly matches the reference audio content",
        )

        tts_text = st.text_area(
            "Text to synthesize",
            height=100,
            help="The text you want the character to speak",
        )
        uploaded_audio = None

# Optional pose video
with st.expander("Advanced: Pose-driven generation"):
    st.markdown("Optionally provide a pose video to control head movement and gestures")
    pose_video = st.file_uploader(
        "Upload pose video (optional)",
        type=["mp4", "avi", "mov"],
        help="DW-Pose sequence for pose-driven generation",
    )
    if pose_video:
        st.video(pose_video)

# Project name
st.subheader("Project")
default_project_name = datetime.now().strftime("s2v_%Y%m%d_%H%M%S")
project_name = st.text_input(
    "Project name",
    value=default_project_name,
    help="All files will be saved in ./output/<project_name>/",
)
project_name = sanitize_project_name(project_name) or default_project_name
project_dir = OUTPUT_ROOT / project_name
st.caption(f"Output folder: `output/{project_name}`")

if project_dir.exists():
    existing_files = list(project_dir.glob("*"))
    if existing_files:
        st.warning(f"Project folder already exists with {len(existing_files)} files. Files may be overwritten.")

# Prompt section
st.subheader("Prompts")

prompt = st.text_area(
    "Your Prompt",
    value=DEFAULT_PROMPTS[TASK],
    height=100,
    help="Describe the speaking style and expressions",
)

# Prompt extension button
col1, col2 = st.columns([1, 4])
with col1:
    extend_clicked = st.button("Extend Prompt", disabled=not use_prompt_extend or not PROMPT_EXTEND_MODEL)

if extend_clicked:
    with st.spinner("Extending prompt..."):
        result = extend_prompt(prompt, TASK)
        if result.success:
            st.session_state[f"{TASK_KEY}_extended_prompt"] = result.extended_prompt
            st.success("Prompt extended successfully!")
        else:
            st.warning(f"Extension failed: {result.message}")

# Show extended prompt if available
if st.session_state.get(f"{TASK_KEY}_extended_prompt"):
    with st.expander("Extended Prompt (from LLM)", expanded=True):
        st.text_area(
            "Extended",
            value=st.session_state[f"{TASK_KEY}_extended_prompt"],
            height=150,
            disabled=True,
            label_visibility="collapsed",
        )

# Generate button
st.divider()

# Validation
can_generate = True
error_messages = []

if not uploaded_image:
    can_generate = False
    error_messages.append("Please upload a reference image")

if audio_mode == "Upload audio file" and not uploaded_audio:
    can_generate = False
    error_messages.append("Please upload an audio file")
elif audio_mode == "Text-to-Speech (TTS)":
    if not tts_prompt_audio:
        can_generate = False
        error_messages.append("Please upload TTS reference audio")
    if not tts_text:
        can_generate = False
        error_messages.append("Please enter text to synthesize")

if st.button("Generate Video", type="primary", use_container_width=True):
    if not can_generate:
        for msg in error_messages:
            st.error(msg)
    else:
        generation_start = datetime.now()

        # Create project directory
        project_dir.mkdir(parents=True, exist_ok=True)
        input_dir = project_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        image_path = save_uploaded_file(uploaded_image, input_dir / "image.jpg")

        audio_path = None
        tts_audio_path = None
        if uploaded_audio:
            audio_ext = Path(uploaded_audio.name).suffix
            audio_path = save_uploaded_file(uploaded_audio, input_dir / f"audio{audio_ext}")

        if enable_tts and tts_prompt_audio:
            tts_ext = Path(tts_prompt_audio.name).suffix
            tts_audio_path = save_uploaded_file(tts_prompt_audio, input_dir / f"tts_reference{tts_ext}")

        pose_video_path = None
        if pose_video:
            pose_ext = Path(pose_video.name).suffix
            pose_video_path = save_uploaded_file(pose_video, input_dir / f"pose{pose_ext}")

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = project_dir / f"output_{timestamp}.mp4"

        # Use extended prompt if available
        generation_prompt = st.session_state.get(f"{TASK_KEY}_extended_prompt") or prompt

        with st.status("Generating video...", expanded=True) as status:
            st.write(f"Running on {num_gpus} GPU(s)...")
            st.write(f"Resolution: {resolution}, Frames per clip: {infer_frames}")
            if enable_tts:
                st.write("Using TTS for audio generation...")
            st.write(f"Prompt: {generation_prompt[:100]}...")

            success, output, generation_time = run_generation(
                task=TASK,
                output_file=final_output,
                prompt=generation_prompt,
                num_gpus=num_gpus,
                resolution=resolution,
                sample_steps=sample_steps,
                sample_solver=sample_solver,
                sample_shift=sample_shift,
                sample_guide_scale=sample_guide_scale,
                seed=seed,
                image_path=image_path,
                audio_path=audio_path,
                use_prompt_extend=False,  # Already extended in UI
                enable_tts=enable_tts,
                tts_prompt_audio=tts_audio_path,
                tts_prompt_text=tts_prompt_text if enable_tts else None,
                tts_text=tts_text if enable_tts else None,
                pose_video=pose_video_path,
                infer_frames=infer_frames,
                start_from_ref=start_from_ref,
                num_clip=num_clip if num_clip > 1 else None,
            )

            if not success:
                status.update(label="Generation failed", state="error")
                st.error(output)
                st.stop()

            status.update(label=f"Generation complete ({format_duration(generation_time)})", state="complete")

        generation_end = datetime.now()

        # Get output video info
        video_info = get_video_info(final_output)

        # Save metadata
        metadata = create_metadata(
            task=TASK,
            model_checkpoint=CONFIG["checkpoint"],
            user_prompt=prompt,
            resolution=resolution,
            num_gpus=num_gpus,
            sample_steps=sample_steps,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_solver=sample_solver,
            seed=seed,
            generation_start=generation_start,
            generation_end=generation_end,
            output_video_path=final_output,
            output_video_length_seconds=video_info["duration"],
            output_video_file_size_bytes=video_info["file_size_bytes"],
            extended_prompt=st.session_state.get(f"{TASK_KEY}_extended_prompt"),
            source_image_path="input/image.jpg",
            source_audio_path=str(audio_path.relative_to(project_dir)) if audio_path else None,
            pose_video_path=str(pose_video_path.relative_to(project_dir)) if pose_video_path else None,
            extra_settings={
                "audio_mode": audio_mode,
                "enable_tts": enable_tts,
                "infer_frames": infer_frames,
                "num_clip": num_clip,
                "start_from_ref": start_from_ref,
                "tts_text": tts_text if enable_tts else None,
            },
        )
        metadata.save(project_dir / "metadata.json")

        # Also save prompt.txt for reference
        (project_dir / "prompt.txt").write_text(prompt)

        # Display result
        if final_output.exists():
            st.success(f"Video generated successfully! Saved to `output/{project_name}`")
            st.video(str(final_output))

            # Show metadata summary
            with st.expander("Generation Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{video_info['duration']:.1f}s")
                with col2:
                    st.metric("File Size", format_file_size(video_info["file_size_bytes"]))
                with col3:
                    st.metric("Generation Time", format_duration(generation_time))

                st.write(f"**GPUs Used:** {num_gpus}")
                st.write(f"**Resolution:** {resolution}")
                st.write(f"**Audio Mode:** {audio_mode}")

            # Show project contents
            with st.expander("Project files"):
                for subdir in [input_dir, project_dir]:
                    if subdir == project_dir:
                        files = [f for f in subdir.glob("*") if f.is_file()]
                    else:
                        files = list(subdir.glob("*"))
                    if files:
                        label = f"**{subdir.relative_to(project_dir)}/**" if subdir != project_dir else "**Root:**"
                        st.write(label)
                        for f in files:
                            st.write(f"  - {f.name}")
        else:
            st.warning("Output file not found at expected location. Check logs for details.")
            st.code(output)

# Footer
st.divider()
st.markdown(
    """
**Tips:**
- Upload a clear frontal image of the person for best lip-sync results
- Audio length determines video length (max limited by clip settings)
- **TTS Mode**: Provide a 5-15 second reference audio sample with matching text for voice cloning
- **Pose Video**: Optionally provide DW-Pose sequence to control head/body movement
- "Start from reference" uses your image as the first frame
- Use 2+ GPUs for faster generation with FSDP parallelism
"""
)
