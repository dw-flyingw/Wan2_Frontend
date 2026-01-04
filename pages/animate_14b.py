"""
Animate-14B page for character animation and replacement.
"""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.common import (
    extract_audio_from_video,
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
from utils.generation import run_generation, run_preprocessing
from utils.metadata import create_metadata
from utils.prompt_utils import extend_prompt

TASK = "animate-14B"
TASK_KEY = get_task_session_key(TASK)
CONFIG = MODEL_CONFIGS[TASK]

st.title("Animate")
st.markdown("Animate a character from reference image using motion from source video")

# Initialize session state for this page
if f"{TASK_KEY}_extended_prompt" not in st.session_state:
    st.session_state[f"{TASK_KEY}_extended_prompt"] = None

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Mode selection
    mode = st.radio(
        "Mode",
        ["replacement", "animation"],
        help="Animation: Mimics motion from video. Replacement: Replaces person in video.",
    )

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
    res_parts = resolution.split("*")
    resolution_tuple = (int(res_parts[0]), int(res_parts[1]))

    st.divider()

    # Preprocessing options
    st.subheader("Preprocessing")
    fps = st.number_input("Target FPS", min_value=1, max_value=60, value=30)

    if mode == "animation":
        use_retarget = st.checkbox(
            "Use pose retargeting",
            help="Recommended when character poses differ significantly",
        )
        use_flux = st.checkbox(
            "Use FLUX image editing",
            disabled=not use_retarget,
            help="Recommended for non-standard poses",
        )
        iterations, k, w_len, h_len = 3, 7, 1, 1
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
        help="Number of frames for temporal guidance",
    )
    sample_steps = st.slider(
        "Sampling steps",
        min_value=10,
        max_value=50,
        value=CONFIG["default_steps"],
    )
    sample_solver = st.selectbox("Solver", ["unipc", "dpm++"], index=0)
    seed = st.number_input("Seed", min_value=-1, max_value=2147483647, value=-1, help="-1 for random")

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
        help=f"Method: {PROMPT_EXTEND_METHOD}",
    )
    if use_prompt_extend and PROMPT_EXTEND_MODEL:
        st.caption(f"Using `{PROMPT_EXTEND_METHOD}` method")
    elif use_prompt_extend and not PROMPT_EXTEND_MODEL:
        st.warning("PROMPT_EXTEND_MODEL not set in .env")

# Main content - Media inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Video")
    uploaded_video = st.file_uploader(
        "Upload driving video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Video containing the motion to transfer",
    )
    if uploaded_video:
        st.video(uploaded_video)

with col2:
    st.subheader("Reference Image")
    uploaded_image = st.file_uploader(
        "Upload reference image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Image of the character to animate",
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
    help="Choose where to get audio for the final video",
)

uploaded_audio = None
if audio_source == "Upload custom audio":
    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "m4a", "aac"],
        help="Audio to merge with the final generated video",
    )
    if uploaded_audio:
        st.audio(uploaded_audio)

# Project name
st.subheader("Project")
default_project_name = datetime.now().strftime("animate_%Y%m%d_%H%M%S")
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
    help="Describe what you want the animated character to do",
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

if st.button("Generate Animation", type="primary", use_container_width=True):
    if not uploaded_video:
        st.error("Please upload a source video")
    elif not uploaded_image:
        st.error("Please upload a reference image")
    elif not prompt.strip():
        st.error("Please enter a prompt")
    else:
        generation_start = datetime.now()

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
        preprocessing_time = None
        with st.status("Preprocessing...", expanded=True) as status:
            st.write("Extracting pose, face, and mask from source video...")

            success, output, preprocessing_time = run_preprocessing(
                video_path=video_path,
                image_path=image_path,
                output_path=processed_dir,
                mode=mode,
                resolution=resolution_tuple,
                fps=fps,
                use_retarget=use_retarget,
                use_flux=use_flux,
                iterations=iterations,
                k=k,
                w_len=w_len,
                h_len=h_len,
            )

            if not success:
                status.update(label="Preprocessing failed", state="error")
                st.error(output)
                st.stop()

            status.update(label=f"Preprocessing complete ({format_duration(preprocessing_time)})", state="complete")
            st.write("Generated files:")
            for f in processed_dir.glob("*"):
                st.write(f"  - {f.name}")

        # Step 2: Generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = project_dir / f"output_{timestamp}.mp4"

        # Use extended prompt if available
        generation_prompt = st.session_state.get(f"{TASK_KEY}_extended_prompt") or prompt

        with st.status("Generating animation...", expanded=True) as status:
            st.write(f"Running on {num_gpus} GPU(s)...")
            st.write(f"Prompt: {generation_prompt[:100]}...")

            success, output, generation_time = run_generation(
                task=TASK,
                output_file=final_output,
                prompt=generation_prompt,
                num_gpus=num_gpus,
                resolution=resolution,
                sample_steps=sample_steps,
                sample_solver=sample_solver,
                sample_shift=CONFIG["default_shift"],
                sample_guide_scale=CONFIG["default_guide_scale"],
                seed=seed,
                audio_path=audio_path,
                use_prompt_extend=False,  # Already extended in UI
                src_root_path=processed_dir,
                replace_flag=(mode == "replacement"),
                refert_num=refert_num,
                use_relighting_lora=use_relighting_lora,
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
            sample_shift=CONFIG["default_shift"],
            sample_guide_scale=CONFIG["default_guide_scale"],
            sample_solver=sample_solver,
            seed=seed,
            generation_start=generation_start,
            generation_end=generation_end,
            output_video_path=final_output,
            output_video_length_seconds=video_info["duration"],
            output_video_file_size_bytes=video_info["file_size_bytes"],
            extended_prompt=st.session_state.get(f"{TASK_KEY}_extended_prompt"),
            source_video_path="input/video.mp4",
            source_image_path="input/image.jpg",
            source_audio_path=str(audio_path.relative_to(project_dir)) if audio_path else None,
            preprocessing_time_seconds=preprocessing_time,
            extra_settings={
                "mode": mode,
                "refert_num": refert_num,
                "fps": fps,
                "use_retarget": use_retarget,
                "use_flux": use_flux,
                "use_relighting_lora": use_relighting_lora,
            },
        )
        metadata.save(project_dir / "metadata.json")

        # Also save prompt.txt for backwards compatibility
        (project_dir / "prompt.txt").write_text(prompt)

        # Display result
        if final_output.exists():
            st.success(f"Animation generated successfully! Saved to `output/{project_name}`")
            st.video(str(final_output))

            # Show metadata summary
            with st.expander("Generation Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{video_info['duration']:.1f}s")
                with col2:
                    st.metric("File Size", format_file_size(video_info["file_size_bytes"]))
                with col3:
                    st.metric("Total Time", format_duration(metadata.total_time_seconds))

                st.write(f"**GPUs Used:** {num_gpus}")
                st.write(f"**Preprocessing:** {format_duration(preprocessing_time)}")
                st.write(f"**Generation:** {format_duration(generation_time)}")

            # Show project contents
            with st.expander("Project files"):
                for subdir in [input_dir, processed_dir, project_dir]:
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
- **Animation mode**: Mimics motion from the source video onto the reference character
- **Replacement mode**: Replaces the person in the video with the reference character
- **Audio options**: Keep original audio from source video, upload custom audio, or generate silent video
- Use 2+ GPUs for faster generation with FSDP parallelism
- Click "Extend Prompt" to enhance your description and auto-generate anti-prompt
"""
)
