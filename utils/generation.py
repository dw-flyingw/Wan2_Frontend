"""
Unified generation runner for all Wan2.2 tasks.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import (
    GENERATE_SCRIPT,
    MODEL_CONFIGS,
    PREPROCESS_SCRIPT,
    PROMPT_EXTEND_LANG,
    PROMPT_EXTEND_METHOD,
    PROMPT_EXTEND_MODEL,
    WAN2_2_ROOT,
)


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
    timeout: int = 1800,
) -> tuple[bool, str, float]:
    """
    Run the preprocessing pipeline for animate task.

    Args:
        video_path: Path to source video
        image_path: Path to reference image
        output_path: Path to save processed files
        mode: "animation" or "replacement"
        resolution: (width, height) tuple
        fps: Target FPS
        use_retarget: Enable pose retargeting
        use_flux: Enable FLUX image editing
        iterations: Mask dilation iterations (replacement mode)
        k: Kernel size (replacement mode)
        w_len: W subdivisions (replacement mode)
        h_len: H subdivisions (replacement mode)
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, output_or_error, elapsed_seconds)
    """
    start_time = time.time()
    process_ckpt = MODEL_CONFIGS["animate-14B"]["process_checkpoint"]

    cmd = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--ckpt_path",
        process_ckpt,
        "--video_path",
        str(video_path),
        "--refer_path",
        str(image_path),
        "--save_path",
        str(output_path),
        "--resolution_area",
        str(resolution[0]),
        str(resolution[1]),
        "--fps",
        str(fps),
    ]

    if mode == "replacement":
        cmd.extend(
            [
                "--replace_flag",
                "--iterations",
                str(iterations),
                "--k",
                str(k),
                "--w_len",
                str(w_len),
                "--h_len",
                str(h_len),
            ]
        )
    else:  # animation mode
        if use_retarget:
            cmd.append("--retarget_flag")
            if use_flux:
                cmd.append("--use_flux")

    # Run from preprocess directory for imports to work
    preprocess_dir = PREPROCESS_SCRIPT.parent

    # Set up environment with Wan2.2 in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WAN2_2_ROOT}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(preprocess_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start_time
        if result.returncode != 0:
            return False, f"Preprocessing failed:\n{result.stderr}\n{result.stdout}", elapsed
        return True, result.stdout, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, f"Preprocessing timed out after {timeout} seconds", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Preprocessing error: {str(e)}", elapsed


def run_generation(
    task: str,
    output_file: Path,
    prompt: str,
    num_gpus: int,
    resolution: str = "1280*720",
    sample_steps: int = 20,
    sample_solver: str = "unipc",
    sample_shift: Optional[float] = None,
    sample_guide_scale: Optional[float] = None,
    seed: int = -1,
    # Optional inputs
    image_path: Optional[Path] = None,
    audio_path: Optional[Path] = None,
    # Prompt extension options
    use_prompt_extend: bool = False,
    # Animate-specific
    src_root_path: Optional[Path] = None,
    replace_flag: bool = False,
    refert_num: int = 5,
    use_relighting_lora: bool = False,
    # S2V-specific
    enable_tts: bool = False,
    tts_prompt_audio: Optional[Path] = None,
    tts_prompt_text: Optional[str] = None,
    tts_text: Optional[str] = None,
    pose_video: Optional[Path] = None,
    infer_frames: int = 80,
    start_from_ref: bool = False,
    num_clip: Optional[int] = None,
    # I2V/T2V-specific
    frame_num: Optional[int] = None,
    # Timeout
    timeout: int = 7200,
) -> tuple[bool, str, float]:
    """
    Run the generation pipeline for any Wan2.2 task.

    Args:
        task: Task type (t2v-A14B, i2v-A14B, s2v-14B, animate-14B)
        output_file: Path to save output video
        prompt: Text prompt
        num_gpus: Number of GPUs to use
        resolution: Output resolution (e.g., "1280*720")
        sample_steps: Number of sampling steps
        sample_solver: Solver type ("unipc" or "dpm++")
        sample_shift: Sampling shift (uses model default if None)
        sample_guide_scale: Guidance scale (uses model default if None)
        seed: Random seed (-1 for random)
        image_path: Path to input image (for i2v, s2v)
        audio_path: Path to audio file (for s2v, animate with audio)
        use_prompt_extend: Enable prompt extension
        src_root_path: Path to preprocessed files (for animate)
        replace_flag: Use replacement mode (for animate)
        refert_num: Number of reference frames (for animate)
        use_relighting_lora: Use relighting LoRA (for animate replacement)
        enable_tts: Enable TTS (for s2v)
        tts_prompt_audio: TTS prompt audio path (for s2v)
        tts_prompt_text: TTS prompt text (for s2v)
        tts_text: Text to synthesize (for s2v)
        pose_video: Path to pose video (for s2v)
        infer_frames: Frames per clip (for s2v)
        start_from_ref: Start from reference image (for s2v)
        num_clip: Number of video clips (for s2v)
        frame_num: Number of frames (for t2v, i2v)
        timeout: Timeout in seconds

    Returns:
        Tuple of (success, output_or_error, elapsed_seconds)
    """
    start_time = time.time()
    config = MODEL_CONFIGS.get(task)
    if config is None:
        return False, f"Unknown task: {task}", 0.0

    ckpt_dir = config["checkpoint"]

    # Set up environment with Wan2.2 in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WAN2_2_ROOT}:{env.get('PYTHONPATH', '')}"

    # Build command
    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(GENERATE_SCRIPT),
        ]
    else:
        cmd = [sys.executable, str(GENERATE_SCRIPT)]

    # Common arguments
    cmd.extend(
        [
            "--task",
            task,
            "--size",
            resolution,
            "--ckpt_dir",
            ckpt_dir,
            "--sample_steps",
            str(sample_steps),
            "--sample_solver",
            sample_solver,
            "--save_file",
            str(output_file),
            "--prompt",
            prompt,
        ]
    )

    # Optional sample parameters
    if sample_shift is not None:
        cmd.extend(["--sample_shift", str(sample_shift)])
    if sample_guide_scale is not None:
        cmd.extend(["--sample_guide_scale", str(sample_guide_scale)])
    if seed >= 0:
        cmd.extend(["--base_seed", str(seed)])
    if frame_num is not None:
        cmd.extend(["--frame_num", str(frame_num)])

    # Image input (for i2v, s2v)
    if image_path and image_path.exists():
        cmd.extend(["--image", str(image_path)])

    # Audio input
    if audio_path and audio_path.exists():
        cmd.extend(["--audio", str(audio_path)])

    # Multi-GPU options
    if num_gpus > 1:
        cmd.extend(["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(num_gpus)])

    # Prompt extension
    if use_prompt_extend and PROMPT_EXTEND_MODEL:
        cmd.append("--use_prompt_extend")
        cmd.extend(["--prompt_extend_method", PROMPT_EXTEND_METHOD])
        cmd.extend(["--prompt_extend_target_lang", PROMPT_EXTEND_LANG])
        cmd.extend(["--prompt_extend_model", PROMPT_EXTEND_MODEL])

    # Task-specific arguments
    if task == "animate-14B":
        if src_root_path:
            cmd.extend(["--src_root_path", str(src_root_path)])
        cmd.extend(["--refert_num", str(refert_num)])
        if replace_flag:
            cmd.append("--replace_flag")
            if use_relighting_lora:
                cmd.append("--use_relighting_lora")

    elif task == "s2v-14B":
        cmd.extend(["--infer_frames", str(infer_frames)])
        if enable_tts:
            cmd.append("--enable_tts")
            if tts_prompt_audio:
                cmd.extend(["--tts_prompt_audio", str(tts_prompt_audio)])
            if tts_prompt_text:
                cmd.extend(["--tts_prompt_text", tts_prompt_text])
            if tts_text:
                cmd.extend(["--tts_text", tts_text])
        if pose_video and pose_video.exists():
            cmd.extend(["--pose_video", str(pose_video)])
        if start_from_ref:
            cmd.append("--start_from_ref")
        if num_clip is not None:
            cmd.extend(["--num_clip", str(num_clip)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WAN2_2_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start_time
        if result.returncode != 0:
            return False, f"Generation failed:\n{result.stderr}\n{result.stdout}", elapsed
        return True, result.stdout, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, f"Generation timed out after {timeout} seconds", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"Generation error: {str(e)}", elapsed
