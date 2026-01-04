"""
Common utility functions shared across all pages.
"""

import subprocess
from pathlib import Path


def get_available_gpus() -> int:
    """Get number of available NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except Exception:
        return 1


def get_gpu_info() -> list[dict]:
    """Get detailed GPU information."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                    }
                )
        return gpus
    except Exception:
        return []


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
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",  # no video
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            # Check if video has no audio
            if (
                "does not contain any stream" in result.stderr
                or "Output file is empty" in result.stderr
            ):
                return False, "Source video has no audio track"
            return False, f"Failed to extract audio: {result.stderr}"
        if output_path.exists() and output_path.stat().st_size > 0:
            return True, str(output_path)
        return False, "Source video has no audio track"
    except subprocess.TimeoutExpired:
        return False, "Audio extraction timed out"
    except Exception as e:
        return False, f"Audio extraction error: {str(e)}"


def get_video_info(video_path: Path) -> dict:
    """Get video information including duration, fps, and dimensions."""
    info = {
        "duration": 0.0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "file_size_bytes": 0,
    }

    if not video_path.exists():
        return info

    info["file_size_bytes"] = video_path.stat().st_size

    try:
        # Get duration
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["duration"] = float(result.stdout.strip())

        # Get fps and dimensions
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                info["width"] = int(parts[0])
                info["height"] = int(parts[1])
                # Parse fps (can be fraction like "30000/1001")
                fps_str = parts[2]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    info["fps"] = float(num) / float(den)
                else:
                    info["fps"] = float(fps_str)
    except Exception:
        pass

    return info


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"


def sanitize_project_name(name: str) -> str:
    """Sanitize project name for use as directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
