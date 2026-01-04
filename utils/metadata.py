"""
Metadata dataclass for tracking generation parameters and output information.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class GenerationMetadata:
    """Complete metadata for a video generation run."""

    # Timestamps
    timestamp: str
    generation_start: str
    generation_end: str

    # Task info
    task: str
    model_checkpoint: str

    # Source files (relative to project directory)
    source_video_path: str | None = None
    source_image_path: str | None = None
    source_audio_path: str | None = None
    pose_video_path: str | None = None

    # Prompts
    user_prompt: str = ""
    extended_prompt: str | None = None

    # Generation settings
    resolution: str = ""
    num_gpus: int = 1
    sample_steps: int = 20
    sample_shift: float = 5.0
    sample_guide_scale: float = 1.0
    sample_solver: str = "unipc"
    seed: int = -1

    # Timing
    preprocessing_time_seconds: float | None = None
    generation_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    # Output
    output_video_path: str = ""
    output_video_length_seconds: float = 0.0
    output_video_file_size_bytes: int = 0

    # Extra task-specific settings
    extra_settings: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "GenerationMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def create_metadata(
    task: str,
    model_checkpoint: str,
    user_prompt: str,
    resolution: str,
    num_gpus: int,
    sample_steps: int,
    sample_shift: float,
    sample_guide_scale: float,
    sample_solver: str,
    seed: int,
    generation_start: datetime,
    generation_end: datetime,
    output_video_path: Path,
    output_video_length_seconds: float,
    output_video_file_size_bytes: int,
    extended_prompt: str | None = None,
    source_video_path: str | None = None,
    source_image_path: str | None = None,
    source_audio_path: str | None = None,
    pose_video_path: str | None = None,
    preprocessing_time_seconds: float | None = None,
    extra_settings: dict | None = None,
) -> GenerationMetadata:
    """Create a GenerationMetadata instance with calculated timing."""
    total_time = (generation_end - generation_start).total_seconds()
    generation_time = total_time
    if preprocessing_time_seconds is not None:
        generation_time = total_time - preprocessing_time_seconds

    return GenerationMetadata(
        timestamp=datetime.now().isoformat(),
        generation_start=generation_start.isoformat(),
        generation_end=generation_end.isoformat(),
        task=task,
        model_checkpoint=model_checkpoint,
        source_video_path=source_video_path,
        source_image_path=source_image_path,
        source_audio_path=source_audio_path,
        pose_video_path=pose_video_path,
        user_prompt=user_prompt,
        extended_prompt=extended_prompt,
        resolution=resolution,
        num_gpus=num_gpus,
        sample_steps=sample_steps,
        sample_shift=sample_shift,
        sample_guide_scale=sample_guide_scale,
        sample_solver=sample_solver,
        seed=seed,
        preprocessing_time_seconds=preprocessing_time_seconds,
        generation_time_seconds=generation_time,
        total_time_seconds=total_time,
        output_video_path=str(output_video_path),
        output_video_length_seconds=output_video_length_seconds,
        output_video_file_size_bytes=output_video_file_size_bytes,
        extra_settings=extra_settings or {},
    )
