"""
Prompt extension utilities.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

# Add wan_overrides and Wan2.2 to path for imports
FRONTEND_ROOT = Path(__file__).parent.parent.resolve()
WAN_OVERRIDES = FRONTEND_ROOT / "wan_overrides"
sys.path.insert(0, str(WAN_OVERRIDES))

from .config import (
    PROMPT_EXTEND_LANG,
    PROMPT_EXTEND_METHOD,
    PROMPT_EXTEND_MODEL,
    WAN2_2_ROOT,
)

# Add Wan2.2 to path after overrides
sys.path.insert(1, str(WAN2_2_ROOT))

from wan.utils.prompt_extend import (
    DashScopePromptExpander,
    OpenAIPromptExpander,
    PromptOutput,
    QwenPromptExpander,
)


@dataclass
class PromptExtensionResult:
    """Result of prompt extension."""

    success: bool
    original_prompt: str
    extended_prompt: str
    message: str


def get_prompt_expander(task: str, is_vl: bool = False):
    """
    Create and return a prompt expander based on configuration.

    Args:
        task: The task type (t2v-A14B, i2v-A14B, s2v-14B, animate-14B)
        is_vl: Whether this is a vision-language task (has image input)

    Returns:
        A prompt expander instance or None if not configured
    """
    if not PROMPT_EXTEND_MODEL:
        logging.warning("PROMPT_EXTEND_MODEL not set, prompt extension disabled")
        return None

    try:
        if PROMPT_EXTEND_METHOD == "openai":
            return OpenAIPromptExpander(
                base_url=PROMPT_EXTEND_MODEL,
                task=task,
                is_vl=is_vl,
            )
        elif PROMPT_EXTEND_METHOD == "dashscope":
            return DashScopePromptExpander(
                task=task,
                is_vl=is_vl,
            )
        elif PROMPT_EXTEND_METHOD == "local_qwen":
            return QwenPromptExpander(
                model_name=PROMPT_EXTEND_MODEL if PROMPT_EXTEND_MODEL else None,
                task=task,
                is_vl=is_vl,
            )
        else:
            logging.warning(f"Unknown prompt extend method: {PROMPT_EXTEND_METHOD}")
            return None
    except Exception as e:
        logging.error(f"Failed to create prompt expander: {e}")
        return None


def extend_prompt(
    prompt: str,
    task: str,
    image: Optional[Image.Image] = None,
    seed: int = -1,
) -> PromptExtensionResult:
    """
    Extend a prompt using the configured LLM.

    Args:
        prompt: The original user prompt
        task: The task type (t2v-A14B, i2v-A14B, s2v-14B, animate-14B)
        image: Optional image for vision-language prompt extension
        seed: Random seed for reproducibility

    Returns:
        PromptExtensionResult with success status and extended prompt
    """
    is_vl = image is not None
    expander = get_prompt_expander(task, is_vl)

    if expander is None:
        return PromptExtensionResult(
            success=False,
            original_prompt=prompt,
            extended_prompt=prompt,
            message="Prompt extension not configured",
        )

    try:
        result: PromptOutput = expander(
            prompt,
            image=image,
            tar_lang=PROMPT_EXTEND_LANG,
            seed=seed,
        )

        if result.status:
            return PromptExtensionResult(
                success=True,
                original_prompt=prompt,
                extended_prompt=result.prompt,
                message="Prompt extended successfully",
            )
        else:
            return PromptExtensionResult(
                success=False,
                original_prompt=prompt,
                extended_prompt=prompt,
                message=f"Extension failed: {result.message}",
            )
    except Exception as e:
        logging.error(f"Prompt extension error: {e}")
        return PromptExtensionResult(
            success=False,
            original_prompt=prompt,
            extended_prompt=prompt,
            message=f"Extension error: {str(e)}",
        )


def get_prompt_extension_status() -> dict:
    """
    Get the current prompt extension configuration status.

    Returns:
        Dictionary with configuration status
    """
    return {
        "enabled": bool(PROMPT_EXTEND_MODEL),
        "method": PROMPT_EXTEND_METHOD,
        "language": PROMPT_EXTEND_LANG,
        "model_url": PROMPT_EXTEND_MODEL[:50] + "..." if len(PROMPT_EXTEND_MODEL) > 50 else PROMPT_EXTEND_MODEL,
    }
