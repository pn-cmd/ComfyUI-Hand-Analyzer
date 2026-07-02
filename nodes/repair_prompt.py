"""ComfyUI HandRepairPrompt — converts a HAND_REPORT into actionable repair prompts.

Takes the structured hand analysis report and outputs:
- repair_prompt: positive prompt for inpainting/re-generation
- negative_prompt: negative prompt additions for the issue types
- should_repair: boolean for pipeline routing (True if severity >= WARNING)
- repair_guidance: structured dict with per-hand bounding boxes, issues, strategy

This node closes the loop between detection and repair — it tells the
downstream inpainting/re-generation pipeline WHAT to fix and HOW.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Custom ComfyUI return type for repair guidance
"REPAIR_GUIDANCE"  # noqa: B018 — string used as type identifier in ComfyUI


class HandRepairPrompt:
    """Generate repair prompts from a hand analysis report."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "report": ("HAND_REPORT", {}),
            },
            "optional": {
                "include_base_prompts": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "REPAIR_GUIDANCE")
    RETURN_NAMES = ("repair_prompt", "negative_prompt", "should_repair", "repair_guidance")
    FUNCTION = "generate"
    CATEGORY = "Hand Analysis"

    def generate(
        self,
        report: dict,
        include_base_prompts: bool = True,
    ) -> tuple[str, str, bool, dict[str, Any]]:
        """Generate repair prompts from the hand analysis report.

        Parameters
        ----------
        report : dict
            The HAND_REPORT dict from HandAnalysisNode.
        include_base_prompts : bool
            If True, includes base positive/negative prompt fragments.
            If False, only issue-specific fragments are included.

        Returns
        -------
        tuple[str, str, bool, dict]
            (repair_prompt, negative_prompt, should_repair, repair_guidance)
        """
        # Import repair_prompts — handle both ComfyUI package and direct test imports
        try:
            from ..src.hand_analyzer.repair_prompts import (
                generate_repair_prompts,
                determine_repair_strategy,
                build_repair_guidance,
                BASE_POSITIVE_PROMPT,
                BASE_NEGATIVE_PROMPT,
            )
        except ImportError:
            from src.hand_analyzer.repair_prompts import (
                generate_repair_prompts,
                determine_repair_strategy,
                build_repair_guidance,
                BASE_POSITIVE_PROMPT,
                BASE_NEGATIVE_PROMPT,
            )

        # Generate prompts
        positive_prompt, negative_prompt = generate_repair_prompts(report)

        # If include_base_prompts is False, strip the base fragments
        if not include_base_prompts and positive_prompt:
            # Remove base prompts (they're at the start, separated by ", ")
            if positive_prompt.startswith(BASE_POSITIVE_PROMPT):
                positive_prompt = positive_prompt[len(BASE_POSITIVE_PROMPT):].lstrip(", ").strip()
            if negative_prompt.startswith(BASE_NEGATIVE_PROMPT):
                negative_prompt = negative_prompt[len(BASE_NEGATIVE_PROMPT):].lstrip(", ").strip()

        # Determine if repair is needed
        guidance = build_repair_guidance(report)
        should_repair = guidance["should_repair"]

        logger.info(
            "HandRepairPrompt: should_repair=%s, strategy=%s, issues=%d",
            should_repair, guidance["strategy"],
            sum(len(h.get("issue_types", [])) for h in guidance.get("hands", [])),
        )

        return (positive_prompt, negative_prompt, should_repair, guidance)