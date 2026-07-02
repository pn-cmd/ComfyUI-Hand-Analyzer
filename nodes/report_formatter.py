"""ComfyUI HandReportFormatter — converts a HAND_REPORT dict to a JSON string.

This makes the report accessible as a text output for downstream nodes
(e.g., ShowText|pysssss) or for the pipeline to parse.
"""
from __future__ import annotations

import json
from typing import Any

class HandReportFormatter:
    """Format a hand analysis report as JSON string for downstream consumption."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "report": ("HAND_REPORT", {}),
                "indent": ("INT", {"default": 2, "min": 0, "max": 8}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_string",)
    FUNCTION = "format"
    CATEGORY = "Hand Analysis"

    def format(self, report: dict, indent: int = 2) -> tuple[str]:
        """Convert the report dict to a JSON string."""
        if isinstance(report, str):
            # Already a JSON string — try to pretty-print
            try:
                parsed = json.loads(report)
                return (json.dumps(parsed, indent=indent, default=str),)
            except json.JSONDecodeError:
                return (report,)
        return (json.dumps(report, indent=indent, default=str),)