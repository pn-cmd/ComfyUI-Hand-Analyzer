"""ComfyUI Hand Analyzer — custom node pack for hand/finger analysis.

This module is the ComfyUI entry point.  ComfyUI scans custom_nodes directories
for ``__init__.py`` files that expose ``NODE_CLASS_MAPPINGS`` and
``NODE_DISPLAY_NAME_MAPPINGS``.
"""
from __future__ import annotations

# ComfyUI discovers these mappings automatically.
# The imports are wrapped in try/except so that the module can be imported
# in test environments without the full ComfyUI/torch/mediapipe stack.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

try:
    from .nodes.hand_analysis import HandAnalysisNode
    from .nodes.report_formatter import HandReportFormatter
    from .nodes.region_cropper import HandRegionCropper
    from .nodes.repair_prompt import HandRepairPrompt

    NODE_CLASS_MAPPINGS = {
        "HandAnalysis": HandAnalysisNode,
        "HandReportFormatter": HandReportFormatter,
        "HandRegionCropper": HandRegionCropper,
        "HandRepairPrompt": HandRepairPrompt,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "HandAnalysis": "Hand Analysis 🔍",
        "HandReportFormatter": "Hand Report → JSON 📋",
        "HandRegionCropper": "Hand Region Cropper ✂️",
        "HandRepairPrompt": "Hand Repair Prompt 🔧",
    }
except ImportError:
    # In test environments without ComfyUI deps, the mappings remain empty.
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]