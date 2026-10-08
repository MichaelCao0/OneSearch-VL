from vision_deepresearch_async_workflow.tool_contract import allowed_names

from vision_deepresearch_async_workflow.tools.shared import DeepResearchTool
from vision_deepresearch_async_workflow.tools.video_tools import (
    SelectFrameTool,
    SelectTimespanTool,
)
from vision_deepresearch_async_workflow.tools.visual_tools import (
    CropTool,
    ImageSearchTool,
    LayoutParsingTool,
    PerspectiveCorrectTool,
    SharpenTool,
    SuperResolutionTool,
    TextSearchTool,
    WebSearchTool,
)

# Registered implementations for the union of the three prompt tool sets.
DEEPRESEARCH_TOOLS = {
    "select_timespan": SelectTimespanTool(),
    "select_frame": SelectFrameTool(),
    "crop": CropTool(),
    "image_search": ImageSearchTool(),
    "text_search": TextSearchTool(),
    "layout_parsing": LayoutParsingTool(),
    "perspective_correct": PerspectiveCorrectTool(),
    "super_resolution": SuperResolutionTool(),
    "sharpen": SharpenTool(),
    "web_search": WebSearchTool(),
}

def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    names = allowed_names(DEEPRESEARCH_TOOLS)
    if names is None:
        return DEEPRESEARCH_TOOLS.copy()
    return {name: tool for name, tool in DEEPRESEARCH_TOOLS.items() if name in names}
