from app.tool.base import BaseTool, ToolResult


class AskHuman(BaseTool):
    """Add a tool to ask human for help."""

    name: str = "ask_human"
    description: str = (
        "Use this tool to ask human for details when you find you need more information."
    )
    parameters: str = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "The question you want to ask human.",
            }
        },
        "required": ["inquire"],
    }

    async def execute(self, inquire: str) -> ToolResult:
        """Show the question in frontend chat interface.

        Args:
            inquire: The question to show in frontend.

        Returns:
            ToolResult with the question that will be shown in frontend.
        """

        return ToolResult(output=inquire)
