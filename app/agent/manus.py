import asyncio
import os
from datetime import datetime
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from app.agent.base import AgentState, BaseAgentEvents
from app.agent.browser import BrowserContextHelper
from app.agent.react import ReActAgent
from app.agent.toolcall import ToolCallContextHelper
from app.config import config
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, PLAN_PROMPT, SYSTEM_PROMPT
from app.sandbox.client import SANDBOX_CLIENT, SANDBOX_MANAGER
from app.sandbox.core.sandbox import DockerSandbox
from app.schema import Message
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.base import BaseTool
from app.tool.bash import Bash
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.create_chat_completion import CreateChatCompletion
from app.tool.deep_research import DeepResearch
from app.tool.planning import PlanningTool
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.web_search import WebSearch

SYSTEM_TOOLS: list[BaseTool] = [
    Bash(),
    WebSearch(),
    DeepResearch(),
    BrowserUseTool(),
    StrReplaceEditor(),
    PlanningTool(),
    CreateChatCompletion(),
    AskHuman(),
]

SYSTEM_TOOLS_MAP = {tool.name: tool.__class__ for tool in SYSTEM_TOOLS}


class McpToolConfig(BaseModel):
    id: str
    name: str
    # for stdio
    command: str
    args: list[str]
    env: dict[str, str]
    # for sse
    url: str
    headers: dict[str, Any]


class Manus(ReActAgent):
    """A versatile general-purpose agent."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT.format(
        directory="/workspace",
        task_id="Not Specified",
        task_dir="Not Specified",
        language="English",
        current_date=datetime.now().strftime("%Y-%m-%d"),
        max_steps=20,
        current_step=0,
    )
    next_step_prompt: str = NEXT_STEP_PROMPT.format(
        max_steps=20,
        current_step=0,
        remaining_steps=20,
        task_dir="Not Specified",
    )
    plan_prompt: str = PLAN_PROMPT.format(
        max_steps=20,
        language="English",
        available_tools="",
    )

    max_steps: int = 20
    task_request: str = ""

    tool_call_context_helper: Optional[ToolCallContextHelper] = None
    browser_context_helper: Optional[BrowserContextHelper] = None

    task_dir: str = ""
    language: Optional[str] = Field(None, description="Language for the agent")
    sandbox: Optional[DockerSandbox] = None

    def initialize(
        self,
        task_id: str,
        language: Optional[str] = None,
        tools: Optional[list[Union[McpToolConfig, str]]] = None,
        max_steps: Optional[int] = None,
        task_request: Optional[str] = None,
    ):
        self.task_id = task_id
        self.language = language
        self.task_dir = f"/workspace/{task_id}"
        self.current_step = 0
        self.tools = tools

        if max_steps is not None:
            self.max_steps = max_steps

        if task_request is not None:
            self.task_request = task_request

        return self

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        return self

    async def prepare(self) -> None:
        """Prepare the agent for execution."""
        self.system_prompt = SYSTEM_PROMPT.format(
            directory="/workspace",
            task_id=self.task_id,
            task_dir=self.task_dir,
            language=self.language or "English",
            current_date=datetime.now().strftime("%Y-%m-%d"),
            max_steps=self.max_steps,
            current_step=self.current_step,
        )

        self.next_step_prompt = NEXT_STEP_PROMPT.format(
            max_steps=self.max_steps,
            current_step=self.current_step,
            remaining_steps=self.max_steps - self.current_step,
            task_dir=self.task_dir,
        )

        await self.update_memory(
            role="system", content=self.system_prompt, base64_image=None
        )

        orgnization_id, task_id = self.task_id.split("/")
        sandbox_id = f"openmanus-sandbox-{orgnization_id}-{task_id}"
        host_workspace_root = str(f"{config.host_workspace_root}/{orgnization_id}")
        volume_bindings = {
            host_workspace_root: f"/workspace/{orgnization_id}",
            host_workspace_root: f"/workspace",
        }

        await SANDBOX_MANAGER.create_sandbox(
            sandbox_id=sandbox_id, volume_bindings=volume_bindings
        )
        self.sandbox = await SANDBOX_MANAGER.get_sandbox(sandbox_id)
        self.browser_context_helper = BrowserContextHelper(self)
        self.tool_call_context_helper = ToolCallContextHelper(self)
        self.tool_call_context_helper.available_tools = ToolCollection(Terminate())

        if self.tools:
            for tool in self.tools:
                if isinstance(tool, str) and tool in SYSTEM_TOOLS_MAP:
                    inst = SYSTEM_TOOLS_MAP[tool]()
                    await self.tool_call_context_helper.add_tool(inst)
                    if hasattr(inst, "llm"):
                        inst.llm = self.llm
                elif isinstance(tool, McpToolConfig):
                    await self.tool_call_context_helper.add_mcp(
                        {
                            "client_id": tool.id,
                            "url": tool.url,
                            "command": tool.command,
                            "args": tool.args,
                            "env": tool.env,
                            "headers": tool.headers,
                        }
                    )

        # Initialize attributes for user input pausing mechanism
        self._user_input_event = asyncio.Event()
        self._next_user_input: Optional[str] = None
        self._pending_input_for_think: Optional[str] = None

    async def plan(self) -> str:
        """Create an initial plan based on the user request."""
        # Create planning message
        self.emit(BaseAgentEvents.LIFECYCLE_PLAN_START, {})

        self.plan_prompt = PLAN_PROMPT.format(
            language=self.language or "English",
            max_steps=self.max_steps,
            available_tools="\n".join(
                [
                    f"- {tool.name}: {tool.description}"
                    for tool in self.tool_call_context_helper.available_tools
                ]
            ),
        )
        planning_message = await self.llm.ask(
            [
                Message.system_message(self.plan_prompt),
                Message.user_message(self.task_request),
            ],
            system_msgs=[Message.system_message(self.system_prompt)],
        )

        # Add the planning message to memory
        await self.update_memory("user", planning_message)
        self.emit(BaseAgentEvents.LIFECYCLE_PLAN_COMPLETE, {"plan": planning_message})
        return planning_message

    async def think(self, user_prompt: Optional[str] = None) -> bool:
        """Process current state and decide next actions with appropriate context.

        Args:
            user_prompt: Optional additional user prompt to consider during thinking
        """
        # Determine the actual prompt to use for this thinking step
        current_step_user_instruction = (
            user_prompt  # Explicitly passed prompt takes precedence
        )

        if (
            not current_step_user_instruction
            and hasattr(self, "_pending_input_for_think")
            and self._pending_input_for_think is not None
        ):
            current_step_user_instruction = self._pending_input_for_think
            self._pending_input_for_think = None  # Consume it

        # Update next_step_prompt with current step information
        original_prompt = self.next_step_prompt  # Store original before modification
        self.next_step_prompt = NEXT_STEP_PROMPT.format(
            max_steps=self.max_steps,
            current_step=self.current_step,
            remaining_steps=self.max_steps - self.current_step,
            task_dir=self.task_dir,
        )

        # Add user prompt if provided
        if current_step_user_instruction:
            self.next_step_prompt = f"{self.next_step_prompt}\n\nAdditional User Instruction for this step: {current_step_user_instruction}"

        browser_in_use = self._check_browser_in_use_recently()

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await self.tool_call_context_helper.ask_tool()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result

    async def act(self) -> str:
        """Execute decided actions"""
        results = await self.tool_call_context_helper.execute_tool()
        return "\n\n".join(results)

    def _check_browser_in_use_recently(self) -> bool:
        """Check if the browser is in use by looking at the last 3 messages."""
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )
        return browser_in_use

    async def receive_user_input(self, input_str: str):
        """Receives user input from the frontend to resume a paused agent."""
        logger.info(f"Agent: Received user input: '{input_str}'")
        self._next_user_input = input_str
        if hasattr(self, "_user_input_event"):
            self._user_input_event.set()
        else:
            logger.error(
                "Agent: _user_input_event not initialized when receiving input."
            )

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously. But add user input handling.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        self.emit(BaseAgentEvents.LIFECYCLE_START, {"request": request})

        results: List[str] = []
        self.emit(BaseAgentEvents.LIFECYCLE_PREPARE_START, {})
        await self.prepare()
        self.emit(BaseAgentEvents.LIFECYCLE_PREPARE_COMPLETE, {})
        async with self.state_context(AgentState.RUNNING):
            if request:
                await self.update_memory("user", request)
                if self.should_plan:
                    await self.plan()

            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(
                    f"Agent: Executing step {self.current_step}/{self.max_steps}"
                )

                # Pass the input received from the last pause to the current step's thinking phase
                if hasattr(self, "_next_user_input"):  # Ensure attribute exists
                    self._pending_input_for_think = self._next_user_input
                    self._next_user_input = None  # Consume

                try:
                    step_result = await self.step()
                except Exception as e:
                    raise

                # Check for stuck state
                if self.is_stuck():
                    self.emit(BaseAgentEvents.STATE_STUCK_DETECTED, {})
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

                if self.should_terminate:
                    self.state = AgentState.FINISHED
                    logger.info(
                        f"Agent: Terminating after step {self.current_step} due to internal signal."
                    )
                    break

                # Pause for user input if more steps are allowed and agent not terminated
                if self.current_step < self.max_steps:
                    self.emit(
                        "agent_paused_for_input",
                        {
                            "current_step": self.current_step,
                            "max_steps": self.max_steps,
                            "message": "Agent is waiting for your input to proceed.",
                        },
                    )
                    if hasattr(self, "_user_input_event"):
                        self._user_input_event.clear()
                    else:  # Should not happen if prepare is called
                        logger.error(
                            "Agent: _user_input_event not initialized before pause."
                        )
                        self._user_input_event = asyncio.Event()  # Defensive init

                    logger.info(
                        f"Agent: Step {self.current_step} completed. Pausing for user input."
                    )
                    # await self._user_input_event.wait()  # Actual pause
                    input_str = input("Enter your input: ")
                    self._next_user_input = input_str
                    logger.info(
                        f"Agent: Resumed by user. Input for next step ({self.current_step + 1 if self.current_step < self.max_steps else 'final'}): '{self._next_user_input}'"
                    )

                    # Check if user input signals termination
                    if (
                        isinstance(self._next_user_input, str)
                        and self._next_user_input.lower() == "terminate"
                    ):
                        self.should_terminate = True
                        self.state = AgentState.FINISHED  # Ensure state is FINISHED
                        logger.info(
                            "Agent: Termination signal 'terminate' received from user input."
                        )
                        break  # Exit the loop
                # If self.current_step == self.max_steps, loop condition handles exit in the next iteration
            # End of while loop

            # Handle max steps reached (user's specific logic from diff)
            if (
                self.current_step >= self.max_steps
                and self.state != AgentState.FINISHED
            ):
                self.current_step = 0  # User's change
                self.state = AgentState.IDLE  # User's change
                self.emit(
                    BaseAgentEvents.STEP_MAX_REACHED, {"max_steps": self.max_steps}
                )
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
                # If max_steps is reached, it's a form of completion.
                # If should_terminate is not set, LIFECYCLE_COMPLETE will be emitted.

        await SANDBOX_CLIENT.cleanup()
        if self.should_terminate or self.state == AgentState.FINISHED:
            self.emit(
                BaseAgentEvents.LIFECYCLE_TERMINATED,
                {
                    "total_input_tokens": (
                        self.llm.total_input_tokens if self.llm else 0
                    ),
                    "total_completion_tokens": (
                        self.llm.total_completion_tokens if self.llm else 0
                    ),
                },
            )
        else:  # This includes the case where max_steps was reached and state became IDLE
            self.emit(
                BaseAgentEvents.LIFECYCLE_COMPLETE,
                {
                    "results": results,
                    "total_input_tokens": (
                        self.llm.total_input_tokens if self.llm else 0
                    ),
                    "total_completion_tokens": (
                        self.llm.total_completion_tokens if self.llm else 0
                    ),
                },
            )
        return "\n".join(results) if results else "No steps executed"

    async def cleanup(self):
        """Clean up Manus agent resources."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        if self.tool_call_context_helper:
            await self.tool_call_context_helper.cleanup_tools()
        if self.sandbox:
            await SANDBOX_MANAGER.delete_sandbox(self.sandbox.id)
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")
