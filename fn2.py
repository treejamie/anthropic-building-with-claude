import json

from typing import cast, Any, Callable
from dotenv import load_dotenv
from datetime import datetime
from anthropic import Anthropic
from anthropic.types import (
    MessageParam,
    Message,
    ToolParam,
    ContentBlockParam,
    ToolResultBlockParam,
    TextBlock,
    ToolUseBlock,
)


# maybe refactor the tools out into their own module and update c1
def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    if not date_format:
        raise ValueError("date_format cannot be empty")

    return datetime.now().strftime(date_format)


get_current_datetime_schema = ToolParam(
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time formatted as a string. Returns the current datetime using Python's strftime formatting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": 'A strftime-compatible format string that controls the output format, e.g. "%Y-%m-%d %H:%M:%S" for "2026-03-25 14:30:00" or "%d/%m/%Y" for "25/03/2026". Must not be empty.',
                }
            },
        },
    }
)


load_dotenv()
client = Anthropic()

TOOLS: dict[str, Callable] = {"get_current_datetime": get_current_datetime}


def inspect(content: str, label: str | None = None) -> None:
    if label:
        print(label, ":\n")
    print(content, "\n----------\n\n")


def text_from_message(message: Message) -> str:
    return "\n".join(
        [block.text for block in message.content if isinstance(block, TextBlock)]
    )


def add_user_message(messages: list[MessageParam], message: Message | str) -> None:
    """Peasant context simulation v2"""
    user_message: MessageParam = {
        "role": "user",
        "content": cast(list[ContentBlockParam], message.content)
        if isinstance(message, Message)
        else message,
    }
    messages.append(user_message)


def add_assistant_message(messages: list[MessageParam], message: Message | str) -> None:
    """Peasant application management v2"""
    assistant_message: MessageParam = {
        "role": "assistant",
        "content": cast(list[ContentBlockParam], message.content)
        if isinstance(message, Message)
        else message,
    }
    messages.append(assistant_message)


def chat(
    messages: list[MessageParam],
    system: str | None = None,
    temperature: float = 0.2,
    stop_sequences: list[str] = [],
    tools: list[ToolParam] | None = None,
) -> Message:
    """Basic text response handling"""

    # define params
    params: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    if tools:
        params["tools"] = tools

    return client.messages.create(**params)


def run_conversation(messages: list) -> Message:
    while True:
        # call claude
        response = chat(messages, tools=[get_current_datetime_schema])

        # make the assiatant message
        add_assistant_message(messages, response)
        print(text_from_message(response))

        # if this is not a tool, we're done.
        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)

        add_user_message(messages, tool_results)
    # done
    return response


def run_tools(message):
    tool_requests: list[ToolUseBlock] = [
        block for block in message.content if block.type == "tool_use"
    ]
    tool_results_block: list[ToolResultBlockParam] = []

    for tool_request in tool_requests:
        if tool_request.name in TOOLS.keys():
            try:
                tool_output = TOOLS[tool_request.name](**tool_request.input)

                tool_result_block: ToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": tool_request.id,
                    "content": json.dumps(tool_output),
                    "is_error": False,
                }
            except Exception as e:
                tool_result_block: ToolResultBlockParam = {
                    "type": "tool_result",
                    "tool_use_id": tool_request.id,
                    "content": f"Error: {e}",
                    "is_error": True,
                }

            tool_results_block.append(tool_result_block)

    return tool_results_block
