"""Codex provider integration for deepagents-cli."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class CodexChatModel(BaseChatModel):
    """Chat model wrapper that routes LangChain turns through Codex Responses API."""

    model: str = Field(default="gpt-5.3-codex")
    timeout_seconds: int = Field(default=120)
    base_url: str = Field(default="https://chatgpt.com/backend-api/codex")
    codex_home: str | None = Field(default=None)

    _bound_tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Store tool schemas in Responses format and return `self`."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            openai_tool = convert_to_openai_tool(tool)
            if not isinstance(openai_tool, dict):
                continue
            fn = openai_tool.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append(
                {
                    "type": "function",
                    "name": name.strip(),
                    "description": str(fn.get("description", "") or ""),
                    "parameters": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                    "strict": False,
                }
            )
        self._bound_tools = converted
        return self

    @property
    def _llm_type(self) -> str:
        """Return the provider identifier for LangChain diagnostics."""
        return "codex"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Generate a response via OpenAI-compatible Responses API for Codex."""
        instructions, input_items = self._messages_to_responses_input(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_items or [{"role": "user", "content": ""}],
            "store": False,
        }
        if self._bound_tools:
            payload["tools"] = self._bound_tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True

        try:
            response = self._request_response_via_stream(payload)
        except Exception as exc:
            msg = f"Codex Responses request failed: {exc}"
            raise RuntimeError(msg) from exc

        message = self._response_to_ai_message(response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _create_client(self) -> Any:
        """Create an OpenAI SDK client configured for the Codex backend."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            msg = "OpenAI SDK is required for Codex provider support"
            raise RuntimeError(msg) from exc

        return OpenAI(
            api_key=self._read_codex_access_token(),
            base_url=self.base_url,
            timeout=float(self.timeout_seconds),
        )

    def _request_response_via_stream(self, payload: dict[str, Any]) -> Any:
        """Issue a streamed Responses request and extract a terminal response object."""
        client = self._create_client()
        responses_api = getattr(client, "responses", None)
        if responses_api is None:
            msg = "OpenAI client is missing `responses` API"
            raise RuntimeError(msg)

        # Mirror Hermes harness behavior: prefer context-managed stream API
        # when the concrete SDK object exposes it.
        stream_fn = getattr(responses_api, "stream", None)
        if callable(stream_fn) and hasattr(type(responses_api), "stream"):
            return self._request_response_from_context_stream(responses_api, payload)
        return self._request_response_from_create_stream(responses_api, payload)

    def _request_response_from_context_stream(
        self, responses_api: Any, payload: dict[str, Any]
    ) -> Any:
        """Use `responses.stream(...)` and `get_final_response()` to complete."""
        collected_output_items: list[Any] = []
        collected_text_deltas: list[str] = []
        has_function_calls = False

        with responses_api.stream(**payload) as stream:
            for event in stream:
                event_type = self._item_get(event, "type", "")
                if event_type == "response.output_item.done":
                    done_item = self._item_get(event, "item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif isinstance(event_type, str) and "output_text.delta" in event_type:
                    delta = self._item_get(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        collected_text_deltas.append(delta)
                elif isinstance(event_type, str) and "function_call" in event_type:
                    has_function_calls = True

            final_response = stream.get_final_response()

        self._backfill_output_from_stream_events(
            final_response=final_response,
            collected_output_items=collected_output_items,
            collected_text_deltas=collected_text_deltas,
            has_function_calls=has_function_calls,
        )
        return final_response

    def _request_response_from_create_stream(
        self, responses_api: Any, payload: dict[str, Any]
    ) -> Any:
        """Fallback for mocked/legacy clients that only support create(stream=True)."""
        request_payload = dict(payload)
        request_payload["stream"] = True
        stream_or_response = responses_api.create(**request_payload)

        # Compatibility with mocks/providers returning a concrete response.
        if isinstance(stream_or_response, dict):
            return stream_or_response
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            msg = "Codex Responses stream did not return an iterable event stream"
            raise RuntimeError(msg)

        terminal_response: Any = None
        collected_output_items: list[Any] = []
        collected_text_deltas: list[str] = []
        has_function_calls = False

        try:
            for event in stream_or_response:
                event_type = self._item_get(event, "type", "")
                if event_type == "response.output_item.done":
                    done_item = self._item_get(event, "item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif isinstance(event_type, str) and "output_text.delta" in event_type:
                    delta = self._item_get(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        collected_text_deltas.append(delta)
                elif isinstance(event_type, str) and "function_call" in event_type:
                    has_function_calls = True

                if event_type in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    terminal_response = self._item_get(event, "response")
                    if terminal_response is not None:
                        break
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                close_fn()

        if terminal_response is None:
            logger.debug("Codex stream had no terminal response event; synthesizing")
            terminal_response = {"output": []}

        self._backfill_output_from_stream_events(
            final_response=terminal_response,
            collected_output_items=collected_output_items,
            collected_text_deltas=collected_text_deltas,
            has_function_calls=has_function_calls,
        )
        return terminal_response

    @staticmethod
    def _backfill_output_from_stream_events(
        *,
        final_response: Any,
        collected_output_items: list[Any],
        collected_text_deltas: list[str],
        has_function_calls: bool,
    ) -> None:
        """Populate `final_response.output` when stream transport omitted it."""
        output = CodexChatModel._item_get(final_response, "output")
        if isinstance(output, list) and output:
            return

        if collected_output_items:
            if isinstance(final_response, dict):
                final_response["output"] = list(collected_output_items)
            else:
                setattr(final_response, "output", list(collected_output_items))
            return

        if not collected_text_deltas or has_function_calls:
            return

        synthesized_output = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "".join(collected_text_deltas)}
                ],
            }
        ]
        if isinstance(final_response, dict):
            final_response["output"] = synthesized_output
        else:
            setattr(final_response, "output", synthesized_output)

    def _read_codex_access_token(self) -> str:
        """Read access token from Codex CLI auth store (`~/.codex/auth.json`)."""
        auth_path = self._codex_auth_path()
        if not auth_path.is_file():
            msg = "Codex session not found. Run `codex login` and try again."
            raise RuntimeError(msg)
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Could not read Codex auth file at {auth_path}: {exc}"
            raise RuntimeError(msg) from exc

        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            msg = "Codex auth file is missing `tokens`; run `codex login` again."
            raise RuntimeError(msg)

        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            msg = "Codex auth file has no `access_token`; run `codex login` again."
            raise RuntimeError(msg)
        return access_token.strip()

    def _codex_auth_path(self) -> Path:
        """Resolve the Codex auth.json path from `codex_home`, env, or default."""
        raw_home = self.codex_home or os.getenv("CODEX_HOME", "").strip()
        codex_home = Path(raw_home).expanduser() if raw_home else Path.home() / ".codex"
        return codex_home / "auth.json"

    def _messages_to_responses_input(
        self, messages: list[BaseMessage]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Translate LangChain messages into Responses API input items."""
        instructions = "You are a helpful assistant."
        items: list[dict[str, Any]] = []

        for message in messages:
            msg_type = getattr(message, "type", "")
            text_content = self._stringify_message_content(message)

            if msg_type == "system":
                if text_content.strip():
                    instructions = text_content
                continue

            if msg_type == "tool":
                tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
                if tool_call_id:
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": text_content,
                        }
                    )
                continue

            if isinstance(message, AIMessage):
                if text_content.strip():
                    items.append({"role": "assistant", "content": text_content})
                for tool_call in message.tool_calls or []:
                    name = str(tool_call.get("name", "") or "").strip()
                    call_id = str(tool_call.get("id", "") or "").strip()
                    if not name or not call_id:
                        continue
                    args = tool_call.get("args", {})
                    arguments = (
                        args
                        if isinstance(args, str)
                        else json.dumps(args, ensure_ascii=False)
                    )
                    arguments = arguments.strip() or "{}"
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
                continue

            items.append({"role": "user", "content": text_content})

        return instructions, items

    def _response_to_ai_message(self, response: Any) -> AIMessage:
        """Map a Responses API response object into LangChain `AIMessage`."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for item in self._item_iter(response):
            item_type = self._item_get(item, "type")
            if item_type == "message":
                for part in self._item_get(item, "content", []) or []:
                    part_type = self._item_get(part, "type")
                    if part_type in {"output_text", "text"}:
                        text = self._item_get(part, "text", "")
                        if isinstance(text, str) and text:
                            text_parts.append(text)
            elif item_type == "function_call":
                name = self._item_get(item, "name", "")
                call_id = self._item_get(item, "call_id", "")
                arguments = self._parse_arguments(
                    self._item_get(item, "arguments", "{}")
                )
                if isinstance(name, str) and isinstance(call_id, str) and name and call_id:
                    tool_calls.append(
                        {
                            "id": call_id,
                            "name": name,
                            "args": arguments,
                            "type": "tool_call",
                        }
                    )

        kwargs: dict[str, Any] = {
            "content": "".join(text_parts).strip(),
            "tool_calls": tool_calls,
        }

        usage = self._item_get(response, "usage")
        if usage is not None:
            kwargs["response_metadata"] = {
                "token_usage": {
                    "prompt_tokens": int(
                        self._item_get(usage, "input_tokens", 0) or 0
                    ),
                    "completion_tokens": int(
                        self._item_get(usage, "output_tokens", 0) or 0
                    ),
                    "total_tokens": int(
                        self._item_get(usage, "total_tokens", 0) or 0
                    ),
                }
            }

        return AIMessage(**kwargs)

    @staticmethod
    def _item_iter(response: Any) -> list[Any]:
        """Extract output item list from SDK object or plain dictionary."""
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        return output if isinstance(output, list) else []

    @staticmethod
    def _item_get(item: Any, key: str, default: Any = None) -> Any:
        """Read an attribute from an SDK object or a dictionary item."""
        value = getattr(item, key, None)
        if value is None and isinstance(item, dict):
            value = item.get(key)
        return default if value is None else value

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        """Parse Responses function-call arguments into a dictionary."""
        if isinstance(raw_arguments, dict):
            return raw_arguments

        text = str(raw_arguments or "").strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"_raw_arguments": text}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw_arguments": text}

    @staticmethod
    def _stringify_message_content(message: BaseMessage) -> str:
        """Flatten LangChain message content to plain text."""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                if isinstance(block, str):
                    chunks.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return " ".join(chunks)
        return str(content)
