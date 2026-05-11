"""LangChain-compatible chat model backed by the openai SDK.

Talks to any OpenAI-compatible base URL (OpenAI, Ollama, Together, vLLM).
Implements the slice of BaseChatModel that LangGraph's StateGraph + ToolNode
actually exercise: bind_tools, _generate (for invoke), _stream (for token
streaming), and message format conversion both directions.

Why this exists: lets us drop langchain-openai entirely while keeping the
existing LangGraph wiring untouched. Swap providers by changing one env var.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, List, Optional, Sequence

from openai import OpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr


def _lc_messages_to_openai(messages: Sequence[BaseMessage]) -> List[dict]:
    out: List[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": _content_str(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _content_str(m.content)})
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": _content_str(m.content)}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args") or {}),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": _content_str(m.content),
                }
            )
        else:
            # Fallback: best-effort by message type string
            out.append({"role": getattr(m, "type", "user"), "content": _content_str(m.content)})
    return out


def _content_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if t:
                    parts.append(t)
        return "".join(parts)
    return str(content)


class OpenAICompatChat(BaseChatModel):
    """Chat model that calls any OpenAI-compatible endpoint via the openai SDK."""

    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    tools: Optional[List[dict]] = Field(default=None)

    _client: OpenAI = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        # Local servers (Ollama, vLLM) accept any non-empty key. For the real
        # OpenAI endpoint, prefer letting the SDK raise its own clear error
        # ("api_key client option must be set...") instead of sending a junk
        # key and getting back a confusing 401.
        if self.api_key:
            key = self.api_key
        elif self.base_url:
            key = "sk-no-key"
        else:
            key = None
        self._client = OpenAI(base_url=self.base_url, api_key=key)

    @property
    def _llm_type(self) -> str:
        return "openai-compat"

    def bind_tools(self, tools: Sequence[BaseTool | dict], **kwargs: Any) -> "OpenAICompatChat":
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.model_copy(update={"tools": formatted})

    def _request_kwargs(self, messages: Sequence[BaseMessage], **kwargs: Any) -> dict:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": _lc_messages_to_openai(messages),
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.tools:
            params["tools"] = self.tools
        params.update(kwargs)
        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = self._request_kwargs(messages, **kwargs)
        if stop:
            params["stop"] = stop
        resp = self._client.chat.completions.create(**params)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args, "type": "tool_call"})
        ai = AIMessage(content=msg.content or "", tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._request_kwargs(messages, **kwargs)
        params["stream"] = True
        if stop:
            params["stop"] = stop
        stream = self._client.chat.completions.create(**params)
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = delta.content or ""
            tc_chunks = []
            for tc in delta.tool_calls or []:
                tc_chunks.append(
                    tool_call_chunk(
                        name=(tc.function.name if tc.function else None),
                        args=(tc.function.arguments if tc.function else None),
                        id=tc.id,
                        index=tc.index,
                    )
                )
            if not text and not tc_chunks:
                continue
            chunk = AIMessageChunk(content=text, tool_call_chunks=tc_chunks)
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager and text:
                run_manager.on_llm_new_token(text, chunk=cg_chunk)
            yield cg_chunk
