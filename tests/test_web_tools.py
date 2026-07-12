"""Tests for the Tavily web search tool (WebTools).

Covers:
  - missing API key returns a clear 'not configured' message (no network)
  - client is lazy: not constructed until the first real search
  - answer-synthesis path returns the answer string
  - empty answer falls back to top result snippets
  - no answer and no results returns a 'No results' message
  - Tavily client errors surface as a friendly failure string
  - the StructuredTool is named web_search and calls through

All network access is mocked — these tests never hit Tavily.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from robot.tools.inner.web_tools import WebTools


def _tools_with_client(response: dict) -> WebTools:
    wt = WebTools(api_key="test-key")
    client = MagicMock()
    client.search.return_value = response
    wt._client = client
    return wt


def test_missing_api_key_returns_not_configured():
    wt = WebTools(api_key="")
    result = wt.web_search("anything")
    assert "TAVILY_API_KEY" in result
    # No client should ever have been built.
    assert wt._client is None


def test_client_is_lazy():
    wt = WebTools(api_key="test-key")
    assert wt._client is None


def test_answer_synthesis_path():
    wt = _tools_with_client({"answer": "The Warriors won 112-104.", "results": []})
    assert wt.web_search("warriors score") == "The Warriors won 112-104."


def test_empty_answer_falls_back_to_snippets():
    wt = _tools_with_client(
        {
            "answer": "",
            "results": [
                {"title": "ESPN", "content": "Warriors beat Lakers 112-104."},
                {"title": "NBA.com", "content": "Final: GSW 112, LAL 104."},
            ],
        }
    )
    result = wt.web_search("warriors score")
    assert "ESPN" in result
    assert "112-104" in result


def test_no_answer_no_results():
    wt = _tools_with_client({"answer": None, "results": []})
    result = wt.web_search("xyzzy")
    assert "No results" in result
    assert "xyzzy" in result


def test_client_error_returns_friendly_message():
    wt = WebTools(api_key="test-key")
    client = MagicMock()
    client.search.side_effect = RuntimeError("connection refused")
    wt._client = client
    result = wt.web_search("anything")
    assert result.startswith("Web search failed:")
    assert "connection refused" in result


def test_search_passes_voice_first_params():
    wt = _tools_with_client({"answer": "ok", "results": []})
    wt.web_search("query")
    kwargs = wt._client.search.call_args.kwargs
    assert kwargs["include_answer"] is True
    assert kwargs["timeout"] == 10


def test_structured_tool_name_and_callthrough():
    wt = _tools_with_client({"answer": "42", "results": []})
    tool = wt.create_web_search_tool()
    assert tool.name == "web_search"
    assert tool.func("meaning of life") == "42"
