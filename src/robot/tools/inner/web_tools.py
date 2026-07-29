"""Web search tool backed by Tavily.

Voice-first design: we use Tavily's answer-synthesis mode (include_answer=True)
so the tool returns one short paragraph the agent can read aloud, not a list of
links. Raw search results are useless over TTS.

The Tavily client is lazy: constructed on first tool call, not at import or
ToolManager construction. So the robot boots fine without TAVILY_API_KEY set —
the tool just returns a clear "not configured" message if called.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from robot.config import TAVILY_API_KEY

# Hard cap on the search round-trip. A desk robot mid-conversation can't sit
# on a 60s default timeout — better to fail fast and say so.
_SEARCH_TIMEOUT_SECONDS = 10


class WebTools:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key if api_key is not None else TAVILY_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def web_search(self, query: str) -> str:
        if not self._api_key:
            return (
                "Web search isn't configured — TAVILY_API_KEY is missing "
                "from .env."
            )
        try:
            response = self._get_client().search(
                query=query,
                search_depth="basic",
                include_answer=True,
                max_results=5,
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )
        except Exception as e:
            return f"Web search failed: {e}"

        answer = (response.get("answer") or "").strip()
        if answer:
            return answer

        # Answer synthesis occasionally comes back empty; fall back to the
        # top result snippets so the agent still has something to work with.
        results = response.get("results") or []
        if not results:
            return f"No results found for '{query}'."
        snippets = [
            f"{r.get('title', '')}: {r.get('content', '')}".strip(": ")
            for r in results[:3]
        ]
        return "\n".join(s for s in snippets if s)

    def lookup_food_calories(self, item: str) -> str:
        """Search the web for one food's calorie count.

        A thin template over web_search rather than a separate integration:
        the generic tool's description tells the model to answer
        conversationally, which is wrong here — we want a number. Phrasing
        the query with "calories per serving" is what reliably gets Tavily's
        answer synthesis to lead with one.
        """
        item = item.strip()
        if not item:
            return "No food given to look up."
        answer = self.web_search(f"{item} calories per serving")
        if answer.startswith(("Web search isn't configured", "Web search failed", "No results")):
            return answer
        return f"Web result for '{item}': {answer}"

    def create_lookup_food_calories_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.lookup_food_calories,
            name="lookup_food_calories",
            description=(
                "Search the web for how many calories a food has. Use for "
                "restaurant, chain, and packaged items the user hasn't logged "
                "before — 'Panda Express orange chicken', 'Chipotle burrito "
                "bowl', a named snack brand.\n\n"
                "Call lookup_food FIRST. Only come here when the user has no "
                "usable past log for this item. Skip this entirely for plain "
                "home foods you can estimate well ('two eggs', 'a bowl of "
                "rice') — a search round-trip is slow and adds nothing.\n\n"
                "Arguments:\n"
                "  item (str): the food, as specific as possible. Include the "
                "restaurant or brand — 'Panda Express chow mein' beats 'chow "
                "mein'. One item per call.\n\n"
                "Returns a short paragraph that should contain a calorie "
                "number. Pull the number out for the portion the user actually "
                "ate and use it; if the result is vague or missing a number, "
                "estimate instead and say that you did."
            ),
        )

    def create_web_search_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.web_search,
            name="web_search",
            description=(
                "Search the web for current information. Use for anything "
                "you don't know or that changes over time: news, sports "
                "scores, prices, release dates, 'is X open right now', "
                "facts you're unsure about.\n\n"
                "  query (str): a plain-language search query. Phrase it "
                "like a question or topic, e.g. 'Warriors game score "
                "tonight' or 'when does the new Zelda come out'.\n\n"
                "Returns a short synthesized answer (one paragraph). "
                "Summarize it conversationally — don't read it verbatim if "
                "it's long, and never read URLs out loud. If the search "
                "fails or isn't configured, tell the user plainly."
            ),
        )
