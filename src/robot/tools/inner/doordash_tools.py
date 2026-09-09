"""DoorDash tools — search, cart building, and the two-step order gate.

The ordering flow the model is expected to follow:

    search_doordash        find the store           -> store_id
    doordash_menu          find the item            -> menu_id + item_id
    add_to_doordash_cart   build the cart           -> cart_uuid
    review_doordash_order  read the order back      -> spoken summary + code
    (the user says yes)
    place_doordash_order   charge and place it

`place_doordash_order` is not a formality the model can skip or fake — the
guards live in DoorDashClient.confirm_and_place, which reads the user's real
transcript rather than anything the model passes. See doordash_client.py.

Failure contract matches clip_tools: every method returns a SPOKEN string.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, StructuredTool

from robot.tools.inner.doordash_client import DoorDashClient, DoorDashError

logger = logging.getLogger(__name__)


class DoorDashTools:
    """`client` is injected by ToolManager. None (or a missing dd-cli binary)
    means the whole group is unavailable and ToolManager registers none of
    these tools, so a disabled feature costs zero tool-budget tokens."""

    def __init__(self, client: DoorDashClient | None = None):
        self.client = client

    @property
    def available(self) -> bool:
        return self.client is not None and self.client.available

    def _guard(self, method: str, *args, **kwargs) -> str:
        """Call `client.<method>(...)`, turning every failure into speech.

        Takes the method NAME rather than a bound method so the client is
        never dereferenced when the feature is off — `self.client.foo` would
        raise on None before the guard could return its message.
        """
        if self.client is None:
            return "DoorDash isn't set up on this machine."
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except DoorDashError as exc:
            return exc.spoken
        except Exception:
            logger.exception("doordash tool failed unexpectedly")
            return "Something went wrong talking to DoorDash."

    # --- discovery --------------------------------------------------------

    def search_doordash(
        self, query: str, category: str = "restaurant", limit: int = 5
    ) -> str:
        return self._guard("search_stores", query, category, limit)

    def doordash_menu(self, store_id: str, query: str = "") -> str:
        return self._guard("browse_menu", store_id, query)

    # --- cart -------------------------------------------------------------

    def add_to_doordash_cart(
        self,
        store_id: str,
        menu_id: str,
        item_id: str,
        item_name: str,
        quantity: int = 1,
        cart_uuid: str = "",
    ) -> str:
        return self._guard(
            "add_to_cart",
            store_id,
            menu_id,
            item_id,
            item_name,
            quantity,
            cart_uuid,
        )

    def show_doordash_cart(self, cart_uuid: str = "") -> str:
        return self._guard("show_cart", cart_uuid)

    def remove_from_doordash_cart(self, cart_uuid: str, cart_item_id: str) -> str:
        return self._guard("remove_from_cart", cart_uuid, cart_item_id)

    def doordash_order_history(self, max_orders: int = 5) -> str:
        return self._guard("order_history", max_orders)

    # --- the gate ---------------------------------------------------------

    def review_doordash_order(
        self,
        cart_uuid: str,
        tip_cents: int = 0,
        fulfillment: str = "",
        priority: bool = False,
        scheduled_time: str = "",
    ) -> str:
        return self._guard(
            "review_order",
            cart_uuid,
            tip_cents,
            fulfillment,
            priority,
            scheduled_time,
        )

    def place_doordash_order(self, confirmation_code: str) -> str:
        return self._guard("confirm_and_place", confirmation_code)

    def cancel_doordash_order(self) -> str:
        return self._guard("cancel_pending")

    # --- tool factories ---------------------------------------------------

    def create_search_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.search_doordash,
            name="search_doordash",
            description=(
                "Search DoorDash for places to order from. Use for 'what's "
                "nearby', 'find me sushi', 'is there a Thai place', 'where "
                "can I get groceries'. `query` is free text like 'ramen' or "
                "'burgers'. `category` is 'restaurant' (default) for food "
                "places; use 'grocery', 'convenience', 'alcohol', 'pets', or "
                "'retail' for non-restaurant stores — for those the query is "
                "ignored and you get the nearest stores of that type, then "
                "call doordash_menu with a query to search inside one. "
                "Returns store names with their store_id, distance, and "
                "delivery estimate. Use the store_id for everything after."
            ),
        )

    def create_menu_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.doordash_menu,
            name="doordash_menu",
            description=(
                "Look up what a store sells. Pass `store_id` from "
                "search_doordash. Leave `query` empty for a restaurant's menu; "
                "pass a `query` like 'oat milk' to search inside a grocery or "
                "retail store (their catalogs are too big to list). Returns a "
                "menu_id and items with their item_id and price — you need "
                "BOTH the menu_id and the item_id to add anything to a cart. "
                "Use this to answer 'what do they have' and to make "
                "recommendations."
            ),
        )

    def create_add_to_cart_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.add_to_doordash_cart,
            name="add_to_doordash_cart",
            description=(
                "Add one item to a DoorDash cart. Requires `store_id`, "
                "`menu_id`, and `item_id` — get all three from doordash_menu "
                "first; never guess them. `item_name` is the readable name, "
                "`quantity` defaults to 1. Pass `cart_uuid` to add to an "
                "existing cart; leave it empty to start one (items append to "
                "any open cart at that store). Returns the cart_uuid — keep "
                "it. This does NOT order anything: adding to the cart is free "
                "and reversible. Call it once per distinct item."
            ),
        )

    def create_show_cart_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.show_doordash_cart,
            name="show_doordash_cart",
            description=(
                "Show what's in a DoorDash cart. With `cart_uuid`, lists that "
                "cart's items. Without one, lists all open carts and their "
                "cart_uuids — use that when the user refers to a cart you "
                "don't have the id for."
            ),
        )

    def create_remove_from_cart_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.remove_from_doordash_cart,
            name="remove_from_doordash_cart",
            description=(
                "Remove one line from a DoorDash cart. `cart_item_id` is the "
                "cart-line id from show_doordash_cart, NOT the menu item_id — "
                "call show_doordash_cart first to get it."
            ),
        )

    def create_order_history_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.doordash_order_history,
            name="doordash_order_history",
            description=(
                "List the user's recent DoorDash orders with store names and "
                "order_uuids. Use for 'what did I order last time', 'my "
                "usual', or to find a store they've used before."
            ),
        )

    def create_review_order_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.review_doordash_order,
            name="review_doordash_order",
            description=(
                "Price a cart and read the order back to the user for "
                "approval. ALWAYS call this before placing any order — it "
                "charges nothing. Returns a spoken summary with the items, "
                "the total, how far away the store is, how long it'll take, "
                "and a confirmation code. Say that summary to the user "
                "essentially as written, then STOP and wait for their answer "
                "— do not call place_doordash_order in the same turn. "
                "`tip_cents` is in CENTS (500 = $5.00); ask the user or pass "
                "0. `fulfillment` is 'delivery' or 'pickup' (empty keeps the "
                "cart's current mode)."
            ),
        )

    def create_place_order_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.place_doordash_order,
            name="place_doordash_order",
            description=(
                "Actually place the order and charge the user's card. Call "
                "this ONLY after review_doordash_order has read the order "
                "back AND the user has said yes in a later turn. Pass the "
                "`confirmation_code` from that review. If the user asked for "
                "any change, or said anything other than a clear yes, do NOT "
                "call this — review the order again instead. It will refuse "
                "and charge nothing if the user hasn't actually confirmed, so "
                "never call it hoping it goes through."
            ),
        )

    def create_cancel_order_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.cancel_doordash_order,
            name="cancel_doordash_order",
            description=(
                "Drop a pending order that was read back but not yet placed, "
                "when the user says no, cancel, never mind, or wants changes. "
                "Nothing is charged. This does not cancel an order that was "
                "already placed — for that, tell the user to use the "
                "DoorDash app."
            ),
        )
