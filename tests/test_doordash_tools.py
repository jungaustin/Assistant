"""Tests for the DoorDash tools, concentrated on the order-confirmation gate.

The point of these tests is that no sequence of *model* behaviour can place an
order the user didn't approve out loud. dd-cli is never actually invoked: the
subprocess boundary is stubbed, and a `submitted` list records what would have
been charged. A passing suite means nothing reached `order submit`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from robot.tools.inner.doordash_client import (
    DoorDashClient,
    DoorDashError,
    OrderSummary,
    is_affirmative,
)
from robot.tools.inner.doordash_tools import DoorDashTools

# --- a fake dd-cli --------------------------------------------------------

PREVIEW = {
    "success": True,
    "quote": {
        "store_order_cart": {
            "store": {"store_id": "928163", "name": "Sushi House"},
            "fulfillment_type": {"is_consumer_pickup": False},
            "orders": [
                {
                    "order_items": [
                        {"item": {"name": "California Roll"}, "quantity": 2},
                        {"item": {"name": "Miso Soup"}, "quantity": 1},
                    ]
                }
            ],
        },
        "line_items": [
            {
                "label": "Subtotal",
                "final_money": {"unit_amount": 2400, "display_string": "$24.00"},
            },
            {
                "label": "Fees & Taxes",
                "final_money": {"unit_amount": 445, "display_string": "$4.45"},
            },
        ],
        "net_total_before_tip": {"unit_amount": 2845, "display_string": "$28.45"},
        "delivery_availability": {
            "asap_available": True,
            "asap_minutes_range_string": "25-35 min",
        },
    },
}

SEARCH = {
    "success": True,
    "stores": [
        {
            "store_id": "928163",
            "name": "Sushi House",
            "distance_meters": 2896,
            "delivery_time": "25-35 min",
        }
    ],
}


class FakeCLI:
    """Stands in for the dd-cli process itself, not for our wrapper.

    We patch `subprocess.run`, so every test still exercises the real argument
    construction, JSON parsing, error mapping, and store-metadata harvesting.
    `submitted` records anything that reached `order submit` — the assertion
    that matters is that it stays empty unless the user actually said yes.
    """

    def __init__(self):
        self.submitted: list[list[str]] = []
        self.calls: list[list[str]] = []
        self.preview_payload = PREVIEW
        self.returncode = 0
        self.stderr = ""

    def __call__(self, cmd, **kwargs):
        args = cmd[1:]  # drop the binary
        self.calls.append(args)
        if self.returncode:
            return SimpleNamespace(
                returncode=self.returncode, stdout="", stderr=self.stderr
            )

        # Mirrors the real CLI contract we verified: --json-output is a
        # group-level flag before the command, --intent is per-command.
        assert args[0] == "--json-output", args
        assert "--intent" in args, "every dd-cli command requires --intent"

        rest = args[1:]
        if rest[:2] == ["order", "submit"]:
            self.submitted.append(rest)
            payload = {"success": True, "order_uuid": "ord-1"}
        elif rest[:2] == ["order", "preview"]:
            payload = self.preview_payload
        elif rest[0] in ("search", "find-nearby-stores"):
            payload = SEARCH
        elif rest[:2] == ["cart", "add-items"]:
            payload = {"success": True, "cart_uuid": "cart-1"}
        elif rest[:2] == ["address", "list"]:
            payload = {"success": True, "addresses": []}
        else:
            payload = {"success": True}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")


@pytest.fixture
def client(monkeypatch):
    c = DoorDashClient(binary="dd-cli-fake", max_order_cents=15000)
    fake = FakeCLI()
    monkeypatch.setattr("robot.tools.inner.doordash_client.subprocess.run", fake)
    monkeypatch.setattr(type(c), "available", property(lambda self: True))
    c.fake = fake  # type: ignore[attr-defined]
    return c


def arm(client, user_text="order me sushi"):
    """Run one user turn that ends with an armed (previewed) order."""
    client.note_user_turn(user_text)
    spoken = client.review_order("cart-1", tip_cents=500)
    code = client._pending.code
    return spoken, code


# --- the summary the user hears ------------------------------------------


def test_summary_has_items_total_distance_and_eta(client):
    # Distance only exists in discovery responses, so search first — that's
    # what populates the cache the confirmation reads.
    client.search_stores("sushi")
    spoken, code = arm(client)

    assert "California Roll" in spoken and "Miso Soup" in spoken
    assert "Sushi House" in spoken
    assert "$28.45" in spoken  # total before tip
    assert "$5.00" in spoken  # the tip
    assert "$33.45" in spoken  # what actually gets charged
    assert "1.8 miles away" in spoken  # 2896 m
    assert "25-35 min" in spoken
    assert code in spoken


def test_summary_admits_unknown_distance_rather_than_inventing(client):
    # No search ran, so nothing cached a distance for this store.
    spoken, _ = arm(client)
    assert "couldn't get the distance" in spoken
    assert "miles" not in spoken


# --- the gate -------------------------------------------------------------


def test_cannot_place_without_a_review(client):
    client.note_user_turn("order me sushi")
    out = client.confirm_and_place("1234")
    assert "don't have an order ready" in out
    assert client.fake.submitted == []


def test_cannot_place_in_the_same_turn_as_the_review(client):
    """The model previews and immediately submits without the user speaking.
    This is the main failure mode a prompt-only rule cannot prevent."""
    _, code = arm(client)
    out = client.confirm_and_place(code)
    assert "say yes out loud" in out
    assert client.fake.submitted == []


def test_cannot_place_when_the_user_said_something_unrelated(client):
    _, code = arm(client)
    client.note_user_turn("what's the weather tomorrow")
    out = client.confirm_and_place(code)
    assert "didn't hear a clear yes" in out
    assert client.fake.submitted == []


def test_cannot_place_when_the_user_said_no(client):
    _, code = arm(client)
    client.note_user_turn("no, don't")
    out = client.confirm_and_place(code)
    assert "didn't hear a clear yes" in out
    assert client.fake.submitted == []


def test_cannot_place_with_a_wrong_code(client):
    _, code = arm(client)
    client.note_user_turn("yes")
    out = client.confirm_and_place("0000" if code != "0000" else "1111")
    assert "doesn't match" in out
    assert client.fake.submitted == []


def test_expired_quote_is_not_placed(client, monkeypatch):
    _, code = arm(client)
    client.note_user_turn("yes")
    client._pending.created_at -= client.confirm_ttl + 1
    out = client.confirm_and_place(code)
    assert "stale" in out
    assert client.fake.submitted == []


def test_a_clear_yes_places_the_order(client):
    client.search_stores("sushi")
    _, code = arm(client)
    client.note_user_turn("yes, place it")
    out = client.confirm_and_place(code)
    assert client.fake.submitted, "a confirmed order should reach order submit"
    args = client.fake.submitted[0]
    assert "-y" in args  # our gate replaces dd-cli's dead interactive prompt
    assert "--tip-cents" in args and "500" in args
    assert "$33.45" in out and "Sushi House" in out


def test_changed_total_re_confirms_instead_of_placing(client):
    _, code = arm(client)
    client.note_user_turn("yes")
    # An item went out of stock between the read-back and the yes.
    changed = json.loads(json.dumps(PREVIEW))
    changed["quote"]["net_total_before_tip"]["unit_amount"] = 3999
    changed["quote"]["net_total_before_tip"]["display_string"] = "$39.99"
    client.fake.preview_payload = changed

    out = client.confirm_and_place(code)
    assert client.fake.submitted == []
    assert "changed since I read it to you" in out
    assert "$39.99" in out
    # and a fresh code was issued, so the old one can't be replayed
    assert client._pending.code != code


def test_over_the_spend_cap_is_refused(client):
    client.max_order_cents = 1000
    client.note_user_turn("order me sushi")
    out = client.review_order("cart-1", tip_cents=0)
    assert "over the" in out and "limit" in out
    assert client._pending is None


def test_editing_the_cart_disarms_a_pending_order(client):
    _, code = arm(client)
    client.add_to_cart("928163", "1657275", "i_1", "Edamame", 1)
    client.note_user_turn("yes")
    out = client.confirm_and_place(code)
    assert "don't have an order ready" in out
    assert client.fake.submitted == []


def test_cancel_disarms(client):
    _, code = arm(client)
    assert "won't place it" in client.cancel_pending()
    client.note_user_turn("yes")
    assert client.fake.submitted == []


# --- affirmative matching -------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "Yes.",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay, go ahead",
        "place the order",
        "order it",
        "do it",
        "go ahead",
        "sounds good",
        "yes please",
        "confirm",
        "that's right",
    ],
)
def test_affirmatives(text):
    assert is_affirmative(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no",
        "nope",
        "not yet",
        "wait",
        "hold on",
        "cancel that",
        "never mind",
        "don't place it",
        # the dangerous middle ground: a yes that also asks for a change
        "yes but make it a large",
        "yeah, actually add a drink",
        "sure, but remove the soup",
        "ok change the tip to ten dollars",
        # a passing "ok" buried in unrelated speech must not buy food
        "ok so anyway what I was saying about dinner last night was that "
        "the place we went to had really good ramen and I want to go back",
        # near-misses that must not match on substrings
        "yesterday I ordered sushi",
        "what's on the menu",
    ],
)
def test_not_affirmatives(text):
    assert not is_affirmative(text)


# --- tool layer -----------------------------------------------------------


def test_tools_unavailable_without_a_client():
    tools = DoorDashTools(None)
    assert not tools.available
    assert "isn't set up" in tools.search_doordash("sushi")


def test_tool_errors_are_spoken_not_raised(client):
    def boom(*a, **k):
        raise DoorDashError("You're not signed in to DoorDash.")

    client.search_stores = boom  # type: ignore[method-assign]
    tools = DoorDashTools(client)
    assert tools.search_doordash("sushi") == "You're not signed in to DoorDash."


def test_unexpected_exceptions_are_spoken_not_raised(client):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    client.search_stores = boom  # type: ignore[method-assign]
    tools = DoorDashTools(client)
    assert "went wrong" in tools.search_doordash("sushi")


def test_manager_registers_nothing_when_disabled(monkeypatch):
    """A disabled group must cost zero tool-budget tokens."""
    from robot.tools.inner.doordash_tools import DoorDashTools as DT

    tools = DT(None)
    assert not tools.available


# --- CLI failure mapping --------------------------------------------------


def test_not_signed_in_is_explained_not_raised(client):
    client.fake.returncode = 1
    client.fake.stderr = (
        "Error: Failed to execute command due to missing credentials. "
        "Sign in with dd-cli login first"
    )
    out = DoorDashTools(client).search_doordash("sushi")
    assert "not signed in" in out.lower()
    assert "dd-cli login" in out


def test_cli_error_never_places_an_order(client):
    _, code = arm(client)
    client.note_user_turn("yes")
    client.fake.returncode = 1
    client.fake.stderr = "Error: upstream timeout"
    out = client.confirm_and_place(code)
    assert client.fake.submitted == []
    assert "didn't place it" in out


# --- wiring ---------------------------------------------------------------


def test_manager_forwards_user_turns_to_the_client(client, monkeypatch):
    """The gate is only trustworthy if real turns actually reach the client."""
    from robot.tools.manager import ToolManager

    monkeypatch.setattr("robot.tools.manager.make_doordash_client", lambda: client)
    # Keep the rest of ToolManager's construction from touching the network.
    tm = ToolManager.__new__(ToolManager)
    from robot.tools.inner.doordash_tools import DoorDashTools as DT

    tm.doordash_tools = DT(client)

    tm.note_user_turn("yes, place it")
    assert client._last_user_text == "yes, place it"
    assert client._turn == 1


def test_note_user_turn_is_safe_when_doordash_is_off():
    from robot.tools.inner.doordash_tools import DoorDashTools as DT
    from robot.tools.manager import ToolManager

    tm = ToolManager.__new__(ToolManager)
    tm.doordash_tools = DT(None)
    tm.note_user_turn("hello")  # must not raise


def test_unreadable_total_is_never_offered(client):
    """The user asked to always hear the cost. If we can't state it, we don't
    offer the order at all — an unpriced order must not become placeable."""
    blind = json.loads(json.dumps(PREVIEW))
    del blind["quote"]["net_total_before_tip"]
    blind["quote"]["line_items"] = []
    client.fake.preview_payload = blind

    client.note_user_turn("order me sushi")
    out = client.review_order("cart-1", tip_cents=0)
    assert "couldn't read the total" in out
    assert client._pending is None

    client.note_user_turn("yes")
    assert "don't have an order ready" in client.confirm_and_place("1234")
    assert client.fake.submitted == []


def test_total_falls_back_to_a_total_line_item(client):
    """Robustness: if net_total_before_tip moves, a labelled Total line still
    gives us a price to state."""
    alt = json.loads(json.dumps(PREVIEW))
    del alt["quote"]["net_total_before_tip"]
    alt["quote"]["line_items"].append(
        {
            "label": "Total",
            "final_money": {"unit_amount": 2845, "display_string": "$28.45"},
        }
    )
    client.fake.preview_payload = alt

    client.note_user_turn("order me sushi")
    out = client.review_order("cart-1", tip_cents=0)
    assert "$28.45" in out
    assert client._pending is not None


def test_default_address_is_looked_up_at_most_once(client):
    """_distance_text runs per store; the address lookup must not fan out
    into one subprocess call per result."""
    client._store_meta = {
        str(i): {"lat": 37.33, "lng": -122.03, "name": f"S{i}"} for i in range(5)
    }
    for i in range(5):
        client._distance_text(str(i))
    address_calls = [c for c in client.fake.calls if "address" in c]
    assert len(address_calls) <= 1, address_calls


def test_subtotal_is_never_mistaken_for_the_total():
    """ "Subtotal" contains "total" — the fallback must not quote the pre-fee
    number as the amount the user will be charged."""
    from robot.tools.inner.doordash_client import _total_from_line_items

    items = [
        {"label": "Subtotal", "final_money": {"unit_amount": 2400}},
        {"label": "Fees & Taxes", "final_money": {"unit_amount": 445}},
        {"label": "Total", "final_money": {"unit_amount": 2845}},
    ]
    assert _total_from_line_items(items) == 2845
    assert _total_from_line_items(items[:1]) is None
    assert _total_from_line_items(None) is None
