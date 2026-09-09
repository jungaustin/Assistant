"""DoorDash ordering via the `dd-cli` beta binary.

Wraps `dd-cli` (github.com/doordash-oss/doordash-cli, v0.2.x) as a subprocess
and adds the thing a voice robot needs and the CLI can't provide: a
**confirmation gate that cannot be talked around**.

Why a gate at all. `dd-cli order submit` charges the default payment method
immediately. Its own safety net is an interactive y/n prompt, which is useless
here — the subprocess has no TTY, so the prompt would hang the robot rather
than protect it. We therefore always pass `-y` and enforce confirmation
ourselves, in Python, where the model cannot route around it.

The gate (`confirm_and_place`) requires ALL of:
  1. an armed preview for this cart (`review_order` ran),
  2. the confirmation code from that preview, echoed back by the model,
  3. a NEW user turn since the preview — the model cannot preview and submit
     inside one turn, so a human always speaks in between,
  4. that turn's *actual* transcript being affirmative. The text is recorded
     by the Agent (see ToolManager.note_user_turn), NOT passed by the model,
     so the model cannot fabricate a "yes",
  5. the re-priced quote still matching the fingerprint the user approved —
     if an item went out of stock or the total moved, we re-confirm,
  6. the total sitting under DOORDASH_MAX_ORDER_CENTS.

Failure contract matches the rest of the tool layer (see clip_tools): every
public method returns a SPOKEN string and never raises into the agent loop.

Privacy note: `dd-cli` requires an `--intent` string on every command and
states that DoorDash may review it for research and product improvement. We
send a fixed, non-personal string by default; set DOORDASH_SHARE_INTENT=true
to include the user's own words instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Sent as --intent (required by every dd-cli command) when the user hasn't
# opted into sharing their own phrasing. Deliberately generic: it satisfies
# the flag without shipping the user's speech to a third party.
GENERIC_INTENT = (
    "Summary: Help the user order food or household items for themselves.\n"
    'user prompt/purpose: "(not shared)"'
)

METERS_PER_MILE = 1609.344

# Distinguishes 'not looked up yet' from 'looked up, no coords available'.
_UNSET = object()


class DoorDashError(Exception):
    """Carries the sentence the robot should say. Mirrors ClipError."""

    def __init__(self, spoken: str):
        super().__init__(spoken)
        self.spoken = spoken


# --- confirmation-phrase matching -----------------------------------------
# The user's real transcript is matched here. Negatives are checked FIRST and
# veto outright: "yes but make it a large" contains "yes" and must NOT place
# the order the user just asked to change.

_AFFIRMATIVE = re.compile(
    r"\b("
    r"yes|yeah|yep|yup|yuh|sure|ok|okay|confirm(ed|s)?|correct|right"
    r"|do it|go ahead|go for it|place (the |it|that )?order|place it|order it"
    r"|send it|buy it|get it|submit it|let'?s do it|sounds good|please do"
    r"|absolutely|definitely|affirmative|that'?s right|perfect|great"
    r")\b",
    re.IGNORECASE,
)

_NEGATIVE = re.compile(
    r"\b("
    r"no|nope|nah|don'?t|do not|cancel|stop|wait|hold on|hold off|not yet"
    r"|never ?mind|forget it|change|instead|actually|different|remove|delete"
    r"|add|swap|replace|scratch that|undo|abort|no thanks|maybe later"
    # hedges: "yes but make it a large" is a request to change the order,
    # not approval of the one that was read back.
    r"|but|except|instead of|also|rather|prefer|make it|can you|could you"
    r"|bigger|smaller|larger|extra|without|hold the"
    r")\b",
    re.IGNORECASE,
)

# An explicit command confirms at any length; a bare "ok" only counts in a
# short utterance, so "ok so what I was saying about dinner..." can't buy food.
_EXPLICIT = re.compile(
    r"\b(place (the |it|that )?order|place it|order it|submit it|send it|buy it|do it|go ahead)\b",
    re.IGNORECASE,
)
_MAX_CASUAL_WORDS = 12


def is_affirmative(text: str) -> bool:
    """True when `text` is a clean yes with no hedge, negation, or edit."""
    if not text or not text.strip():
        return False
    t = text.strip()
    if _NEGATIVE.search(t):
        return False
    if _EXPLICIT.search(t):
        return True
    if not _AFFIRMATIVE.search(t):
        return False
    return len(t.split()) <= _MAX_CASUAL_WORDS


# --- JSON spelunking -------------------------------------------------------
# dd-cli's --json-output shape is stable but deep, and we can only verify the
# paths the bundled formatters use. Everything below tries the known path
# first and falls back to a recursive search, so a field moving one level
# doesn't silently drop the total or the ETA out of a confirmation.


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _find_key(node: Any, key: str) -> Any:
    """First value for `key` anywhere in the tree (breadth-ish, shallow wins)."""
    queue = [node]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            if key in cur and cur[key] is not None:
                return cur[key]
            queue.extend(cur.values())
        elif isinstance(cur, list):
            queue.extend(cur)
    return None


def _money(node: Any) -> tuple[Optional[int], Optional[str]]:
    """(cents, '$12.34') from a dd-cli monetary object."""
    if not isinstance(node, dict):
        return None, None
    cents = node.get("unit_amount")
    disp = node.get("display_string")
    return (
        cents if isinstance(cents, int) else None,
        disp if isinstance(disp, str) and disp.strip() else None,
    )


def _dollars(cents: Optional[int]) -> str:
    if cents is None:
        return "an unknown amount"
    return f"${cents / 100:,.2f}"


def _pluralize(name: str, qty: int) -> str:
    """'2 California Rolls', not '2 California Roll'. Conservative: names that
    already look plural (or sibilant) are left alone rather than risking
    'French Friess'."""
    if not qty or qty == 1:
        return name
    if name.lower().endswith(("s", "x", "ch", "sh", "z")):
        return f"{qty} {name}"
    return f"{qty} {name}s"


def _total_from_line_items(line_items: Any) -> Optional[int]:
    """Grand total from a quote's line items, when the usual field is absent.

    Careful with the substring: "Subtotal" contains "total" and is emphatically
    NOT the amount the user gets charged. Exact "total" wins; otherwise take
    the largest total-ish line, since fees and tax only ever add.
    """
    if not isinstance(line_items, list):
        return None
    candidates: list[int] = []
    for li in line_items:
        if not isinstance(li, dict):
            continue
        label = str(li.get("label") or "").strip().lower()
        cents, _ = _money(li.get("final_money"))
        if cents is None or "total" not in label:
            continue
        if label == "total":
            return cents
        if "subtotal" not in label:
            candidates.append(cents)
    return max(candidates) if candidates else None


def _haversine_miles(lat1, lng1, lat2, lng2) -> Optional[float]:
    try:
        p = math.pi / 180
        a = (
            0.5
            - math.cos((lat2 - lat1) * p) / 2
            + math.cos(lat1 * p)
            * math.cos(lat2 * p)
            * (1 - math.cos((lng2 - lng1) * p))
            / 2
        )
        return 7917.5 * math.asin(math.sqrt(a)) / 2
    except (TypeError, ValueError):
        return None


@dataclass
class OrderSummary:
    """What the user hears before anything is charged."""

    store_name: str
    items: list[tuple[str, int]]  # (name, quantity)
    subtotal_lines: list[tuple[str, str]]  # (label, display) from the quote
    total_before_tip_cents: Optional[int]
    tip_cents: int
    eta_text: Optional[str]
    distance_text: Optional[str]
    is_pickup: bool
    fingerprint: str

    @property
    def charge_cents(self) -> Optional[int]:
        if self.total_before_tip_cents is None:
            return None
        return self.total_before_tip_cents + self.tip_cents

    def spoken(self, code: str) -> str:
        """Voice-shaped confirmation. Deliberately NOT dd-cli's --beautify
        output: that's a terminal block with box-drawing rules and CLI hints,
        which is wrong to read aloud."""
        parts = [_pluralize(name, qty) for name, qty in self.items]
        if not parts:
            item_text = "the items in your cart"
        elif len(parts) == 1:
            item_text = parts[0]
        elif len(parts) == 2:
            item_text = f"{parts[0]} and {parts[1]}"
        else:
            item_text = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        lines = [f"Here's the order: {item_text} from {self.store_name}."]

        if self.tip_cents:
            lines.append(
                f"That's {_dollars(self.total_before_tip_cents)} "
                f"plus a {_dollars(self.tip_cents)} tip, "
                f"so {_dollars(self.charge_cents)} total."
            )
        else:
            lines.append(f"Total is {_dollars(self.total_before_tip_cents)}, no tip.")

        # Distance and time, as one natural sentence rather than a field dump.
        ready = "ready in" if self.is_pickup else "here in"
        if self.distance_text and self.eta_text:
            lines.append(
                f"It's {self.distance_text} and should be {ready} {self.eta_text}."
            )
        elif self.distance_text:
            lines.append(f"It's {self.distance_text}.")
        elif self.eta_text:
            lines.append(f"Should be {ready} {self.eta_text}.")
        if not self.distance_text:
            lines.append("I couldn't get the distance for this one.")
        if not self.eta_text:
            lines.append("I don't have a time estimate for it either.")
        if self.is_pickup:
            lines.append("This one's pickup, not delivery.")

        lines.append(f"Say yes to place it. Confirmation code {code}.")
        return " ".join(lines)


@dataclass
class PendingOrder:
    cart_uuid: str
    code: str
    fingerprint: str
    summary: OrderSummary
    submit_args: dict
    created_turn: int
    created_at: float = field(default_factory=time.time)


class DoorDashClient:
    """Thin, defensive wrapper around the `dd-cli` binary."""

    def __init__(
        self,
        binary: str = "dd-cli",
        *,
        timeout: float = 60.0,
        max_order_cents: int = 15000,
        confirm_ttl: float = 600.0,
        share_intent: bool = False,
    ):
        self.binary = binary
        self.timeout = timeout
        self.max_order_cents = max_order_cents
        self.confirm_ttl = confirm_ttl
        self.share_intent = share_intent

        self._pending: Optional[PendingOrder] = None
        # Harvested from every discovery response we see. `order preview` does
        # not carry distance, so this cache is how the confirmation gets it.
        self._store_meta: dict[str, dict] = {}
        self._home_coords: Any = _UNSET
        # Set by ToolManager.note_user_turn on each real user utterance.
        self._turn: int = 0
        self._last_user_text: str = ""

    # --- lifecycle --------------------------------------------------------

    @property
    def available(self) -> bool:
        """True only when the binary really exists — an enabled-but-missing
        dd-cli registers no tools rather than failing at the first order.
        `shutil.which` handles both a bare name on PATH and an absolute path.
        """
        return shutil.which(self.binary) is not None

    def note_user_turn(self, text: str) -> None:
        self._turn += 1
        self._last_user_text = text or ""

    # --- subprocess -------------------------------------------------------

    def _intent(self) -> str:
        if self.share_intent and self._last_user_text:
            safe = self._last_user_text.strip().replace('"', "'")[:200]
            return (
                "Summary: Help the user order food or household items for "
                f'themselves.\nuser prompt/purpose: "{safe}"'
            )
        return GENERIC_INTENT

    def _run(self, args: list[str], *, timeout: Optional[float] = None) -> Any:
        """Run `dd-cli --json-output <args> --intent <intent>` and parse JSON.

        `--json-output` is a group-level flag so it precedes the command;
        `--intent` is per-command (required=True on every leaf) so it trails.
        """
        cmd = [self.binary, "--json-output", *args, "--intent", self._intent()]
        logger.info("dd-cli run args=%s", " ".join(args))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                stdin=subprocess.DEVNULL,  # never let a prompt block the robot
            )
        except FileNotFoundError:
            raise DoorDashError(
                "The DoorDash command line tool isn't installed on this machine."
            )
        except subprocess.TimeoutExpired:
            raise DoorDashError("DoorDash took too long to answer. Try again.")

        blob = (proc.stdout or "") + ("\n" + proc.stderr if proc.returncode else "")
        if proc.returncode != 0:
            raise DoorDashError(self._spoken_error(blob))

        payload = self._parse_json(proc.stdout)
        if isinstance(payload, dict) and payload.get("success") is False:
            msg = payload.get("message") or "DoorDash couldn't complete that."
            raise DoorDashError(str(msg)[:300])
        self._harvest_stores(payload)
        return payload

    @staticmethod
    def _parse_json(text: str) -> Any:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Tolerate a banner or trailing note around the payload.
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise DoorDashError("DoorDash sent back something I couldn't read.")

    @staticmethod
    def _spoken_error(blob: str) -> str:
        low = (blob or "").lower()
        if "missing credentials" in low or "dd-cli login" in low:
            return (
                "You're not signed in to DoorDash. Run dd-cli login in a "
                "terminal, then ask me again."
            )
        if "expired" in low and "token" in low:
            return "Your DoorDash sign-in expired. Run dd-cli login again."
        if "no such option" in low or "unexpected extra" in low:
            return "I called DoorDash the wrong way. That's a bug on my side."
        for line in reversed([ln.strip() for ln in (blob or "").splitlines()]):
            if line.startswith("Error:"):
                return line[6:].strip()[:300] or "DoorDash returned an error."
        return "DoorDash returned an error."

    # --- store metadata / distance ---------------------------------------

    def _harvest_stores(self, payload: Any) -> None:
        """Cache anything store-shaped. Distance is only ever present in
        discovery responses, so we grab it whenever it floats past."""
        for node in _walk(payload):
            if not isinstance(node, dict):
                continue
            sid = node.get("store_id") or node.get("id")
            if sid is None or not isinstance(sid, (str, int)):
                continue
            if not any(
                k in node
                for k in (
                    "name",
                    "distance_meters",
                    "delivery_time",
                    "printable_address",
                )
            ):
                continue
            meta = self._store_meta.setdefault(str(sid), {})
            for key in (
                "name",
                "distance_meters",
                "distance",
                "delivery_time",
                "printable_address",
                "lat",
                "lng",
                "latitude",
                "longitude",
            ):
                if node.get(key) not in (None, ""):
                    meta[key] = node[key]

    def _default_address_coords(self) -> Optional[tuple[float, float]]:
        """Lat/lng of the default delivery address, fetched at most once.

        Memoized because _distance_text runs per store in a result list, and
        each miss would otherwise cost a whole subprocess round-trip.
        """
        if self._home_coords is not _UNSET:
            return self._home_coords  # type: ignore[return-value]
        self._home_coords = None
        try:
            payload = self._run(["address", "list"], timeout=30)
        except DoorDashError:
            return None
        addresses = _find_key(payload, "addresses") or []
        if not isinstance(addresses, list):
            return None
        chosen = next(
            (a for a in addresses if isinstance(a, dict) and a.get("is_default")), None
        )
        if chosen is None and addresses:
            chosen = addresses[0] if isinstance(addresses[0], dict) else None
        if not chosen:
            return None
        lat = chosen.get("lat", chosen.get("latitude"))
        lng = chosen.get("lng", chosen.get("longitude"))
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            self._home_coords = (float(lat), float(lng))
        return self._home_coords  # type: ignore[return-value]

    def _distance_text(self, store_id: Optional[str]) -> Optional[str]:
        """Miles to the store, or None. Never guesses — an absent distance is
        reported as absent rather than invented."""
        if not store_id:
            return None
        meta = self._store_meta.get(str(store_id), {})

        meters = meta.get("distance_meters")
        if isinstance(meters, (int, float)) and meters > 0:
            return self._format_miles(meters / METERS_PER_MILE)

        raw = meta.get("distance")
        if isinstance(raw, (int, float)) and raw > 0:
            # Discovery renders `distance` in metres alongside distance_meters.
            return self._format_miles(raw / METERS_PER_MILE)

        lat = meta.get("lat", meta.get("latitude"))
        lng = meta.get("lng", meta.get("longitude"))
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            home = self._default_address_coords()
            if home:
                miles = _haversine_miles(home[0], home[1], float(lat), float(lng))
                if miles is not None:
                    return self._format_miles(miles)
        return None

    @staticmethod
    def _format_miles(miles: float) -> str:
        if miles < 0.1:
            return "just around the corner"
        if miles < 10:
            return f"{miles:.1f} miles away"
        return f"{miles:.0f} miles away"

    # --- discovery --------------------------------------------------------

    def search_stores(
        self, query: str, category: str = "restaurant", limit: int = 5
    ) -> str:
        args: list[str]
        if category and category.lower() not in ("restaurant", "restaurants", ""):
            args = [
                "find-nearby-stores",
                "--vertical",
                category.lower(),
                "--max",
                str(max(1, min(int(limit or 5), 20))),
            ]
        else:
            args = [
                "search",
                "--query",
                query,
                "--limit",
                str(max(1, min(int(limit or 5), 20))),
            ]
        payload = self._run(args)
        stores = _find_key(payload, "stores") or []
        if not isinstance(stores, list) or not stores:
            return (
                f"I didn't find anything for {query}."
                if query
                else "I didn't find any stores nearby."
            )

        lines = []
        for s in stores[:limit]:
            if not isinstance(s, dict):
                continue
            name = s.get("name") or "Unknown store"
            sid = s.get("store_id") or s.get("id")
            bits = [f"{name} (store_id {sid})"]
            dist = self._distance_text(str(sid) if sid is not None else None)
            if dist:
                bits.append(dist)
            if s.get("delivery_time"):
                bits.append(f"about {s['delivery_time']}")
            if s.get("description"):
                bits.append(str(s["description"])[:80])
            lines.append(" — ".join(bits))
        return "\n".join(lines)

    def browse_menu(self, store_id: str, query: str = "") -> str:
        """Restaurant menu, or an in-store item search when `query` is given.

        Retail and grocery catalogs are too large to enumerate, so a query
        routes to find-items — that's dd-cli's own guidance.
        """
        if query:
            payload = self._run(
                ["find-items", "--store-id", str(store_id), "--query", query]
            )
        else:
            payload = self._run(["menu", "--store-id", str(store_id)])

        menu_id = _find_key(payload, "menu_id")
        lines = []
        if menu_id:
            lines.append(f"menu_id: {menu_id}")
        count = 0
        for node in _walk(payload):
            if count >= 25:
                break
            if not isinstance(node, dict):
                continue
            name = node.get("name") or node.get("item_name")
            iid = node.get("item_id") or node.get("id")
            if not name or iid is None:
                continue
            price = None
            for k in (
                "price",
                "display_price",
                "unit_price_monetary_fields",
                "price_monetary_fields",
            ):
                _, disp = _money(node.get(k))
                if disp:
                    price = disp
                    break
            # dd-cli prefixes restaurant menu item ids with `i_`; cart
            # add-items wants it stripped.
            iid = str(iid)
            lines.append(
                f"{name} — item_id {iid.removeprefix('i_')}"
                + (f" — {price}" if price else "")
            )
            count += 1
        if not lines or (len(lines) == 1 and menu_id):
            return "I couldn't find any items there."
        return "\n".join(lines)

    # --- cart -------------------------------------------------------------

    def add_to_cart(
        self,
        store_id: str,
        menu_id: str,
        item_id: str,
        item_name: str,
        quantity: int = 1,
        cart_uuid: str = "",
    ) -> str:
        items = [
            {
                "item_id": str(item_id).removeprefix("i_"),
                "item_name": item_name,
                "quantity": max(1, int(quantity or 1)),
            }
        ]
        args = [
            "cart",
            "add-items",
            "--store-id",
            str(store_id),
            "--menu-id",
            str(menu_id),
            "--items-json",
            json.dumps(items),
        ]
        if cart_uuid:
            args += ["--cart-uuid", cart_uuid]
        payload = self._run(args)
        new_uuid = _find_key(payload, "cart_uuid") or cart_uuid
        # Cart changed: any armed confirmation is now stale by definition.
        self._pending = None
        qty = items[0]["quantity"]
        return (
            f"Added {qty} {item_name} to the cart. cart_uuid: {new_uuid}"
            if new_uuid
            else f"Added {qty} {item_name} to the cart."
        )

    def show_cart(self, cart_uuid: str = "") -> str:
        if not cart_uuid:
            payload = self._run(["cart", "list"])
            carts = _find_key(payload, "carts") or []
            if not isinstance(carts, list) or not carts:
                return "You don't have any open carts."
            lines = []
            for c in carts:
                if not isinstance(c, dict):
                    continue
                name = _find_key(c, "store_name") or _find_key(c, "name") or "a store"
                _, total = _money(_find_key(c, "total") or {})
                lines.append(
                    f"{name} — cart_uuid {c.get('cart_uuid') or c.get('id')}"
                    + (f" — {total}" if total else "")
                )
            return "\n".join(lines) or "You don't have any open carts."

        payload = self._run(["cart", "show", "--cart-uuid", cart_uuid])
        items = self._extract_items(payload)
        if not items:
            return "That cart is empty."
        lines = [f"{qty} × {name}" for name, qty in items]
        _, total = _money(_find_key(payload, "total") or {})
        if total:
            lines.append(f"Total so far: {total}")
        return "\n".join(lines)

    def remove_from_cart(self, cart_uuid: str, cart_item_id: str) -> str:
        self._run(
            [
                "cart",
                "remove-item",
                "--cart-uuid",
                cart_uuid,
                "--cart-item-id",
                str(cart_item_id),
            ]
        )
        self._pending = None
        return "Removed."

    def order_history(self, max_orders: int = 5) -> str:
        payload = self._run(
            ["order", "history", "--max", str(max(1, min(int(max_orders or 5), 20)))]
        )
        orders = _find_key(payload, "orders") or []
        if not isinstance(orders, list) or not orders:
            return "I don't see any past orders."
        lines = []
        for o in orders[:max_orders]:
            if not isinstance(o, dict):
                continue
            name = _find_key(o, "store_name") or _find_key(o, "name") or "a store"
            when = o.get("created_at") or o.get("order_date") or ""
            lines.append(
                f"{name} — order_uuid {o.get('order_uuid')}"
                + (f" — {str(when)[:10]}" if when else "")
            )
        return "\n".join(lines)

    # --- the confirmation gate -------------------------------------------

    @staticmethod
    def _extract_items(payload: Any) -> list[tuple[str, int]]:
        items: list[tuple[str, int]] = []
        for node in _walk(payload):
            if not isinstance(node, dict):
                continue
            order_items = node.get("order_items") or node.get("items")
            if not isinstance(order_items, list):
                continue
            for oi in order_items:
                if not isinstance(oi, dict):
                    continue
                inner = oi.get("item") if isinstance(oi.get("item"), dict) else {}
                name = inner.get("name") or oi.get("name") or oi.get("item_name")
                qty = oi.get("quantity") or inner.get("quantity") or 1
                if name:
                    items.append((str(name), int(qty) if str(qty).isdigit() else 1))
            if items:
                break
        return items

    def _summarize(self, payload: Any, tip_cents: int) -> OrderSummary:
        quote = payload.get("quote") if isinstance(payload, dict) else None
        quote = quote if isinstance(quote, dict) else (payload or {})
        soc = quote.get("store_order_cart")
        soc = soc if isinstance(soc, dict) else {}
        store = soc.get("store") if isinstance(soc.get("store"), dict) else {}

        store_id = store.get("store_id") or _find_key(soc, "store_id")
        store_name = (
            store.get("name") or _find_key(soc, "store_name") or "the restaurant"
        )

        ft = soc.get("fulfillment_type")
        if isinstance(ft, dict):
            is_pickup = bool(ft.get("is_consumer_pickup"))
        else:
            is_pickup = "pickup" in str(ft or "").lower()

        items = self._extract_items(soc) or self._extract_items(quote)

        subtotal_lines: list[tuple[str, str]] = []
        for li in quote.get("line_items") or []:
            if not isinstance(li, dict):
                continue
            label = li.get("label")
            _, disp = _money(li.get("final_money"))
            if label and disp:
                subtotal_lines.append((str(label), disp))

        total_cents, _ = _money(quote.get("net_total_before_tip"))
        if total_cents is None:
            total_cents, _ = _money(_find_key(quote, "net_total_before_tip"))
        if total_cents is None:
            # Last resort: a line item that calls itself the total. Better than
            # arming an order whose price we can't state (review_order refuses
            # outright if this is still None).
            total_cents = _total_from_line_items(quote.get("line_items"))

        da = quote.get("delivery_availability")
        da = da if isinstance(da, dict) else {}
        eta = (
            da.get("asap_pickup_minutes_range_string")
            if is_pickup
            else da.get("asap_minutes_range_string")
        ) or _find_key(da, "eta_minutes_range")
        eta_text = str(eta) if eta else None

        # Distance matters either way — more so for pickup, since the user is
        # the one driving there.
        distance_text = self._distance_text(
            str(store_id) if store_id is not None else None
        )

        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "store": str(store_id),
                    "items": sorted((n, q) for n, q in items),
                    "total": total_cents,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]

        return OrderSummary(
            store_name=str(store_name),
            items=items,
            subtotal_lines=subtotal_lines,
            total_before_tip_cents=total_cents,
            tip_cents=tip_cents,
            eta_text=eta_text,
            distance_text=distance_text,
            is_pickup=is_pickup,
            fingerprint=fingerprint,
        )

    def _preview(
        self,
        cart_uuid: str,
        *,
        fulfillment: str = "",
        priority: bool = False,
        scheduled_time: str = "",
        no_apply_credits: bool = False,
    ) -> Any:
        args = ["order", "preview", "--cart-uuid", cart_uuid]
        if fulfillment:
            args += ["--fulfillment", fulfillment]
        if priority:
            args.append("--priority")
        if scheduled_time:
            args += ["--scheduled-time", scheduled_time]
        if no_apply_credits:
            args.append("--no-apply-credits")
        return self._run(args)

    def review_order(
        self,
        cart_uuid: str,
        tip_cents: int = 0,
        fulfillment: str = "",
        priority: bool = False,
        scheduled_time: str = "",
        no_apply_credits: bool = False,
    ) -> str:
        """Price the cart and ARM the confirmation. Charges nothing."""
        tip_cents = max(0, int(tip_cents or 0))
        payload = self._preview(
            cart_uuid,
            fulfillment=fulfillment,
            priority=priority,
            scheduled_time=scheduled_time,
            no_apply_credits=no_apply_credits,
        )
        summary = self._summarize(payload, tip_cents)

        if summary.total_before_tip_cents is None:
            # The user asked to always hear the cost before approving. If we
            # can't state it, we don't offer the order at all.
            self._pending = None
            return (
                "I couldn't read the total for that cart, so I won't offer to "
                "order it. Check it in the DoorDash app."
            )

        charge = summary.charge_cents
        if charge is not None and charge > self.max_order_cents:
            self._pending = None
            return (
                f"That order comes to {_dollars(charge)}, which is over the "
                f"{_dollars(self.max_order_cents)} limit I'm allowed to place. "
                "You'll have to finish this one yourself."
            )

        code = f"{secrets.randbelow(9000) + 1000}"
        self._pending = PendingOrder(
            cart_uuid=cart_uuid,
            code=code,
            fingerprint=summary.fingerprint,
            summary=summary,
            submit_args={
                "tip_cents": tip_cents,
                "fulfillment": fulfillment,
                "priority": priority,
                "scheduled_time": scheduled_time,
                "no_apply_credits": no_apply_credits,
            },
            created_turn=self._turn,
        )
        logger.info(
            "doordash order armed cart=%s code=%s total=%s turn=%s",
            cart_uuid,
            code,
            charge,
            self._turn,
        )
        return summary.spoken(code)

    def confirm_and_place(self, confirmation_code: str) -> str:
        """Place the armed order. Every guard in the module docstring runs
        here; any failure returns a spoken refusal and charges nothing."""
        pending = self._pending
        if pending is None:
            return (
                "I don't have an order ready to place. Let me price the cart "
                "first and read it back to you."
            )

        if time.time() - pending.created_at > self.confirm_ttl:
            self._pending = None
            return "That quote is stale now. Let me price the cart again."

        if str(confirmation_code or "").strip() != pending.code:
            return (
                "That confirmation code doesn't match the order I have ready, "
                "so I'm not placing it. Let me read the order back again."
            )

        # Guard 3: a real user turn must separate preview from submit.
        if self._turn <= pending.created_turn:
            return "I need you to say yes out loud before I place this order."

        # Guard 4: and that turn must actually have been a yes. This reads the
        # recorded transcript, not anything the model handed us.
        if not is_affirmative(self._last_user_text):
            return (
                "I didn't hear a clear yes, so I haven't placed anything. "
                "Say 'yes, place the order' when you're ready."
            )

        # Guard 5: re-price and require the approved order to be unchanged.
        try:
            payload = self._preview(
                pending.cart_uuid,
                fulfillment=pending.submit_args["fulfillment"],
                priority=pending.submit_args["priority"],
                scheduled_time=pending.submit_args["scheduled_time"],
                no_apply_credits=pending.submit_args["no_apply_credits"],
            )
        except DoorDashError as exc:
            return f"I couldn't re-check the order, so I didn't place it. {exc.spoken}"

        fresh = self._summarize(payload, pending.submit_args["tip_cents"])
        if fresh.fingerprint != pending.fingerprint:
            code = f"{secrets.randbelow(9000) + 1000}"
            self._pending = PendingOrder(
                cart_uuid=pending.cart_uuid,
                code=code,
                fingerprint=fresh.fingerprint,
                summary=fresh,
                submit_args=pending.submit_args,
                created_turn=self._turn,
            )
            return (
                "The order changed since I read it to you, so I stopped. "
                + fresh.spoken(code)
            )

        charge = fresh.charge_cents
        if charge is not None and charge > self.max_order_cents:
            self._pending = None
            return (
                f"That's {_dollars(charge)}, over my {_dollars(self.max_order_cents)} "
                "limit. I didn't place it."
            )

        args = ["order", "submit", "--cart-uuid", pending.cart_uuid, "-y"]
        tip = pending.submit_args["tip_cents"]
        if tip:
            args += ["--tip-cents", str(tip)]
        if pending.submit_args["fulfillment"]:
            args += ["--fulfillment", pending.submit_args["fulfillment"]]
        if pending.submit_args["priority"]:
            args.append("--priority")
        if pending.submit_args["scheduled_time"]:
            args += ["--scheduled-time", pending.submit_args["scheduled_time"]]
        if pending.submit_args["no_apply_credits"]:
            args.append("--no-apply-credits")

        logger.info("doordash submitting cart=%s total=%s", pending.cart_uuid, charge)
        self._run(args, timeout=max(self.timeout, 90))
        self._pending = None

        eta = fresh.eta_text
        tail = (
            f" Should be {'ready' if fresh.is_pickup else 'here'} in {eta}."
            if eta
            else ""
        )
        return f"Ordered — {_dollars(charge)} from {fresh.store_name}.{tail}"

    def cancel_pending(self) -> str:
        if self._pending is None:
            return "There's nothing waiting to be placed."
        self._pending = None
        return "Cancelled — I won't place it."
