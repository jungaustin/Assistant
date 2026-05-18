"""Tests for the sentence chunker.

Covers: hard boundaries, soft boundaries, multi-boundary in a single token,
force-flush at length, no-whitespace pathological case, partial tokens,
empty tokens, end-of-stream drain, async variant.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable, List

import pytest

from robot.core.chunker import (
    MAX_CHARS,
    achunk_tokens,
    chunk_tokens,
)


def _chunks(tokens: Iterable[str]) -> List[str]:
    return list(chunk_tokens(tokens))


# ---------- Hard boundaries ----------


def test_period_space_emits_phrase():
    out = _chunks(["Hello", " world", ". ", "Next."])
    # First chunk ends with ". ", remainder is held until exhaustion drains it.
    assert out == ["Hello world. ", "Next."]


def test_exclamation_emits():
    out = _chunks(["Wow", "! ", "More"])
    assert out == ["Wow! ", "More"]


def test_question_emits():
    out = _chunks(["Why? ", "Because"])
    assert out == ["Why? ", "Because"]


def test_newline_emits():
    out = _chunks(["Line one\n", "Line two"])
    assert out == ["Line one\n", "Line two"]


# ---------- Soft boundary ----------


def test_comma_emits():
    out = _chunks(["First", ", ", "second"])
    assert out == ["First, ", "second"]


# ---------- Multi-boundary in a single token ----------


def test_single_token_with_two_boundaries_emits_both():
    # The plan's main motivation: small local models sometimes emit whole
    # sentences in one streaming "delta". We must split, not buffer.
    out = _chunks(["Hello. How are you? Fine."])
    assert out == ["Hello. ", "How are you? ", "Fine."]


def test_period_then_comma_in_one_token():
    out = _chunks(["Done. Also, more"])
    assert out == ["Done. ", "Also, ", "more"]


# ---------- Force-flush at length ----------


def test_long_run_no_punct_force_flushes_at_whitespace():
    # 100 chars of "word " repeated, no terminal punctuation.
    words = ["word "] * 20  # 100 chars total
    out = _chunks(words)
    # First chunk must be ≤ MAX_CHARS, end at a whitespace.
    assert out[0].endswith(" ")
    assert len(out[0]) <= MAX_CHARS
    # No data loss: joined output must equal joined input.
    assert "".join(out) == "".join(words)


def test_pathological_no_whitespace_at_all():
    # A 200-char token with no spaces. Split at MAX_CHARS (no other choice).
    blob = "x" * 200
    out = _chunks([blob])
    assert len(out[0]) == MAX_CHARS
    assert "".join(out) == blob


# ---------- Partial / empty tokens ----------


def test_empty_tokens_are_skipped():
    out = _chunks(["", "Hello", "", ". ", ""])
    assert out == ["Hello. "]


def test_single_char_tokens_accumulate_to_boundary():
    # Streaming LLMs sometimes emit one char at a time.
    out = _chunks(list("Hi. Bye."))
    assert out == ["Hi. ", "Bye."]


def test_token_split_across_boundary():
    # Boundary char arrives in a later token (period in token A, space in B).
    out = _chunks(["End.", " Next"])
    assert out == ["End. ", "Next"]


def test_token_split_mid_boundary():
    # ". " split across two tokens. Only emits when both arrive.
    out = _chunks(["End", ".", " ", "Next"])
    assert out == ["End. ", "Next"]


# ---------- End-of-stream drain ----------


def test_trailing_unflushed_buffer_emits():
    out = _chunks(["No terminal punct"])
    assert out == ["No terminal punct"]


def test_empty_iterator_yields_nothing():
    assert _chunks([]) == []


def test_only_empty_tokens_yield_nothing():
    assert _chunks(["", "", ""]) == []


# ---------- Numbers and abbreviations (boundary edge cases) ----------


def test_decimal_number_does_not_split():
    # "1.5" has no ". " — the period isn't followed by a space.
    out = _chunks(["The number is 1.5 here."])
    assert out == ["The number is 1.5 here."]


def test_abbreviation_does_split_acceptably():
    # "Dr. Smith" DOES match ". ". Acceptable — known cost noted in chunker docs.
    out = _chunks(["Dr. Smith is here."])
    assert out == ["Dr. ", "Smith is here."]


# ---------- Async variant ----------


async def _aiter(items: Iterable[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_async_chunker_matches_sync():
    tokens = ["Hello", " world", ". ", "How are you?", " Fine, ", "thanks."]
    sync_out = list(chunk_tokens(tokens))
    async_out = [chunk async for chunk in achunk_tokens(_aiter(tokens))]
    assert sync_out == async_out


@pytest.mark.asyncio
async def test_async_chunker_drains_trailing_buffer():
    out = [chunk async for chunk in achunk_tokens(_aiter(["No punct"]))]
    assert out == ["No punct"]
