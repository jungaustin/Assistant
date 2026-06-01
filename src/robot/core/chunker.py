"""Sentence chunker. Turns a stream of LLM tokens into phrase-sized chunks.

Why: TTS engines that synthesize per-phrase (Piper, RealtimeTTS) play audio
as each chunk finishes. If you feed them tokens one-at-a-time, they either
chunk internally (Piper does, inconsistently) or speak each token (robotic).
Doing the chunking once, upstream, gives every engine the same well-formed
input and lets us tune phrasing in one place.

Boundary rules:
  - Hard boundary: ``. ``, ``! ``, ``? ``, ``\\n``  → emit immediately
  - Soft boundary: ``, ``                          → emit immediately
  - Force-flush: buffer length ≥ MAX_CHARS         → emit at last whitespace
                                                     (avoids mid-word split)

A single token can contain multiple boundaries (``"Hello. How are you?"``
arrives as one token from some models); the chunker emits one phrase per
boundary inside the token.

Numbers like ``1.5`` don't trigger ``. `` (needs the trailing space).
Abbreviations like ``Dr. Smith`` DO trigger; the cost is a small mid-sentence
beat, which is fine and arguably correct prosody.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterable, Iterator

MAX_CHARS = 80
HARD_BOUNDARIES = (". ", "! ", "? ", "\n")
SOFT_BOUNDARIES = (", ",)
ALL_BOUNDARIES = HARD_BOUNDARIES + SOFT_BOUNDARIES


def _find_first_boundary(buf: str) -> int:
    """Return the index of the FIRST boundary in `buf` (right after it), or -1.

    "First" so a single token with multiple boundaries emits in order rather
    than buffering the whole token until something else arrives.
    """
    earliest = -1
    for b in ALL_BOUNDARIES:
        idx = buf.find(b)
        if idx == -1:
            continue
        end = idx + len(b)
        if earliest == -1 or end < earliest:
            earliest = end
    return earliest


def _force_flush_index(buf: str) -> int:
    """When the buffer is too long with no boundary, split at the last
    whitespace. Returns the split index (everything before it gets emitted).
    Falls back to splitting at MAX_CHARS if there's no whitespace at all.
    """
    cutoff = buf.rfind(" ", 0, MAX_CHARS)
    if cutoff == -1:
        # Pathological: a single un-spaced run > MAX_CHARS. Emit at MAX_CHARS.
        return MAX_CHARS
    return cutoff + 1  # include the trailing space in the emitted chunk


def chunk_tokens(token_iter: Iterable[str]) -> Iterator[str]:
    """Yield phrase-sized chunks from a sync iterator of tokens.

    Each yielded chunk ends at a boundary or a forced flush. Empty tokens are
    skipped. Trailing buffer is yielded on exhaustion (no trailing data lost).
    """
    buf = ""
    for token in token_iter:
        if not token:
            continue
        buf += token

        # Drain every boundary that's currently in the buffer (a single
        # token can carry multiple).
        while True:
            idx = _find_first_boundary(buf)
            if idx != -1:
                yield buf[:idx]
                buf = buf[idx:]
                continue
            if len(buf) >= MAX_CHARS:
                idx = _force_flush_index(buf)
                yield buf[:idx]
                buf = buf[idx:]
                continue
            break

    if buf:
        yield buf


async def achunk_tokens(token_aiter: AsyncIterator[str]) -> AsyncIterator[str]:
    """Async variant. Same rules, for an async iterator of tokens."""
    buf = ""
    async for token in token_aiter:
        if not token:
            continue
        buf += token
        while True:
            idx = _find_first_boundary(buf)
            if idx != -1:
                yield buf[:idx]
                buf = buf[idx:]
                continue
            if len(buf) >= MAX_CHARS:
                idx = _force_flush_index(buf)
                yield buf[:idx]
                buf = buf[idx:]
                continue
            break
    if buf:
        yield buf
