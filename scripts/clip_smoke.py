"""Integrated clip-that smoke check (`just clip-smoke`, eng plan T6).

Runs the REAL stack end to end — Swift capture helper, RAM disk, mic ring,
snapshot, save — for ~15 seconds, then asserts the saved clip is sane:
file exists, duration within tolerance of what was buffered, one video and
two audio streams (system + mic). This is the check the unit suite can't
give you: TCC, ScreenCaptureKit, hdiutil, and ffmpeg all for real.

Needs Screen Recording permission for the app that launches it (first run
prompts once — the helper's stable signing identity keeps later rebuilds
quiet). Exits nonzero with a reason on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from robot.config import CLIP_RECORDER_BIN
from robot.core.clip import ClipError, ClipService

CAPTURE_SECONDS = 15


def fail(reason: str) -> None:
    print(f"FAIL: {reason}")
    sys.exit(1)


def main() -> None:
    binary = Path(CLIP_RECORDER_BIN)
    if not binary.exists():
        fail(f"capture helper missing at {binary} — run `just build-clip-recorder`")

    save_dir = Path(tempfile.mkdtemp(prefix="nemo-clip-smoke-"))
    service = ClipService(
        recorder_binary=binary,
        save_dir=save_dir,
        speak=lambda msg: print(f"[speak] {msg}"),
    )
    print(f"[smoke] starting capture ({CAPTURE_SECONDS}s)...")
    service.start()
    try:
        # Wait for the first segment so a TCC denial fails fast and clearly.
        # (Private-attr peeking is fine here — this script lives in-repo and
        # exists precisely to exercise the real internals.)
        deadline = time.time() + 20
        while time.time() < deadline:
            rec = service._recorder
            if rec is not None and rec.permission_denied:
                fail(
                    "Screen Recording permission denied — grant it to the "
                    "app that launched this (System Settings → Privacy → "
                    "Screen Recording), then re-run."
                )
            if rec is not None and rec.snapshot_segments():
                break
            time.sleep(0.5)
        else:
            fail("no segment arrived within 20s — capture never started")

        time.sleep(CAPTURE_SECONDS)
        print("[smoke] snapshot (the on_recording_start moment)")
        service.take_snapshot()
        snap = service._snapshot
        if snap is None:
            fail("snapshot slot empty after take_snapshot()")
        expected = 5.0 * len(snap.segments)

        time.sleep(2)  # the user finishes saying "clip that"
        print("[smoke] save")
        try:
            clip = service.save()
        except ClipError as exc:
            fail(f"save raised its spoken error: {exc.spoken}")

        info = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(clip),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        duration = float(info["format"]["duration"])
        kinds = [s["codec_type"] for s in info["streams"]]
        size_mb = clip.stat().st_size / 1e6

        print(
            f"[smoke] clip={clip} duration={duration:.1f}s "
            f"(snapshot covered ~{expected:.0f}s) streams={kinds} "
            f"size={size_mb:.1f}MB"
        )
        # Duration = snapshot window + up to ~2 boundary segments that landed
        # during the 2s pause and the boundary wait.
        if not (expected * 0.9 <= duration <= expected + 12):
            fail(
                f"duration {duration:.1f}s outside [{expected * 0.9:.1f}, {expected + 12:.1f}]"
            )
        if kinds.count("video") != 1 or kinds.count("audio") != 2:
            fail(f"expected 1 video + 2 audio streams, got {kinds}")
        if size_mb < 1.0:
            fail(f"clip suspiciously small ({size_mb:.2f}MB)")
        print("PASS")
    finally:
        service.stop()


if __name__ == "__main__":
    main()
