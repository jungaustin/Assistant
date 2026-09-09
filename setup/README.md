# setup/

One-time bootstrap scripts. Not part of the runtime.

## spotify_oauth_bootstrap.py

Obtain a Spotify refresh token. Open http://127.0.0.1:5000, log in, and the
refresh token is written straight into `.env` as `REFRESH_TOKEN` — no manual
copy needed. After that the runtime uses `SpotifyClient` directly with the
token; Flask is not needed again.

```
source .venv/bin/activate
python setup/spotify_oauth_bootstrap.py
```

### Re-authorizing (six-month expiry)

As of 2026-07-20 Spotify expires refresh tokens after six months. When that
happens the token endpoint returns `invalid_grant`; the runtime detects this
and discards the dead token (it does not retry).

**You normally don't need this script for that.** The runtime re-authorizes
itself: `robot/tools/inner/spotify_auth.py` runs the same authorization-code
exchange in-process, so the robot opens the Spotify login in your browser,
waits up to a minute for you to click Agree, saves the new token to `.env`,
and then finishes the request that triggered it — ask for a playlist with an
expired token and you just get the playlist. You can also trigger it by voice
("reconnect Spotify" → the `reauthorize_spotify` tool).

Run this script by hand when the automatic flow can't work:

- `CLIENT_ID` / `CLIENT_SECRET` are missing from `.env`.
- The callback port is taken — on macOS, System Settings → General →
  AirDrop & Handoff → AirPlay Receiver squats on 5000. Turn it off, or set
  `SPOTIFY_REDIRECT_URI` to another port *and* register that URI on the
  Spotify app dashboard.
- The Brain is running headless (a Pi) with no browser. The robot logs the
  authorize URL in that case, so you can also finish it from a phone.

See the `SPOTIFY_*` block in `.env.example` for the knobs.

(The runtime also auto-saves rotated refresh tokens back to `.env` whenever
Spotify hands out a new one on a normal refresh.)
