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
happens the token endpoint returns `invalid_grant`; the runtime detects this,
discards the dead token (it does not retry), and Nemo will say "Spotify needs
to be re-authorized" instead of crashing. To fix it, just run this script
again and log in — the new token overwrites the old one in `.env`.

(The runtime also auto-saves rotated refresh tokens back to `.env` whenever
Spotify hands out a new one on a normal refresh, so this is the only manual
step, and only every six months.)
