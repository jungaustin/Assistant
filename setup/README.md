# setup/

One-time bootstrap scripts. Not part of the runtime.

## spotify_oauth_bootstrap.py

Run once to obtain a Spotify refresh token. Open http://127.0.0.1:5000, log in, copy the refresh token from the console, and put it in `.env` as `REFRESH_TOKEN`. After that the runtime uses `SpotifyClient` directly with the refresh token — Flask is not needed again.

```
source .venv/bin/activate
python setup/spotify_oauth_bootstrap.py
```
