"""One-shot OAuth bootstrap for Google Calendar.

Prerequisites (you do these once in the Google Cloud Console — see the
README "Google Calendar setup" section for screenshots/links):

1. Create a Google Cloud project (or reuse an existing one).
2. Enable the Google Calendar API for the project.
3. Configure the OAuth consent screen as "External", add yourself as a
   test user (status: Testing — fine for a personal-use app, no review
   needed; up to 100 test users).
4. Create an OAuth 2.0 Client ID with type "Desktop app".
5. Download the client_secret JSON and save it to
   `robot/state/google-credentials.json`.

Then run:

    just oauth-calendar

This script will open your browser, you approve the scope, and a
refresh-token JSON gets written to `robot/state/google-calendar-token.json`.
After that the runtime client picks it up automatically and refreshes
the access token as needed — you should never need to run this again
unless you revoke access or change scopes.

Why a separate file from `setup/spotify_oauth_bootstrap.py`: Google's
SDK ships its own loopback OAuth flow (`InstalledAppFlow`), no Flask
needed. The Spotify version predates that pattern and uses a manual
redirect-URI dance.
"""

from __future__ import annotations

import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow


# Single scope: read + write events in calendars the user has access to.
# `calendar.events` is narrower than `calendar` (which can create/delete
# entire calendars) — we don't need that.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _resolve_paths() -> tuple[Path, Path]:
    """Read paths from config so a custom STATE_DB_PATH-style override
    via env var actually works."""
    # Import lazily so this script can run even if the package isn't
    # installed in editable mode (e.g. someone clones and runs it before
    # `uv sync`).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from robot.config import (
        GOOGLE_CALENDAR_CREDENTIALS_PATH,
        GOOGLE_CALENDAR_TOKEN_PATH,
    )

    return Path(GOOGLE_CALENDAR_CREDENTIALS_PATH), Path(GOOGLE_CALENDAR_TOKEN_PATH)


def main() -> int:
    credentials_path, token_path = _resolve_paths()

    if not credentials_path.exists():
        print(
            f"ERROR: client credentials not found at {credentials_path}.\n"
            f"Download the OAuth client JSON from Google Cloud Console "
            f"(see the docstring at the top of this file for steps) and "
            f"save it to that path."
        )
        return 1

    token_path.parent.mkdir(parents=True, exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    # port=0 → bind any free port. The library prints the local URL it's
    # listening on, opens the browser, and waits for the auth redirect.
    creds = flow.run_local_server(port=0)

    token_path.write_text(creds.to_json())
    print(f"\n✓ token saved to {token_path}")
    print(f"  scope: {' '.join(SCOPES)}")
    print(f"  expires at: {creds.expiry}")
    print(
        "\nThe runtime client will refresh the access token automatically "
        "from now on. You shouldn't need to run this script again unless "
        "you revoke access or change scopes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
