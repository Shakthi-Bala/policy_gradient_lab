"""Send experiment summaries and media to a configured Discord webhook."""

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def webhook_url():
    value = os.environ.get("DISCORD_WEBHOOK_URL")
    if value:
        return value
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DISCORD_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def send_update(message, attachments=()):
    url = webhook_url()
    if not url:
        print("Discord skipped: DISCORD_WEBHOOK_URL is unset.")
        return False
    for attempt in range(3):
        stack = ExitStack()
        try:
            files = {}
            for i, path in enumerate(attachments):
                handle = stack.enter_context(Path(path).open("rb"))  # noqa: SIM115 - ExitStack closes dynamic attachments
                files[f"files[{i}]"] = (Path(path).name, handle)
            response = requests.post(
                url,
                data={"payload_json": json.dumps({"content": message[:2000]})},
                files=files or None,
                timeout=45,
            )
            if response.status_code == 429:
                time.sleep(min(float(response.json().get("retry_after", 1)), 10))
                continue
            if not response.ok:
                print(f"Discord delivery failed (HTTP {response.status_code}).")
                return False
            print("Discord update delivered.")
            return True
        except (requests.RequestException, OSError, ValueError) as error:
            # Exceptions can embed the secret URL, so print only their class.
            print(f"Discord delivery failed ({type(error).__name__}).")
            return False
        finally:
            stack.close()
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message")
    parser.add_argument("--attach", type=Path, action="append", default=[])
    args = parser.parse_args()
    raise SystemExit(0 if send_update(args.message, args.attach) else 1)
