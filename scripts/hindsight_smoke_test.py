"""Exercise the documented Hindsight retain and recall operations.

Before running this script, set HINDSIGHT_API_KEY and HINDSIGHT_BASE_URL in
the project-root .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from hindsight_client import Hindsight


BANK_ID = "bull-bear-desk-smoke-test"
RECALL_QUERY = "oversold bounce failed before earnings"


def main() -> None:
    """Retain example trade memories and print matching recall results."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("HINDSIGHT_API_KEY")
    base_url = os.getenv("HINDSIGHT_BASE_URL")
    if not api_key:
        raise RuntimeError("HINDSIGHT_API_KEY is missing from .env")
    if not base_url:
        raise RuntimeError("HINDSIGHT_BASE_URL is missing from .env")

    client = Hindsight(base_url=base_url, api_key=api_key)

    memories = (
        "Bought a tech stock after an oversold RSI bounce; the bounce failed "
        "within two sessions before earnings.",
        "Avoided a second oversold bounce setup before earnings after the prior "
        "trade failed and volatility expanded.",
        "A small-cap earnings trade lost money when an oversold bounce faded "
        "into the report; wait for confirmation next time.",
    )
    for memory in memories:
        client.retain(bank_id=BANK_ID, content=memory)

    response = client.recall(bank_id=BANK_ID, query=RECALL_QUERY)
    print(f"Recall query: {RECALL_QUERY}")
    print(f"Results returned: {len(response.results)}")
    for result in response.results:
        print(f"- {result.text}")


if __name__ == "__main__":
    main()
