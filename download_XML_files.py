"""
download_vital_articles.py
==========================

Create one XML dump file *per* Level‑4 Vital Article using Wikipedia’s
Special:Export endpoint.

• 10 030 requests → 10 030 files like  "Albert_Einstein.xml"
• Resumable (re‑run safely; existing files are skipped)
• Polite (1 request/second, custom User‑Agent, limited retries)
"""

from __future__ import annotations
import time
import urllib.parse
from pathlib import Path
from typing import Iterable
from sknetwork.data import load_netset


import requests
from slugify import slugify
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1)  Load your title list  (replace with your own loader)
# ---------------------------------------------------------------------------
def load_vital_titles() -> list[str]:
    """Return a list of page titles (strings)."""
    # Your original helper:
    return sorted(set(load_netset("wikivitals").names))      # type: ignore


# ---------------------------------------------------------------------------
# 2)  Settings
# ---------------------------------------------------------------------------
OUT_DIR = Path("vital_xml")
OUT_DIR.mkdir(exist_ok=True)

USER_AGENT = (
    "(clustering research; contact: lawrence.leitgib@epfl.ch)"
)
REQUEST_DELAY = 1.0          # seconds between *successful* requests
MAX_RETRIES = 3              # network retries per page
TIMEOUT = 30                 # seconds for the HTTP request


# ---------------------------------------------------------------------------
# 3)  Core download helper
# ---------------------------------------------------------------------------
def fetch_xml(page_title: str) -> bytes:
    """
    Return the XML (bytes) for a single Wikipedia page in current‑revision form.
    Raises `requests.HTTPError` on permanent failures.
    """
    quoted = urllib.parse.quote(page_title, safe="")
    url = (
        "https://en.wikipedia.org/wiki/Special:Export/"
        f"{quoted}"
        "?action=submit"
    )

    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# 4)  Download loop — resumable & robust
# ---------------------------------------------------------------------------
def download_all(titles: Iterable[str]) -> None:
    count = 0
    for title in tqdm(titles, desc="Downloading", ncols=90):
        fn = OUT_DIR / f"{count}.xml"

        # Skip if already present and > 0 B
        if fn.exists() and fn.stat().st_size:
            continue

        # Try up to MAX_RETRIES times
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                fn.write_bytes(fetch_xml(title))
                count += 1
                time.sleep(REQUEST_DELAY)   # polite gap *after* success
                break                       # done, go to next title
            except (requests.RequestException, IOError) as exc:
                if attempt == MAX_RETRIES:
                    print(f"\nFAILED after {MAX_RETRIES} attempts → {title}: {exc}")
                else:
                    backoff = 2 ** (attempt - 1)
                    time.sleep(backoff)


# ---------------------------------------------------------------------------
# 5)  Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_titles = load_vital_titles()
    download_all(all_titles)
    print("✅  Finished – every requested title now has its own .xml file.")
