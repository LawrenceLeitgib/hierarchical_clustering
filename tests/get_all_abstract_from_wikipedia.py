import os
import time
import urllib.parse
from pathlib import Path

import requests
from slugify import slugify
from tqdm import tqdm   # optional, just for nice progress bars


from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from sknetwork.embedding import Spectral
from sknetwork.visualization import visualize_dendrogram




# ----------------------------------------------------------------------
# Your list of Level‑4 Vital Article titles
# ----------------------------------------------------------------------
wikivitals = load_netset("wikivitals")   # whatever gave you the list
titles: list[str] = sorted(set(wikivitals.names))   # dedupe & sort (optional)

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
OUT_DIR = Path("vital_xml")        # will contain one XML per page
BATCH_SIZE = 200                   # how many titles per export POST
SLEEP_BETWEEN_CALLS = 1.0          # be polite to Wikipedia
USER_AGENT = "YourAwesomeBot/0.1 (research; your.email@example.com)"

# ----------------------------------------------------------------------
# Helper: fetch one export batch and write it to disc
# ----------------------------------------------------------------------
def export_pages(titles_batch: list[str], batch_idx: int) -> None:
    """
    Call Special:Export with a POST that contains newline‑separated titles.
    Saves the returned XML as 'batch_{idx:04d}.xml'.
    """
    payload = {
        "action": "submit",
        "pages": "\n".join(titles_batch),
        "curonly": "1",       # current revision only (no full history)
        "templates": "0",     # don't expand templates; keeps XML smaller
    }

    resp = requests.post(
        "https://en.wikipedia.org/wiki/Special:Export",
        data=payload,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    resp.raise_for_status()

    out_file = OUT_DIR / f"batch_{batch_idx:04d}.xml"
    out_file.write_bytes(resp.content)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    batches = [
        titles[i : i + BATCH_SIZE] for i in range(0, len(titles), BATCH_SIZE)
    ]
    print(batches)
    for idx, batch in enumerate(tqdm(batches, desc="Exporting")):
        print(f"Exporting batch {idx + 1}/{len(batches)}")
        print(f"Titles: {batch}")
        print(f"Number of titles: {len(batch)}")
        export_pages(batch, idx)
        time.sleep(SLEEP_BETWEEN_CALLS)   # politeness delay


if __name__ == "__main__":
    main()
