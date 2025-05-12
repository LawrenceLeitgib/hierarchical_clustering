"""
extract_leads.py
----------------
Read numbered MediaWiki XML dumps (0.xml, 1.xml, …) produced by
Special:Export and write the plain‑text lead section of each article
to vital_abstracts.jsonl.

Output line format:
{"idx": 0, "title": "0", "lead": "<plain text>"}
"""

from __future__ import annotations
import json, re, xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import mwparserfromhell as mwp
from sknetwork.data import load_netset

XML_DIR   = Path("vital_xml")           #folder of 0.xml … N.xml
OUT_FILE  = Path("utils/vital_abstracts.jsonl")
NS        = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}

# -- 1) your original title list so idx ↔ title stays aligned ----------
wikivitals: list[str] = load_netset("wikivitals").names
#assert len(wikivitals) == len(list(XML_DIR.glob("*.xml"))), \
 #      "Title list and XML count differ!"

lead_pat  = re.compile(r"(^|\n)==")    # first heading marks end of lead
para_pat  = re.compile(r"\n\s*\n")     # split on blank line(s)

def lead_plain_text(raw_wikitext: str) -> str:
    """Return plain‑text lead (intro section) stripped of markup."""
    # keep only text BEFORE first '==', i.e. before first section heading
    m = lead_pat.search(raw_wikitext)
    if m:
        raw_wikitext = raw_wikitext[:m.start()]
    # parse & strip markup
    plain = mwp.parse(raw_wikitext).strip_code(normalize=True, collapse=True)
    # collapse multiple blank lines; trim ends
    paragraphs = [p.strip() for p in para_pat.split(plain) if p.strip()]
    return "\n\n".join(paragraphs)

with OUT_FILE.open("w", encoding="utf‑8") as sink:
    for idx, title in tqdm(list(enumerate(wikivitals)), ncols=90, desc="Parsing"):
        xml_path = XML_DIR / f"{idx}.xml"
        if not xml_path.exists():
            print(f"⚠️  Missing {xml_path} — skipping")
            continue

        try:
            tree  = ET.parse(xml_path)
            text  = tree.find(".//mw:text", NS).text or ""
            lead  = lead_plain_text(text)
        except Exception as exc:
            print(f"⚠️  Failed on {xml_path}: {exc}")
            continue

        sink.write(json.dumps({"idx": idx, "title": title, "lead": lead},
                              ensure_ascii=False) + "\n")

print(f"✅  Done – leads written to {OUT_FILE}")
