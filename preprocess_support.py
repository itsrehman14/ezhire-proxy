"""Pre-generate support_docs.json from FAQ.html and TnC.html.

Run once (or whenever the HTML source files change):
    python preprocess_support.py
"""

import json
import os

from server import SUPPORT_PAGES_DIR, load_support_documents


def main() -> None:
    docs = load_support_documents()

    out_path = os.path.join(SUPPORT_PAGES_DIR, "support_docs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    faq_count = sum(1 for d in docs if d.get("source") == "FAQ")
    tnc_count = sum(1 for d in docs if d.get("source") == "TnC")
    print(f"Wrote {out_path}: {faq_count} FAQ entries, {tnc_count} TnC chunks ({len(docs)} total)")


if __name__ == "__main__":
    main()
