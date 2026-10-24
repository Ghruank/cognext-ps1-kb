"""Page-recall check for the page-card navigator, with gold documents given (oracle routing).

    python -m alpha.exp_pages "CEATIntegratedAnnualReport2526Re.pdf" "Metal Coatings_Annual Report FY 2026.pdf"
"""

import json
import pickle
import statistics as st
import sys
from pathlib import Path

from eval import bench
from alpha.navigate import select_pages
from alpha.pagecards import build_page_cards

CARDS = Path("neo/storage/train_pagecards.pkl")


def main() -> None:
    docs = set(sys.argv[1:])
    units = [u for d in pickle.load(open("storage/train_docs.pkl", "rb")) for u in d["units"]]
    catalog = json.load(open("storage/train_catalog.json"))
    existing = pickle.load(open(CARDS, "rb")) if CARDS.exists() else {}
    cards, usd = build_page_cards(units, catalog, docs=docs, existing=existing)
    pickle.dump(cards, open(CARDS, "wb"))
    print(f"page cards: {len(cards)} total, this run ${usd:.3f}")
    rows = [q for q in bench.load() if q["gold"] and {c["doc"] for c in q["gold"]} <= docs]
    rec, n_sel, cost = [], [], []
    for q in rows:
        gold = {(c["doc"], p) for c in q["gold"] for p in c["pages"]}
        chosen, usage = select_pages(q["question"], [], sorted({c["doc"] for c in q["gold"]}), cards, catalog)
        sel = {(d, p) for d, ps in chosen.items() for p in ps}
        rec.append(len(sel & gold) / len(gold))
        n_sel.append(len(sel))
        cost.append(usage["usd"])
        print(f"{q['id']}: gold pages {len(gold):3d}, selected {len(sel):3d}, page recall {rec[-1]:.2f}, "
              f"page precision {len(sel & gold) / max(1, len(sel)):.2f}, ${usage['usd']:.4f}", flush=True)
    print(f"\n{len(rows)} questions: mean page recall {st.mean(rec):.3f}, median selected {st.median(n_sel)}, "
          f"mean cost ${st.mean(cost):.4f}")


if __name__ == "__main__":
    main()
