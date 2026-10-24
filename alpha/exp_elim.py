"""Chunk-elimination experiments on STAR's candidate sets, scored against the judge itself.

    python -m alpha.exp_elim label        # judge every candidate (exact judge prompt, dev mode, batches of 40)
    python -m alpha.exp_elim report       # baseline + elimination variants

Ground truth is the official judge's own verdict, not gold-text overlap: the judge accepts
only ~76% of the gold chunks themselves, so a gold-mimicking filter caps precision there.
"""

import json
import pickle
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import bench
from eval.official import make_evaluator

DIR = Path(__file__).resolve().parent / "storage" / "elim"
BATCH = 40


def load_testbed(ids: set[str] | None = None) -> list[dict]:
    tb = pickle.load(open(DIR / "testbed.pkl", "rb"))
    return [t for t in tb if ids is None or t["id"] in ids]


def label(ids: set[str]) -> dict[str, list[bool]]:
    ev = make_evaluator("dev")
    path = DIR / "judge_labels.json"
    labels = json.loads(path.read_text()) if path.exists() else {}
    jobs = []
    for t in load_testbed(ids):
        if t["id"] in labels:
            continue
        for s in range(0, len(t["units"]), BATCH):
            jobs.append((t, s))

    def run(job):
        t, s = job
        chunks = [u["text"] for u in t["units"][s:s + BATCH]]
        try:
            v = ev._judge_chunk_relevance(t["question"], chunks)
        except RuntimeError:
            return t["id"], s, None
        rel = {int(x["id"]) for x in v if x.get("relevant") is True and str(x.get("id")).isdigit()}
        return t["id"], s, [i in rel for i in range(len(chunks))]

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(run, jobs))
    parts: dict[str, dict[int, list]] = {}
    for qid, s, v in res:
        parts.setdefault(qid, {})[s] = v
    for t in load_testbed(ids):
        if t["id"] in parts and all(v is not None for v in parts[t["id"]].values()):
            labels[t["id"]] = [x for s in sorted(parts[t["id"]]) for x in parts[t["id"]][s]]
        elif t["id"] in parts:
            print(f"{t['id']}: judge failed on some batch, skipped")
    path.write_text(json.dumps(labels))
    return labels


def score(rows: list[tuple[list[bool], list[bool], int]]) -> str:
    """rows: (judge verdicts, keep mask, gold count). Precision/recall as the judge would score the kept set."""
    P, R, ret, k = [], [], [], []
    for y, keep, g in rows:
        kept = [a for a, m in zip(y, keep) if m]
        P.append(sum(kept) / len(kept) if kept else 0.0)
        R.append(sum(kept) / g if g else 0.0)
        ret.append(sum(kept) / sum(y) if sum(y) else 1.0)
        k.append(len(kept))
    p, r = st.mean(P), st.mean(R)
    return (f"P={p:.3f} R={r:.3f} F1={2*p*r/(p+r+1e-9):.3f}  judge-relevant kept={st.mean(ret):.3f}  "
            f"median k={st.median(k):.0f}")


def report(variants: dict) -> None:
    labels = json.loads((DIR / "judge_labels.json").read_text())
    tb = [t for t in load_testbed(set(labels)) if t["n_gold"]]
    print(f"{len(tb)} questions with judge labels\n")
    for name, fn in variants.items():
        rows = [(labels[t["id"]], fn(t), t["n_gold"]) for t in tb]
        print(f"{name:34} {score(rows)}")


if __name__ == "__main__":
    if sys.argv[1] == "label":
        ids = set(bench.subset()) if len(sys.argv) < 3 else set(sys.argv[2].split(","))
        labels = label(ids)
        for t in load_testbed(ids):
            if t["id"] in labels:
                y = labels[t["id"]]
                prox = [u["proxy"] for u in t["units"]]
                print(f"{t['id']}: cand {len(y):3d} judge-rel {sum(y):3d} (P={sum(y)/max(1,len(y)):.2f} "
                      f"R={sum(y)/max(1,t['n_gold']):.2f})  proxy-pos {sum(prox):3d}  gold {t['n_gold']}")


def run_elimination(tag: str, **kw) -> dict[str, list[bool]]:
    """Runs neo.eliminate on every labelled question; caches keep masks under `tag`."""
    from alpha.eliminate import eliminate
    path = DIR / f"keep_{tag}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    labels = json.loads((DIR / "judge_labels.json").read_text())
    todo = [t for t in load_testbed(set(labels)) if t["id"] not in out]

    def one(t):
        keep, info = eliminate(t["question"], [], t["units"], **kw)
        ks = set(keep)
        return t["id"], [i in ks for i in range(len(t["units"]))], info

    with ThreadPoolExecutor(4) as pool:
        for qid, mask, info in pool.map(one, todo):
            out[qid] = mask
            path.write_text(json.dumps(out))   # long free-tier runs get interrupted; keep finished questions
            print(f"{qid}: kept {sum(mask)}/{len(mask)}  {info['structural']} llm-removed {info['llm_removed']} failed {info['failed_batches']}/{info['calls']} ${info['usd']:.3f}", flush=True)
    path.write_text(json.dumps(out))
    return out
