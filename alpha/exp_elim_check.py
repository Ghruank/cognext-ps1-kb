"""Two checks on the elimination labels.

    python -m alpha.exp_elim_check consistency   # judge re-run with chunks reshuffled into different batches
    python -m alpha.exp_elim_check full TAG...    # official-style scoring: the whole kept set in ONE judge call
"""

import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

from eval.official import make_evaluator
from alpha.exp_elim import BATCH, DIR, load_testbed


def consistency() -> None:
    ev = make_evaluator("dev")
    labels = json.loads((DIR / "judge_labels.json").read_text())
    tb = [t for t in load_testbed(set(labels)) if len(t["units"]) > 1]
    jobs = []
    for t in tb:
        order = list(range(len(t["units"])))
        random.Random(1).shuffle(order)
        for s in range(0, len(order), BATCH):
            jobs.append((t, order[s:s + BATCH]))

    def run(job):
        t, idx = job
        try:
            v = ev._judge_chunk_relevance(t["question"], [t["units"][i]["text"] for i in idx])
        except RuntimeError:
            return t["id"], {}
        rel = {int(x["id"]) for x in v if x.get("relevant") is True and str(x.get("id")).isdigit()}
        return t["id"], {i: (n in rel) for n, i in enumerate(idx)}

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(run, jobs))
    second: dict[str, dict] = {}
    for qid, d in res:
        second.setdefault(qid, {}).update(d)
    (DIR / "judge_labels_rerun.json").write_text(json.dumps({q: {str(k): v for k, v in d.items()} for q, d in second.items()}))
    pp = pn = np_ = nn = 0
    for t in tb:
        for i, a in enumerate(labels[t["id"]]):
            if i in second[t["id"]]:
                b = second[t["id"]][i]
                pp += a and b; pn += a and not b; np_ += (not a) and b; nn += (not a) and not b
    print(f"run1 relevant -> run2 relevant {pp/(pp+pn):.3f}   run1 irrelevant -> run2 relevant {np_/(np_+nn):.3f}   "
          f"(n={pp+pn+np_+nn})")


def full(tags: list[str]) -> None:
    ev = make_evaluator("dev")
    labels = json.loads((DIR / "judge_labels.json").read_text())
    tb = [t for t in load_testbed(set(labels)) if t["n_gold"]]
    masks = {tag: (None if tag == "baseline" else json.loads((DIR / f"keep_{tag}.json").read_text())) for tag in tags}

    def run(job):
        tag, t = job
        m = masks[tag]
        chunks = [u["text"] for n, u in enumerate(t["units"]) if m is None or m[t["id"]][n]]
        if not chunks:
            return tag, t["id"], 0.0, 0.0, 0, False
        try:
            p, r = ev.context_precision_recall(t["question"], chunks, t["n_gold"])
            return tag, t["id"], p, r, len(chunks), False
        except RuntimeError:
            return tag, t["id"], 0.0, 0.0, len(chunks), True

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(run, [(tag, t) for tag in tags for t in tb]))
    for tag in tags:
        rs = [r for r in res if r[0] == tag]
        p, r = st.mean(x[2] for x in rs), st.mean(x[3] for x in rs)
        rc = st.mean(min(1, x[3]) for x in rs)
        print(f"{tag:15} P={p:.3f} R={r:.3f} F1={2*p*r/(p+r+1e-9):.3f} | Rcap={rc:.3f} F1cap={2*p*rc/(p+rc+1e-9):.3f} "
              f"| judge failed {sum(x[5] for x in rs)}/{len(rs)} | median k={st.median(x[4] for x in rs):.0f}")
    (DIR / "full_scores.json").write_text(json.dumps(res))


if __name__ == "__main__":
    consistency() if sys.argv[1] == "consistency" else full(sys.argv[2:])
