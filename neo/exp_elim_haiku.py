"""Score Claude Haiku elimination (agent outputs in storage/elim/haiku/<id>.out.json) on the judge labels.

    python -m neo.exp_elim_haiku
"""

import json
import statistics as st

from neo.eliminate import structural_reason
from neo.exp_elim import DIR, load_testbed

H = DIR / "haiku"


def masks() -> dict[str, list[bool]]:
    out = {}
    for t in load_testbed():
        f = H / f"{t['id']}.out.json"
        if not f.exists():
            continue
        drop = {int(i) for i in json.loads(f.read_text()).get("irrelevant_ids", [])}
        out[t["id"]] = [structural_reason(u) is None and k not in drop for k, u in enumerate(t["units"])]
    (DIR / "keep_haiku_strict.json").write_text(json.dumps(out))
    return out


def table(variants: dict) -> None:
    labels = json.loads((DIR / "judge_labels.json").read_text())
    tb = [t for t in load_testbed(set(labels)) if t["n_gold"] and all(t["id"] in m for m in variants.values() if m)]
    print(f"{len(tb)} questions (deepseek judge labels, batches of 40)\n")
    for name, m in variants.items():
        P, R, Rc, ret, K = [], [], [], [], []
        for t in tb:
            y = labels[t["id"]]
            keep = [True] * len(y) if m is None else m[t["id"]]
            k = [a for a, b in zip(y, keep) if b]
            P.append(sum(k) / len(k) if k else 0)
            R.append(sum(k) / t["n_gold"])
            Rc.append(min(1, R[-1]))
            ret.append(sum(k) / sum(y) if sum(y) else 1)
            K.append(len(k))
        p, r, rc = st.mean(P), st.mean(R), st.mean(Rc)
        print(f"{name:16} P={p:.3f} R={r:.3f} F1={2*p*r/(p+r+1e-9):.3f} | Rcap={rc:.3f} F1cap={2*p*rc/(p+rc+1e-9):.3f} "
              f"| judge-rel kept={st.mean(ret):.3f} | median k={st.median(K):.0f}")


if __name__ == "__main__":
    m = masks()
    v = {"baseline": None}
    for tag in ("e1_deepseek", "e1_lenient_v2"):
        v[tag] = json.loads((DIR / f"keep_{tag}.json").read_text())
    v["haiku_strict"] = m
    table(v)
