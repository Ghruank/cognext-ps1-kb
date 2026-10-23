# Set Selection Math: How Many and Which Passages to Retrieve (PS1)

Scope: calibrated relevance, expected-F1-optimal set selection, conformal guarantees, adaptive k / stopping rules (including TAR), multi-facet coverage, and small-data learning-to-rank. The section ends with a recipe for PS1's metric: per-question context precision = judged-relevant / returned, and context recall = relevant / |gold|. Gold sets hold a median of 81 paragraphs, with a range of 0-352.

Notation used throughout: for one question, candidates are sorted by calibrated probability p_1 >= p_2 >= ... >= p_n, where p_i = P(paragraph i is relevant | q). The returned set is S, k = |S|, and G = |gold|, which is random from the model's point of view.

---

## 1. Expected-F1-optimal set selection given per-item probabilities

### Takeaway
If labels are independent given the per-paragraph probabilities, the set that maximizes expected F-beta is always a top-k prefix by probability (Lewis 1995's "Probability Ranking Principle for F-measure"). The best k can be found exactly in O(n^2) time with Ye et al.'s (2012) algorithm, so a per-query adaptive k falls straight out of calibrated probabilities. Without independence, Dembczynski et al.'s GFM is exact but needs a quadratic number of joint-distribution parameters. A cheap ratio-of-expectations approximation is usually adequate, and Lipton et al. show that the optimal fixed threshold is F*/2, which is low when the attainable F1 is modest.

### Cited Findings
- **Ye, Chai, Lee & Chieu (ICML 2012), "Optimizing F-Measures: A Tale of Two Approaches."** The paper contrasts two approaches:
  - EUM (empirical utility maximization) learns a classifier or threshold that maximizes F on training data.
  - DTA (decision-theoretic approach) learns P(y|x) and then predicts the set with maximum expected F.

  The two are asymptotically equivalent with accurate models and large train and test sets. EUM is more robust to model misspecification. DTA is better for rare classes, small datasets and a common domain-adaptation scenario. — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625); [NUS PDF](https://www.comp.nus.edu.sg/~leews/publications/fscore.pdf)
- **Theorem 9 (PRP for F-measure, Lewis 1995, restated by Ye et al.).** If s* = argmax_s E[F_beta(s, y)] under independent labels, then min{p_i : s*_i = 1} >= max{p_i : s*_i = 0}. So the optimal prediction is the top k for some k in {0..n}, which reduces the candidates from 2^n to n+1. — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625)
- **Exact expected F_beta of the top-k set (Ye et al., Sec. 3.2.1).** Let S_{1:k} = sum_{i<=k} y_i and S_{k+1:n} = the remaining positives. Then:
  `f_{beta;k} = sum_{0<=k1<=k, 0<=k2<=n-k} P(S_{1:k}=k1) P(S_{k+1:n}=k2) (1+beta^2) k1 / (k + beta^2 (k1+k2))`
  P(S_{1:k}=i) is the coefficient of x^i in prod_{j<=k} [p_j x + (1-p_j)], which is a Poisson-binomial distribution. Computing each f_{beta;k} costs O(n^2), so all of them together cost O(n^3) time and O(n) space. — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625)
- **O(n^2) algorithm for rational beta^2 = q/r (Algorithm 1).** It rewrites f_{beta;k} = sum_{k1} (1+beta^{-2}) k1 P(S_{1:n}=k1) s(k, k beta^{-2} + k1), where s(k, a) = sum_{k2} P(S_{k+1:n}=k2)/(a+k2). It uses the recurrence s(k-1, a) = p_k s(k, a+1) + (1-p_k) s(k, a), with s(n, a) = 1/a. The authors warn that the polynomial division is numerically unstable and recommend precomputing all the C[i] in O(n^2) time and space. They report that it handles "tens of thousands of instances." — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625)
- **Prior algorithms (as related by Ye et al.).**
  - Lewis characterized the optimum and gave an O(n) bound that can be very loose.
  - Chai (2005) gave an O(n^3) exact algorithm.
  - Jansche (2007) solved the same problem in O(n^4).
  - Dembczynski et al. (2011) gave an O(n^3) algorithm for the general, non-independent case. They also showed that independence-assuming methods can be suboptimal in some cases, yet empirically are at least as good as methods that do not assume independence.

  — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625)
- **General F-measure Maximizer (Dembczynski, Waegeman, Cheng & Hüllermeier, NIPS 2011).** GFM is exact "regardless of the underlying distribution." It needs only a quadratic number of joint-distribution parameters, namely P(y_i=1, sum y = s) for all i and s, plus P(y=0). — [NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html). The exact parameterization is from my memory of the paper and was not verified on the fetched abstract page.
- **Lipton, Elkan & Narayanaswamy (2014), "Thresholding Classifiers to Maximize F1."** With well-calibrated probabilities, the F1-optimal threshold is half the optimal F1 score. An uninformative classifier maximizes F1 by labeling everything positive. — [arXiv 1402.1892](https://arxiv.org/abs/1402.1892)

### Inferences
- **Our metric is not exactly F1.** The challenge reports precision and recall separately. If the leaderboard combines them as P+R, or F1 of the two means, the same machinery applies with a different utility. For top-k sets with calibrated p_i:
  - Expected precision is exact: E[P_k] = (sum_{i<=k} p_i)/k.
  - Expected recall is approximately E[R_k] = (sum_{i<=k} p_i) / Ĝ, where Ĝ = sum_{i=1}^{N} p_i over all paragraphs in the corpus, or a separately regressed gold count.
  - Evaluate every k with prefix sums in O(n) after an O(n log n) sort, then take the argmax.
  - For a P+R objective, adding item k+1 helps iff p_{k+1} > P_k · Ĝ / (Ĝ + k + 1). This threshold falls toward P_k/2 as k approaches Ĝ, which is consistent with Lipton's F*/2.
- **Use exact DTA for expected F1.** The O(n^2) Ye algorithm on n <= 3000 candidates is about 10^7 operations, trivial per query. A ratio-of-expectations shortcut, F1_k ≈ 2 sum_{i<=k} p_i / (k + Ĝ), is a good first approximation when Ĝ is large. Ĝ is large for the 81-median gold sets, so the law of large numbers applies.
- **Large-G questions.** Because Ĝ appears in the denominator, a large expected gold count automatically pushes k up, toward 300 for the annual-report questions. Small Ĝ shrinks k toward 1-3. This is exactly the bidirectional adaptivity that fixed top-k lacks.
- **Feasibility on 100 training questions.** Ye et al.'s finding that DTA wins for rare classes and under domain adaptation argues for DTA (calibrated p, then optimal k) over EUM (one global tuned threshold) when moving to an unseen corpus. Their finding that EUM is more robust to misspecification argues for keeping a tuned-threshold fallback and comparing the two in cross-validation.
- **Correlated relevance violates independence.** Paragraphs from the same section are correlated. GFM handles this but needs joint estimates we cannot get from 100 questions. Following Dembczynski's empirical note, independence-based DTA is a reasonable default.

### Gaps
- I did not verify GFM's exact complexity statement (O(n^3) per Ye et al.'s citation) on the primary PDF.
- I found no paper that optimizes the exact "judge-marked precision plus count-based recall" metric. The P+R derivation above is my own.

---

## 2. Probability calibration for rankers, classifiers and LLM judges

### Takeaway
Use Platt scaling (2 parameters) or temperature scaling (1 parameter) when calibration data is small. Isotonic regression overfits with limited data. Thousands of (query, paragraph) pairs from 100 questions are enough for Platt or isotonic in pair count. The effective sample size, however, is about 100 queries, because of the per-query correlation. So calibrate with question-grouped cross-validation and prefer low-parameter maps. Raw verbalized LLM confidences are overconfident, but temperature scaling repairs much of the miscalibration.

### Cited Findings
- **Niculescu-Mizil & Caruana (ICML 2005)** compare Platt scaling and isotonic regression and quantify how much data each needs. Isotonic is better when enough data is available, while Platt (a sigmoid fit) is the safer choice for small calibration sets. — [mlanthology](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting); [Wikipedia: Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling)
- A later summary notes that isotonic regression "can suffer from overfitting, especially with limited calibration data." — [HKUST IJCAI'13 PDF](https://cse.hkust.edu.hk/~jamesk/papers/ijcai13a.pdf)
- **Platt scaling formula:** p = 1 / (1 + exp(A·s + B)), with A and B fit by maximum likelihood on held-out scores. — [Wikipedia: Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling)
- **Temperature scaling** (Guo et al. 2017): p = σ(z/T), with a single T fit on validation negative log-likelihood. — [arXiv 1706.04599](https://arxiv.org/abs/1706.04599). This is cited from known literature and was not fetched in this session.
- **Calibrating LLM judges.** For comparative LLM-as-a-judge, choosing the temperature that minimizes ECE "can almost entirely resolve the miscalibration." — [arXiv 2505.15240](https://arxiv.org/pdf/2505.15240)
- **Verbalized LLM-reranker confidence** is severely overconfident. Supervised verbalized methods are proposed to fix it. — [arXiv 2606.03535](https://arxiv.org/pdf/2606.03535)
- **Calibrators for relevance decisions.** A 2026 study compares temperature, spline and other calibrators on reranking datasets and reports large ECE reductions. — [arXiv 2608.10406](https://arxiv.org/pdf/2608.10406). I saw only the snippet, not the full paper.
- **Rank-correct is not calibrated.** Most rerankers are "rank-correct" but not calibrated. — [ZeroEntropy explainer](https://www.zeroentropy.dev/concepts/score-calibration/), a vendor source and informal.

### Inferences
- **Calibrate the target that is actually scored.** The judge's relevant/not-relevant call is the target, not just gold membership. Where possible, label training pairs with the same judge prompt, or treat gold as a proxy and estimate the gold-vs-judge disagreement rate.
- **Fit the calibrator on out-of-fold scores.** Use GroupKFold by question, then fit Platt on the pooled out-of-fold scores. Also check that sum_i p_i tracks the true G per question; this "count calibration" matters for recall. If Σp is biased, fit a 1-D correction, for example a log-linear regression of G on Σp.
- **Use Brier score and per-query count error as model-selection metrics, not only AUC.**

### Gaps
- No primary source fetched gives an exact minimum sample size for isotonic regression. The often-quoted figure of about 1,000 was not verified.

---

## 3. Conformal prediction for retrieval and RAG

### Takeaway
Split conformal gives distribution-free, finite-sample guarantees that hold marginally over exchangeable queries. The guarantees are recall-type: "each relevant passage is kept with probability >= 1-α," or "the output is fully factual with probability >= 1-α." They are not precision guarantees. In practice, conformal filters cut context by roughly 20-58% at α = 0.05-0.20, with flat or slightly better downstream F1. Conformal risk control is the right tool to bound the expected per-question miss rate, 1 - recall, directly.

### Cited Findings
- **Conformal filtering for RAG (Chakraborty et al., arXiv 2511.17908, JHU HLTCOE).**
  - Nonconformity scores: A = 1 - cos(emb(q), emb(s)) with Qwen3-Embedding-8B, or A = 1 - an LLM relevance rating from GPT-4o in [0,1].
  - Threshold: τ̂_α is the ⌈(n+1)(1-α)⌉-th smallest score among the n relevant calibration snippets.
  - Kept set: K_q = {s : A(q,s) <= τ̂_α}.
  - Guarantee: P(s ∈ K_q | r(q,s)=1) >= 1-α under exchangeability.
  - Calibration sets: about 1,440 or 740 snippets (NeuCLIR) and 1,710 or 560 (RAGTIME); the cal/test ordering is ambiguous.
  - NeuCLIR downstream ARGUE F1, with the unfiltered baseline at 0.69:

    | Method | α=0.05 (F1 / context removed) | α=0.10 | α=0.20 |
    |---|---|---|---|
    | Embedding | 0.72 / 22.2% | 0.70 / 35% | 0.68 / 52.8% |
    | LLM rating | 0.71 / 46.5% | 0.70 / 58% | 0.68 / 57.8% |

    F1 gains at α <= 0.10 were significant (p < 0.05).
  - Caveats:
    - Labels came from Llama-3.3-70B with only a 10% manual check.
    - Quantized LLM scores over-cover.
    - Gains are modest.
    - The text and tables are inconsistent on reduction figures, and the abstract claims a "2-3x" reduction.

  — [arXiv 2511.17908v2](https://arxiv.org/html/2511.17908v2)
- **C-RAG (Kang et al., ICML 2024, arXiv 2402.03181).**
  - Certifies an upper confidence bound on "conformal generation risk" for RAG via conformal risk analysis, including under test distribution shift.
  - Proves RAG achieves lower conformal generation risk than a single LLM when retriever and transformer quality are non-trivial.
  - Bounds held empirically on four datasets with four retrievers.
  - This is a risk-certification tool, not a set-construction recipe.

  — [arXiv 2402.03181](https://arxiv.org/html/2402.03181v4)
- **Conformal factuality (Mohri & Hashimoto, ICML 2024, arXiv 2402.10978).**
  - Method: split the output into sub-claims, score each one, and back off by removing claims below threshold t. The per-example score is r(x, y*) = inf{t : for all j >= t, y* ∈ E(F_j(x))}, and q̂ is the ⌈(n+1)(1-α)⌉/n quantile.
  - Guarantee: P(output entailed by truth) >= 1-α, with upper bound 1-α + 1/(n+1) under nestedness.
  - Calibration size: about 25 calibration examples (first 50 inputs, randomly halved 1,000 times).
  - Results: on FActScore, factuality rose from about 25% to about 80% while keeping about half of the sub-claims. On NQ, it rose from about 78% to about 93% while removing about a quarter of the claims.

  — [arXiv 2402.10978](https://arxiv.org/html/2402.10978)
- **TRAQ** applies conformal prediction to both retriever and generator to give a correctness guarantee for RAG QA. — [arXiv 2307.04642](https://arxiv.org/html/2307.04642v2)
- **Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei & Schuster).** For a monotone bounded loss L_i(λ) in [0,B], choose
  λ̂ = inf{λ : (n/(n+1)) R̂_n(λ) + B/(n+1) <= α}.
  This gives E[L_{n+1}(λ̂)] <= α. With L = 1 - recall_q(λ), it bounds the expected per-question miss rate. — [arXiv 2208.02814](https://arxiv.org/abs/2208.02814). This is cited from known literature and was not fetched this session.

### Inferences
- **Per-snippet coverage versus per-question recall.** The per-snippet guarantee pools snippets, so questions with 300 gold paragraphs dominate the calibration quantile. Our recall is averaged per question, so conformal risk control with per-question L = 1 - recall_q is the better-matched guarantee.
- **What n = 100 calibration questions buys.** The B/(n+1) slack is about 0.01, so achievable α ≈ 0.1-0.2 is realistic. Using 100 questions for both training and calibration requires cross-conformal or jackknife+ style splits.
- **Conformal only gives a recall floor.** It controls recall, not precision. Use it as a minimum-k constraint, then let expected-utility optimization (§1) choose a larger k only if that improves the objective.
- **Exchangeability breaks on a new corpus.** The guarantee is void under shift. Treat it as a heuristic on the new corpus, or recalibrate if any labels are available.

### Gaps
- No found paper reports conformal retrieval set sizes when gold sets reach hundreds of passages per query.
- I did not fetch C-RAG's numeric bounds or its retrieval set sizes.

---

## 4. Adaptive k and stopping rules, including TAR / e-discovery

### Takeaway
Score-gap heuristics such as Adaptive-k (largest gap plus a buffer of 5) are training-free and cut tokens 2-10x. However, they recall only about 70% of relevant content and are fragile across embedders and dataset types. TAR stopping rules are designed for near-total recall over long ranked lists, which matches our 300-paragraph questions. The knee rule and the target or quantile rules give principled stops, but only the sample-based rules (Target, QBCB) carry statistical guarantees.

### Cited Findings
- **Adaptive-k (Taguchi, Maekawa & Bhutani, arXiv 2506.08479, 2025).**
  - Rule: sort cosine scores, compute g_i = s_i - s_{i+1}, set k = argmax g_i searched over the top 90% of candidates, then add a buffer B = 5. Cost is O(n log n).
  - Aggregation results (HoloBench): context recall is about 66-76%, against Self-Route falling to 21.5%, with 2-10x fewer tokens than full context.
  - Factoid results: recall with BGE is about 71% on HotpotQA but only about 27% on NQ and 31% on TriviaQA. With GTE, HotpotQA recall drops to about 5.5%.
  - Best-tuned fixed k sometimes wins.
  - The authors say the method is unsuitable for summarization-style queries.

  — [arXiv 2506.08479](https://arxiv.org/html/2506.08479v1)
- **Related adaptive-cutoff methods.**
  - Cluster-based Adaptive Retrieval (CAR) cuts at a cluster boundary in sorted distances and reports about 60% fewer LLM tokens. — [arXiv 2511.14769](https://arxiv.org/html/2511.14769v1)
  - A classifier that predicts the optimal k per query. — [arXiv 2512.14313](https://arxiv.org/html/2512.14313v1)
  - Tail-Aware Adaptive-k uses extreme value theory for truncation. — [arXiv 2606.11907](https://arxiv.org/pdf/2606.11907). I did not verify its content.
- **Knee method (Cormack & Grossman 2016).**
  - Plot cumulative relevant documents found against rank and detect the knee.
  - Stop when the slope ratio ρ (slope before the knee divided by slope after) is at least 6, with bound_t = 156 - min(relevant found, 150).
  - The rule is target-agnostic and overshoots at low recall targets.
  - Its effectiveness depends heavily on ρ.

  — [Heuristic Stopping Rules for TAR, arXiv 2106.09871](https://arxiv.org/pdf/2106.09871); [Cormack & Grossman 2016 PDF](https://plg.uwaterloo.ca/~gvcormac/reliability/cormackgrossman16.pdf)
- **Target method (Cormack & Grossman).** Draw a random sample until 10 relevant documents are found (the "target set"), then review until all targets are found. This guarantees recall >= 0.70 with 95% confidence. A later analysis says 9 targets suffice, and notes the rule allows no flexibility in recall goal or confidence. — [Lewis, Yang & Frieder, "Certifying One-Phase TAR" (CIKM 2021), arXiv 2108.12746](https://arxiv.org/pdf/2108.12746)
- **QBCB (Quantile Binomial Coefficient Bound).**
  - Uses a labeled control set and quantile estimation to set how many control positives must be found before stopping.
  - It is statistically valid for any recall target and confidence level.
  - Overshooting the recall target is "a major source of excess cost."

  — [arXiv 2108.12746](https://arxiv.org/pdf/2108.12746)
- **Other stopping approaches.**
  - Point-process stopping. — [arXiv 2311.08597](https://arxiv.org/pdf/2311.08597)
  - Chao's estimator stopping. — [arXiv 2404.01176](https://arxiv.org/pdf/2404.01176)
  - Decision-theoretic stopping for document screening. — [arXiv 2606.07071](https://arxiv.org/pdf/2606.07071). I did not read it.

### Inferences
- **What carries over from TAR.** TAR's core idea is to estimate how many relevant items remain from the score distribution, then stop when the expected marginal gain falls below cost. In our setting with calibrated p this reduces to §1: the expected remaining relevant count is Σ_{i>k} p_i. The knee rule is a useful label-free fallback when calibration on a new corpus is untrustworthy.
- **Gap rules are risky for the 300-paragraph questions.** A sharp gap often appears after the first few near-duplicates, so gap rules alone would truncate badly there. Use the gap as a feature for the k-model, not as the decision rule.

### Gaps
- I did not find TAR results that use LLM-judged relevance or paragraph-level units.
- I did not find empirical comparisons of Adaptive-k against calibrated expected-F1 selection.

---

## 5. Multi-facet coverage: submodular selection

### Takeaway
Monotone submodular objectives that pair relevance or coverage with diversity can be maximized greedily with a (1 - 1/e) guarantee. Facet coverage, meaning one evidence passage per question facet, fits a weighted-coverage or facility-location objective. For our metric, though, diversity should only shape the order and guarantee a minimum per facet. Count-based recall rewards returning all relevant near-duplicates, not removing them.

### Cited Findings
- **Lin & Bilmes (ACL 2011)** propose monotone submodular summarization objectives F(S) = L(S) + λR(S), a coverage term plus a diversity reward. A scalable greedy algorithm has a constant-factor optimality guarantee. They achieved state-of-the-art results on DUC 2004-2007, both generic and query-focused. — [ACL Anthology P11-1052](https://aclanthology.org/P11-1052/)
- **Coverage function:** L(S) = Σ_i min(C_i(S), α C_i(V)), where C_i(S) = Σ_{j∈S} w_ij. Diversity function: R(S) = Σ_k sqrt(Σ_{j∈P_k∩S} r_j) over clusters P_k. — [ACL Anthology P11-1052](https://aclanthology.org/P11-1052/). These formulas come from my knowledge of the paper; the fetched snippet confirmed only the coverage-plus-diversity structure.
- **Greedy guarantee.** Greedy maximization of a monotone submodular function under a cardinality constraint achieves at least (1-1/e) of the optimum (Nemhauser, Wolsey & Fisher 1978). — [Lin & Bilmes slides](https://people.ece.uw.edu/bilmes/p/mypubs/lin2011-class-submod-sum.slides.pdf)
- **MMR (Carbonell & Goldstein 1998):** select argmax_{d∉S} [λ sim(d,q) - (1-λ) max_{d'∈S} sim(d,d')]. This is cited from known literature and was not fetched.

### Inferences
- **Facet decomposition for PS1.**
  - Split the question into facets f_1..f_m, using an LLM or rules for entity, time and metric.
  - Score p_{i,f} for each paragraph and facet.
  - Use the objective Σ_f w_f · (1 - Π_{i∈S} (1 - p_{i,f})), the probabilistic coverage: the expected number of facets with at least one relevant passage. It is monotone submodular, so greedy gives (1-1/e).
  - Use it to guarantee at least one high-p passage per facet, then fill the rest by §1 expected utility.
- **Do not penalize redundancy.** Precision and recall are counted per relevant paragraph, so redundancy is not penalized by the metric. MMR's diversity penalty would hurt recall on 300-paragraph questions.

### Gaps
- I did not fetch recent (2024-2026) facet-aware RAG papers, such as nugget-coverage retrieval. I have no empirical numbers for facet-coverage retrieval under LLM-judged metrics.

---

## 6. Learning-to-rank and relevance classification with small labeled sets

### Takeaway
With 100 questions, a heavily regularized gradient-boosted classifier or ranker (LightGBM) over a small set of corpus-agnostic features is appropriate. Query-relative, rank-normalized features generalize to unseen corpora far better than lexical identity features. Because we need calibrated probabilities, a binary objective (or LambdaRank followed by Platt) is preferable, validated with question-grouped CV.

### Cited Findings
- **LightGBM lambdarank parameters.**
  - `lambdarank_truncation_level` (default 30) sets how many top results training focuses on. It should be slightly above the NDCG@k cutoff.
  - `lambdarank_norm` (default true) normalizes lambdas per query, which helps unbalanced queries.
  - `label_gain` defaults to 2^i - 1.
  - `min_data_in_leaf` defaults to 20 and `num_leaves` to 31.
  - `rank_xendcg` is faster than lambdarank with similar quality.

  — [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- **Truncation level versus our sets.** Lambdarank's default focus on the top 30 conflicts with sets of 81-300 paragraphs. It must be raised, or a pointwise binary objective used instead, if the full list matters. — [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- **Ye et al.:** DTA with a good probabilistic model is better under domain adaptation, a relevant point for an unseen corpus. — [arXiv 1206.4625](https://arxiv.org/abs/1206.4625)

### Inferences
These are my recommendations and are not sourced.
- **Features expected to generalize:**
  1. Raw scores from bi-encoder, BM25 and an LLM or cross-encoder judge (banned as the core design, but usable as a feature).
  2. Within-query normalized versions of each score: z-score, rank, percentile, and gap from the top-1 score.
  3. Document-level aggregates: the max and mean paragraph score in the same document, and the paragraph's rank within its document.
  4. Neighbor scores (i-1, i+1 paragraphs), which capture section contiguity in annual reports.
  5. Query-type features: predicted answer cardinality ("list all", "summarize", numeric lookup) and query length.
  6. Structural features: heading-match flags, table/number density, and position in the document.
- **Features to avoid:** lexical token identities, document IDs and corpus-specific term statistics.
- **Regularization:** num_leaves 7-15, min_data_in_leaf >= 50, feature_fraction 0.7, early stopping on GroupKFold(5) by question, and monotone constraints on score features (LightGBM `monotone_constraints`).

### Gaps
- I did not find a primary study quantifying LambdaMART generalization from 100 queries to a new corpus.

---

## Recommended recipe for our system
This section is my synthesis, built on the cited pieces above.

**Step 0. Candidate pool sized for recall.** For each question, route to the top documents (Ĝ can be about 300 from one report). Then score every paragraph in those documents, not a fixed top-k. n of about 1-5k paragraphs per query is fine for every step below.

**Step 1. Relevance model.**
- Model: a LightGBM binary classifier on the generalizing features in §6, trained on all labeled (q, paragraph) pairs from the 100 questions.
- Labels: gold membership. Ideally relabel a sample with the actual judge prompt to align with judge-marked relevance.
- Validation: GroupKFold(5) by question; never split one question's paragraphs across folds.
- Regularization: small trees, monotone constraints on similarity features, early stopping on fold log-loss.

**Step 2. Calibration.**
- Fit Platt scaling on out-of-fold scores pooled across folds. Use isotonic only if the reliability diagram shows clear non-sigmoid shape and out-of-fold Brier improves.
- Check count calibration: regress the true G_q on Σ_i p_i,q, and apply a linear or log correction if biased.
- On a new corpus, refit only the 1-2 Platt parameters if any labeled questions exist. Otherwise keep the parameters, which rely on the rank-normalized features to transfer.

**Step 3. Choose k per question by expected utility (§1).**
```
sort candidates by p desc; prefix S_k = Σ_{i<=k} p_i ; Ĝ = corrected Σ_all p_i
for k in 1..n:
    EP_k = S_k / k
    ER_k = S_k / max(Ĝ, ε)
    U_k  = objective(EP_k, ER_k)   # P+R, harmonic mean, or exact Ye O(n^2) E[F1]
k* = argmax_k U_k
```
- If the official aggregate is F1 per question, use the exact Ye et al. DTA (O(n^2)).
- If Ĝ < 0.5 (likely a zero-gold question), return a minimal set of 1-2 items, since recall is undefined there and precision dominates.

**Step 4. Recall floor via conformal risk control (§3).**
- On the out-of-fold predictions for the 100 questions, choose the probability threshold λ̂ such that (n/(n+1)) · mean_q(1 - recall_q(λ)) + 1/(n+1) <= α, with α ≈ 0.2.
- Final set = top-k* ∪ {i : p_i >= λ̂}.
- Take the union only if the cross-validated objective improves. Otherwise report the floor as a diagnostic.

**Step 5. Facet guarantee (§5).**
- Decompose the question into facets and greedily maximize probabilistic facet coverage for the first few slots, so every facet gets at least one passage with p_{i,f} above its median.
- Then fill the rest by Step 3. Do not apply MMR redundancy penalties.

**Step 6. Label-free fallback for a new corpus (§4).**
- If calibration diagnostics fail (the mean predicted p differs wildly from training), switch to the TAR knee rule on the cumulative Σp curve, or use the Adaptive-k largest gap as a sanity bound.
- Log both choices and compare them.

**Avoiding overfitting on 100 questions:**
- Report nested CV: the outer GroupKFold for evaluation, the inner one for hyperparameters and calibration.
- Stratify the folds by gold-set size bucket (0, 1-10, 11-100, >100) so that every fold contains the 300-paragraph cases.
- Keep fewer than about 30 features.
- Prefer DTA, which has few tunable decision parameters, over tuning a global threshold. Ye et al. show DTA is better under domain adaptation.

**Generalizing to a new corpus:**
- Use only query-relative or rank-normalized features plus document-structure features.
- Recheck Σp-versus-G count calibration on any available labeled questions from the new corpus.
- Keep the conformal guarantees as heuristics until recalibrated, because exchangeability does not hold across corpora.
