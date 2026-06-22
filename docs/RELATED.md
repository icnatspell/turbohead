# Related methods

How FlashHead/TurboHead sits next to other ways of speeding up a large language-model head, and
which neighbors are worth comparing against or borrowing from. Written so a junior ML engineer
with no prior exposure to these methods can follow it: every technique gets a one or two sentence
plain-language explanation before we relate it to our work.

## What "the head" is and why it is slow

A language model ends with a **head** (also called `lm_head`). It takes the model's hidden state,
a vector `h` of length `D`, and multiplies it by the output embedding matrix, which has one row per
vocabulary token (`V` rows, each of length `D`). The result is one score per token, called a
**logit**. The next token is the highest-scoring one (an `argmax`), or a random draw from
`softmax(logits)`.

`V` is huge (about 152,000 for Qwen3-0.6B), so this single matrix-vector multiply reads `V·D`
weights every time the model emits a token. At decode time the model emits one token per step, so
the head is reading a giant matrix to produce one number per token. The work is limited by reading
those weights from memory, not by the arithmetic, so the only way to make it faster is to **read
fewer bytes**. Every method on this page does that somehow.

## Terms used throughout

- **Logit**: a token's raw score before softmax. It equals that token's embedding row dotted with
  `h`. "Finding the best token" is "finding the embedding row with the largest dot product".
- **top-1 / recall@1 / agreement**: did the fast approximate method return the *same single best
  token* as the exact full matrix multiply? This is our headline quality metric. 97% means it
  agrees with the exact head 97 times out of 100.
- **Candidate count**: how many token rows the fast method actually scores exactly. Fewer scored
  rows means less work but a higher chance of missing the true best token. Quality-versus-cost
  curves on this page plot recall@1 against candidate count (or against dot-products per query).
- **ANN / MIPS**: Approximate Nearest Neighbor search finds the vectors closest to a query
  quickly, accepting a small error rate in exchange for speed. Maximum Inner Product Search is the
  version where "closest" means "largest dot product". Since a logit *is* a dot product, picking
  the top token is exactly an MIPS problem, and the ANN/MIPS field has ready-made tools for it.

## The framing: FlashHead is an IVF index on the head

An **inverted file index (IVF)** is a classic fast-search structure. Offline, you group all the
items (here, the `V` token embedding rows) into clusters. At query time you first compare the
query against the small set of cluster centers, pick the few nearest clusters, and only scan the
items inside those clusters instead of all `V`. The two pieces have standard names: the **coarse
quantizer** is the set of cluster centers you compare against first, and the **inverted lists**
are the per-cluster lists of items.

FlashHead is exactly this, applied to the head:

| IVF concept | FlashHead piece |
|---|---|
| coarse quantizer | the `K` cluster centroids (stage 1) |
| inverted lists | the `cap`-token clusters |
| list scan / re-rank | stage 2, exact dot over the `P·cap` shortlist |

In words: stage 1 scores `h` against `K` cluster centroids and keeps the top `P` clusters; stage 2
takes the `P·cap` tokens in those clusters and computes their exact logits, then returns the best.
This framing matters because it places FlashHead inside the ANN/MIPS literature, so that field's
standard upgrades (graph indices, product quantization, both explained below) become candidate
improvements with a clear home in our cost model. See `IDEAS.md` for the ones we have prototyped.

## Direct structural cousins (cheap shortlist, then exact refine)

These share FlashHead's shape: a cheap approximate first pass picks a small candidate set, then
exact logits are computed only for that set. Because the final scores are exact, they preserve
top-1, which makes them the fair comparisons. (Methods that approximate the final logit itself
instead lose top-1; they are the "weaker baselines" at the end.)

- **SVD-Softmax** (Shim et al., NeurIPS 2017). *What it is:* SVD (singular value decomposition) is
  the standard way to approximate a big matrix by the product of two thin ones, which is cheap to
  multiply. SVD-Softmax uses that cheap low-rank product only to *guess* which tokens are likely
  winners, then computes exact logits for just those few. *Relation to us:* this is the low-rank
  analog of our clustering router, and the honest form of "low-rank" because it approximates only
  the shortlisting step, not the final score. It is a fairer low-rank baseline than a fully
  low-rank head. Compare on top-1 versus candidate count.

- **L2S: Learning to Screen** (Chen et al., ICLR 2019). *What it is:* it groups the *context*
  vectors `h` (not the token embeddings) into clusters, and for each context cluster it stores,
  from training data, the set of tokens that actually turned out to be the answer for hidden states
  in that cluster. At inference it routes `h` to its context cluster and scores only that stored
  candidate set. *Relation to us:* almost our method, except it *learns* which tokens to consider
  from data, rather than grouping token embeddings by geometry. That learned routing is what our
  cheap data-aware routing prototype failed to beat (`logs/dataaware_routing_poc.py`); L2S is what
  doing it properly looks like, and on paper the method most likely to beat fixed clustering on
  coverage.

  **POC result: coverage-bound at our data scale, lost to flat IVF.** See `logs/l2s_poc.py`
  (group the context vectors into `G` clusters, candidate set per cluster = the union of observed
  best-tokens, plus a top-`F` most-frequent-tokens backstop; fit on 10,000 positions, evaluate on
  2,000).

  | method | candidates/query | recall@1 |
  |---|---|---|
  | flat IVF (FlashHead) | 1024 | **94.4%** |
  | L2S, G=64, F=1024 | 1035 | 77.5% |
  | L2S, G=256, F=0 | 19 | 62.2% |

  In every L2S row `recall@1` equals the candidate-set coverage exactly: once the true token is in
  the stored set, scoring finds it, so the stored set itself is the wall. Ten thousand fit
  positions cover only 62 to 78% of held-out true tokens, so L2S loses badly per candidate. The
  cause is data scale, not a flaw in the idea (the original paper calibrates on far more text), but
  it exposes the structural edge of geometry-based IVF: every token sits in some cluster and is
  therefore reachable, while a learned candidate set can only reach tokens it happened to observe
  during fitting. A fair test needs a large calibration corpus, and even then L2S has to beat that
  built-in full-vocabulary coverage. Lower priority than the graph index below.

## Index-structure upgrades from ANN/MIPS

We already run IVF, so the ANN field's two standard improvements map straight onto our two knobs:
recall versus number of candidates, and bytes read per candidate.

- **Graph indices (HNSW; FGD "Fast Graph Decoding," Zhang et al., NeurIPS 2018).** *What it is:*
  HNSW (Hierarchical Navigable Small World) builds a graph that links each item to a handful of
  nearby items. To search, you start at some node and greedily walk toward the query, hopping to
  whichever neighbor is closer, visiting only a tiny fraction of all items. Two terms from the
  experiment: `efSearch` is how wide a frontier the walk keeps (larger means more accurate and
  slower), and `ndis` is the number of distance computations the walk performed, which is the
  cost. *Relation to us:* a graph reaches a given recall while scoring fewer candidates than flat
  clustering, which is exactly our recall-versus-candidate trade-off. It also connects to the
  parked hierarchical idea: a graph is the kind of near-lossless coarse navigation that idea needed
  (`IDEAS.md` #1).

  **POC result: promising on the algorithmic axis, conditional on hardware.** See
  `logs/graph_mips_poc.py` (FAISS `IndexHNSWFlat` with inner-product distance, built over the
  V=151,936 token rows; cost measured as `ndis`, the dot-products per query that FAISS reports). At
  matched top-1 recall, HNSW computes about 3x fewer dot products than flat IVF:

  | recall@1 | flat IVF (dots/query) | HNSW (dots/query) | ratio |
  |---|---|---|---|
  | ~88% | 10520 (P=64) | 2115 (ef=128) | ~5x |
  | ~97% | 13592 (P=256) | 3693 (ef=256) | ~3.7x |
  | ~98% | 17688 (P=512) | 6770 (ef=512) | ~2.6x |

  IVF always pays a fixed `K=9496` dot-products in stage 1 no matter how small `P` is, because it
  scores every cluster centroid; the graph has no such floor and navigates straight toward the
  answer, so it wins most at low recall and the gap narrows as recall nears 100%. The catch is that
  a dot product is **not** a fixed-cost unit on our deployment. IVF's stage 1 is one dense,
  contiguous int4 matrix multiply (`MatMulNBits` is ONNX Runtime's kernel for multiplying by
  4-bit-quantized weights, dequantizing on the fly; it is the kernel that makes FlashHead fast on
  CPU). HNSW's dot products instead jump around memory following graph links (random access),
  run in fp32, and get no 4-bit speedup. So the 3x fewer-dots advantage will not become a 3x
  wall-clock speedup on this CPU and could even lose. The promise is real but conditional: it pays
  off where `V` is very large, on hardware that does not punish random memory access (GPU), or if
  the graph's vectors are themselves quantized. Pursue it as the recall-per-candidate lever, not as
  a drop-in CPU speedup.

- **Product quantization** (Jégou et al., 2011). *What it is:* compress each vector by splitting it
  into chunks and replacing each chunk with the id of the nearest entry in a small prototype
  table, so scoring reads far fewer bytes per token. *Relation to us:* this shrinks the bytes read
  in stage 2. It is `IDEAS.md` #2, and on CPU it stacks with the int4 stage 1.

## The ancestor

- **Adaptive softmax** (Grave et al., ICML 2017). *What it is:* it splits the vocabulary by word
  frequency. Common words live in a fast full-size path; rare words sit in smaller secondary groups
  that are only computed when needed. *Relation to us:* the same skeleton as FlashHead, but it
  routes by token *frequency* rather than embedding *similarity*. It is the natural baseline to
  cite, and a reminder that our coverage hole (rare true tokens landing in clusters we did not
  probe, see `IDEAS.md` #4) is the exact problem its frequency tiering was built to handle.

## Orthogonal, composable, not competitors

These change the byte width or the timing of the head, not its retrieval structure, so they stack
with FlashHead rather than replace it.

- **Quantization.** *What it is:* store weights in fewer bits (for example int8 or int4) instead
  of 16- or 32-bit floats, which directly cuts the bytes read. *Relation to us:* our dense
  baselines (`head16`, `head8g128`, `head4g128`, `head4g32`) and the int4 stage-1 router are this
  lever applied to the head.
- **Speculative decoding** (EAGLE-3, Medusa). *What it is:* a small fast "draft" model proposes
  several next tokens, and the big model checks them all in a single pass, keeping the ones it
  agrees with. This produces several tokens per big-model step instead of one. *Relation to us:* it
  changes *when* the head runs, not how fast one head call is, and because the big model now runs
  its head on several drafted positions per step, the head's share of the work goes up, which makes
  FlashHead's saving matter more. The catch is the acceptance check, which needs the big model's
  probability for each drafted token; our coverage correction supplies exactly that at exact
  quality (`IDEAS.md` #4, `THESIS_ADAPTIVE_PROBING.md`).

## Weaker baselines (already outperformed on top-1)

- **Vocabulary pruning / contextual vocabulary selection.** *What it is:* decide up front that only
  a subset of tokens are possible for the current context and ignore the rest. *Why it is weaker:*
  a token wrongly dropped early can never be recovered, so top-1 suffers.
- **Pure low-rank / factorized heads.** *What it is:* replace the head matrix with a low-rank
  approximation and use its approximate scores directly. *Why it is weaker:* it approximates the
  final logit itself, which caps top-1 accuracy. SVD-Softmax above is the stronger version that
  uses low-rank only to shortlist and then scores exactly.

## What to compare against

For related work and a likely improvement axis, **L2S** (learned routing) and **HNSW/graph MIPS**
(better recall per candidate) are the two worth real attention. Both attack the one metric where
FlashHead has headroom, top-1 at small candidate counts, which is also where the adaptive-probing
thesis lives (`THESIS_ADAPTIVE_PROBING.md`). SVD-Softmax and adaptive softmax are the baselines to
cite for completeness. Of the two, the graph index is the one with a measured upside in our own
prototype; L2S was coverage-bound at the data scale we tested.
