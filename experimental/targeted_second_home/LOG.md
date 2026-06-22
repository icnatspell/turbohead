# targeted_second_home — run log

## 2026-06-20 — bake-off vs the shipped always-score lever. Always-score wins.

Built the targeted-second-home idea as a head-to-head against `recall_lift` lever 4 (always-score,
the shipped win), not against a blanket r=2. Same most-missed token set (fit on the train half,
`turbohead-calibrate-misses`-style), same held-out eval, same deploy P=256. The two mechanisms differ
only in HOW they rescue those tokens.

- **always-score:** score the token's row every step (rides the Wspec path). Flat N rows/step.
  Unconditional rescue: in the set ⇒ always caught.
- **targeted second home:** add the token to its next-best cluster, reachable through a second route.
  Amortized cost: its row is gathered only on steps where that second cluster is in the top-P.
  Conditional rescue: caught only when the second cluster is probed.

12000 positions (6000 fit / 6000 eval), Qwen3-0.6B, baseline agree@256 = 97.47%, 152 eval misses.

| N    | always rescue | always agree | always rows/step | 2nd-home rescue | 2nd-home agree | 2nd-home rows/step |
|------|---------------|--------------|------------------|-----------------|----------------|--------------------|
| 64   | 52.0%         | 98.78%       | 64               | 43.4%           | 98.57%         | 7.4                |
| 256  | 53.3%         | 98.82%       | 256              | 44.7%           | 98.60%         | 8.5                |
| 1024 | 53.3%         | 98.82%       | 1024             | 44.7%           | 98.60%         | 8.5                |
| 4096 | 53.3%         | 98.82%       | 4096             | 44.7%           | 98.60%         | 8.5                |

**Finding. Targeted second home works but is dominated by always-score.** It captures ~84% of
always-score's lift (98.57% vs 98.78% over a 97.47% base) at ~1/9 the extra rows (7.4 vs 64), so per
extra row it is ~7× more efficient. But:

1. **The cost it saves does not exist.** Always-score's 64 rows/step ride on top of P·cap = 4096
   candidate rows already gathered — about 1.6% more stage-2 work, and through the existing Wspec path
   with zero graph change. Saving 56 of those rows does not move the decode clock.
2. **It gives up agreement and can't catch up.** Second home rescues conditionally (only when the
   second cluster is probed), so it sits 0.22pp below always-score at every N and can never reach its
   ceiling.
3. **Its one real edge is moot.** Second home would win if you needed a LARGE rescue set, where
   always-score's flat N becomes a real cost. But the lift plateaus at N=64: about half the misses are
   frequent tokens a tiny list catches, the rest are idiosyncratic one-offs no fixed list reaches
   (same plateau the lever-4 study found). So a large set is never wanted, and the edge never applies.
4. **It costs build complexity.** Variable cluster size breaks the equal-`cap` kernel assumption (the
   fused op gathers rows with plain arithmetic because every cluster has exactly `cap` members), and
   it needs dedup when a token lands in two probed clusters. Always-score needs none of that.

**Verdict: PARKED.** Always-score already ships, is already ~free, and gives strictly higher
agreement for this workload. Targeted second home loses the only contest that matters.

### When it could come back
- A workload whose miss tail does NOT plateau (many distinct frequent misses), where always-score's
  flat N becomes a real per-step cost. Not Qwen3-0.6B on WikiText.
- If the fused op ever moves to variable-`cap` clusters for another reason, the build complexity
  objection (point 4) goes away and this becomes a cheaper way to spend that capacity than r=2.
