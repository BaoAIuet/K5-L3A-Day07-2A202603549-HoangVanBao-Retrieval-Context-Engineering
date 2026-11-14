# Submission results

## Attribution and license

Retrieved excerpts in the traces are from *Principles of Management*, OpenStax,
Rice University, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The book's [license and attribution notice](https://openstax.org/books/principles-management/pages/preface)
permits non-commercial redistribution with attribution and the same license.
The source URL and page are preserved in each trace. This results bundle is
shared non-commercially under CC BY-NC-SA 4.0 to the extent it contains adapted
OpenStax material.

## Contents

- `retrieval/`: per-query ranking, final context, evidence labels, and aggregate
  metrics for TF-IDF, dense Day 6, and hybrid RRF runs on dev/test.
- `agent-dense-{dev,test}/`: dense baseline cases, full tool-call traces,
  manually reviewed citations/answers, metrics, and run metadata.
- `agent-hybrid-{dev,test}/`: selected hybrid RRF run with the same per-query
  artifacts, manually reviewed labels, metrics, token usage, elapsed time, and
  estimated API cost.

Agent traces include the user question, each search query, ranked chunks,
context sent to the model, final answer, citations, status, elapsed time, and
token usage and estimated cost per query. `reviews.json` records human grades
and rationale; `metrics.json` is calculated from those labels.

## Reproducibility and cost

The local Day 6 index and exported chunk JSON are not included. Recreate them
with the Day 6 environment using the instructions in
[`starter/README.md`](../../starter/README.md), then run `agent_runner.py`.
Configure `OPENAI_API_KEY` only in a local, ignored `.env`.

The selected hybrid agent runs used `gpt-4o-mini`, Dense Day 6 strategy B plus
TF-IDF, `candidate_k=10`, `rrf_k=60`, `top_k=5`, up to 3 searches, and a
1,600-character context budget. Standard pricing as of 2026-11-14 used for estimates was
$0.15/1M uncached input tokens, $0.075/1M cached input tokens, and $0.60/1M
output tokens, as listed in the [OpenAI API pricing documentation](https://developers.openai.com/api/docs/pricing/).

The two dense baseline runs are estimated at $0.00526875 total, assuming no
cached input because cached-token details were not saved for those runs. The
two hybrid runs are estimated at $0.00545310 total; cached input was reported
as zero. Combined estimated cost for these four full-split runs is $0.01072185.
These are standard-rate estimates, not account invoices; actual billing may
differ. Smoke-test usage is excluded.
