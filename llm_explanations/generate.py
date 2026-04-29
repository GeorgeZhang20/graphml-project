"""Generate per-node LLM explanations + top-k label predictions for a TAG.

Output contract (matches what TAPE expects, see TAPE/core/data_utils/load.py):
  TAPE/gpt_responses/<DATASET>/<i>.json   one file per node, OpenAI chat-completion shape:
                                           {"choices": [{"message": {"content": "<JSON string>"}}]}
                                           where the inner content is JSON with keys
                                           "top5_labels" (list[str]) and "explanation" (str).
  TAPE/gpt_preds/<DATASET>.csv             one row per node, comma-separated label IDs (top-k).

Resumable: skips nodes whose JSON already exists. Re-running generate.py picks
up where it left off, so you can interrupt freely (or run seed 0 first then
resume for seeds 1-3).

Usage:
  export OPENAI_API_KEY=sk-...
  python llm_explanations/generate.py \
      --dataset goodreads_children \
      --node_texts new_dataset/data/goodreads_children/node_texts.jsonl \
      --labels    new_dataset/data/goodreads_children/labels.txt \
      --model     gpt-4o-mini

Smoke test (5 nodes, no API call needed if --dry_run):
  python llm_explanations/generate.py ... --max_nodes 5 --dry_run
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROMPT_TEMPLATE = """\
You are a careful expert classifier for {domain}. Given the text describing a single \
{node_kind}, predict the most likely categories and explain your reasoning.

Categories (use these exact labels):
{label_list}

{node_kind} text:
\"\"\"
{text}
\"\"\"

Respond strictly as a JSON object with these keys:
- "top5_labels": list of up to 5 categories ranked from most to least likely (each \
must be one of the categories above, copied verbatim).
- "explanation": 3-6 sentences of reasoning that cite concrete evidence from the text.
"""

# Per-dataset domain/node-kind hints. Add new datasets here when extending.
DATASET_HINTS = {
    "goodreads_children": dict(
        domain="children's books on Goodreads",
        node_kind="book",
    ),
    "ogbn-arxiv": dict(
        domain="computer science research",
        node_kind="paper",
    ),
    "cora": dict(
        domain="computer science research",
        node_kind="paper",
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--node_texts", required=True, type=Path,
                   help="path to a .jsonl file of {id, text} per node")
    p.add_argument("--labels", required=True, type=Path,
                   help="path to a .txt file with one label per line")
    p.add_argument("--responses_dir", type=Path, default=None,
                   help="default: TAPE/gpt_responses/<DATASET>/")
    p.add_argument("--preds_csv", type=Path, default=None,
                   help="default: TAPE/gpt_preds/<DATASET>.csv")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_nodes", type=int, default=None,
                   help="cap for dry runs / cost estimates")
    p.add_argument("--num_workers", type=int, default=8,
                   help="thread pool size for concurrent API calls")
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--dry_run", action="store_true",
                   help="don't call the API; write fake responses for plumbing tests")
    p.add_argument("--start", type=int, default=0,
                   help="start index (inclusive) for partial runs")
    p.add_argument("--end", type=int, default=None,
                   help="end index (exclusive) for partial runs")
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_node_texts(path: Path) -> list[str]:
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows.append((row["id"], row["text"]))
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    if [r[0] for r in rows] != list(range(n)):
        sys.exit(f"node_texts.jsonl ids are not contiguous 0..N-1 in {path}")
    return [r[1] for r in rows]


def load_labels(path: Path) -> list[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def labels_to_index(labels: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(labels)}


def build_prompt(text: str, labels: list[str], dataset: str) -> str:
    hint = DATASET_HINTS.get(dataset, dict(domain="this domain", node_kind="item"))
    return PROMPT_TEMPLATE.format(
        domain=hint["domain"],
        node_kind=hint["node_kind"],
        label_list="\n".join(f"- {lbl}" for lbl in labels),
        text=text[:8000],  # hard cap on prompt body to bound cost; LLM rarely needs more
    )


def call_openai_with_retry(client, model: str, prompt: str, max_retries: int) -> dict:
    """Returns the raw OpenAI chat completion response as a plain dict."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return resp.model_dump()
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


def fake_response(prompt: str, labels: list[str], top_k: int) -> dict:
    """Mimic the OpenAI shape so downstream code works in --dry_run mode."""
    content_obj = {
        "top5_labels": labels[:top_k],
        "explanation": "[dry_run placeholder] No LLM call was made; this text exists "
                       "only to exercise the file-writing pipeline.",
    }
    return {
        "id": "fake-" + str(abs(hash(prompt)) % 10**8),
        "choices": [{"message": {"role": "assistant", "content": json.dumps(content_obj)}}],
        "model": "dry-run",
    }


def parse_top_k_ids(resp: dict, label_to_idx: dict[str, int], top_k: int) -> list[int]:
    """Extract top-k label IDs from the LLM response, robust to label-name typos."""
    try:
        content_str = resp["choices"][0]["message"]["content"]
        content = json.loads(content_str)
        names = content.get("top5_labels") or content.get("top_labels") or []
    except (KeyError, json.JSONDecodeError, IndexError):
        names = []

    ids = []
    seen = set()
    for name in names:
        # exact match first
        if name in label_to_idx:
            i = label_to_idx[name]
        else:
            # case-insensitive fallback
            lower = {k.lower(): v for k, v in label_to_idx.items()}
            i = lower.get(str(name).lower(), -1)
        if i >= 0 and i not in seen:
            ids.append(i)
            seen.add(i)
        if len(ids) >= top_k:
            break

    # backfill if the LLM gave us fewer than k *valid* labels. If the entire
    # label space has fewer than top_k classes we just return what we have
    # (CSV row will be shorter, which TAPE's load_gpt_preds handles via padding).
    n_classes = len(label_to_idx)
    if len(ids) < top_k:
        for fallback in range(n_classes):
            if fallback not in seen:
                ids.append(fallback)
                seen.add(fallback)
            if len(ids) >= top_k or len(seen) >= n_classes:
                break
    return ids[:top_k]


def process_one(i: int, text: str, args, labels, label_to_idx, client, responses_dir):
    out_path = responses_dir / f"{i}.json"
    if out_path.exists():
        # already done; just re-parse the cached top-k for the CSV
        with open(out_path) as f:
            resp = json.load(f)
        top = parse_top_k_ids(resp, label_to_idx, args.top_k)
        return i, top, "cached"

    prompt = build_prompt(text, labels, args.dataset)
    if args.dry_run:
        resp = fake_response(prompt, labels, args.top_k)
    else:
        resp = call_openai_with_retry(client, args.model, prompt, args.max_retries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(resp, f)
    os.replace(tmp, out_path)

    top = parse_top_k_ids(resp, label_to_idx, args.top_k)
    return i, top, "fresh"


def main():
    args = parse_args()

    root = repo_root()
    responses_dir = args.responses_dir or (root / "TAPE" / "gpt_responses" / args.dataset)
    preds_csv = args.preds_csv or (root / "TAPE" / "gpt_preds" / f"{args.dataset}.csv")
    responses_dir.mkdir(parents=True, exist_ok=True)
    preds_csv.parent.mkdir(parents=True, exist_ok=True)

    texts = load_node_texts(args.node_texts)
    labels = load_labels(args.labels)
    label_to_idx = labels_to_index(labels)
    n = len(texts)
    end = args.end if args.end is not None else n
    if args.max_nodes is not None:
        end = min(end, args.start + args.max_nodes)
    indices = range(args.start, end)
    print(f"[generate] dataset={args.dataset} model={args.model} dry_run={args.dry_run}")
    print(f"[generate] {n} total nodes; processing [{args.start}, {end})  "
          f"= {len(indices)} calls (cached ones are skipped)")
    print(f"[generate] {len(labels)} classes")
    print(f"[generate] responses_dir = {responses_dir}")
    print(f"[generate] preds_csv     = {preds_csv}")

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package not installed. `pip install openai>=1.30`")
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set in env.")
        client = OpenAI()

    # We always rebuild the top-k CSV from on-disk JSONs at the end, so partial runs are safe.
    top_k_per_node: dict[int, list[int]] = {}
    n_cached = 0
    n_fresh = 0
    n_failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(process_one, i, texts[i], args, labels, label_to_idx, client, responses_dir): i
            for i in indices
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                idx, top, status = fut.result()
                top_k_per_node[idx] = top
                if status == "cached":
                    n_cached += 1
                else:
                    n_fresh += 1
            except Exception as e:
                n_failed += 1
                print(f"  [fail] node {i}: {type(e).__name__}: {e}")

            done = n_cached + n_fresh + n_failed
            if done % 200 == 0 or done == len(indices):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f"  [progress] {done}/{len(indices)} "
                      f"(cached={n_cached}, fresh={n_fresh}, failed={n_failed}) "
                      f"rate={rate:.1f}/s  elapsed={elapsed:.0f}s")

    # Skip the CSV rebuild for --dry_run; nothing useful is being verified by it.
    if args.dry_run:
        print(f"[csv] skipping CSV rebuild for --dry_run")
    else:
        # Rebuild full CSV from disk (covers prior runs too).
        # IMPORTANT: on Drive FUSE, sequential reads of N JSONs are very slow
        # (~50-100ms/file metadata RPC). For 76k JSONs that's an hour. We
        # parallelize the per-file reads with a thread pool (Drive RPCs are
        # I/O-bound, GIL is fine here). We also reuse `top_k_per_node` for
        # nodes processed in this session so we don't reread their JSONs.
        print(f"[csv] rebuilding {preds_csv} from on-disk JSONs (parallel reads)...")

        def _read_top(i: int):
            if i in top_k_per_node:
                return i, top_k_per_node[i]
            jpath = responses_dir / f"{i}.json"
            if jpath.exists():
                with open(jpath) as jf:
                    resp = json.load(jf)
                return i, parse_top_k_ids(resp, label_to_idx, args.top_k)
            return i, list(range(args.top_k))  # placeholder for missing nodes

        rows = [None] * n
        # 32 workers is fine for I/O-bound Drive FUSE; bump higher if Drive is the bottleneck.
        with ThreadPoolExecutor(max_workers=32) as pool:
            t_csv = time.time()
            for k, (i, top) in enumerate(pool.map(_read_top, range(n))):
                rows[i] = top
                if (k + 1) % 5000 == 0:
                    elapsed = time.time() - t_csv
                    print(f"  [csv] {k + 1}/{n} read  elapsed={elapsed:.0f}s")
        with open(preds_csv, "w", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
        print(f"[csv] wrote {preds_csv} ({n} rows)")

    if n_failed:
        print(f"[done] WITH FAILURES: {n_failed} nodes failed; re-run to retry "
              f"(cached nodes will be skipped).")
        sys.exit(1)
    print(f"[done] cached={n_cached}, fresh={n_fresh}, failed={n_failed}")


if __name__ == "__main__":
    main()
