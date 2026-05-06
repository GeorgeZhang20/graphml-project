# LLM Explanation Generation

Calls an LLM once per node in our new dataset, asks it to predict a category and
explain its reasoning, and writes the results in the format TAPE expects.

## Output contract (must match upstream TAPE)
- **`TAPE/gpt_responses/<DATASET>/<i>.json`** — one file per node `i`. The TAPE
  loader at [TAPE/core/data_utils/load.py:60-65](../TAPE/core/data_utils/load.py)
  reads `json_data["choices"][0]["message"]["content"]` (i.e., the raw OpenAI
  chat-completion response shape).
- **`TAPE/gpt_preds/<DATASET>.csv`** — comma-separated **top-k label ids**, one
  row per node. See `TAPE/gpt_preds/cora.csv` for an example.

## Backend choice
| Backend       | Cost (~30k nodes) | Quality | Notes                                |
|---------------|-------------------|---------|--------------------------------------|
| GPT-4o-mini   | ~$5–15            | high    | needs `OPENAI_API_KEY`               |
| Llama-3-8B    | free              | medium  | run on Colab T4 via `transformers`   |
| Llama-3-70B   | free if local     | high    | needs serious compute                |

We use GPT-4o-mini for the runs reported in the paper (cost was acceptable; ~$15 on the full 76k-node run). The `--model` flag accepts any chat-completion-compatible name, so swapping in a local Llama is a one-line change once the OPENAI_BASE_URL/HF integration is wired up on your end.

## How to run
```bash
export OPENAI_API_KEY=sk-...
python llm_explanations/generate.py \
    --dataset goodreads_children \
    --node_texts new_dataset/data/goodreads_children/node_texts.jsonl \
    --labels    new_dataset/data/goodreads_children/labels.txt \
    --model     gpt-4o-mini
```

`generate.py` is resumable: re-running it skips nodes whose JSON already lives
under `TAPE/gpt_responses/<DATASET>/`, so interrupting and re-running is safe.
