# paper_summarizer

Paste arXiv links (or keyword searches) and get structured Markdown summaries of
the papers, written into a folder of your choice — summarized entirely by a
**local** language model. The output is plain Markdown with YAML front matter, so
it drops straight into an [Obsidian](https://obsidian.md) vault but works with
any Markdown app or as plain files.

arXiv papers download over plain HTTP with no interaction; metadata (title /
authors / year) comes from the arXiv API.

## Requirements
- **Python 3.10+**
- Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- A **local model server** running on your machine (see *Local model* below).

## Local model
The summarizer never calls a cloud API — it talks to a model server running on
your own machine. Pick one of two backends in `config.ini` under `[model]`:

| `backend` | Talks to | Endpoint |
|-----------|----------|----------|
| `openai`  | Any server speaking the **OpenAI-compatible Chat Completions API** — LM Studio, llama.cpp's server, vLLM, LocalAI, Jan, KoboldCpp, text-generation-webui, GPT4All, **and** Ollama's `/v1` endpoint. | `POST <base_url>/chat/completions` |
| `ollama`  | **Ollama's native API**. | `POST <base_url>/api/generate` |

`openai` is the default and the most portable choice: almost every local runtime
exposes that API, so the same config works across tools. Examples:

```ini
# LM Studio (start its local server, load a model)
[model]
backend  = openai
base_url = http://localhost:1234/v1
model    = <model id shown in LM Studio>

# Ollama via its OpenAI-compatible endpoint
[model]
backend  = openai
base_url = http://localhost:11434/v1
model    = qwen2.5:7b

# Ollama via its native API
[model]
backend  = ollama
base_url = http://localhost:11434
model    = qwen2.5:7b
```

For Ollama, pull the model first: `ollama pull qwen2.5:7b`.

### Choosing a model
The script asks the model to (1) follow a fixed Markdown section layout and
(2) emit a small JSON object for metadata. It does **not** need a giant model,
but it does need an **instruction-tuned chat model** that follows formatting
reasonably well. Guidelines:

- Use an **instruct/chat** model, not a base/completion model.
- ~**7B parameters or larger** is a good baseline (e.g. Qwen2.5-7B-Instruct,
  Llama-3.1-8B-Instruct, Mistral-7B-Instruct). Smaller models often ignore the
  section structure or produce invalid JSON.
- A context window of **8k tokens or more** is plenty — the paper is processed in
  chunks, never sent whole.

Metadata extraction degrades gracefully: if the model returns malformed JSON, the
tool falls back to the arXiv API, then the PDF's embedded metadata, then a
largest-font-on-page-1 heuristic. So a weaker model mainly costs summary quality,
not crashes.

## Usage
1. Put arXiv references into `links.txt`, one per line:
   ```
   https://arxiv.org/abs/2301.00001
   https://arxiv.org/pdf/2301.00001
   2301.00001            # bare arXiv id
   ```
2. Run:
   ```bash
   python paper_summarizer.py
   ```

You can also pass links directly (this overrides `links.txt`):
```bash
python paper_summarizer.py https://arxiv.org/abs/2301.00001 1312.0514
```

## Keyword search
Instead of (or alongside) links, put **search lines** into `links.txt`. The top
N arXiv results are resolved and summarized automatically:

```
search: limit order book liquidity | 5      # top 5 results
search arxiv: hawkes process | 3            # explicit, same thing
search: market making                       # count optional (default 3, max 25)
```

Quoted phrases work: `search: "order flow" toxicity | 3`. Or from the CLI:

```bash
python paper_summarizer.py --search "limit order book liquidity" --top 5
```

Search semantics:
- A search line is **one-shot**, like a link: its resolved papers are recorded in
  `.processed.json`. Re-running the same line does not re-query; it only re-queues
  results that are still unfinished. Change the keywords/number or use `--force`
  for fresh results.
- Already-analyzed papers don't eat the quota: the search fetches extra
  candidates and picks the top N **new** ones.

### Flags
- `--search QUERY` — keyword search (repeatable); `--top N` results (default 3)
- `--force` — re-process already-done links / re-query already-run searches
- `--keep-pdf` — keep the downloaded PDFs in `pdfs/` (deleted by default)

## How it works
1. Recognizes each line as an arXiv link/id and extracts its identifier.
2. Downloads the PDF from arXiv over plain HTTP.
3. Extracts the text with PyMuPDF.
4. Derives metadata: the arXiv API first, then a model JSON extraction from the
   PDF text, backed by the PDF's metadata and a largest-font heuristic.
5. Map-reduce summarization with your local model: each text chunk is condensed
   to notes, then merged into a structured summary
   (TL;DR / Core Idea / Method / Key Results / Relevance).
6. Writes `<Title> (arXiv <id>).md` into the output folder with YAML front
   matter (`source`, `url`, `paper_id`, `summarized_with`, tags).

Processed links are tracked in `.processed.json` (keyed `arxiv:<id>`), so re-runs
only handle new papers.

## Resuming interrupted runs
A downloaded PDF is kept in `pdfs/` until its summary is fully written and
recorded in `.processed.json`; only then is it deleted. So if a run is
interrupted during summarization (Ctrl+C, model timeout, sleep), the PDF stays
behind as an "unfinished" marker. On the next start the tool first scans `pdfs/`
and **completes any paper that was downloaded but not summarized** —
reconstructing the id/URL from the PDF filename, so it works even if you removed
the link from `links.txt`.

## Manual fallback
If a download ever fails, save the PDF yourself as `pdfs/arxiv_<id>.pdf`
(e.g. `pdfs/arxiv_2301.00001.pdf`) and re-run. It is picked up by the resume scan
above and summarized directly — no link needed.

## Configuration
All machine-specific settings live in **`config.ini`** next to the script, so the
project is portable. On first use, copy the template and edit it:

```bash
cp config.example.ini config.ini
```

`config.ini` is gitignored, so your personal paths never get committed. Every key
is optional except `[paths] output_dir`; anything you leave out falls back to the
default in `config.example.ini`. If `config.ini` is missing entirely, the tool
still runs on built-in defaults and prints a reminder. Keys:

- `[paths] output_dir` — destination folder for the notes (absolute path or `~`)
- `[model] backend` — `openai` or `ollama`
- `[model] base_url`, `model`, `api_key`, `temperature`, `max_tokens`,
  `num_ctx` (ollama only), `chunk_chars`, `timeout`
- `[summary] language`, `domain`, `relevance_focus`
- `[search] default_top`, `max_top`
