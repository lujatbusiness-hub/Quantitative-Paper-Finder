#!/usr/bin/env python3
"""
arXiv -> Markdown paper summarizer (local-model powered).

Reads paper references from links.txt (one per line) or from the command line,
downloads each arXiv PDF, extracts the text, derives bibliographic metadata and
a structured summary with a *local* language model, and writes a Markdown note
into a configurable output folder (works great as an Obsidian vault folder, but
it is plain Markdown + YAML front matter, so any Markdown app or plain files
work too).

The summarizer talks to a local model through one of two backends, selected in
config.ini:

  * openai  -- any server speaking the OpenAI-compatible Chat Completions API
               (POST <base_url>/chat/completions). This covers virtually every
               local runtime: LM Studio, llama.cpp's server, vLLM, LocalAI,
               Jan, KoboldCpp, text-generation-webui, GPT4All, and Ollama's own
               /v1 endpoint. This is the default and the most portable choice.
  * ollama  -- Ollama's native API (POST <base_url>/api/generate).

Everything is driven by config.ini, so nothing in this file is tied to one
machine or one research field; copy config.example.ini to config.ini and edit.

Usage:
    python paper_summarizer.py                  # process every new link in links.txt
    python paper_summarizer.py <url> [<url>..]  # process the given link(s) instead
    python paper_summarizer.py --search "<kw>"  # keyword search arXiv
    python paper_summarizer.py --force          # re-process links already done
    python paper_summarizer.py --keep-pdf       # keep the downloaded PDFs

If a PDF already exists at pdfs/arxiv_<id>.pdf (e.g. dropped there manually),
the download step is skipped and the file is summarized directly.
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import fitz  # PyMuPDF
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
LINKS_FILE = SCRIPT_DIR / "links.txt"
PDF_DIR = SCRIPT_DIR / "pdfs"
PROCESSED_FILE = SCRIPT_DIR / ".processed.json"

# Built-in fallbacks for every user-configurable setting. Anything missing from
# config.ini falls back to the matching value here, so a partial config is fine.
_CONFIG_DEFAULTS = {
    "paths": {"output_dir": "~/Paper Summaries"},
    "model": {
        # openai = OpenAI-compatible /chat/completions (most portable); ollama = native.
        "backend": "openai",
        # OpenAI-compatible: include the /v1 (e.g. http://localhost:11434/v1 for
        # Ollama, http://localhost:1234/v1 for LM Studio). Ollama native: host only.
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        # Optional bearer token; most local servers ignore it. Leave blank if unused.
        "api_key": "",
        "temperature": "0.2",
        # Max tokens generated per response. Raise if summaries get cut off; 0 = server default.
        "max_tokens": "2048",
        # Context window, in tokens. Only the 'ollama' backend uses this (sent as
        # num_ctx); OpenAI-compatible servers set context when the model loads.
        "num_ctx": "8192",
        # Characters per map chunk (~1300 tokens at 5000 chars).
        "chunk_chars": "5000",
        # Seconds to wait for a model response.
        "timeout": "600",
    },
    "summary": {
        "language": "English",
        # Field of study, used to prime the model (e.g. "quantitative finance",
        # "machine learning"). Keep it broad if you summarize mixed topics.
        "domain": "general academic research",
        # What the closing "Relevance / Application" section should focus on.
        "relevance_focus": "how the paper could inform future research and practical work in this field",
    },
    "search": {"default_top": "3", "max_top": "25"},
}


def load_user_config(script_dir: Path) -> dict:
    """Load settings from config.ini next to this script.

    All user-specific values (output path, model backend/endpoint/name, summary
    language and domain, search limits) live in config.ini so the project is
    portable: copy config.example.ini to config.ini and edit it. config.ini is
    gitignored, so personal paths never get committed. If config.ini is absent,
    values fall back to config.example.ini and then to the defaults above; a
    partial config.ini is fine -- only the keys you set override the defaults.
    """
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_dict(_CONFIG_DEFAULTS)  # seed defaults so missing keys never raise

    config_file = script_dir / "config.ini"
    example_file = script_dir / "config.example.ini"
    if config_file.exists():
        parser.read(config_file, encoding="utf-8")
    else:
        if example_file.exists():
            parser.read(example_file, encoding="utf-8")
        print(
            "· no config.ini found -- using defaults. Copy config.example.ini to "
            "config.ini and set [paths] output_dir and your [model] backend."
        )

    return {section: dict(parser.items(section)) for section in parser.sections()}


_cfg = load_user_config(SCRIPT_DIR)

# Destination folder for the Markdown notes.
OUTPUT_DIR = Path(_cfg["paths"]["output_dir"]).expanduser()

# Local model backend + parameters.
BACKEND = _cfg["model"]["backend"].strip().lower()
BASE_URL = _cfg["model"]["base_url"].strip()
MODEL = _cfg["model"]["model"].strip()
API_KEY = _cfg["model"]["api_key"].strip()
TEMPERATURE = float(_cfg["model"]["temperature"])
MAX_TOKENS = int(_cfg["model"]["max_tokens"])         # 0 = let the server decide
NUM_CTX = int(_cfg["model"]["num_ctx"])               # ollama backend only
CHUNK_CHARS = int(_cfg["model"]["chunk_chars"])       # ~1300 tokens per map chunk
LLM_TIMEOUT = int(_cfg["model"]["timeout"])

# Summary shaping.
SUMMARY_LANGUAGE = _cfg["summary"]["language"]
SUMMARY_DOMAIN = _cfg["summary"]["domain"]
RELEVANCE_FOCUS = _cfg["summary"]["relevance_focus"]

HTTP_TIMEOUT = 60

SOURCE_LABEL = {"arxiv": "arXiv"}
ARXIV_API = "http://export.arxiv.org/api/query"

# Keyword search uses arXiv's official API (no bot protection).
DEFAULT_SEARCH_TOP = int(_cfg["search"]["default_top"])  # results when no count given
MAX_SEARCH_TOP = int(_cfg["search"]["max_top"])          # hard cap (each costs minutes)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Local model client (OpenAI-compatible or native Ollama)
# --------------------------------------------------------------------------- #
def _openai_chat_url() -> str:
    """Build the chat-completions URL, tolerating a base_url with or without it."""
    url = BASE_URL.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _ollama_generate_url() -> str:
    """Build the native Ollama generate URL, tolerating a trailing /v1 or path."""
    url = BASE_URL.rstrip("/")
    if url.endswith("/api/generate"):
        return url
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return f"{url}/api/generate"


def _openai_generate(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "stream": False,
    }
    if MAX_TOKENS > 0:
        payload["max_tokens"] = MAX_TOKENS
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    resp = requests.post(_openai_chat_url(), json=payload, headers=headers, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _ollama_generate(prompt: str) -> str:
    options = {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}
    if MAX_TOKENS > 0:
        options["num_predict"] = MAX_TOKENS
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": options}
    resp = requests.post(_ollama_generate_url(), json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def llm_generate(prompt: str) -> str:
    """Send a single prompt to the configured local model and return its text."""
    if BACKEND == "ollama":
        return _ollama_generate(prompt)
    return _openai_generate(prompt)


def check_backend() -> None:
    """Validate the backend choice early with a clear message."""
    if BACKEND not in ("openai", "ollama"):
        sys.exit(
            f"! unknown [model] backend = {BACKEND!r}. "
            "Use 'openai' (OpenAI-compatible servers) or 'ollama' (native API)."
        )


# --------------------------------------------------------------------------- #
# Link classification (arXiv)
# --------------------------------------------------------------------------- #
ARXIV_NEW_RE = re.compile(r"\d{4}\.\d{4,5}")               # 2301.00001 / 1412.7515
ARXIV_OLD_RE = re.compile(r"[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7}", re.IGNORECASE)


def _arxiv_id_from(text: str) -> str | None:
    match = ARXIV_NEW_RE.search(text) or ARXIV_OLD_RE.search(text)
    return match.group(0) if match else None


def classify_link(raw: str) -> tuple[str, str, str] | None:
    """Recognize an arXiv reference.

    Returns (source, identifier, landing_url) with source == "arxiv", or None if
    the link is not a recognizable arXiv paper.
    """
    url = raw.strip()
    if not url:
        return None

    if "arxiv.org" in url.lower():
        aid = _arxiv_id_from(url)
        return ("arxiv", aid, f"https://arxiv.org/abs/{aid}") if aid else None

    # Bare identifiers (no host).
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", url):
        aid = ARXIV_NEW_RE.match(url).group(0)
        return ("arxiv", aid, f"https://arxiv.org/abs/{aid}")
    if re.fullmatch(r"[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7}(v\d+)?", url, re.IGNORECASE):
        aid = ARXIV_OLD_RE.match(url).group(0)
        return ("arxiv", aid, f"https://arxiv.org/abs/{aid}")

    return None


# --------------------------------------------------------------------------- #
# Keyword search (arXiv API)
# --------------------------------------------------------------------------- #
# links.txt syntax:  search[ arxiv]: <keywords> [| N]
SEARCH_LINE_RE = re.compile(
    r"^search(?:\s+arxiv)?\s*:\s*(.+?)\s*(?:\|\s*(\d+)\s*)?$", re.IGNORECASE
)


def parse_search_line(raw: str) -> dict | None:
    """Parse 'search: <keywords> [| N]' into a search spec."""
    match = SEARCH_LINE_RE.match(raw.strip())
    if not match:
        return None
    top = int(match.group(2)) if match.group(2) else DEFAULT_SEARCH_TOP
    if top > MAX_SEARCH_TOP:
        print(f"· capping search at {MAX_SEARCH_TOP} results")
        top = MAX_SEARCH_TOP
    return {"query": match.group(1).strip(), "top": max(1, top)}


def search_arxiv(query: str, max_results: int) -> list[dict]:
    """Top arXiv papers for the keywords, by relevance. Quoted phrases allowed."""
    tokens = re.findall(r'"[^"]+"|\S+', query)
    search_query = " AND ".join(f"all:{t}" for t in tokens)
    resp = requests.get(
        ARXIV_API,
        params={
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results: list[dict] = []
    for entry in ET.fromstring(resp.text).findall("a:entry", ns):
        aid = _arxiv_id_from(entry.findtext("a:id", default="", namespaces=ns))
        if not aid:
            continue
        title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=ns)).strip()
        published = entry.findtext("a:published", default="", namespaces=ns)
        results.append({
            "source": "arxiv",
            "id": aid,
            "url": f"https://arxiv.org/abs/{aid}",
            "title": title or aid,
            "authors": [
                name.strip()
                for author in entry.findall("a:author", ns)
                if (name := author.findtext("a:name", default="", namespaces=ns)).strip()
            ],
            "year": published[:4] if published[:4].isdigit() else None,
        })
    return results


# --------------------------------------------------------------------------- #
# arXiv download + metadata (no bot protection)
# --------------------------------------------------------------------------- #
def download_arxiv_pdf(identifier: str, dest: Path) -> bool:
    """Download an arXiv PDF over plain HTTP. Returns True on success."""
    pdf_url = f"https://arxiv.org/pdf/{identifier}"
    try:
        resp = requests.get(pdf_url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        print(f"  ! arXiv download error: {exc}")
        return False
    if resp.status_code != 200 or resp.content[:5] != b"%PDF-":
        print(f"  ! arXiv download failed (status {resp.status_code})")
        return False
    dest.write_bytes(resp.content)
    print(f"  · downloaded arXiv PDF ({len(resp.content) // 1024} KB)")
    return True


def arxiv_metadata(identifier: str) -> dict | None:
    """Fetch title/authors/year from the arXiv API. None on any failure."""
    try:
        resp = requests.get(
            ARXIV_API,
            params={"id_list": identifier},
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except (requests.RequestException, ET.ParseError):
        return None

    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    title = re.sub(r"\s+", " ", entry.findtext("a:title", default="", namespaces=ns)).strip()
    if not title or title.lower() == "error":
        return None
    published = entry.findtext("a:published", default="", namespaces=ns)
    year = published[:4] if published[:4].isdigit() else None
    authors = [
        name.strip()
        for author in entry.findall("a:author", ns)
        if (name := author.findtext("a:name", default="", namespaces=ns)).strip()
    ]
    return {"title": title, "authors": authors, "year": year}


# --------------------------------------------------------------------------- #
# PDF text & metadata
# --------------------------------------------------------------------------- #
def extract_text(pdf_path: Path) -> str:
    """Extract plain text from all pages of the PDF."""
    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()


def heuristic_title(pdf_path: Path) -> str | None:
    """Pick the largest-font text line on page 1 as a title fallback."""
    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0:
            return None
        blocks = doc[0].get_text("dict")["blocks"]
    best_size, best_text = 0.0, ""
    for block in blocks:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            size = max((span["size"] for span in line["spans"]), default=0)
            if len(text) > 8 and size > best_size:
                best_size, best_text = size, text
    return best_text or None


def derive_metadata(text: str, pdf_path: Path, identifier: str) -> dict:
    """Title/authors/year from the PDF, using the model for extraction with fallbacks."""
    title = authors = year = None

    raw = llm_generate(textwrap.dedent(f"""\
        Extract bibliographic metadata from the first pages of an academic paper
        below. Return ONLY a JSON object with keys:
          "title"   : string,
          "authors" : array of strings,
          "year"    : a 4-digit string or null.
        Use only information explicitly present in the text. Do not guess.

        TEXT:
        {text[:3500]}
    """))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            title = (data.get("title") or "").strip() or None
            authors = [a.strip() for a in (data.get("authors") or []) if a and a.strip()]
            year = str(data.get("year")).strip() if data.get("year") else None
        except (json.JSONDecodeError, AttributeError):
            pass

    if not title:
        meta_title = (fitz.open(pdf_path).metadata or {}).get("title", "")
        title = meta_title.strip() if meta_title and len(meta_title.strip()) > 5 else None
    if not title:
        title = heuristic_title(pdf_path)
    if not title:
        title = identifier

    if year and not re.fullmatch(r"\d{4}", year):
        year = None

    return {"title": title, "authors": authors or [], "year": year}


def get_metadata(source: str, identifier: str, text: str, pdf_path: Path,
                 hint: dict | None = None) -> dict:
    """Source-aware metadata: search-result hint first (authoritative title/
    authors/year from the arXiv API), then the arXiv API, then PDF/model
    extraction."""
    meta = None
    if hint and hint.get("title"):
        meta = {
            "title": hint["title"],
            "authors": hint.get("authors") or [],
            "year": hint.get("year"),
        }
    if meta is None and source == "arxiv":
        meta = arxiv_metadata(identifier)
    if not meta or not meta.get("title"):
        meta = derive_metadata(text, pdf_path, identifier)
    meta["source"] = source
    meta["identifier"] = identifier
    return meta


# --------------------------------------------------------------------------- #
# Summarization (map-reduce)
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split text on paragraph boundaries into <= `size` character chunks."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > size and current:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
        while len(current) > size:  # a single oversized paragraph
            chunks.append(current[:size].strip())
            current = current[size:]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def summarize(text: str, meta: dict) -> str:
    """Map-reduce summarize the paper into the structured Markdown body."""
    chunks = chunk_text(text)
    print(f"  · summarizing in {len(chunks)} chunk(s) with {MODEL} ...")

    notes: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        notes.append(llm_generate(textwrap.dedent(f"""\
            You are a research assistant reading a research paper in the field of
            {SUMMARY_DOMAIN}. Extract the important technical content from the
            section below as concise bullet points: research question, hypotheses,
            data/sample, methodology, and key results with concrete numbers. Do
            not add commentary or invent anything. Output bullet points only.

            SECTION:
            {chunk}
        """)))
        print(f"    - mapped chunk {i}/{len(chunks)}")

    authors = ", ".join(meta["authors"]) if meta["authors"] else "Unknown"
    body = llm_generate(textwrap.dedent(f"""\
        You are an expert analyst in {SUMMARY_DOMAIN}. Using ONLY the extracted
        notes below from the paper titled "{meta['title']}" by {authors}, write a
        structured summary in {SUMMARY_LANGUAGE} Markdown.

        Use EXACTLY these sections and headers, in this order:

        ## TL;DR
        (2-3 sentence plain-language summary)

        ## Core Idea
        (the central thesis / contribution)

        ## Method
        (data, sample period, models, key assumptions)

        ## Key Results
        (bullet points; include concrete numbers and effect sizes where given)

        ## Relevance / Application
        ({RELEVANCE_FOCUS})

        Rules: do not invent results that are not in the notes. Be concise and
        specific. Output only the Markdown, starting directly with "## TL;DR".

        NOTES:
        {chr(10).join(notes)}
    """))

    idx = body.find("## TL;DR")
    return body[idx:].strip() if idx != -1 else body


# --------------------------------------------------------------------------- #
# Note writing
# --------------------------------------------------------------------------- #
def slugify(title: str, max_len: int = 80) -> str:
    slug = re.sub(r'[\\/:*?"<>|]+', "", title)
    slug = re.sub(r"\s+", " ", slug).strip()
    return slug[:max_len].strip()


def write_note(meta: dict, body: str, landing_url: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = SOURCE_LABEL.get(meta["source"], meta["source"])
    disp_id = meta["identifier"].replace("/", "_")
    filename = f"{slugify(meta['title'])} ({label} {disp_id}).md"
    path = OUTPUT_DIR / filename

    author_lines = [f"  - {a}" for a in meta["authors"]] or ["  - Unknown"]
    safe_title = meta["title"].replace('"', "'")
    lines = [
        "---",
        f'title: "{safe_title}"',
        "authors:",
        *author_lines,
        f"year: {meta['year'] or 'unknown'}",
        f"url: {landing_url}",
        f"source: {meta['source']}",
        f"paper_id: {meta['identifier']}",
        "type: paper-summary",
        f"tags: [{meta['source']}, paper, research]",
        f"summarized_with: {MODEL}",
        f"created: {date.today().isoformat()}",
        "---",
    ]
    frontmatter = "\n".join(lines)

    path.write_text(f"{frontmatter}\n\n# {meta['title']}\n\n{body}\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_links(args_links: list[str]) -> list[str]:
    if args_links:
        return args_links
    if not LINKS_FILE.exists():
        LINKS_FILE.write_text(
            "# Paste one arXiv link per line. Lines starting with # are ignored.\n",
            encoding="utf-8",
        )
        print(f"Created {LINKS_FILE}. Add links and re-run.")
        return []
    return [
        line.strip()
        for line in LINKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_processed() -> dict:
    if PROCESSED_FILE.exists():
        return json.loads(PROCESSED_FILE.read_text(encoding="utf-8"))
    return {}


def save_processed(processed: dict) -> None:
    PROCESSED_FILE.write_text(json.dumps(processed, indent=2), encoding="utf-8")


def process_item(source: str, identifier: str, landing_url: str, pdf_path: Path,
                 hint: dict | None = None) -> dict | None:
    text = extract_text(pdf_path)
    if len(text) < 500:
        print(f"  ! extracted text too short ({len(text)} chars); skipping")
        return None

    print("  · deriving metadata ...")
    meta = get_metadata(source, identifier, text, pdf_path, hint)
    print(f"  · {meta['title']}")

    body = summarize(text, meta)
    note_path = write_note(meta, body, landing_url)
    print(f"  ✓ wrote note: {note_path.name}")

    return {
        "title": meta["title"],
        "file": note_path.name,
        "source": source,
        "date": datetime.now().isoformat(timespec="seconds"),
    }


def finalize(processed: dict, key: str, result: dict, pdf_path: Path, keep_pdf: bool) -> None:
    """Record a finished paper, then drop its PDF (kept on disk until now so an
    interrupted run leaves the PDF behind as an "unfinished" marker)."""
    processed[key] = result
    save_processed(processed)
    if not keep_pdf:
        pdf_path.unlink(missing_ok=True)


def resume_unfinished(processed: dict, keep_pdf: bool) -> int:
    """Finish any paper whose PDF was downloaded but never summarized.

    Such a PDF sits in pdfs/ with a key that is not yet in .processed.json --
    the sign of a run interrupted during summarization. Source, id and landing
    URL are reconstructed from the PDF filename, so this works even if the link
    is no longer in links.txt. No network download is needed.
    """
    if not PDF_DIR.exists():
        return 0
    resumed = 0
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        source, _, safe_id = pdf.stem.partition("_")
        if source not in SOURCE_LABEL or not safe_id:
            continue
        identifier = safe_id.replace("_", "/")   # restore old-style arXiv slash
        key = f"{source}:{identifier}"
        if key in processed:
            continue                              # already finished; leave as-is
        if pdf.stat().st_size < 1024:
            continue                              # truncated download; skip
        landing_url = f"https://arxiv.org/abs/{identifier}"
        print(f"\n↻ resuming unfinished {SOURCE_LABEL[source]} paper: {identifier}")
        try:
            result = process_item(source, identifier, landing_url, pdf)
        except Exception as exc:  # noqa: BLE001  (model, fitz, etc. -- keep the PDF for a later retry)
            print(f"  ! could not finish: {exc}")
            result = None
        if result:
            finalize(processed, key, result, pdf, keep_pdf)
            resumed += 1
    return resumed


def queue_search(spec: dict, processed: dict, seen: set, queue: list, force: bool) -> None:
    """Resolve a keyword search into queue entries.

    A search line is one-shot like a link: its resolved papers are recorded in
    .processed.json under a search key. Re-running the same line re-queues only
    those resolved papers that are still unfinished (no fresh API query), so the
    result set stays stable across runs. Use --force to query fresh.
    """
    skey = f'search:arxiv:{spec["query"].lower()}|{spec["top"]}'
    print(f'\n🔎 search "{spec["query"]}" (arXiv, top {spec["top"]})')

    def enqueue(result: dict) -> None:
        key = f'{result["source"]}:{result["id"]}'
        seen.add(key)
        safe_id = result["id"].replace("/", "_")
        queue.append((result["source"], result["id"], result["url"],
                      PDF_DIR / f'{result["source"]}_{safe_id}.pdf', key, result))

    if skey in processed and not force:
        requeued = 0
        for result in processed[skey].get("results", []):
            key = f'{result["source"]}:{result["id"]}'
            if key in seen or key in processed:
                continue
            enqueue(result)
            requeued += 1
        if requeued:
            print(f"  · search ran before; re-queuing {requeued} unfinished result(s)")
        else:
            print("  · search ran before and all its results are analyzed; "
                  "change the keywords/number or use --force for fresh results")
        return

    # Fetch more than needed so already-analyzed papers don't eat the quota.
    fetch_n = min(50, max(2 * spec["top"], spec["top"] + 10))
    try:
        found = search_arxiv(spec["query"], fetch_n)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"  ! arXiv search failed: {exc}")
        return

    chosen: list[dict] = []
    local_seen: set[str] = set()
    for result in found:
        key = f'{result["source"]}:{result["id"]}'
        if key in local_seen or key in seen or (key in processed and not force):
            continue
        local_seen.add(key)
        chosen.append(result)
        if len(chosen) >= spec["top"]:
            break

    if not found:
        print("  · no arXiv results")
    elif not chosen:
        print("  · all top arXiv results are already analyzed")
    for i, result in enumerate(chosen, 1):
        print(f'    {i}. [arXiv {result["id"]}] {result["title"][:90]}')

    if not chosen:
        return
    for result in chosen:
        enqueue(result)
    processed[skey] = {
        "query": spec["query"],
        "source": "arxiv",
        "top": spec["top"],
        "results": [
            {k: r.get(k) for k in ("source", "id", "url", "title", "authors", "year")}
            for r in chosen
        ],
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    save_processed(processed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize arXiv papers into Markdown notes with a local model.")
    parser.add_argument("links", nargs="*",
                        help="arXiv URLs/ids or 'search: <keywords> | <N>' lines (overrides links.txt)")
    parser.add_argument("--search", action="append", default=[], metavar="QUERY",
                        help="keyword search (repeatable); combine with --top")
    parser.add_argument("--top", type=int, default=DEFAULT_SEARCH_TOP,
                        help=f"results for --search (default {DEFAULT_SEARCH_TOP})")
    parser.add_argument("--force", action="store_true", help="re-process already-done links")
    parser.add_argument("--keep-pdf", action="store_true", help="keep downloaded PDFs")
    args = parser.parse_args()

    check_backend()
    processed = load_processed()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Startup check: complete any paper whose PDF was downloaded but never
    # summarized (a previous run interrupted during summarization).
    resumed = resume_unfinished(processed, args.keep_pdf)
    if resumed:
        print(f"↻ completed {resumed} unfinished paper(s) from a previous run.")

    # Input lines: CLI positionals override links.txt; `--search` flags alone
    # run just those searches without draining links.txt.
    if args.links:
        lines = args.links
    elif args.search:
        lines = []
    else:
        lines = load_links([])

    cli_searches = [
        {"query": q.strip(), "top": max(1, min(args.top, MAX_SEARCH_TOP))}
        for q in args.search
        if q.strip()
    ]
    if not lines and not cli_searches:
        print("No new links to process.")
        return 0

    # Build the work queue. Each line is either a keyword search (expanded via
    # the arXiv API) or a direct link. Two dedup layers:
    #   * `processed` (.processed.json) -> already analyzed in a previous run
    #   * `seen`                        -> duplicates within this same run
    queue: list[tuple[str, str, str, Path, str, dict | None]] = []  # source, id, url, pdf, key, hint
    seen: set[str] = set()
    for raw in lines:
        spec = parse_search_line(raw)
        if spec:
            queue_search(spec, processed, seen, queue, args.force)
            continue
        info = classify_link(raw)
        if not info:
            print(f"! unrecognized link (not an arXiv paper): {raw}")
            continue
        source, identifier, landing_url = info
        key = f"{source}:{identifier}"
        if key in seen:
            print(f"· duplicate of an earlier link this run; skipping: {key}")
            continue
        seen.add(key)
        if key in processed and not args.force:
            print(f"· already analyzed ({processed[key]['file']}); use --force to redo: {key}")
            continue
        safe_id = identifier.replace("/", "_")
        queue.append((source, identifier, landing_url, PDF_DIR / f"{source}_{safe_id}.pdf", key, None))
    for spec in cli_searches:
        queue_search(spec, processed, seen, queue, args.force)

    if not queue:
        print("Nothing new to do.")
        return 0

    print(f"\nQueued: {len(queue)} arXiv paper(s).")

    done = failed = 0
    for source, identifier, landing_url, pdf_path, key, hint in queue:
        print(f"\n→ [{SOURCE_LABEL[source]}] {landing_url}")

        if pdf_path.exists() and pdf_path.stat().st_size >= 1024:
            print(f"  · using existing PDF: {pdf_path.name}")
        elif not download_arxiv_pdf(identifier, pdf_path):
            print(f"  ! no PDF obtained; skipping (drop it as pdfs/{pdf_path.name} and re-run)")
            failed += 1
            continue

        try:
            result = process_item(source, identifier, landing_url, pdf_path, hint)
        except requests.RequestException as exc:
            print(f"  ! model request error: {exc}")
            result = None

        if result:
            finalize(processed, key, result, pdf_path, args.keep_pdf)
            done += 1
        else:
            failed += 1

    print(f"\nDone. {done} summarized, {failed} failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
