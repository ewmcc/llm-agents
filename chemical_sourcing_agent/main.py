"""
main.py — Entry point for the chemical sourcing agent.

Run with:
    python main.py --query "acetonitrile HPLC grade"

The pipeline has 7 numbered steps, each clearly separated below.
Steps 2 and 6 call the LLM. Steps 3-5 and 7 are pure Python.

Output (saved to outputs/ in the same folder as this script):
    outputs/<query_slug>_products.csv   — full ranked product table
    outputs/<query_slug>_report.md      — markdown table + LLM summary
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from llm import build_chains
from search import search_for_products
from extraction import extract_from_page
from normalize import normalize_records, rank_products


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from the same directory as this script."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    """
    Convert a string to a safe filename slug.
    "Acetonitrile HPLC Grade" -> "acetonitrile_hplc_grade"
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # remove non-word chars
    text = re.sub(r"[\s-]+", "_", text)    # spaces/hyphens to underscores
    return text


def df_to_markdown(df) -> str:
    """
    Convert a DataFrame to a markdown table string.
    We select only the human-readable columns for the report.
    """
    display_cols = [
        "vendor", "chemical_name", "product_name", "purity", "grade",
        "size", "price", "price_per_unit", "size_base_unit", "availability", "url",
    ]
    # Only include columns that actually exist in the DataFrame
    cols = [c for c in display_cols if c in df.columns]
    return df[cols].to_markdown(index=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(query: str):
    print(f"\n=== Chemical Sourcing Agent ===")
    print(f"Query: {query!r}\n")
    slug = slugify(query)

    # -----------------------------------------------------------------------
    # Step 1: Load environment variables and configuration
    # -----------------------------------------------------------------------
    load_dotenv()  # reads .env file into os.environ

    # Check that required API keys are present before doing any work.
    missing = [k for k in ("TAVILY_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
               if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}")
        print("Copy .env.example to .env and fill in your credentials.")
        sys.exit(1)

    config = load_config()
    output_dir = Path(__file__).parent / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 2: Expand the query with the LLM
    #   The expansion chain returns: canonical_name, cas_number, synonyms,
    #   and 2-4 search_queries optimized for vendor product pages.
    # -----------------------------------------------------------------------
    print("[Step 2] Building LangChain chains and expanding query via LLM...")
    chains = build_chains(config)

    try:
        expansion = chains["expansion"].invoke({"query": query})
    except Exception as e:
        print(f"ERROR: Query expansion failed — {e}")
        sys.exit(1)

    canonical_name = expansion.get("canonical_name", query)
    cas_number     = expansion.get("cas_number") or "unknown"
    search_queries = expansion.get("search_queries", [query])

    print(f"  Canonical name : {canonical_name}")
    print(f"  CAS number     : {cas_number}")
    print(f"  Search queries : {search_queries}\n")

    # -----------------------------------------------------------------------
    # Step 3: Search for product pages using Tavily
    #   Returns a list of {url, title} dicts, deduplicated across all queries.
    # -----------------------------------------------------------------------
    print("[Step 3] Searching for product pages via Tavily with raw content...")
    search_results = search_for_products(search_queries, config)
    print()

    if not search_results:
        print("No search results returned. Exiting.")
        sys.exit(0)

    # Save the full list of Tavily URLs before extraction so the list is
    # preserved even if the script fails during Steps 4-6.
    sources_path = output_dir / f"{slug}_sources.csv"
    with open(sources_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["url", "title"])
        writer.writeheader()
        writer.writerows([{"url": r["url"], "title": r["title"]} for r in search_results])
    print(f"  Sources saved  : {sources_path}\n")

    # -----------------------------------------------------------------------
    # Step 4: Extract product records from Tavily content via LLM
    #   For each result: use the raw_content from Tavily, then call the
    #   extraction LLM chain. Failed pages are skipped (extractor returns empty).
    #   No Playwright scraping needed — Tavily handles JS rendering.
    # -----------------------------------------------------------------------
    print("[Step 4] Extracting products from Tavily content via LLM...")
    all_records = []

    for i, result in enumerate(search_results, start=1):
        url = result["url"]
        print(f"  [{i}/{len(search_results)}] {url}")

        # 4a. Use content from Tavily (raw_content for vendors, summary for non-vendors)
        page_text = result.get("content", "").strip()
        if not page_text:
            print(f"      WARNING: No content from Tavily for this URL")
            continue

        # Strategy 1: truncate to max_page_chars before sending to the LLM.
        # max_page_chars is set in config.yaml (default 4000). This keeps token
        # costs predictable — most product info appears near the top of a vendor page.
        max_chars = config.get("max_page_chars", 4000)
        page_text = page_text[:max_chars]
        print(f"      Sending {len(page_text):,} chars to LLM")

        # 4b. Send the text to the LLM, get back a list of product dicts
        records = extract_from_page(page_text, url, chains)
        all_records.extend(records)

    print(f"\n  Total raw records collected: {len(all_records)}\n")

    if not all_records:
        print("No product records were extracted. Try a different query.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 5: Normalize and rank products (pure Python — no LLM)
    #   - Parse price/size/purity strings into numbers
    #   - Compute price_per_unit
    #   - Deduplicate by (vendor, product_name, size)
    #   - Sort by purity desc, then price_per_unit asc
    # -----------------------------------------------------------------------
    print("[Step 5] Normalizing and ranking products...")
    df = normalize_records(all_records)

    if df.empty:
        print("No records survived normalization (no price or purity data found).")
        sys.exit(0)

    df = rank_products(df)
    print(f"  {len(df)} unique products after deduplication and ranking.\n")

    # -----------------------------------------------------------------------
    # Step 6: Generate a procurement summary with the LLM
    #   We pass the ranked markdown table and ask for: best value, highest
    #   purity, multi-vendor strategy, and any cautions.
    # -----------------------------------------------------------------------
    print("[Step 6] Generating procurement summary via LLM...")
    table_md = df_to_markdown(df)

    try:
        summary = chains["summary"].invoke({
            "chemical": canonical_name,
            "cas_number": cas_number,
            "table": table_md,
        })
    except Exception as e:
        print(f"  WARNING: Summary generation failed — {e}")
        summary = "(Summary unavailable due to an error.)"

    print()

    # -----------------------------------------------------------------------
    # Step 7: Save outputs
    #   - CSV: full DataFrame with all numeric columns (for further analysis)
    #   - Markdown report: human-readable table + LLM summary block
    #   Output is written to the outputs/ folder in the same directory as main.py
    # -----------------------------------------------------------------------
    print("[Step 7] Saving outputs...")

    csv_path = output_dir / f"{slug}_products.csv"
    md_path  = output_dir / f"{slug}_report.md"

    df.to_csv(csv_path, index=False)

    sources_lines = []
    for i, s in enumerate(search_results, start=1):
        title = s.get("title") or s["url"]
        sources_lines.append(f"{i}. [{title}]({s['url']})")

    report_lines = [
        f"# Chemical Sourcing Report: {canonical_name}",
        f"\n**CAS Number:** {cas_number}",
        f"**Query:** {query}",
        f"\n---\n",
        f"## Ranked Products\n",
        table_md,
        f"\n---\n",
        f"## Procurement Recommendation\n",
        summary,
        f"\n---\n",
        f"## Sources Searched\n",
        "\n".join(sources_lines),
    ]
    md_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"  CSV saved    : {csv_path}")
    print(f"  Report saved : {md_path}")
    print("\nDone.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chemical sourcing agent using LangChain + Bedrock + Tavily."
    )
    parser.add_argument(
        "--query",
        required=True,
        help='Chemical to search for, e.g. "acetonitrile HPLC grade"',
    )
    args = parser.parse_args()
    run(args.query)
