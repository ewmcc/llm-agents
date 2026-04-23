"""
extraction.py — Sends page text to the LLM and parses out product records.

This is where the LLM does its main "reading" work: given raw page text,
it identifies product listings and returns them as structured JSON.

The extraction chain uses the prompts/extraction.txt prompt and returns a
list of dicts. If the LLM returns invalid JSON, or the page had no products,
we log a warning and return an empty list so the pipeline keeps running.
"""


def extract_from_page(page_text: str, url: str, chains: dict) -> list[dict]:
    """
    Use the extraction LLM chain to pull product records from page text.

    Args:
        page_text : cleaned text from scraper.fetch_page_text()
        url       : the source URL (injected into any records that omit it)
        chains    : the dict returned by llm.build_chains()

    Returns:
        A list of product dicts. May be empty if the page had no products
        or if the LLM response could not be parsed.
    """

    # Skip pages where scraping failed (empty string returned by scraper).
    if not page_text.strip():
        print(f"  [extract] Skipping (empty page text): {url}")
        return []

    try:
        # Invoke the extraction chain with the two required template variables.
        # The chain returns a Python list (parsed from JSON by JsonOutputParser).
        result = chains["extraction"].invoke(
            {"page_text": page_text, "url": url}
        )
    except Exception as e:
        # Common causes: Bedrock throttle, network error, malformed JSON from LLM.
        print(f"  [extract] WARNING: Extraction failed for {url} — {e}")
        return []

    # The prompt asks for a JSON array, but occasionally the LLM wraps it in a dict.
    # Handle the most common case: {"products": [...]}
    if isinstance(result, dict):
        # Try common wrapper keys before giving up.
        for key in ("products", "results", "items", "data"):
            if key in result and isinstance(result[key], list):
                result = result[key]
                break
        else:
            print(f"  [extract] WARNING: Unexpected JSON structure from LLM for {url}")
            return []

    if not isinstance(result, list):
        print(f"  [extract] WARNING: Expected a list from LLM, got {type(result)} for {url}")
        return []

    # Ensure every record has a 'url' field — the LLM sometimes omits it even
    # though the prompt asks for it.
    for record in result:
        if isinstance(record, dict) and not record.get("url"):
            record["url"] = url

    print(f"  [extract] Extracted {len(result)} record(s) from {url}")
    return result
