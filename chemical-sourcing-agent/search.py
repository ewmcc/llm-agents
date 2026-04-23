"""
search.py — Uses the Tavily API to find chemical product pages.

Tavily is a search API designed for LLM applications. It returns clean,
structured results rather than raw HTML, so we can quickly get a list of
URLs to scrape without dealing with Google's bot protection.

The main function iterates over multiple search queries (produced by the
expansion step) and deduplicates results by URL so we don't scrape the
same page twice.
"""

import os
import re
from tavily import TavilyClient

# Domains that never contain purchasable products — use Tavily's short summary
# instead of raw_content to avoid sending thousands of irrelevant tokens to the LLM.
NON_VENDOR_DOMAINS = re.compile(
    r'pubmed\.ncbi|ncbi\.nlm|wikipedia\.org|sciencedirect\.com'
    r'|nature\.com|pmc\.ncbi|hero\.epa|researchgate\.net'
    r'|chemspider\.com|drugbank\.ca|ebi\.ac\.uk|rsc\.org',
    re.IGNORECASE,
)


def search_for_products(queries: list[str], config: dict) -> list[dict]:
    """
    Run each query through Tavily and collect unique result URLs.

    Args:
        queries : list of search query strings (from the expansion LLM step)
        config  : the loaded config.yaml dict (needs 'max_search_results')

    Returns:
        A list of dicts, each with 'url' and 'title' keys.
        Duplicates (same URL appearing in multiple queries) are removed.
    """

    # TavilyClient reads TAVILY_API_KEY from the environment automatically.
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    results = []
    seen_urls = set()  # track URLs we've already added to avoid duplicates

    for query in queries:
        print(f"  [search] Querying Tavily: {query!r}")

        try:
            # search_depth="advanced" gives better results at slightly higher cost.
            # include_raw_content=True captures the full page content directly from Tavily
            # instead of requiring separate Playwright scraping — avoids bot detection
            # and JS rendering issues on vendor sites.
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=config["max_search_results"],
                include_raw_content=True,
            )
        except Exception as e:
            # A single failed query shouldn't stop the whole run.
            print(f"  [search] WARNING: Query failed ({e}), skipping.")
            continue

        # response["results"] is a list of dicts with keys: url, title, content, raw_content, score
        for item in response.get("results", []):
            url = item.get("url", "").strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                # Strategy 3: non-vendor/academic pages get Tavily's short summary
                # only — they won't have purchasable products, so raw_content would
                # just waste tokens sending journal abstracts or Wikipedia prose to the LLM.
                if NON_VENDOR_DOMAINS.search(url):
                    content = item.get("content", "")  # Tavily short summary (~200 chars)
                    print(f"    [search] (summary only — non-vendor domain): {url}")
                else:
                    content = item.get("raw_content") or item.get("content", "")
                results.append({
                    "url": url,
                    "title": item.get("title", ""),
                    "content": content,
                })

    print(f"  [search] Found {len(results)} unique URLs across all queries.")
    return results
