# chemical_sourcing_agent

A Python agent that finds, extracts, and compares chemical product listings from vendor websites.

Built as a **learning example** for LangChain LCEL chains + Amazon Bedrock + Tavily search.

---

## Quick Start

### Prerequisites

- Python 3.11+
- An [AWS account](https://aws.amazon.com/) with Bedrock access enabled for a supported Claude model in your region
- A [Tavily API key](https://app.tavily.com)

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the `chemical_sourcing_agent/` directory with your credentials:
   ```
   TAVILY_API_KEY=your_tavily_key
   AWS_ACCESS_KEY_ID=your_aws_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret
   AWS_DEFAULT_REGION=us-east-1
   ```

3. Run:
   ```bash
   python main.py --query "acetonitrile HPLC grade"
   ```

Output files are written to `outputs/` in this directory.

---

## Architecture

The pipeline runs in 7 steps. The LLM is called in steps 2 and 6 only.
All data transforms in steps 3–5 and 7 are pure Python.

```
User query
    │
    ▼
[Step 2] LLM (expansion chain)
    Expand query → canonical name, CAS number, 2–4 targeted search strings
    │
    ▼
[Step 3] Tavily search
    Run each search string → deduplicated list of vendor URLs + page content
    • Vendor/unknown URLs    → raw_content (full rendered page text)
    • Non-vendor/academic URLs (PubMed, Wikipedia, ScienceDirect, etc.)
                             → content (Tavily short summary ~200 chars)
    │
    ▼
[Step 4] LLM (extraction chain) — one page at a time
    Content truncated to max_page_chars (config.yaml, default 4000 chars)
    extract_from_page() → LLM reads text → JSON list of product dicts
    │
    ▼
[Step 5] Python normalization + ranking (no LLM)
    normalize_records() → parse price/size/purity → compute price_per_unit → deduplicate
    rank_products()     → sort by purity desc, price_per_unit asc
    │
    ▼
[Step 6] LLM (summary chain)
    Ranked markdown table → procurement recommendation (best value, purity, strategy)
    │
    ▼
[Step 7] Save outputs
    outputs/<slug>_products.csv
    outputs/<slug>_report.md
```

---

## File Reference

| File | What it does |
|---|---|
| `main.py` | Entry point; orchestrates all 7 steps |
| `llm.py` | Builds the three LangChain LCEL chains (`expansion`, `extraction`, `summary`) |
| `search.py` | Calls Tavily API with `include_raw_content=True`, deduplicates results by URL |
| `extraction.py` | Calls the extraction chain, validates and returns product dicts |
| `normalize.py` | Parses strings → numbers, computes price_per_unit, deduplicates, ranks |
| `config.yaml` | Model ID, region, search/scrape limits, output directory |
| `prompts/expansion.txt` | Prompt for query expansion (returns JSON) |
| `prompts/extraction.txt` | Prompt for product extraction from page text (returns JSON array) |
| `prompts/summary.txt` | Prompt for procurement recommendation (returns plain text) |

---

## Token Optimization

Sending full raw page content to the LLM for every result is expensive. Two strategies are applied to keep token usage predictable:

### Strategy 1 — `max_page_chars` truncation (Step 4)

Before any page text is sent to the extraction LLM, it is hard-truncated to `max_page_chars` characters (configured in `config.yaml`, default `4000`). Product information is almost always near the top of a vendor page — catalog name, CAS, price, purity, package sizes — so truncating the tail loses very little signal while eliminating large amounts of boilerplate (footer links, legal text, related products carousels, etc.).

Adjust in `config.yaml`:
```yaml
max_page_chars: 4000   # increase for verbose pages, decrease to cut costs further
```

The runtime output logs the character count sent for each URL:
```
  [1/12] https://www.sigmaaldrich.com/...
      Sending 4,000 chars to LLM
  [2/12] https://pubmed.ncbi.nlm.nih.gov/...
    [search] (summary only — non-vendor domain): ...
      Sending 198 chars to LLM
```

### Strategy 3 — Non-vendor domain routing (Step 3)

Academic literature sites, databases, and encyclopaedias will never contain purchasable products with prices or catalog numbers. Sending their full text to the LLM wastes the entire token budget for that page.

`search.py` maintains a `NON_VENDOR_DOMAINS` regex that matches known non-vendor domains:

```
pubmed.ncbi, ncbi.nlm, wikipedia.org, sciencedirect.com,
nature.com, pmc.ncbi, hero.epa, researchgate.net,
chemspider.com, drugbank.ca, ebi.ac.uk, rsc.org
```

Matching URLs receive only Tavily's short `content` summary (~200 chars) instead of `raw_content`. They are still passed to the extraction LLM so the agent can confirm there are no products, but the cost is negligible.

**Combined impact:** On a typical 8-result query, 2–4 results are usually non-vendor pages. Routing those to short summaries and capping all content at 4,000 chars reduces total extraction-step token consumption by roughly 50–70% compared to sending uncapped `raw_content` for every page.

---

## Runtime Example

```bash
python main.py --query "isotope labeled testosterone"
```

**Step 2 — Query expansion (LLM)**

```
  Canonical name : Isotope-labeled testosterone
  CAS number     : unknown
  Search queries : ['isotope labeled testosterone buy supplier',
                    'deuterated testosterone d3 research grade',
                    '13C testosterone labeled standard price',
                    'testosterone isotope labeled internal standard']
```

**Step 3 — Tavily search**

28 unique URLs returned across all 4 queries. One non-vendor URL is automatically downgraded to a short summary:

```
  [search] Querying Tavily: 'isotope labeled testosterone buy supplier'
  [search] Querying Tavily: 'deuterated testosterone d3 research grade'
  [search] Querying Tavily: '13C testosterone labeled standard price'
  [search] Querying Tavily: 'testosterone isotope labeled internal standard'
    [search] (summary only — non-vendor domain): https://pubmed.ncbi.nlm.nih.gov/23431483/
  [search] Found 28 unique URLs across all queries.
```

**Step 4 — Extraction (LLM, per page)**

Each vendor page is truncated to 4,000 chars before the LLM sees it. Non-vendor summaries are much shorter:

```
  [1/28] https://www.fishersci.com/shop/products/acetonitrile-2-13c-99-1-g/501630158
      Sending 4,000 chars to LLM
      [extract] Extracted 1 record(s) from ...
  [2/28] https://www.musechem.com/isotope-labelled-compound/isotope-labeled-steroids/...
      Sending 4,000 chars to LLM
      [extract] Extracted 11 record(s) from ...
  [3/28] https://www.targetmol.com/compound/testosterone_d3
      Sending 4,000 chars to LLM
      [extract] Extracted 4 record(s) from ...
  ...
  [25/28] https://pubmed.ncbi.nlm.nih.gov/23431483/
      Sending 2,241 chars to LLM       ← short summary, not raw_content
      [extract] Extracted 0 record(s) from ...

  Total raw records collected: 26
```

**Step 5 — Normalization and ranking**

```
  19 unique products after deduplication and ranking.
```

Top results found across vendors (purity desc, price/g asc):

| vendor | product_name | purity | size | price | availability |
|---|---|---|---|---|---|
| Fisher Scientific | Cambridge Isotope Laboratories TESTOSTERONE (3,4-¹³C₂, 99%) | 99% | 0.01 g | — | — |
| Cambridge Isotope Laboratories | Testosterone (2,3,4-¹³C₃, 99%), CLM-9164-C | 99% | 100 µg/mL | — | — |
| Sigma-Aldrich | Testosterone-2,3,4-¹³C₃ solution (#730610), 98% (CP) | 98% (CP) | 1 mL in ampule | — | — |
| Cayman Chemical | Testosterone-d3 (CAS 77546-39-5) | — | — | — | — |
| TargetMol | Testosterone-D3 (TMIH-0560) | — | 25 mg | $4,220 | 7–10 days |
| TargetMol | Testosterone-D3 (TMIH-0560) | — | 10 mg | $2,280 | 7–10 days |
| TargetMol | Testosterone-D3 (TMIH-0560) | — | 5 mg | $1,370 | 7–10 days |
| TargetMol | Testosterone-D3 (TMIH-0560) | — | 1 mg | $457 | In Stock |

**Step 6 — Procurement recommendation (LLM)**

The LLM identified Cambridge Isotope Laboratories + Sigma-Aldrich as the recommended multi-vendor strategy for highest purity (99% vs 98% CP respectively), flagged TargetMol pricing as requiring verification ($168,800–$457,000/g), and noted that rows from MUSECHEM were related steroids rather than testosterone — search result contamination worth investigating with more targeted queries or `include_domains` filtering.

**Outputs**

```
outputs/isotope_labeled_testosterone_sources.csv    ← all 28 URLs searched
outputs/isotope_labeled_testosterone_products.csv   ← 19 ranked products
outputs/isotope_labeled_testosterone_report.md      ← markdown table + LLM procurement summary
```

---


The CSV contains these columns:

| Column | Description |
|---|---|
| `chemical_name` | Name as extracted from the page |
| `cas_number` | CAS registry number |
| `vendor` | Vendor name |
| `product_name` | Full product/catalog name |
| `purity` | Purity string as shown (e.g. "≥99.9%") |
| `grade` | Quality grade (e.g. "HPLC", "ACS", "reagent") |
| `size` | Package size string (e.g. "500 mL") |
| `price` | Price string as shown (e.g. "$45.20") |
| `availability` | Stock status if shown |
| `url` | Source product page URL |
| `price_usd` | Price parsed to float |
| `size_normalized` | Size converted to base unit (g or mL) |
| `size_base_unit` | "g" or "mL" |
| `purity_pct` | Purity parsed to float 0–100 |
| `price_per_unit` | `price_usd / size_normalized` ($/g or $/mL) |

---

## Common Issues

**`Missing required environment variables`**
→ Make sure `.env` exists in the `chemical_sourcing_agent/` directory and all three keys are set.

**`No product records were extracted`**
→ Tavily may not have returned relevant content for those URLs.
Try a more specific query, or increase `max_search_results` in `config.yaml`.

**`Bedrock throttle / ResourceNotFoundException`**
→ Confirm the model is enabled in your AWS account under Bedrock > Model access.
The model ID must exactly match what's in `config.yaml`.

**LLM returns invalid JSON**
→ `extraction.py` and `expansion` steps will print a warning and continue.
If this happens frequently, lower `max_page_chars` in `config.yaml` to reduce context length.
