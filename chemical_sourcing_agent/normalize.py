"""
normalize.py — Pure Python/pandas transforms on raw LLM-extracted records.

This module intentionally has NO LLM calls. All the operations here are
deterministic: given the same input, you always get the same output.
This makes them easy to test, debug, and trust.

Key transforms:
  1. Parse price strings like "$45.20" or "USD 120" into floats.
  2. Parse size strings like "500 mL" or "1 kg" into a number + unit.
  3. Convert sizes to a common base unit (grams or mL) so we can compare
     e.g. "500 g" vs "1 kg" on equal footing.
  4. Compute price_per_unit = price / normalized_size.
  5. Parse purity strings like "≥99.5%" or "99%" into a float (0–100).
  6. Deduplicate records that represent the same product+size from the same vendor.
  7. Rank by purity (high to low), then price_per_unit (low to high).
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: parse price
# ---------------------------------------------------------------------------

def _parse_price(price_str) -> float | None:
    """
    Extract a numeric dollar amount from a price string.

    Examples:
      "$45.20"    -> 45.20
      "USD 120"   -> 120.0
      "45,200.00" -> 45200.0
      None        -> None
    """
    if not price_str or not isinstance(price_str, str):
        return None

    # Remove currency symbols and words, keep digits, dots, and commas.
    # Then remove commas used as thousands separators.
    cleaned = re.sub(r"[^\d.,]", "", price_str)
    cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Helper: parse size  →  (numeric value in base unit, display unit)
# ---------------------------------------------------------------------------

# Conversion factors to a base unit:
#   mass  → grams
#   volume → mL
SIZE_CONVERSIONS = {
    "kg":  1000.0,
    "g":   1.0,
    "mg":  0.001,
    "l":   1000.0,   # litre → mL
    "ml":  1.0,
    "µl":  0.001,
    "ul":  0.001,
}

# Base unit labels shown in output
BASE_UNIT_LABELS = {
    "kg": "g", "g": "g", "mg": "g",
    "l": "mL", "ml": "mL", "µl": "mL", "ul": "mL",
}


def _parse_size(size_str) -> tuple[float | None, str | None, str | None]:
    """
    Parse a size string into (normalized_value, base_unit, display_unit).

    Examples:
      "500 mL"  -> (500.0,  "mL", "mL")
      "1 kg"    -> (1000.0, "g",  "kg")
      "2.5 g"   -> (2.5,    "g",  "g")
      "100mg"   -> (0.1,    "g",  "mg")
      None      -> (None, None, None)
    """
    if not size_str or not isinstance(size_str, str):
        return None, None, None

    # Match a number (with optional decimal) followed by a unit.
    # re.IGNORECASE handles "ML", "G", "KG", etc.
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(kg|g|mg|µl|ul|ml|l)\b",
        size_str,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None, None

    value = float(match.group(1))
    unit = match.group(2).lower()

    factor = SIZE_CONVERSIONS.get(unit, 1.0)
    base_unit = BASE_UNIT_LABELS.get(unit, unit)

    return value * factor, base_unit, unit


# ---------------------------------------------------------------------------
# Helper: parse purity
# ---------------------------------------------------------------------------

def _parse_purity(purity_str) -> float | None:
    """
    Extract a numeric purity percentage from a string.

    Examples:
      "99.9%"     -> 99.9
      "≥99%"      -> 99.0
      ">99.5%"    -> 99.5
      "HPLC grade"-> None  (no numeric value)
      None        -> None
    """
    if not purity_str or not isinstance(purity_str, str):
        return None

    # Find the first number in the string (handles "≥99.5%" or ">99%").
    match = re.search(r"(\d+(?:\.\d+)?)", purity_str)
    if not match:
        return None

    value = float(match.group(1))

    # Sanity check: purity as a percentage should be between 0 and 100.
    # If someone wrote "0.999" (a fraction), convert it.
    if value <= 1.0:
        value *= 100.0

    return value if 0.0 <= value <= 100.0 else None


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert a flat list of raw LLM-extracted product dicts into a clean DataFrame.

    Adds columns: price_usd, size_normalized, size_base_unit, size_display_unit,
                  price_per_unit, purity_pct.

    Args:
        records: list of dicts, each with the 10 fields from the extraction prompt.

    Returns:
        A pandas DataFrame. Rows missing both price_usd AND purity_pct are dropped
        because they have no useful information for comparison.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure all expected columns exist (fill missing ones with None).
    for col in ["chemical_name", "cas_number", "vendor", "product_name",
                "purity", "grade", "size", "price", "availability", "url"]:
        if col not in df.columns:
            df[col] = None

    # --- Parse numeric fields ---
    df["price_usd"] = df["price"].apply(_parse_price)

    parsed_sizes = df["size"].apply(_parse_size)
    df["size_normalized"]   = parsed_sizes.apply(lambda t: t[0])  # e.g. 500.0
    df["size_base_unit"]    = parsed_sizes.apply(lambda t: t[1])  # e.g. "mL"
    df["size_display_unit"] = parsed_sizes.apply(lambda t: t[2])  # e.g. "mL"

    df["purity_pct"] = df["purity"].apply(_parse_purity)

    # --- Compute price per unit ---
    # Only possible when we have both a price and a size.
    # Result is in $/g or $/mL depending on the size unit.
    df["price_per_unit"] = df.apply(
        lambda row: (
            round(row["price_usd"] / row["size_normalized"], 4)
            if pd.notna(row["price_usd"]) and pd.notna(row["size_normalized"])
            and row["size_normalized"] > 0
            else None
        ),
        axis=1,
    )

    # --- Drop rows with no useful comparison data ---
    df = df.dropna(subset=["price_usd", "purity_pct"], how="all")

    # --- Deduplicate ---
    # Keep the first occurrence of each (vendor, product_name, size) combination.
    # This handles cases where the same product appears via different search queries.
    df = df.drop_duplicates(subset=["vendor", "product_name", "size"], keep="first")

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Ranking function
# ---------------------------------------------------------------------------

def rank_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort products so the most useful options appear first.

    Sorting priority:
      1. Purity (descending) — higher purity first
      2. Price per unit (ascending) — cheaper per unit first

    Records missing both sort keys are placed at the end.

    Args:
        df: the DataFrame returned by normalize_records()

    Returns:
        A new DataFrame sorted by the above criteria with a fresh integer index.
    """
    if df.empty:
        return df

    # pandas sort_values treats NaN as the largest value by default.
    # na_position="last" makes NaN rows go to the bottom for both sort directions.
    df_sorted = df.sort_values(
        by=["purity_pct", "price_per_unit"],
        ascending=[False, True],  # purity: high first; price_per_unit: low first
        na_position="last",
    )

    return df_sorted.reset_index(drop=True)
