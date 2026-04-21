"""
llm.py — Builds the three LangChain LCEL chains used in this project.

Each chain follows the same simple pattern:
    PromptTemplate | ChatBedrock | OutputParser

The three chains are:
    - expansion : expands the user's query into synonyms + search queries (returns dict)
    - extraction: pulls structured product records from page text (returns list)
    - summary   : generates a purchasing recommendation from a ranked table (returns str)
"""

import os
from pathlib import Path

from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


def _load_prompt(filename: str) -> str:
    """Read a prompt template from the prompts/ directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text(encoding="utf-8")


def build_chains(config: dict) -> dict:
    """
    Instantiate ChatBedrock and build all three LCEL chains.

    Returns a dict with keys: 'expansion', 'extraction', 'summary'.
    Each value is a runnable chain you can call with .invoke({...}).

    Args:
        config: the loaded config.yaml dict (needs 'bedrock_model_id', 'region')
    """

    # --- LLM setup ---
    # ChatBedrock wraps Amazon Bedrock's InvokeModel API.
    # It handles converting LangChain messages into the format Bedrock expects.
    # Credentials come from environment variables (set via .env / AWS CLI).
    llm = ChatBedrock(
        model_id=config["bedrock_model_id"],
        region_name=config["region"],
        # Keep temperature low: we want deterministic, structured output.
        model_kwargs={"temperature": 0.0, "max_tokens": 2048},
    )

    # --- Output parsers ---
    # JsonOutputParser: expects the LLM to return raw JSON, parses it into
    #   a Python dict or list automatically. Raises an error if JSON is invalid.
    # StrOutputParser: just extracts the text content of the LLM response as-is.
    json_parser = JsonOutputParser()
    str_parser = StrOutputParser()

    # --- Chain 1: Query expansion ---
    # Input variables: {query}
    # Output: dict with canonical_name, cas_number, synonyms, search_queries
    expansion_chain = (
        PromptTemplate.from_template(_load_prompt("expansion.txt"))
        | llm
        | json_parser
    )

    # --- Chain 2: Product extraction ---
    # Input variables: {page_text}, {url}
    # Output: list of product dicts (one per pack size found on the page)
    extraction_chain = (
        PromptTemplate.from_template(_load_prompt("extraction.txt"))
        | llm
        | json_parser
    )

    # --- Chain 3: Procurement summary ---
    # Input variables: {chemical}, {cas_number}, {table}
    # Output: plain text recommendation (str)
    summary_chain = (
        PromptTemplate.from_template(_load_prompt("summary.txt"))
        | llm
        | str_parser
    )

    return {
        "expansion": expansion_chain,
        "extraction": extraction_chain,
        "summary": summary_chain,
    }
