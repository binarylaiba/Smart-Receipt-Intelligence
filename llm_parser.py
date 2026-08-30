"""LLM Parser Module for AI Receipt Analyser.

This module utilizes Pydantic v2 and the official Groq Python SDK
(using model `llama-3.3-70b-versatile`) to parse and structure raw OCR
receipt text into a validated financial contract.
"""

from __future__ import annotations

import os
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError

from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
    # Load .env from workspace or parent directory
    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file, override=True)
    else:
        load_dotenv(find_dotenv(), override=True)
except ImportError:
    pass

try:
    from groq import Groq, APIError, AuthenticationError
except ImportError:
    Groq = None  # type: ignore
    APIError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore


# Define allowed category taxonomy
ReceiptCategory = Literal[
    "Produce",
    "Dairy",
    "Meat",
    "Bakery",
    "Snacks",
    "Beverages",
    "Household",
    "Personal Care",
    "Other",
]


class ReceiptItem(BaseModel):
    """Represents an individual line item extracted from a receipt."""

    name: str = Field(
        ...,
        description="Cleaned, human-readable name of the purchased item, free from OCR artifacts.",
    )
    category: ReceiptCategory = Field(
        ...,
        description="Standardized item category classification.",
    )
    quantity: float = Field(
        default=1.0,
        description="Quantity of item purchased (defaults to 1.0 if not specified).",
    )
    price: float = Field(
        ...,
        description="Total price charged for this item (or total line price).",
    )


class ReceiptAnalysis(BaseModel):
    """Complete structured financial analysis of a receipt."""

    merchant: str = Field(
        ...,
        description="Name of the store, supermarket, or merchant.",
    )
    date: str = Field(
        ...,
        description="Date of purchase in YYYY-MM-DD or standard recognizable format.",
    )
    items: List[ReceiptItem] = Field(
        default_factory=list,
        description="List of structured items purchased on the receipt.",
    )
    subtotal: float = Field(
        ...,
        description="Subtotal amount before taxes/discounts.",
    )
    tax: float = Field(
        default=0.0,
        description="Tax or surcharge amount.",
    )
    total: float = Field(
        ...,
        description="Final total payment amount.",
    )
    spending_summary: str = Field(
        ...,
        description="Concise 1-2 sentence executive summary of total spending and dominant categories.",
    )
    discretionary_flags: List[str] = Field(
        default_factory=list,
        description="List of flagged non-essential / high-cost discretionary items with reasoning.",
    )
    budget_advice: List[str] = Field(
        default_factory=list,
        description="Exactly 3 realistic, personalized money-saving tips based on items in this receipt.",
    )


def _build_system_prompt() -> str:
    """Generate the strict system instructions with embedded JSON schema."""
    return (
        "You are an expert financial auditor and OCR correction engine.\n"
        "Your task is to analyze noisy, raw OCR text from a physical receipt and output a pristine, structured JSON object.\n\n"
        "### STRICT JSON OUTPUT STRUCTURE:\n"
        "You MUST include every single key shown below in your JSON response:\n"
        "{\n"
        '  "merchant": "Store Name",\n'
        '  "date": "YYYY-MM-DD",\n'
        '  "items": [\n'
        "    {\n"
        '      "name": "Item Name",\n'
        '      "category": "One of: Produce, Dairy, Meat, Bakery, Snacks, Beverages, Household, Personal Care, Other",\n'
        '      "quantity": 1.0,\n'
        '      "price": 0.00\n'
        "    }\n"
        "  ],\n"
        '  "subtotal": 0.00,\n'
        '  "tax": 0.00,\n'
        '  "total": 0.00,\n'
        '  "spending_summary": "1-2 sentence summary of spending and dominant categories",\n'
        '  "discretionary_flags": ["Non-essential item 1 with short reason", "Item 2..."],\n'
        '  "budget_advice": ["Tip 1", "Tip 2", "Tip 3"]\n'
        "}\n\n"
        "### GUIDELINES:\n"
        "1. Fix OCR mistakes (e.g. 'O' vs '0', 'l' vs '1', missing decimal points like 499 -> 4.99).\n"
        "2. Item category MUST be strictly one of: 'Produce', 'Dairy', 'Meat', 'Bakery', 'Snacks', 'Beverages', 'Household', 'Personal Care', 'Other'.\n"
        "3. Ensure all numbers (quantity, price, subtotal, tax, total) are floats.\n"
        "4. Always provide spending_summary, discretionary_flags, and exactly 3 budget_advice tips."
    )


def resolve_groq_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve Groq API key from parameter, Streamlit secrets, or environment variables.

    Args:
        api_key: Optional explicitly provided key.

    Returns:
        Optional[str]: Cleaned API key string if found, otherwise None.
    """
    if api_key and str(api_key).strip():
        return str(api_key).strip().strip("'\"")

    # 1. Check Streamlit Secrets (for Streamlit Community Cloud deployments)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key_name in ["GROQ_API_KEY", "groq_api_key", "groq_api", "GROQ_KEY"]:
                if key_name in st.secrets:
                    val = st.secrets[key_name]
                    if val and str(val).strip():
                        return str(val).strip().strip("'\"")
            # Also check nested dictionary [groq] or [secrets]
            if "groq" in st.secrets and isinstance(st.secrets["groq"], dict):
                for sub_key in ["api_key", "GROQ_API_KEY", "groq_api_key"]:
                    if sub_key in st.secrets["groq"]:
                        val = st.secrets["groq"][sub_key]
                        if val and str(val).strip():
                            return str(val).strip().strip("'\"")
    except Exception:
        pass

    # 2. Check direct environment variables
    for env_var in ["GROQ_API_KEY", "groq_api_key", "groq_api", "GROQ_KEY"]:
        val = os.environ.get(env_var)
        if val and val.strip():
            return val.strip().strip("'\"")

    # 3. Try reloading from local .env file
    try:
        from dotenv import load_dotenv, find_dotenv
        _env_file = Path(__file__).resolve().parent / ".env"
        if _env_file.exists():
            load_dotenv(dotenv_path=_env_file, override=True)
        else:
            load_dotenv(find_dotenv(), override=True)
        for env_var in ["GROQ_API_KEY", "groq_api_key", "groq_api", "GROQ_KEY"]:
            val = os.environ.get(env_var)
            if val and val.strip():
                return val.strip().strip("'\"")
    except Exception:
        pass

    return None


def parse_receipt_with_groq(
    raw_ocr_text: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> ReceiptAnalysis:
    """Parse raw receipt OCR text into a validated ReceiptAnalysis Pydantic model using Groq.

    Args:
        raw_ocr_text: Unstructured text extracted by OCR engine.
        api_key: Optional Groq API key. If omitted, resolves from Streamlit secrets, environment, or .env.
        model_name: Optional Groq model identifier (defaults to available high-performing models).

    Returns:
        ReceiptAnalysis: Validated Pydantic model with structured receipt data.

    Raises:
        ImportError: If the 'groq' package is not installed.
        ValueError: If OCR text is empty or API key is missing.
        ValidationError: If model output does not conform to the ReceiptAnalysis schema.
        RuntimeError: If Groq API request fails or response cannot be parsed.
    """
    if Groq is None:
        raise ImportError(
            "The 'groq' package is required. Install it with: `py -m pip install groq`"
        )

    if not raw_ocr_text or not raw_ocr_text.strip():
        raise ValueError("raw_ocr_text cannot be empty.")

    resolved_api_key = resolve_groq_api_key(api_key)

    if not resolved_api_key:
        raise ValueError(
            "Groq API key not found. Please provide `api_key`, enter it in the Streamlit sidebar, "
            "or configure `GROQ_API_KEY` in Streamlit Secrets (App Settings -> Secrets) or your .env file."
        )

    # Valid Groq model candidates in priority order
    candidate_models = (
        [model_name]
        if model_name
        else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]
    )

    client = Groq(api_key=resolved_api_key)
    system_prompt = _build_system_prompt()
    user_prompt = f"Here is the raw OCR text of the receipt:\n---\n{raw_ocr_text}\n---"

    last_error: Optional[Exception] = None

    for model in candidate_models:
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            response_content = chat_completion.choices[0].message.content
            if not response_content:
                continue

            # Validate with Pydantic v2
            receipt_data = ReceiptAnalysis.model_validate_json(response_content)
            return receipt_data

        except ValidationError:
            raise
        except Exception as err:
            last_error = err
            continue

    if last_error:
        raise RuntimeError(f"Groq API Error: {last_error}") from last_error
    raise RuntimeError("Failed to obtain a valid response from Groq.")


if __name__ == "__main__":
    import sys
    # Ensure Windows console handles UTF-8 characters gracefully
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Quick demonstration with sample noisy OCR text
    sample_noisy_ocr = """
    W4LM4RT SUPERCENTER #1024
    DATE: 2O24-O8-15
    
    ORG BANANAS        1.89
    2% REDUCED MILK    3.49
    DORITOS CHIPS NACHO 4.99
    CHKN BREAST BNLSS  11.20
    COCA COLA 12PK     7.99
    TIDE LAUNDRY PODS 19.99
    
    SUBT0TAL          49.55
    TAX 7.5%           3.72
    T0TAL             53.27
    """

    print("--- Sample Noisy OCR Input ---")
    print(sample_noisy_ocr)

    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api")
    if groq_key:
        print("\nSending to Groq LLM...")
        try:
            analysis = parse_receipt_with_groq(sample_noisy_ocr, api_key=groq_key)
            print("\n=== Validated Receipt Analysis ===")
            print(f"Merchant: {analysis.merchant}")
            print(f"Date:     {analysis.date}")
            print(f"Total:    ${analysis.total:.2f}")
            print("\nItems:")
            for itm in analysis.items:
                print(f"  - [{itm.category}] {itm.name} (x{itm.quantity}): ${itm.price:.2f}")
            print(f"\nSpending Summary:\n{analysis.spending_summary}")
            print("\nDiscretionary Flags:")
            for flag in analysis.discretionary_flags:
                print(f"  * {flag}")
            print("\nBudget Advice:")
            for tip in analysis.budget_advice:
                print(f"  - {tip}")
        except Exception as e:
            print(f"Error during parsing: {e}")
    else:
        print("\nNote: Set GROQ_API_KEY in your .env file to test live inference.")
