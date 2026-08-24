"""Analytics Module for AI Receipt Analyser.

Calculates spending breakdowns, category metrics, and validates
arithmetic consistency on structured receipt data using Pandas.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import pandas as pd
from llm_parser import ReceiptAnalysis, ReceiptItem


def compute_metrics(
    receipt_data: ReceiptAnalysis,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute financial analytics and category summaries from parsed receipt data.

    Args:
        receipt_data: Validated ReceiptAnalysis Pydantic model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - items_df: DataFrame containing each purchased item (name, category, quantity, price).
            - category_summary_df: Aggregated DataFrame grouped by category with:
                * 'category': Item category
                * 'total_spent': Sum of prices in that category
                * 'item_count': Number of items purchased in that category
                * 'percentage': Percentage share of the receipt's total spend
              Sorted in descending order of total_spent.
    """
    # Define canonical column schemas for consistent outputs
    items_columns = ["name", "category", "quantity", "price"]
    summary_columns = ["category", "total_spent", "item_count", "percentage"]

    # Handle empty items edge case
    if not receipt_data.items:
        items_df = pd.DataFrame(columns=items_columns)
        category_summary_df = pd.DataFrame(columns=summary_columns)
        return items_df, category_summary_df

    # Convert items list to records
    items_records: List[Dict[str, Any]] = [
        {
            "name": item.name,
            "category": item.category,
            "quantity": float(item.quantity),
            "price": float(item.price),
        }
        for item in receipt_data.items
    ]

    items_df = pd.DataFrame(items_records, columns=items_columns)

    # Edge case: If items_df is empty for any reason
    if items_df.empty:
        category_summary_df = pd.DataFrame(columns=summary_columns)
        return items_df, category_summary_df

    # Group by category and compute metrics
    grouped = (
        items_df.groupby("category", as_index=False)
        .agg(
            total_spent=("price", "sum"),
            item_count=("name", "count"),
        )
    )

    # Calculate percentage share of total receipt spend
    # Use receipt_data.total or fallback to sum of item prices if total is 0
    receipt_total = receipt_data.total if receipt_data.total > 0 else items_df["price"].sum()
    
    if receipt_total > 0:
        grouped["percentage"] = (grouped["total_spent"] / receipt_total) * 100.0
    else:
        grouped["percentage"] = 0.0

    # Round metrics for presentation
    grouped["total_spent"] = grouped["total_spent"].round(2)
    grouped["percentage"] = grouped["percentage"].round(2)

    # Sort categories in descending order of spend
    category_summary_df = (
        grouped.sort_values(by="total_spent", ascending=False)
        .reset_index(drop=True)
    )

    return items_df, category_summary_df


def verify_arithmetic(
    receipt_data: ReceiptAnalysis,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Perform audit arithmetic checks on receipt subtotal and total against itemized sums.

    Args:
        receipt_data: Validated ReceiptAnalysis Pydantic model.
        tolerance: Allowed difference threshold in dollars (default 0.05 for rounding variances).

    Returns:
        Dict[str, Any]: Detailed audit report containing:
            - 'calculated_items_sum': Sum of individual item prices
            - 'declared_subtotal': Subtotal extracted from receipt
            - 'declared_tax': Tax extracted from receipt
            - 'declared_total': Final total extracted from receipt
            - 'expected_total': calculated_items_sum + declared_tax
            - 'subtotal_discrepancy': Difference between declared subtotal and items sum
            - 'total_discrepancy': Difference between declared total and expected total
            - 'is_subtotal_accurate': Boolean indicating if subtotal matches within tolerance
            - 'is_total_accurate': Boolean indicating if total matches within tolerance
            - 'has_discrepancy': True if either subtotal or total fails tolerance test
            - 'discrepancy_flags': List of human-readable warnings if discrepancies exist
    """
    calculated_items_sum = round(sum(item.price for item in receipt_data.items), 2)
    declared_subtotal = round(receipt_data.subtotal, 2)
    declared_tax = round(receipt_data.tax, 2)
    declared_total = round(receipt_data.total, 2)

    expected_total = round(calculated_items_sum + declared_tax, 2)

    subtotal_diff = round(abs(declared_subtotal - calculated_items_sum), 2)
    total_diff = round(abs(declared_total - expected_total), 2)

    is_subtotal_accurate = subtotal_diff <= tolerance
    is_total_accurate = total_diff <= tolerance
    has_discrepancy = not (is_subtotal_accurate and is_total_accurate)

    discrepancy_flags: List[str] = []
    if not is_subtotal_accurate:
        discrepancy_flags.append(
            f"Subtotal mismatch: Sum of item prices (${calculated_items_sum:.2f}) "
            f"differs from declared subtotal (${declared_subtotal:.2f}) by ${subtotal_diff:.2f}."
        )
    if not is_total_accurate:
        discrepancy_flags.append(
            f"Total mismatch: Expected total (${expected_total:.2f}) "
            f"differs from declared total (${declared_total:.2f}) by ${total_diff:.2f}."
        )

    return {
        "calculated_items_sum": calculated_items_sum,
        "declared_subtotal": declared_subtotal,
        "declared_tax": declared_tax,
        "declared_total": declared_total,
        "expected_total": expected_total,
        "subtotal_discrepancy": subtotal_diff,
        "total_discrepancy": total_diff,
        "is_subtotal_accurate": is_subtotal_accurate,
        "is_total_accurate": is_total_accurate,
        "has_discrepancy": has_discrepancy,
        "discrepancy_flags": discrepancy_flags,
    }


if __name__ == "__main__":
    # Self-test demonstration with sample receipt data
    sample_receipt = ReceiptAnalysis(
        merchant="Fresh Harvest Market",
        date="2024-08-20",
        items=[
            ReceiptItem(name="Organic Apples", category="Produce", quantity=2.0, price=5.98),
            ReceiptItem(name="Whole Milk 1 Gal", category="Dairy", quantity=1.0, price=3.89),
            ReceiptItem(name="Ribeye Steak", category="Meat", quantity=1.0, price=14.50),
            ReceiptItem(name="Sourdough Bread", category="Bakery", quantity=1.0, price=4.25),
            ReceiptItem(name="Potato Chips", category="Snacks", quantity=2.0, price=6.00),
            ReceiptItem(name="Sparkling Water", category="Beverages", quantity=1.0, price=2.99),
        ],
        subtotal=37.61,
        tax=2.39,
        total=40.00,
        spending_summary="Total spend of $40.00 focused primarily on fresh meat and produce.",
        discretionary_flags=["Potato Chips – snack item ($6.00)"],
        budget_advice=["Buy snacks in bulk", "Look for bakery discounts on evening visits"],
    )

    print("=== 1. Items DataFrame ===")
    df_items, df_summary = compute_metrics(sample_receipt)
    print(df_items.to_string(index=False))

    print("\n=== 2. Category Summary DataFrame (Sorted by Spend) ===")
    print(df_summary.to_string(index=False))

    print("\n=== 3. Arithmetic Verification Audit ===")
    audit_report = verify_arithmetic(sample_receipt)
    for key, value in audit_report.items():
        print(f"  {key}: {value}")
