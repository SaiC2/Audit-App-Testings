# modules/revenue.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class RevenueModule(Module):
    name = "Revenue Testing"

    input_template_path = "Revenue_template.xlsx"

    template_columns = [
        "Order No", "Invoice No", "Customer", "Invoice Date",
        "Amount", "Product", "Salesperson"
    ]

    # --- Tests ---
    def _test_revenue_cutoff(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Revenue Cutoff", df)

    def _test_negative_lines(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Negative Lines", df)

    def _test_unusual_discounts(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Unusual Discounts", df)

    def _test_duplicate_invoices(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Duplicate Invoices", df)

    tests: List[TestSpec] = [
        TestSpec(id="cutoff", label="Revenue Cutoff", description="Cutoff around period end", func=_test_revenue_cutoff),
        TestSpec(id="negative_lines", label="Negative Lines", description="Negative line items in invoices", func=_test_negative_lines),
        TestSpec(id="unusual_discounts", label="Unusual Discounts", description="Abnormally high discounts", func=_test_unusual_discounts),
        TestSpec(id="dup_invoices", label="Duplicate Invoices", description="Potential duplicate invoicing", func=_test_duplicate_invoices),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results