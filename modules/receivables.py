# modules/receivables.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class ReceivablesModule(Module):
    name = "Receivables Testing"

    input_template_path = "receivables_template.xlsx"

    template_columns = [
        "Invoice No", "Customer", "Invoice Date", "Due Date",
        "Amount", "Allocation", "Status"
    ]

    def _test_aging(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Aging", df)

    def _test_top_overdue(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Top Overdue Customers", df)

    def _test_missing_customer(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Missing Customer", df)

    def _test_negative_balances(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Negative Balances", df)

    def _test_duplicate_invoices(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Duplicate Invoices", df)

    tests: List[TestSpec] = [
        TestSpec(id="aging",             label="Aging",                    description="Aging buckets",    func=_test_aging),
        TestSpec(id="top_overdue",       label="Top Overdue Customers",    description="Top overdue",      func=_test_top_overdue),
        TestSpec(id="missing_customer",  label="Missing Customer",         description="Missing names",    func=_test_missing_customer),
        TestSpec(id="negative_balances", label="Negative Balances",        description="Negatives",        func=_test_negative_balances),
        TestSpec(id="dup_invoices",      label="Duplicate Invoices",       description="Potential dups",   func=_test_duplicate_invoices),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results