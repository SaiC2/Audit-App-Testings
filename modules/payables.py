# modules/payables.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class PayablesModule(Module):
    name = "Payables Testing"
    input_template_path = "Payables_template.xlsx"

    template_columns = [
        "Voucher No", "Vendor", "Invoice Date", "Due Date",
        "Amount", "GL Account", "User"
    ]

    # --- Tests ---
    def _test_duplicate_vendors(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Duplicate Vendors", df)

    def _test_early_late_payments(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Early/Late Payments", df)

    def _test_high_value_payments(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("High Value Payments", df)

    def _test_weekend_postings(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Weekend Postings", df)

    tests: List[TestSpec] = [
        TestSpec(id="dup_vendors", label="Duplicate Vendors", description="Potential duplicate suppliers", func=_test_duplicate_vendors),
        TestSpec(id="early_late", label="Early/Late Payments", description="Payments outside expected terms", func=_test_early_late_payments),
        TestSpec(id="high_value", label="High Value Payments", description="Unusually large payments", func=_test_high_value_payments),
        TestSpec(id="weekend", label="Weekend Postings", description="Payables posted on weekends", func=_test_weekend_postings),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results