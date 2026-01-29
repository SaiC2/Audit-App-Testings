# modules/expenses.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class ExpensesModule(Module):
    name = "Expenses Testing"
    input_template_path = "Expenses_template.xlsx"

    template_columns = [
        "Expense ID", "Employee", "Date", "Category",
        "Amount", "Description", "Approver"
    ]

    # --- Tests ---
    def _test_policy_breach(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Policy Breach", df)

    def _test_weekend_expenses(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Weekend Expenses", df)

    def _test_round_amounts(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Round Amounts", df)

    def _test_missing_receipts(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Missing Receipts", df)

    tests: List[TestSpec] = [
        TestSpec(id="policy_breach", label="Policy Breach", description="Claims breaching policy", func=_test_policy_breach),
        TestSpec(id="weekend", label="Weekend Expenses", description="Expenses claimed on weekends", func=_test_weekend_expenses),
        TestSpec(id="round_amounts", label="Round Amounts", description="Suspicious round-number expenses", func=_test_round_amounts),
        TestSpec(id="missing_receipts", label="Missing Receipts", description="Claims without receipts", func=_test_missing_receipts),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results