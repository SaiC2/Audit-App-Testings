# modules/payroll.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class PayrollModule(Module):
    name = "Payroll Testing"
    input_template_path = "Payroll_template.xlsx"

    template_columns = [
        "Employee ID", "Employee Name", "Pay Date", "Pay Period",
        "Gross Pay", "Tax", "Net Pay", "Department"
    ]

    # --- Tests ---
    def _test_overtime_anomalies(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Overtime Anomalies", df)

    def _test_duplicate_employees(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Duplicate Employees", df)

    def _test_net_gt_gross(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Net>Gross Check", df)

    def _test_weekend_payments(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Weekend Payments", df)

    tests: List[TestSpec] = [
        TestSpec(id="overtime_anomalies", label="Overtime Anomalies", description="Unusual overtime patterns", func=_test_overtime_anomalies),
        TestSpec(id="dup_employees", label="Duplicate Employees", description="Potential duplicate employees", func=_test_duplicate_employees),
        TestSpec(id="net_gt_gross", label="Net>Gross Check", description="Net pay greater than gross", func=_test_net_gt_gross),
        TestSpec(id="weekend_payments", label="Weekend Payments", description="Payments processed on weekends", func=_test_weekend_payments),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results
