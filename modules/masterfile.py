# modules/masterfile.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class MasterfileModule(Module):
    name = "Masterfile Testing"
    input_template_path = "Masterfile_template.xlsx"

    template_columns = [
        "Record ID", "Type", "Name", "Created Date",
        "Modified Date", "Status", "Owner"
    ]

    # --- Tests ---
    def _test_stale_records(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Stale Records", df)

    def _test_missing_required(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Missing Required Fields", df)

    def _test_unauthorized_owners(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Unauthorized Owners", df)

    def _test_recent_mass_changes(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Recent Mass Changes", df)

    tests: List[TestSpec] = [
        TestSpec(id="stale", label="Stale Records", description="Aged records without updates", func=_test_stale_records),
        TestSpec(id="missing_required", label="Missing Required Fields", description="Mandatory fields missing", func=_test_missing_required),
        TestSpec(id="unauth_owners", label="Unauthorized Owners", description="Ownership outside allowed list", func=_test_unauthorized_owners),
        TestSpec(id="mass_changes", label="Recent Mass Changes", description="Unusual volume of updates", func=_test_recent_mass_changes),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results