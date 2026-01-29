# modules/inventory.py
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from core.types import Module, TestSpec, DataFrame
from core.utils import placeholder_report_df


class InventoryModule(Module):
    name = "Inventory Testing"

    input_template_path = "inventory_template.xlsx"

    template_columns = [
        "Item", "Item Description", "Txn Date", "Txn Type",
        "Quantity", "Unit Cost", "Location", "User"
    ]

    # --- Tests ---
    def _test_negative_on_hand(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Negative On-hand", df)

    def _test_cost_variance(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Cost Variance", df)

    def _test_inactive_items(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Inactive Items", df)

    def _test_weekend_movements(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        return placeholder_report_df("Weekend Movements", df)

    tests: List[TestSpec] = [
        TestSpec(id="negative_on_hand", label="Negative On-hand", description="Items with negative balance", func=_test_negative_on_hand),
        TestSpec(id="cost_variance", label="Cost Variance", description="Unusual unit cost variance", func=_test_cost_variance),
        TestSpec(id="inactive_items", label="Inactive Items", description="Transactions on inactive items", func=_test_inactive_items),
        TestSpec(id="weekend_movements", label="Weekend Movements", description="Movements posted on weekends", func=_test_weekend_movements),
    ]

    def run_tests(self, df: Optional[DataFrame], selected_test_ids: List[str]) -> Dict[str, DataFrame]:
        results: Dict[str, DataFrame] = {}
        for spec in self.tests:
            if spec.id in selected_test_ids:
                results[spec.label] = spec.func(self, df)  # type: ignore[misc]
        return results
