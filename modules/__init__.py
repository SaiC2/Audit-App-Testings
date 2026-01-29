# modules/__init__.py
from typing import Dict
from core.types import Module
from .manual_journals import ManualJournalsModule
from .receivables import ReceivablesModule
from .inventory import InventoryModule
from .payables import PayablesModule
from .revenue import RevenueModule
from .expenses import ExpensesModule
from .payroll import PayrollModule
from .masterfile import MasterfileModule

REGISTRY: Dict[str, Module] = {
    "Manual Journals Testing": ManualJournalsModule(),
    "Receivables Testing": ReceivablesModule(),
    "Inventory Testing": InventoryModule(),
    "Payables Testing": PayablesModule(),
    "Revenue Testing": RevenueModule(),
    "Expenses Testing": ExpensesModule(),
    "Payroll Testing": PayrollModule(),
    "Masterfile Testing": MasterfileModule(),
}