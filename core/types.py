# core/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Optional
import pandas as pd


DataFrame = pd.DataFrame


@dataclass(frozen=True)
class TestSpec:
    """
    Represents a test the user can run within a module.
    - id: stable programmatic ID
    - label: user-facing label (checkbox text)
    - description: optional explainer
    - func: callable implementing the test
    """
    id: str
    label: str
    description: str
    func: Callable[[Optional[DataFrame]], DataFrame]


class Module(Protocol):
    """
    A program module must provide:
      - name: Used for sidebar selection and headings
      - template_columns: The input template columns for this module
      - tests: A list of TestSpec entries
    """
    name: str
    template_columns: List[str]
    tests: List[TestSpec]

    def run_tests(
        self, df: Optional[DataFrame], selected_test_ids: List[str]
    ) -> Dict[str, DataFrame]:
        ...