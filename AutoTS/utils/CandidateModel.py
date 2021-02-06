from dataclasses import dataclass
from typing import Any

from pandas import DataFrame


@dataclass
class CandidateModel:
    error: float
    fit_model: Any
    model_type: str
    predictions: DataFrame
