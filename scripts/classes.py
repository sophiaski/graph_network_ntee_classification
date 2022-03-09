"""For specifiying schema of method input/output variables."""

from typing import TypedDict, Tuple, List, Sequence, Dict
from torch.utils.data import TensorDataset
from torch import Tensor
import numpy as np
import pandas as pd


class RowData(TypedDict, total=False):
    object_id: np.int64
    ein: str
    form_type: str
    submission_ts: str
    business_name: str
    tax_period_begin: str
    tax_period_end: str
    tax_year: str
    formation_year: str
    mission_descript: str
    mission_descript_2: str
    mission_descript_3: str
    mission_descript_4: str
    mission_descript_5: str
    program_descript: str
    program_descript_2: str
    program_descript_3: str
    program_descript_4: str


class ExperimentDataSplit(TypedDict, total=False):
    train: Tuple[np.ndarray, np.ndarray]
    validation: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]
    train_tensor: TensorDataset
    train_class_weights: Tensor
    validation_tensor: TensorDataset
    test_tensor: TensorDataset
    size: Tuple[int, int, int]


class ExperimentData(TypedDict, total=False):
    data: pd.DataFrame
    target2group: Dict[int, str]
    group2name: Dict[str, str]
    stratify_sklearn: ExperimentDataSplit
    stratify_none: ExperimentDataSplit


class ExperimentDict(TypedDict, total=True):
    broad: ExperimentData
    ntee: ExperimentData
