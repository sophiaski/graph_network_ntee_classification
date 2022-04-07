"""For specifiying schema of method input/output variables."""

from typing import TypedDict

from numpy import int64


class RowData(TypedDict, total=False):
    object_id: int64
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


from numpy import ndarray
from torch.utils.data import TensorDataset
from torch import Tensor
from typing import Tuple
from pandas import DataFrame
from typing import Dict

from bert_data import NGODataset


class ExperimentDataSplit(TypedDict, total=False):
    train: DataFrame
    valid: DataFrame
    test: DataFrame
    split_size: Tuple[int, int, int]
    dataset_train: NGODataset
    dataset_valid: NGODataset
    dataset_test: NGODataset
    class_weights_train: Tensor


class ExperimentData(TypedDict, total=False):
    data: DataFrame
    data_unlabeled: DataFrame
    num_labels: int
    target2group: Dict[int, str]
    group2name: Dict[str, str]
    stratify_sklearn: ExperimentDataSplit
    stratify_none: ExperimentDataSplit


# class ExperimentDict(TypedDict, total=True):
#     broad: ExperimentData
#     ntee: ExperimentDsata


from torch.utils.data import DataLoader


class DataLoaderDict(TypedDict, total=False):
    train: DataLoader
    valid: DataLoader
    test: DataLoader
