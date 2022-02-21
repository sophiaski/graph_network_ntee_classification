"""For specifiying schema of method input/output variables."""

from typing import TypedDict
import numpy as np


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
