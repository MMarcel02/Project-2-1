from dataclasses import dataclass
from enum import StrEnum


# Adding this as a col in our data so we can easily separate for each research question
class ExperimentType(StrEnum):
    TEST_ONLY = "test_only"
    ALTER_HARDWARE = "alter_hardware"
    ALTER_PARAMETERS = "alter_parameters"
    OTHER_GROUPS = "other_groups"
    ALTER_BOTH = "alter_both"


# Avoids assigning dict for every object and read only
@dataclass(slots=True, frozen=True)
class RunData:
    id: str
    name: str
    experiment_type: ExperimentType
