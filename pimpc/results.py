from dataclasses import dataclass
from typing import NamedTuple


class SolveInfo(NamedTuple):
    solve_time: float
    iterations: int
    converged: bool
    obj_val: float


@dataclass(frozen=True)
class Results:
    x: object
    u: object
    du: object
    info: SolveInfo
