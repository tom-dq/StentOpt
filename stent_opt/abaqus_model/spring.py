import dataclasses

from stent_opt.abaqus_model import instance


@dataclasses.dataclass(frozen=True)
class SpringA:
    inst1: instance.Instance
    inst2: instance.Instance
    n1: int
    n2: int
    k: float
