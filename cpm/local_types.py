from typing import TypedDict, Dict

class Cell_Type(TypedDict):
    cell_id: int
    adhesion_penalties: Dict
    target_volume: float
    vol_scaling: float
    target_perimeter: float
    perim_scaling: float