from typing import Dict
import torch as t


class CellKind:
    """
    Abstraction that holds all relevant parameters for
    a cell of a given type.
    """

    def __init__(
        self, type_id, target_perimeter, target_volume, lambda_volume, adhesion_cost
    ):
        self.target_perimeter = target_perimeter
        self.target_volume = target_volume
        self.lambda_volume = lambda_volume
        self.adhesion_cost = adhesion_cost
        self.type_id = type_id


class CellMap:
    """
    Class that holds a map of all cells on the grid and the celltype
    they belong to.
    """

    def __init__(self):
        self.map = {}

    def add(self, cell_id: int, cell_type: CellKind):
        self.map[cell_id] = cell_type

    def get_map(self):
        return self.map

    def get_item(self, id: int):
        return self.map[id]

    def edit_entry(self, cell_id: CellKind, cell_type: CellKind):
        self.map[cell_id] = cell_type
