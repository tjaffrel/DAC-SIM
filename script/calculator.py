from typing import Union, List
from pathlib import Path
from collections import namedtuple

from tqdm import tqdm
import numpy as np
from ase import Atoms
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.calculators import mace_mp

import torch

MACEResult = namedtuple("MACEResult", ["energies", "forces", "stresses"])


class MACECalculator:
    def __init__(
        self,
        model: str,
        batch_size: int = 10,
        device: str = "cpu",
        default_dtype: str = "float64",
    ):
        if model == "mace-mp":
            model = mace_mp().models[0]
            print(f"load mace-mp model")
        else:
            path_model = Path(model)
            if not path_model.exists():
                raise FileNotFoundError(f"Model path {path_model} does not exist.")
            model = torch.load(path_model)
            print(f"Model loaded from {path_model}")
        model.eval()
        self.model = model
        self.batch_size = batch_size

        # set device
        self.device = device
        self.model.to(device)

        # set default dtype
        self.default_dtype = default_dtype
        if default_dtype == "float32":
            torch.set_default_dtype(torch.float32)
        elif default_dtype == "float64":
            torch.set_default_dtype(torch.float64)
        else:
            raise ValueError(f"Invalid default_dtype: {default_dtype}")

    def calculate(
        self, atoms: Union[Atoms, List[Atoms]], compute_stress: bool = False
    ) -> MACEResult:
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        atoms_list = atoms

        # set configs
        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

        # set data loader
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=float(self.model.r_max)
                )
                for config in configs
            ],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # collect data
        energies_list = []
        stresses_list = []
        forces_collection = []

        for batch in tqdm(data_loader):
            batch.to(self.device)
            output = self.model(batch.to_dict(), compute_stress=compute_stress)
            energies_list.append(torch_tools.to_numpy(output["energy"]))
            if compute_stress:
                stresses_list.append(torch_tools.to_numpy(output["stress"]))

            forces = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            forces_collection.append(forces[:-1])  # drop last as its empty

        energies = np.concatenate(energies_list, axis=0)
        forces_list = [
            forces for forces_list in forces_collection for forces in forces_list
        ]
        assert len(atoms_list) == len(energies) == len(forces_list)

        if compute_stress:
            stresses = np.concatenate(stresses_list, axis=0)
            assert len(atoms_list) == stresses.shape[0]

        return MACEResult(
            energies=energies,
            forces=forces_list,
            stresses=stresses if compute_stress else None,
        )
