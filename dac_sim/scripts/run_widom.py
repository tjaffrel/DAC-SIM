from typing import Optional, Union, IO, Literal
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from fire import Fire
from ase.io import read
from ase.build import molecule

import torch
from dac_sim.widom_insertion import WidomInsertion


def run_widom(
    path_cif: str,
    gas: str = "CO2",
    temperature: float = 300.0,
    init_structure_optimize: bool = True,
    init_gas_optimize: bool = True,
    num_insertions: int = 5000,
    grid_spacing: float = 0.15,
    cutoff_distance: float = 1.50,
    min_interplanar_distance: float = 6.0,
    fold: int = 2,
    random_seed: int = None,
    model_path: Optional[str] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    default_dtype: Literal["float32", "float64"] = "float32",
    dispersion: bool = True,
    save_dir: str = None,
    save_logfile: bool = True,
    save_trajectory: bool = True,
    trajectory: Optional[str] = None,
    logfile: Optional[Union[IO, str]] = None,
    append_trajectory: bool = False,
):
    """Perform Widom insertion simulation to calculate Henry coefficient and heat of adsorption.

    Parameters
    ----------
    path_cif : str
        Path to the CIF file or directory containing CIF files.
    gas : str, default="CO2"
        Gas type to insert into the framework.
    temperature : float, default=300.0
        Simulation temperature, in Kelvin.
    init_structure_optimize : bool, default=True
        If True, optimize the structure before the Widom insertion simulation.
    init_gas_optimize : bool, default=True
        If True, optimize the gas molecule before the Widom insertion simulation.
    num_insertions : int, default=5000
        Number of random insertions of the gas molecule during simulation.
    grid_spacing : float, default=0.15
        Spacing of the grid for possible gas insertion points, in angstroms.
    cutoff_distance : float, default=1.50
        When the distance between framework atoms and the gas molecule is less than this value, the insertion is rejected. In angstroms.
    min_interplanar_distance : float, default=6.0
        When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.
    fold : int, default=2
        Number of repetitions of Widom insertion to improve statistics.
    random_seed : int, optional
        Seed for the random number generator for reproducibility.
    model_path : str, optional
        Path to the MACE model file. If None, the default model is used.
    device : {"cuda", "cpu"}, default="cuda"
        Computational device for running simulations, either "cuda" for GPU or "cpu".
    default_dtype : {"float32", "float64"}, default="float32"
        Default data type for the MACE model.
    dispersion : bool, default=True
        Whether to include dispersion correction in the energy calculations.
    save_dir : str, optional
        Directory where output files (e.g., results, log files, trajectory files) will be saved.
    save_logfile : bool, default=True
        Whether to save a log file of the simulation.
    save_trajectory : bool, default=True
        Whether to save a trajectory file recording the insertions.
    trajectory : str, optional
        Path to an existing trajectory file to continue or reference during the simulation.
    logfile : Union[IO, str], optional
        File object or path to a log file where logs will be written.
    append_trajectory : bool, default=False
        If True, append new trajectory data to an existing trajectory file.
    """
    # Configure device
    if device not in ["cuda", "cpu"]:
        raise ValueError("Device must be either 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        device = "cpu"

    # Initialize gas molecule
    try:
        gas_atoms = molecule(gas)
    except ValueError:
        raise ValueError(f"Gas molecule {gas} is not supported")

    # Set up save directory
    if save_dir is None:
        save_dir = Path(path_cif)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # Collect CIF files
    path_cif = Path(path_cif)
    if path_cif.is_dir():
        cif_files = list(path_cif.glob("*.cif"))
    else:
        cif_files = [path_cif]

    # Check if the results file exist
    result_files = [
        f.stem.replace(f"result_widom_{gas.lower()}_", "")
        for f in save_dir.glob(f"result_widom_{gas.lower()}_*.csv")
    ]
    print(f"Found {len(result_files)} results in {save_dir}")
    cif_files = [f for f in cif_files if f.stem not in result_files]
    print(f"{len(cif_files)} CIF files to run Widom insertion")

    # Run Widom insertion for each CIF file
    for filename in tqdm(cif_files):
        print(f"Running Widom insertion for {filename}")
        if save_logfile:
            if logfile is None:
                seed_part = f"_seed_{random_seed}" if random_seed is not None else ""
                logfile_name = f"widom_{gas.lower()}_{filename.stem}_num_{num_insertions}{seed_part}.log"
                path_logfile = (save_dir / logfile_name).as_posix()
            else:
                path_logfile = logfile
        else:
            path_logfile = None
        if save_trajectory:
            if trajectory is None:
                seed_part = f"_seed_{random_seed}" if random_seed is not None else ""
                trajectory_name = f"widom_{gas.lower()}_{filename.stem}_num_{num_insertions}{seed_part}.traj"
                path_trajectory = (save_dir / trajectory_name).as_posix()
            else:
                path_trajectory = trajectory
        else:
            path_trajectory = None

        widom = WidomInsertion(
            structure=read(filename),
            gas=gas_atoms,
            temperature=temperature,
            init_structure_optimize=init_structure_optimize,
            init_gas_optimize=init_gas_optimize,
            trajectory=path_trajectory,
            logfile=path_logfile,
            append_trajectory=append_trajectory,
            model_path=model_path,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )
        result = widom.run(
            num_insertions=num_insertions,
            grid_spacing=grid_spacing,
            cutoff_distance=cutoff_distance,
            min_interplanar_distance=min_interplanar_distance,
            fold=fold,
            random_seed=random_seed,
        )

        # Save the result
        result_data = {
            "filename": filename.name,
            "henry_coefficient[mol/kg Pa]": result.get("henry_coefficient"),
            "averaged_interaction_energy[eV]": result.get(
                "averaged_interaction_energy"
            ),
            "heat_of_adsorption[kJ/mol]": result.get("heat_of_adsorption"),
        }
        path_result = save_dir / f"result_widom_{gas.lower()}_{filename.stem}.csv"
        result_df = pd.DataFrame([result_data])
        result_df.to_csv(path_result, index=False)
        print(f"saving result to {path_result}")


if __name__ == "__main__":
    Fire(run_widom)
