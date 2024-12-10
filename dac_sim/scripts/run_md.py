from typing import Optional, Union, IO, Literal
from pathlib import Path

from tqdm import tqdm
from fire import Fire
from ase.io import read
from ase.build import molecule

import torch
from dac_sim.molecule_dynamic import MolecularDynamic


def run_md(
    path_cif: str,
    gas_list: list[str] = ["CO2", "H2O"],
    timesteps: float = 1.0,
    friction: float = 0.01,
    temperature: float = 300.0,
    init_structure_optimize: bool = True,
    init_gas_optimize: bool = True,
    num_init_steps: int = 5000,
    num_md_steps: int = 10000,
    min_interplanar_distance: float = 6.0,
    model_path: Optional[str] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    default_dtype: Literal["float32", "float64"] = "float32",
    dispersion: bool = True,
    save_dir: str = None,
    save_logfile: bool = True,
    save_trajectory: bool = True,
    trajectory: Optional[str] = None,
    logfile: Optional[Union[IO, str]] = None,
    loginterval: int = 10,
):
    """Perform Molecular Dynamic simulation with gas molecules in MOFs.

    Parameters
    ----------
    path_cif : str
        Path to the CIF file or directory containing CIF files.
    gas_list : List[Atoms], default=["CO2", "H2O"]
        List of Atoms objects for gas molecules e.g. ["CO2", "H2O"] or ["CO2", "CO2"]
    timesteps : float, default=1.0
        Timestep for the molecular dynamics simulation, in fs.
    temperature : float, default=300.0
        Simulation temperature, in Kelvin.
    friction : float, default=0.01
        Friction coefficient for the molecular dynamics simulation, in fs^-1.
    init_structure_optimize : bool, default=True
        If True, optimize the structure before the Widom insertion simulation.
    init_gas_optimize : bool, default=True
        If True, optimize the gas molecule before the Widom insertion simulation.
    num_init_steps : int, default=5000
        Number of steps for initialization to reach equilibrium.
    num_md_steps : int, default=10000
        Number of steps for the main molecular dynamics simulation.
    min_interplanar_distance : float, default=6.0
        When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.
    timesteps : float, optional
        Timestep of the simulation in femtoseconds. Default is 1.0.
    friction : float, optional
        Friction coefficient of the simulation in fs^-1. Default is 0.01 / units.fs.
    temperature : float, optional
        Temperature of the simulation in Kelvin. Default is 300.0.
    min_interplanar_distance : float, optional
        Minimum interplanar distance for the supercell. Default is 6.0.
    init_structure_optimize : bool, optional
        Whether to optimize the structure before running the MD simulation. Default is True.
    init_gas_optimize : bool, optional
        Whether to optimize the gas molecules before running the MD simulation. Default is True.
    model_path : str, optional
        Path to the MACE model file. If None, the default model will be used.
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
        Whether to save a trajectory file.
    trajectory : str, optional
        Path to an existing trajectory file to continue or reference during the simulation.
    logfile : Union[IO, str], optional
        File object or path to a log file where logs will be written.
    loginterval : int, default=10
        Interval for logging in the log file.
    """
    # Configure device
    if device not in ["cuda", "cpu"]:
        raise ValueError("Device must be either 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        device = "cpu"

    # Initialize gas molecules
    try:
        gas_atoms_list = [molecule(gas) for gas in gas_list]
    except Exception as e:
        raise Exception(f"Error initializing gas molecules: {e}")

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
        print(f"Found {len(cif_files)} CIF files in {path_cif}")
    else:
        cif_files = [path_cif]

    # Check if the results file exist
    result_files = [
        f.stem.replace(f"result_md_{''.join(gas_list)}_", "")
        for f in save_dir.glob(f"result_md_{''.join(gas_list)}_*.csv")
    ]
    print(f"Found {len(result_files)} results in {save_dir}")
    cif_files = [f for f in cif_files if f.stem not in result_files]
    print(f"{len(cif_files)} CIF files to run Molecular Dynamic")

    for cif_file in tqdm(cif_files):
        print(f"Running Molecular Dynamic for {cif_file.stem}")
        # load the structure and gas
        structure = read(cif_file)

        # run the molecular dynamic simulation
        if save_trajectory:
            if trajectory is None:
                trajectory_name = f"md_{cif_file.stem}.traj"
                path_trajectory = (save_dir / trajectory_name).as_posix()
            else:
                path_trajectory = trajectory
        else:
            path_trajectory = None
        if save_logfile:
            if logfile is None:
                logfile_name = f"md_{cif_file.stem}.log"
                path_logfile = (save_dir / logfile_name).as_posix()
            else:
                path_logfile = logfile
        else:
            path_logfile = None
        md = MolecularDynamic(
            structure=structure,
            gas_list=gas_atoms_list,
            timesteps=timesteps,
            temperature=temperature,
            friction=friction,
            init_structure_optimize=init_structure_optimize,
            init_gas_optimize=init_gas_optimize,
            trajectory=path_trajectory,
            logfile=path_logfile,
            loginterval=loginterval,
            model_path=model_path,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )
        md.run(
            num_init_steps=num_init_steps,
            num_md_steps=num_md_steps,
            min_interplanar_distance=min_interplanar_distance,
        )

        # Calculate the diffusion coefficient
        result = md.calculate_diffusion_coefficient()

        # Save the result
        path_result = save_dir / f"result_md_{''.join(gas_list)}_{cif_file.stem}.csv"
        result.to_csv(path_result)
        print(f"saving result to {path_result}")


if __name__ == "__main__":
    Fire(run_md)
