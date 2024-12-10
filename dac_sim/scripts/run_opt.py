from typing import Optional, Literal
from pathlib import Path
from contextlib import redirect_stdout

from tqdm import tqdm
from fire import Fire
from ase.io import read

import torch
from dac_sim.optimize import GeometryOptimization


def run_opt(
    path_cif: str,
    num_total_optimization: int = 30,
    num_internal_steps: int = 50,
    num_cell_steps: int = 50,
    fmax: float = 0.05,
    cell_relax: bool = True,
    model_path: Optional[str] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    default_dtype: Literal["float32", "float64"] = "float32",
    dispersion: bool = True,
    save_dir: str = None,
    save_logfile: bool = True,
    save_trajectory: bool = False,
):
    """Run geometry optimization

    Parameters
    ----------
    path_cif : str
        Path to the CIF file or directory containing CIF files.
    num_total_optimization : int, default=30
        The number of optimization steps including internal and cell relaxation
    num_internal_steps : int, default=50
        The number of internal steps (freezing the cell)
    num_cell_steps : int, default=50
        The number of optimization steps (relaxing the cell)
    fmax : float, default=0.05
        The threshold for the maximum force to stop the optimization
    cell_relax : bool, default=True
        If True, relax the cell
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
        Whether to save a trajectory file.
    """
    # Configure device
    if device not in ["cuda", "cpu"]:
        raise ValueError("Device must be either 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        device = "cpu"

    # Set up output directory
    if save_dir is None:
        save_dir = Path(path_cif)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # Collect CIF files
    path_cif = Path(path_cif)
    if path_cif.is_dir():
        # get all cif files in the directory
        cif_files = list(path_cif.glob("*.cif"))
        cif_files = [
            cif_file
            for cif_file in cif_files
            if not (save_dir / f"opt_{cif_file.stem}.cif").exists()
        ]
        print(f"Found {len(cif_files)} CIF files to run Geometry Optimization")
    else:
        cif_files = [path_cif]

    go = GeometryOptimization(
        num_total_optimization=num_total_optimization,
        num_internal_steps=num_internal_steps,
        num_cell_steps=num_cell_steps,
        fmax=fmax,
        cell_relax=cell_relax,
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
    )
    # Run geometry optimization for each CIF file
    for cif_file in tqdm(cif_files):
        print(f"Running Geometry Optimization for {cif_file.stem}")
        structure = read(cif_file)

        if save_trajectory:
            trajectory_file = save_dir / f"{cif_file.stem}.traj"
        else:
            trajectory_file = None

        if save_logfile:
            log_file = save_dir / f"{cif_file.stem}.log"
            with open(log_file, "w") as f_out, redirect_stdout(f_out):
                try:
                    opt_atoms = go.run(structure, trajectory_file=trajectory_file)
                except Exception as e:
                    print(f"Optimization failed for {cif_file.stem}: {e}")
                    continue
        else:
            try:
                opt_atoms = go.run(structure, trajectory_file=trajectory_file)
            except Exception as e:
                print(f"Optimization failed for {cif_file.stem}: {e}")
                continue
        if opt_atoms is None:
            print(f"Optimization failed for {cif_file.stem}")
            continue

        # Save the optimized structure
        opt_atoms.write(save_dir / f"opt_{cif_file.name}")
        print(f"saving optimized structure to {save_dir / f'opt_{cif_file.name}'}")


if __name__ == "__main__":
    Fire(run_opt)
