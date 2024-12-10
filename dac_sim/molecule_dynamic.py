from typing import IO, Union, Literal, Optional, List
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.analysis import DiffusionCoefficient
from mace.calculators import mace_mp

import torch
from dac_sim.grid import add_gas_in_accessible_positions
from dac_sim.optimize import optimize_atoms
from dac_sim import DEFAULT_MODEL_PATH


class MolecularDynamic:
    def __init__(
        self,
        structure: Atoms,
        gas_list: List[Atoms],
        timesteps: float = 1.0,
        temperature: float = 300.0,
        friction: float = 0.01,
        init_structure_optimize: bool = True,
        init_gas_optimize: bool = True,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 10,
        append_trajectory: bool = False,
        model_path: Optional[str] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        default_dtype: Literal["float32", "float64"] = "float32",
        dispersion: bool = True,
    ):
        """
        Molecular dynamics simulation with gas molecules in MOFs

        Parameters
        ----------
        structure : Atoms
            Atoms object for structure (or Framework).
        gas_list : List[Atoms]
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
        trajectory : str, optional
            Path to the trajectory file. If None, the trajectory will be saved in the same directory as the CIF file.
        logfile : Union[IO, str], optional
            Path to the log file or file-like object. If None, no log file is created.
        loginterval : int, default=10
            Interval for logging in the log file.
        append_trajectory : bool, default=False
            If True, append to the existing trajectory file, otherwise overwrite.
        model_path : str, optional
            Path to the MACE model file. If None, the default model is used.
        device : {"cuda", "cpu"}, default="cuda"
            Computational device for running simulations, either "cuda" for GPU or "cpu".
        default_dtype : {"float32", "float64"}, default="float32"
            Default data type for the MACE model.
        dispersion : bool, default=True
            Whether to include dispersion correction in the energy calculations.
        """
        self.atoms = structure
        self.gas_list = gas_list
        self.timesteps = timesteps
        self.temperature = temperature
        self.friction = friction
        self.init_structure_optimize = init_structure_optimize
        self.init_gas_optimize = init_gas_optimize
        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.append_trajectory = append_trajectory

        # Configure device
        device = self._configure_device(device)

        # Set up calculator
        self.calculator = self._initialize_calculator(
            model_path, device, default_dtype, dispersion
        )

        # Set up trajectory
        self.num_init_step = None

    def _configure_device(self, device: str) -> str:
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
            device = "cpu"
        return device

    def _initialize_calculator(
        self,
        model_path: Optional[str],
        device: Literal["cuda", "cpu"],
        default_dtype: Literal["float32", "float64"],
        dispersion: bool,
    ):
        model_path = Path(model_path) if model_path else Path(DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Using model at: {model_path}")
        return mace_mp(
            model=model_path,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )

    def run(
        self,
        num_init_steps: int = 5000,
        num_md_steps: int = 10000,
        min_interplanar_distance: float = 6.0,
    ):
        """
        Run the molecular dynamics simulation.

        Parameters
        ----------
        num_init_steps : int, default=5000
            Number of steps for initialization to reach equilibrium.
        num_md_steps : int, default=10000
            Number of steps for the main molecular dynamics simulation.
        min_interplanar_distance : float, default=6.0
            When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.
        """
        # Update the number of initialization steps
        self.num_init_step = num_init_steps

        structure = self.atoms.copy()
        gas_list = self.gas_list.copy()

        # Optimize the structure and gas molecules
        if self.init_structure_optimize:
            print("Optimizing the structure...")
            structure = optimize_atoms(calculator=self.calculator, atoms=structure)
            if structure is None:
                raise ValueError("Failed to optimize structure")
        if self.init_gas_optimize:
            print("Optimizing the gas molecules...")
            for gas in gas_list:
                gas = optimize_atoms(
                    calculator=self.calculator, atoms=gas, cell_relax=False
                )
                if gas is None:
                    raise ValueError("Failed to optimize gas molecule")

        # Calculate the accessible positions and add the gas molecules to the accessible sites in the structure
        structure_with_gas = add_gas_in_accessible_positions(
            structure,
            gas_list,
            grid_spacing=0.5,
            cutoff_distance=3.0,
            min_interplanar_distance=min_interplanar_distance,
        )

        # Initialize the velocities
        structure_with_gas.calc = self.calculator
        MaxwellBoltzmannDistribution(structure_with_gas, self.temperature * units.kB)
        # Run the molecular dynamics simulation
        dyn = Langevin(
            structure_with_gas,
            timestep=self.timesteps * units.fs,
            temperature_K=self.temperature,
            friction=self.friction / units.fs,
            logfile=self.logfile,
            trajectory=self.trajectory,
            append_trajectory=self.append_trajectory,
            loginterval=self.loginterval,
        )

        # Run the initialization steps
        if num_init_steps > 0:
            print(f"Running {num_init_steps} steps for initialization...")
            dyn.run(num_init_steps)

        # Run the main molecular dynamics simulation
        print(f"Running {num_md_steps} steps for the main simulation...")
        dyn.run(num_md_steps)

    def calculate_diffusion_coefficient(self, show_plot: bool = False):
        """Calculate the diffusion coefficient of the gas molecules.

        Parameters
        ----------
        show_plot : bool, default=False
            If True, show the plot of the diffusion coefficient.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the diffusion coefficients of the gas molecules.
        """
        print("Calculating the diffusion coefficient...")
        traj = read(self.trajectory, ":")
        print(f"Number of total steps in the trajectory: {len(traj)}")
        # skip the num_init_steps
        traj = traj[int(self.num_init_step / self.loginterval) :]
        print(f"Number of run_md_steps in the trajectory: {len(traj)}")
        max_atom_idx = len(self.atoms)
        results = []
        for gas in self.gas_list:
            num_atoms = len(gas)
            atom_indices = list(range(max_atom_idx, max_atom_idx + num_atoms))
            max_atom_idx += num_atoms
            dc = DiffusionCoefficient(
                traj=traj,
                timestep=self.timesteps * self.loginterval,
                atom_indices=atom_indices,
                molecule=True,
            )
            slopes, stds = dc.get_diffusion_coefficients()
            # To convert from Å^2/fs to cm^2/s => multiply by (10^-8)^2 / 10^-15 = 10^-1
            # (Ref: https://wiki.fysik.dtu.dk/ase/_modules/ase/md/analysis.html#DiffusionCoefficient)
            diff_coeff = slopes[0] * units.fs * 1e-1  # Å^2/fs -> cm^2/s
            std = stds[0] * units.fs * 1e-1  # Å^2/fs -> cm^2/s
            results.append(
                {
                    "atom_indices": atom_indices,
                    "chemical_formula": gas.get_chemical_formula(),
                    "diffusion_coefficient": diff_coeff,
                    "std": std,
                }
            )
            print(
                f"Diffusion coefficient ({atom_indices}): {diff_coeff} ± {std} cm^2/s"
            )
            if show_plot:
                dc.plot()
        results = pd.DataFrame(results)
        return results
