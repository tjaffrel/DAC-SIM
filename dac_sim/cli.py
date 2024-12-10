from typing import IO, Optional, Union, Literal
import click
from dac_sim.scripts.run_widom import run_widom
from dac_sim.scripts.run_md import run_md
from dac_sim.scripts.run_opt import run_opt


@click.group(help="DAC-SIM simulation commands")
def cli():
    pass


### CLI groups for Widom insertion ###
@cli.command(
    name="widom",
    help="""
    Run Widom insertion simulation with single or multiple CIF files.

    Examples:
    
    dac-sim widom examples/irmof-1 --gas CO2 --temperature 300.0 --num_insertions 5000
    
    dac-sim widom examples/ --gas CO2 --temperature 300.0 --num_insertions 5000
    """,
)
@click.argument("path_cif", type=click.Path(exists=True))
@click.option("--gas", default="CO2", help="Gas type to insert into the framework.")
@click.option("--temperature", default=300.0, help="Simulation temperature, in Kelvin.")
@click.option(
    "--init_structure_optimize",
    default=True,
    help="Optimize the structure before the Widom insertion simulation.",
)
@click.option(
    "--init_gas_optimize",
    default=True,
    help="Optimize the gas molecule before the Widom insertion simulation.",
)
@click.option(
    "--num_insertions",
    default=5000,
    help="Number of random insertions of the gas molecule during simulation.",
)
@click.option(
    "--grid_spacing",
    default=0.15,
    help="Spacing of the grid for possible gas insertion points, in angstroms.",
)
@click.option(
    "--cutoff_distance",
    default=1.50,
    help="When the distance between framework atoms and the gas molecule is less than this value, the insertion is rejected. In angstroms.",
)
@click.option(
    "--min_interplanar_distance",
    default=6.0,
    help="When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.",
)
@click.option(
    "--fold",
    default=2,
    help="Number of repetitions of Widom insertion to improve statistics.",
)
@click.option(
    "--random_seed",
    default=None,
    help="Seed for the random number generator for reproducibility.",
)
@click.option(
    "--model_path",
    default=None,
    help="Path to the MACE model file. If None, the default model is used.",
)
@click.option(
    "--device",
    default="cuda",
    help="Computational device for running simulations, either 'cuda' for GPU or 'cpu'.",
)
@click.option(
    "--default_dtype",
    default="float32",
    help="Default data type for the MACE model.",
)
@click.option(
    "--dispersion",
    default=True,
    help="Whether to include dispersion correction in the energy calculations.",
)
@click.option(
    "--save_dir", default=None, help="Directory where output files will be saved."
)
@click.option(
    "--save_logfile", default=True, help="Whether to save a log file of the simulation."
)
@click.option(
    "--save_trajectory",
    default=True,
    help="Whether to save a trajectory file.",
)
@click.option("--trajectory", default=None, help="Path to an existing trajectory file.")
@click.option("--logfile", default=None, help="File object or path to a log file.")
@click.option(
    "--append_trajectory",
    default=False,
    help="Append new trajectory data to an existing trajectory file.",
)
def widom(
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
    run_widom(
        path_cif=path_cif,
        gas=gas,
        temperature=temperature,
        init_structure_optimize=init_structure_optimize,
        init_gas_optimize=init_gas_optimize,
        num_insertions=num_insertions,
        grid_spacing=grid_spacing,
        cutoff_distance=cutoff_distance,
        min_interplanar_distance=min_interplanar_distance,
        fold=fold,
        random_seed=random_seed,
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        save_dir=save_dir,
        save_logfile=save_logfile,
        save_trajectory=save_trajectory,
        trajectory=trajectory,
        logfile=logfile,
        append_trajectory=append_trajectory,
    )


### CLI groups for Molecular Dynamics ###
@cli.command(
    name="md",
    help="""
    Run Molecular Dynamics simulation with gas molecules in MOFs.

    Examples:
    
    dac-sim md examples/irmof-1 --gas_list="CO2,H2O" --temperature 300.0 --num_md_steps 10000

    dac-sim md examples/ --gas_list="CO2,H2O" --temperature 300.0 --num_md_steps 10000
    """,
)
@click.argument("path_cif", type=click.Path(exists=True))
@click.option(
    "--gas_list",
    default=["CO2", "H2O"],
    help="List of gas molecules to insert into the framework.",
)
@click.option(
    "--timesteps",
    default=1.0,
    help="Timestep for the molecular dynamics simulation, in fs.",
)
@click.option(
    "--friction",
    default=0.01,
    help="Friction coefficient for the molecular dynamics simulation, in fs^-1.",
)
@click.option(
    "--temperature",
    default=300.0,
    help="Simulation temperature, in Kelvin.",
)
@click.option(
    "--init_structure_optimize",
    default=True,
    help="Optimize the structure before the Widom insertion simulation.",
)
@click.option(
    "--init_gas_optimize",
    default=True,
    help="Optimize the gas molecule before the Widom insertion simulation.",
)
@click.option(
    "--num_init_steps",
    default=5000,
    help="Number of steps for initialization to reach equilibrium.",
)
@click.option(
    "--num_md_steps",
    default=10000,
    help="Number of steps for the main molecular dynamics simulation.",
)
@click.option(
    "--min_interplanar_distance",
    default=6.0,
    help="When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.",
)
@click.option(
    "--model_path",
    default=None,
    help="Path to the MACE model file. If None, the default model is used.",
)
@click.option(
    "--device",
    default="cuda",
    help="Computational device for running simulations, either 'cuda' for GPU or 'cpu'.",
)
@click.option(
    "--default_dtype",
    default="float32",
    help="Default data type for the MACE model.",
)
@click.option(
    "--dispersion",
    default=True,
    help="Whether to include dispersion correction in the energy calculations.",
)
@click.option(
    "--save_dir", default=None, help="Directory where output files will be saved."
)
@click.option(
    "--save_logfile", default=True, help="Whether to save a log file of the simulation."
)
@click.option(
    "--save_trajectory",
    default=True,
    help="Whether to save a trajectory file recording the insertions.",
)
@click.option("--trajectory", default=None, help="Path to an existing trajectory file.")
@click.option("--logfile", default=None, help="File object or path to a log file.")
@click.option(
    "--loginterval",
    default=10,
    help="Interval for logging the simulation progress.",
)
def md(
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
    gas_list = [gas for gas in gas_list.split(",")]
    run_md(
        path_cif=path_cif,
        gas_list=gas_list,
        timesteps=timesteps,
        friction=friction,
        temperature=temperature,
        init_structure_optimize=init_structure_optimize,
        init_gas_optimize=init_gas_optimize,
        num_init_steps=num_init_steps,
        num_md_steps=num_md_steps,
        min_interplanar_distance=min_interplanar_distance,
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        save_dir=save_dir,
        save_logfile=save_logfile,
        save_trajectory=save_trajectory,
        trajectory=trajectory,
        logfile=logfile,
        loginterval=loginterval,
    )


### CLI groups for Geometry Optimization ###
@cli.command(
    name="opt",
    help="""
    Run geometry optimization with single or multiple CIF files.

    Examples:
    
    dac-sim opt examples/irmof-1 --num_total_optimization 30 --num_internal_steps 50 --num_cell_steps 50
    
    dac-sim opt examples/ --num_total_optimization 30 --num_internal_steps 50 --num_cell_steps 50
    """,
)
@click.argument("path_cif", type=click.Path(exists=True))
@click.option(
    "--num_total_optimization",
    default=30,
    help="The number of optimization steps including internal and cell relaxation.",
)
@click.option(
    "--num_internal_steps",
    default=50,
    help="The number of internal steps (freezing the cell).",
)
@click.option(
    "--num_cell_steps",
    default=50,
    help="The number of optimization steps (relaxing the cell).",
)
@click.option(
    "--fmax",
    default=0.05,
    help="The threshold for the maximum force to stop the optimization.",
)
@click.option(
    "--cell_relax",
    default=True,
    help="If True, relax the cell.",
)
@click.option(
    "--model_path",
    default=None,
    help="Path to the MACE model file. If None, the default model is used.",
)
@click.option(
    "--device",
    default="cuda",
    help="Computational device for running simulations, either 'cuda' for GPU or 'cpu'.",
)
@click.option(
    "--default_dtype",
    default="float32",
    help="Default data type for the MACE model.",
)
@click.option(
    "--dispersion",
    default=True,
    help="Whether to include dispersion correction in the energy calculations.",
)
@click.option(
    "--save_dir", default=None, help="Directory where output files will be saved."
)
@click.option(
    "--save_logfile", default=True, help="Whether to save a log file of the simulation."
)
@click.option(
    "--save_trajectory",
    default=False,
    help="Whether to save a trajectory file recording the insertions.",
)
def opt(
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
    run_opt(
        path_cif=path_cif,
        num_total_optimization=num_total_optimization,
        num_internal_steps=num_internal_steps,
        num_cell_steps=num_cell_steps,
        fmax=fmax,
        cell_relax=cell_relax,
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        dispersion=dispersion,
        save_dir=save_dir,
        save_logfile=save_logfile,
        save_trajectory=save_trajectory,
    )
