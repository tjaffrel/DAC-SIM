# Molecular Dynamics Simulation

## 1. Python API

Perform molecular dynamics simulation to calculate the diffusion coefficient and transport properties of gas molecules using [ASE molecular dynamics](https://wiki.fysik.dtu.dk/ase/ase/md/md.html).

The `MolecularDyamic` class involves setting intial configuration of gas molecules in the accessible sites of the structure using [`add_gas_in_accessible_positions`](https://github.com/hspark1212/DAC-SIM/blob/57aad4f2122f30553d43adc4287453a0568b8023/dac_sim/grid.py#L11) function.

### Example Usage

Below is an example of how to use the `MolecularDynamic` class:

```python
from ase.io import read
from ase.build import molecule
from ase.visualize import view
from dac_sim.molecule_dynamic import MolecularDynamic

# Load the structure and build the gas
structure = read("examples/mg-mof-74.cif")
gas_list = [molecule("CO2"), molecule("H2O")]

# Run the molecular dynamics simulation
timestep = 1.0  # [fs]
temperature = 300  # [K]
trajectory = "md_co2_h2o_mg-mof-74.traj"
logfile = "md_co2_h2o_mg-mof-74.log"
md = MolecularDynamic(
    structure,
    gas_list=gas_list,
    timesteps=timestep,
    temperature=temperature,
    trajectory=trajectory,
    logfile=logfile,
    loginterval=10,
    device="cuda",
    default_dtype="float32",
    dispersion=True,
)
md.run(num_init_steps=5000, num_md_steps=10000)

# Calculate diffusion coefficient
result = md.calculate_diffusion_coefficient(show_plot=True)
print(result)

# Visualize the trajectory
traj = read("md_co2_h2o_mg-mof-74.traj", ":")
view(traj, viewer="ngl")
```

The `calculate_diffusion_coefficient` method calculates the diffusion coefficient of the gas molecules and displays a plot of the mean square displacement (MSD) of the gas molecules over time utilizing the [`DiffusionCoefficient`](https://wiki.fysik.dtu.dk/ase/_modules/ase/md/analysis.html#DiffusionCoefficient) class from the ASE package.

### API Reference

```{eval-rst}
.. automodule:: dac_sim.molecule_dynamic
    :members:
    :undoc-members:
    :show-inheritance:
```

## 2. command line interface

The CLI for molecular dynamics simulations is available through the `dac-sim md` command. The detailed description of the command line options can be found by running `dac-sim md --help`.

```bash
dac-sim md examples --gas_list="CO2,H2O" --timesteps=1.0 --temperature=300 --num_steps=1000 --save_dir=results
```

### CLI Reference

```{eval-rst}
.. autofunction:: dac_sim.scripts.run_md.run_md
```
