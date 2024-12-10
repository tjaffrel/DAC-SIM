# Widom Insertion Monte Carlo Simulation

## 1. Python API

The `WidomInsertion` class is designed to perform Widom insertion simulations to calculate the Henry's law coefficient (K$_{H}$) and the averaged interaction energy of gas molecules. This class extends the `Dynamics` class from the [ASE](https://github.com/qsnake/ase/blob/master/ase/optimize/optimize.py) package, facilitating the simulation of molecular dynamics in a manner similar to an **ASE Calculator**.

### Example Usage

Below is an example demonstrating how to use the `WidomInsertion` class:

```python
from ase.io import read
from ase.build import molecule

from dac_sim.widom_insertion import WidomInsertion

# Load the structure and build the gas
structure = read("examples/mg-mof-74.cif")
gas = molecule("CO2")

# Create the WidomInsertion object
temperature = 300  # [K]
trajectory = "widom_co2_mg-mof-74.traj"
logfile = "widom_co2_mg-mof-74.log"
widom_insertion = WidomInsertion(
    structure,
    gas=gas,
    temperature=temperature,
    trajectory=trajectory,
    logfile=logfile,
    device="cuda",
    default_dtype="float32",
)
result = widom_insertion.run(num_insertions=5000)
print(result)
```

### API Reference

```{eval-rst}
.. automodule:: dac_sim.widom_insertion
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: log, log_results, todict
```

## 2. command line interface (CLI)

The CLI for Widom insertion simulations is available through the `dac-sim widom` command. The detailed description of the command line options can be found by running `dac-sim widom --help`.

```bash
dac-sim widom examples/ --gas=CO2 --temperature=300 --num_insertions=5000 --fold=2 --save_dir=results
```

### CLI Reference

```{eval-rst}
.. autofunction:: dac_sim.scripts.run_widom.run_widom
```
