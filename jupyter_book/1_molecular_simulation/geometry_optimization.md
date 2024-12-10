# Geometry Optimization

## 1. Python API

Perform geometry optimization to relax the structure with internal and cell optimization steps. The `GeometryOptimization` class utilizes a joint [`FIRE`](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.FIRE) (optimizer) and [`FrechetCellFilter`](https://wiki.fysik.dtu.dk/ase/ase/filters.html#the-frechetcellfilter-class) (cell optimzation) from the [ASE](https://wiki.fysik.dtu.dk/ase/ase/optimize/optimize.html) package to optimize the structure.

### Example Usage

```python
from ase.io import read
from dac_sim.optimize import GeometryOptimization

# Load the structure
structure = read("examples/mg-mof-74.cif")

# Run geometry optimization
go = GeometryOptimization(
    num_total_optimization=30,
    num_internal_steps=50,
    num_cell_steps=50,
    fmax=0.05,
    cell_relax=True,
    device="cuda",
    default_dtype="float32",
    dispersion=True,
)
opt_structure = go.run(structure)
```

### API Reference

```{eval-rst}
.. automodule:: dac_sim.optimize
    :members:
    :undoc-members:
    :show-inheritance:
```

## 2. command line interface

The CLI for geometry optimization is available through the `dac-sim opt` command. The detailed description of the command line options can be found by running `dac-sim opt --help`.

```bash
dac-sim opt examples --num_total_optimization=100 --num_internal_steps=50 --num_cell_steps=50 --save_dir=results
```

### CLI Reference

```{eval-rst}
.. autofunction:: dac_sim.scripts.run_opt.run_opt
```
