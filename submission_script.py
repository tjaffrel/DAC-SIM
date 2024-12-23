from ase.io import read
from ase.build import molecule
import numpy as np
from scipy.interpolate import interp1d
from dac_sim.widom_insertion import WidomInsertion
import matplotlib.pyplot as plt
import pandas as pd

# Load the structure and build the gas
structure = read("examples/mg-mof-74.cif")
gas = molecule("CO2")


temperatures = [288, 293, 298, 308, 313]  # Temperatures in Kelvin
henry_coeff = []
pressure_data = []
qst_data = []
uptake_data = []

for temp in temperatures:
    widom_insertion = WidomInsertion(
        structure,
        gas=gas,
        temperature=temp,
        trajectory=None,
        logfile=None,
        model_path="/home/theoj/programs/DAC-SIM/dac_sim/models/mace-dac-1.model",
        device="cuda",
        default_dtype="float32",
        dispersion=True,
        gpu_preprocessing=False,
        calculator_type="mace_mp"
    )
    result = widom_insertion.run(num_insertions=5000, random_seed=0, fold=2, grid_spacing=0.2)
    henry_coeff.append(np.mean(result["henry_coefficient"]))  
    pressure_data.append(np.mean(result["pressure"]))
    qst_data.append(np.mean(result["heat_of_adsorption"]))
    uptake_data.append(np.mean(result["uptake"]))

qst_data = np.abs(np.array(qst_data))
uptake_data = np.array(uptake_data)*1e-3


plt.figure()
plt.plot(uptake_data, qst_data, label="COF-999", marker="o")
plt.ylim(0, 100)
plt.xlabel("Uptake (mmol/g)")
plt.ylabel("Isosteric Heat of Adsorption (kJ/mol)")
plt.legend()
plt.grid()
plt.savefig("isosteric_heat_of_adsorption_vs_uptake.png")
plt.show()
