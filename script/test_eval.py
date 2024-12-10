from collections import defaultdict
from pathlib import Path

from fire import Fire
from tqdm import tqdm
import pandas as pd
import numpy as np
from ase.io import read
from mace.calculators import mace_mp


def evaluate(
    model: str,
    path_data: str,
    save_dir: str,
    device: str = "cuda",
    default_dtype: str = "float64",
    dispersion: bool = False,
):
    if model == "mace-mp":
        mace_calc = mace_mp(
            model="medium",
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )
        model = f"mace-mp-medium-dispersion-{dispersion}"
        model_path = Path(model)
    else:
        model_path = Path(model)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found")
        mace_calc = mace_mp(
            model=model_path,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )
        print(f"load model from {model}")

    path_data = Path(path_data)
    if not path_data.exists():
        raise FileNotFoundError(f"{path_data} not found")
    atoms_list = read(path_data, index=":")

    # collect
    results = defaultdict(list)
    for atoms in tqdm(atoms_list):
        atoms.calc = mace_calc
        num_atoms = len(atoms)
        # true
        true_e = atoms.info["energy"]
        true_f = atoms.arrays["forces"]
        true_f_mag = np.linalg.norm(true_f, axis=1)
        # pred
        try:
            pred_e = atoms.get_potential_energy()
            pred_f = atoms.get_forces()
            pred_f_mag = np.linalg.norm(pred_f, axis=1)
        except ValueError as e:
            print(f"error: {e}")
            continue
        # mae
        mae_e_per_atoms = abs(pred_e - true_e) / num_atoms
        mae_f = abs(pred_f - true_f).mean()

        results["name"].append(f"{atoms.info['name']}+{atoms.info['fid']}")
        results["num_atoms"].append(num_atoms)
        results["true_e"].append(true_e)
        results["pred_e"].append(pred_e)
        results["diff_e"].append(pred_e - true_e)
        results["diff_f"].append((pred_f_mag - true_f_mag).tolist())
        results["values_mae_e_per_atoms"].append(mae_e_per_atoms)
        results["values_mae_f"].append(mae_f)
    results["mae_e_per_atoms"] = np.array(results["values_mae_e_per_atoms"]).mean()
    results["mae_f"] = np.array(results["values_mae_f"]).mean()
    results_df = pd.DataFrame(results)

    # save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"test_eval_{model_path.stem}.csv"
    results_df.to_csv(save_path, index=False)
    print(f"save results to {save_path}")


if __name__ == "__main__":
    Fire(evaluate)
