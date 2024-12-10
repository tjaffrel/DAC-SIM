# Model Card

## Model Details

(1) `mace-dac-1.model`:

```
mace_run_train \
  --name="mace_gold_dac_continual" \
  --foundation_model="medium" \
  --train_file="dataset/gold_dac/train.xyz" \
  --valid_file="dataset/gold_dac/val.xyz" \
  --test_file="dataset/gold_dac/val.xyz" \
  --energy_weight=1.0 \
  --forces_weight=1.0 \
  --energy_key="REF_energy" \
  --forces_key="REF_forces" \
  --continual_learning=True \
  --atomic_numbers="foundation" \
  --E0s="foundation" \
  --lr=1e-4 \
  --weight_decay=0 \
  --error_table='PerAtomMAE' \
  --scaling="rms_forces_scaling" \
  --batch_size=4 \
  --valid_batch_size=2 \
  --max_num_epochs=100 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float32" \
  --device=cuda \
  --eval_interval=1 \
  --seed=42 \
  --save_all_checkpoints
```

train/val/test = 40, 20, 26 (# of MOFs)

- train.xyz: (Framework+gas, Framework, gas [H2O, CO2]): 480 + 40 + 2 = 522
- val.xyz: (Framework+gas, Framework, gas [H2O, CO2]): 240 + 20 + 2 = 262
