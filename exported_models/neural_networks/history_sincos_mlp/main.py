#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Orchestrates training:
- Loads YAML config
- Prepares data (with episode-based split and yaw sin/cos)
- Builds the MLP
- Trains and saves artifacts (model, scalers, plots, config snapshot)
"""

import os
import torch
import yaml
from datetime import datetime

from mlp_model import MLPDynamicsModel
from data_preparation import prepare_data
from training_runner import train_model
from plotting import plot_losses, plot_component_l1

def main(config: dict):
    # ---------------------------- Subconfigs ----------------------------
    paths_config = config["paths"]
    train_config = config["training_params"]
    model_config = config["model_params"]

    # ------------------------- Run directory name -----------------------
    lr_token = str(train_config["learning_rate"]).replace("0.", "")
    hl_token = "_".join(str(h) for h in model_config["hidden_layers"])
    vs_token = int(round(train_config["val_split"] * 100))
    hist_token = train_config.get("history_length", 1)
    p_drop = model_config.get("p_dropout", 0.0)
    drop_token = f"_drop_{str(p_drop).replace('0.', '')}" if p_drop > 0 else ""

    run_dir = (
        f"model"
        f"_hist_{hist_token}"
        f"_epoch_{train_config['epochs']}"
        f"_batch_{train_config['batch_size']}"
        f"_lr_{lr_token}"
        f"_vs_{vs_token}"
        f"_hl_{hl_token}"
        f"{drop_token}"
    )
    save_dir = os.path.join(paths_config["base_save_dir"], run_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "mlp_dynamics.pth")

    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]")
    print(f"Artifacts will be saved in: {save_dir}")

    # ------------------------ Snapshot current config -------------------
    with open(os.path.join(save_dir, "config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # ---------------------------- Data prep -----------------------------
    print("\n--- 1. Preparing Data ---")
    history_length = train_config.get("history_length", 1)
    train_data, val_data, _, _, input_dim, output_dim = prepare_data(
        csv_path=paths_config["csv_path"],
        scalers_dir=save_dir,
        val_split_ratio=train_config["val_split"],
        normalize_data=True,
        history_length=history_length
    )
    if train_data is None:
        print("Data preparation failed. Aborting.")
        return
    print("Data ready.")

    # ------------------------------ Model -------------------------------
    print("\n--- 2. Initializing Model ---")
    model = MLPDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=model_config["hidden_layers"],
        p_dropout=p_drop
    )
    print(f"Model: Input({input_dim}) -> {model_config['hidden_layers']} -> Output({output_dim}) (dropout={p_drop})")
    print(model)

    # ------------------------------ Train -------------------------------
    print("\n--- 3. Starting Training ---")
    train_losses, val_losses, train_l1_per_comp, val_l1_per_comp = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=train_config["epochs"],
        learning_rate=train_config["learning_rate"],
        batch_size=train_config["batch_size"]
    )
    print("Training complete.")

    # ---------------------------- Artifacts -----------------------------
    print("\n--- 4. Saving Artifacts ---")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_losses(train_losses, val_losses, save_dir=save_dir)
    plot_component_l1(train_l1_per_comp, val_l1_per_comp, save_dir=save_dir)
    print("\nProcess finished successfully.")

if __name__ == "__main__":
    CONFIG_PATH = "parameters.yaml"
    print("\n" + "=" * 80)
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Starting run")
    print("=" * 80)
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"Configuration loaded from {CONFIG_PATH}")
        main(cfg)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML: {e}")
    except Exception as e:
        print(f"Unexpected ERROR: {e}")
