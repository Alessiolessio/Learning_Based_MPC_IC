#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/

Main script to orchestrate the MLP model training using YAML configuration.
...
"""

import os
import torch
import yaml
from datetime import datetime  # Opcional: para logs

# Import from our local modules
from mlp_model import MLPDynamicsModel
from data_preparation import prepare_data
from training_runner import train_model
from plotting import plot_losses, plot_component_l1
from randomizer import generate_and_save_random_yaml


def main(config):
    """
    Main function that orchestrates the entire process based on a config dictionary.
    """

    # Extrai os sub-dicionários para facilitar o acesso
    paths_config = config['paths']
    train_config = config['training_params']
    model_config = config['model_params']

    # --- 0. Directory Setup (parametrizado) ---
    lr_token = str(train_config['learning_rate']).replace('0.', '')
    hl_token = "_".join(str(h) for h in model_config['hidden_layers'])
    vs_token = int(round(train_config['val_split'] * 100))
    hist_token = train_config.get('history_length', 1) 

    # Adiciona p_dropout ao nome do diretório
    p_drop_token = model_config.get('p_dropout', 0.0)
    drop_str = f"_drop_{str(p_drop_token).replace('0.', '')}" if p_drop_token > 0 else ""

    run_dir_name = (
        f"model"
        f"_hist_{hist_token}" 
        f"_epoch_{train_config['epochs']}"
        f"_batch_{train_config['batch_size']}"
        f"_lr_{lr_token}"
        f"_vs_{vs_token}"
        f"_hl_{hl_token}"
        f"{drop_str}"
    )

    # Usa o 'base_save_dir' do YAML
    save_dir = os.path.join(paths_config['base_save_dir'], run_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Define o caminho de salvamento do modelo
    model_save_path = os.path.join(save_dir, "mlp_dynamics.pth")

    print(f"All artifacts (model, scalers, plots) will be saved in: {save_dir}")

    # Salva uma cópia do config.yaml no diretório do run para rastreabilidade
    with open(os.path.join(save_dir, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # --- 1. Data Preparation ---
    print("\n--- 1. Preparing Data ---")
    
    history_length = train_config.get('history_length', 1)

    train_data, val_data, _, _, input_dim, output_dim = prepare_data(
        csv_path=paths_config['csv_path'],
        scalers_dir=save_dir,        # scalers vão para a pasta parametrizada
        val_split_ratio=train_config['val_split'],
        normalize_data=True,
        history_length=history_length # Passa o parâmetro
    )

    if train_data is None:
        print("Data preparation failed. Aborting.")
        return

    print(f"Data ready (already split by data_preparation).")

    # --- 2. Model Initialization ---
    print("\n--- 2. Initializing Model ---")
    
    # Passando p_dropout para o construtor do modelo
    model = MLPDynamicsModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=model_config['hidden_layers'],
        p_dropout=model_config.get('p_dropout', 0.0) 
    )

    print(f"Model created with architecture: Input({input_dim}) -> {model_config['hidden_layers']} -> Output({output_dim})")
    if model_config.get('p_dropout', 0.0) > 0:
        print(f"Using Dropout with p={model_config.get('p_dropout', 0.0)}")
    print(model)

    # --- 3. Running the Training ---
    print("\n--- 3. Starting Training ---")
    train_losses, val_losses, train_l1_per_comp, val_l1_per_comp = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=train_config['epochs'],
        learning_rate=train_config['learning_rate'],
        batch_size=train_config['batch_size']
    )
    print("Training complete.")

    # --- 4. Saving Artifacts ---
    print("\n--- 4. Saving Artifacts ---")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_losses(train_losses, val_losses, save_dir=save_dir)
    plot_component_l1(train_l1_per_comp, val_l1_per_comp, save_dir=save_dir)

    print("\nProcess finished successfully.")


if __name__ == "__main__":
    config_path = 'parameters.yaml'
    run_idx = 1
    stop = 0

    try:
        while stop == 0:
            print("\n" + "=" * 80)
            print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Starting run #{run_idx}")
            print("=" * 80)

            # 1) Randomiza e salva novo YAML
            try:
                generate_and_save_random_yaml(config_path)
                print(f"Random config generated and saved to {config_path}")
            except Exception as e:
                print(f"ERRO ao gerar random config: {e}")
                # Continua para tentar a próxima iteração
                run_idx += 1
                continue

            # 2) Lê o YAML e roda o pipeline completo
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                print(f"Configuration loaded successfully from {config_path}")
                main(config)

            except FileNotFoundError:
                print(f"ERRO: Arquivo de configuração não encontrado em {config_path}")
            except yaml.YAMLError as e:
                print(f"ERRO: Falha ao parsear o arquivo YAML: {e}")
            except Exception as e:
                print(f"Ocorreu um erro inesperado durante a execução do run #{run_idx}: {e}")

            print(f"Run #{run_idx} finished.")
            run_idx += 1
            stop = 1

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Encerrando loop com segurança.")