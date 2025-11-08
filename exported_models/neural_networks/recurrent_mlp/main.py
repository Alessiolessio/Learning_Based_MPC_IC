#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Location: /home/nexus/VQ_PMCnmpc/VQ_PMC/exported_models/neural_networks/

Script principal que chama os módulos para treinar a rede.
*** MODIFICADO para Lógica Sequencial ***
"""

import argparse
import os
import torch

# Importa as funções/classes dos outros arquivos
# --- MUDANÇA: Nomes dos arquivos atualizados ---
from mlp_model import MLPDynamicsModel
from data_preparation import load_and_prep_episodes # <-- MUDANÇA
from training_runner import train_model     # <-- MUDANÇA
from plotting import plot_losses

def main(args):
    """
    Função principal que orquestra o processo.
    """
    
    # --- 0. Directory Setup ---
    save_dir = os.path.dirname(args.model_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"All artifacts (model, scalers, plots) will be saved in: {save_dir}")

    # --- 1. Data Preparation ---
    print("\n--- 1. Preparing Data (Modo Sequencial) ---")
    
    # Chama a nova função 'load_and_prep_episodes'
    train_episodes, val_episodes, _, _ = load_and_prep_episodes(
        args.csv_path, 
        scalers_dir=save_dir,
        val_split_ratio=args.val_split,
        normalize_data=True # Mantém a normalização
    )
    
    if train_episodes is None:
        print("Data preparation failed. Aborting.")
        return
        
    print(f"Data ready (dividido por episódio).")

    # --- 2. Model Initialization ---
    print("\n--- 2. Initializing Model ---")
    
    # --- MUDANÇA: Input dim agora é (features * seq_len) ---
    input_features = 5 # (x, y, yaw, v, w)
    output_dim = 3     # (x_next, y_next, yaw_next)
    input_dim_seq = input_features * args.sequence_length # ex: 5 * 3 = 15
    
    model = MLPDynamicsModel(
        input_dim=input_dim_seq, # <-- MUDANÇA
        output_dim=output_dim,
        hidden_layers=args.hidden_layers
    )
    
    print(f"Model created with architecture: Input({input_dim_seq}) -> {args.hidden_layers} -> Output({output_dim})")
    print(model)

    # --- 3. Running the Training ---
    print("\n--- 3. Starting Training ---")
    
    # Chama o NOVO 'train_model' de 'training_runner.py'
    train_losses, val_losses = train_model(
        model=model,
        train_episodes=train_episodes, # <-- MUDANÇA (passa lista de episódios)
        val_episodes=val_episodes,     # <-- MUDANÇA (passa lista de episódios)
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seq_len=args.sequence_length   # <-- MUDANÇA (novo parâmetro)
    )
    print("Training complete.")

    # --- 4. Saving Artifacts ---
    print("\n--- 4. Saving Artifacts ---")
    # (Salva o modelo final. A lógica de 'melhor' modelo pode ser
    #  readicionada ao training_runner.py se necessário)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
    
    plot_losses(train_losses, val_losses, save_dir=save_dir)

    print("\nProcess finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains an MLP model to predict robot dynamics.")
    
    # --- Path Parameters ---
    parser.add_argument("--csv_path", type=str, 
                        default="/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv",
                        help="Path to the dataset.csv file.")
    parser.add_argument("--model_path", type=str, 
                        default="trained_models/mlp_dynamics.pth",
                        help="Path to save the trained model (.pth).")

    # --- Training Parameters ---
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, # <-- MUDANÇA (batch de episódios, menor)
                        help="Number of EPISODES per optimizer step (batch size).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of the dataset to use for validation (e.g., 0.2 for 20%).")

    # --- Model Architecture Parameters ---
    # --- MUDANÇA: Adicionado 'sequence_length' ---
    parser.add_argument("--sequence_length", type=int, default=3,
                        help="Number of past steps to use as input (ex: 3).")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[64, 64],
                        help="Sizes of the hidden layers. Ex: --hidden_layers 64 64 128")
    
    args = parser.parse_args()
    
    main(args)