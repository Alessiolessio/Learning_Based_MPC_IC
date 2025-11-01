#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_comparativo.py

Carrega o modelo MLP treinado E o modelo Unicycle para
comparar ambos lado a lado com a trajetória REAL.

Plota:
  1. Trajetórias: Real vs. Unicycle vs. MLP
  2. Erro: Erro(Real vs. Unicycle) vs. Erro(Real vs. MLP)
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import argparse
import math

# Use backend não interativo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. Definição da Classe MLP ---
# (Necessário para carregar o modelo salvo)
class MLPDynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2):
        super(MLPDynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.model(x)

# --- 2. Funções de Rollout (Simulação) ---

def wrap_to_pi(angle):
    """Normaliza ângulo para faixa [-pi, pi]."""
    return (angle + np.pi) % (2*np.pi) - np.pi

def rollout_unicycle(x0, y0, th0, v_arr, w_arr, dt):
    """
    Propaga o modelo unicycle clássico passo a passo.
    """
    n = len(v_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)

    x, y, th = float(x0), float(y0), float(th0)
    x_pred[0], y_pred[0] = x, y

    for k in range(n - 1):
        v = float(v_arr[k])
        w = float(w_arr[k])
        
        # Modelo Unicycle
        x = x + v * math.cos(th) * dt
        y = y + v * math.sin(th) * dt
        th = wrap_to_pi(th + w * dt)

        x_pred[k + 1] = x
        y_pred[k + 1] = y

    return x_pred, y_pred

def rollout_mlp(x0, y0, th0, v_arr, w_arr, model, input_scaler, target_scaler):
    """
    Propaga o modelo MLP treinado passo a passo.
    """
    n = len(v_arr)
    x_pred = np.empty(n, dtype=float)
    y_pred = np.empty(n, dtype=float)
    
    current_state = np.array([x0, y0, th0])
    x_pred[0], y_pred[0] = x0, y0

    # --- MUDANÇA: Define os nomes das colunas que o scaler espera ---
    # Isso corrige o 'UserWarning'
    input_cols = ['x', 'y', 'yaw', 'v', 'w']
    # -----------------------------------------------------------

    model.eval()
    with torch.no_grad():
        for k in range(n - 1):
            v = float(v_arr[k])
            w = float(w_arr[k])
            
            # Monta o input (não normalizado)
            input_vector = np.array([
                current_state[0], # x_k
                current_state[1], # y_k
                current_state[2], # yaw_k
                v,                # v_k
                w                 # w_k
            ])
            
            # --- MUDANÇA: Cria um DataFrame para o scaler ---
            # Isso "anexa" os nomes das colunas ao nosso array
            input_df = pd.DataFrame([input_vector], columns=input_cols)
            
            # 3. NORMALIZA o input (agora usando o DataFrame)
            input_scaled = input_scaler.transform(input_df)
            # ----------------------------------------------------
            
            # 4. Converte para Tensor
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            
            # 5. Roda a PREVISÃO
            pred_scaled_tensor = model(input_tensor)
            
            # 6. Converte para NumPy
            pred_scaled = pred_scaled_tensor.numpy()
            
            # 7. DE-NORMALIZA a previsão
            pred_unscaled = target_scaler.inverse_transform(pred_scaled)
            
            # 8. Obtém o estado previsto
            next_state = pred_unscaled[0]
            
            # 9. Salva o resultado
            x_pred[k + 1] = next_state[0] # x_k+1
            y_pred[k + 1] = next_state[1] # y_k+1
            
            # 10. ATUALIZA o estado atual para o próximo loop
            current_state = next_state

    return x_pred, y_pred


# --- 3. Função Principal (Main) ---

def main(args):
    # --- Carregar Modelo e Scalers ---
    print(f"Carregando modelo de: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Erro: Arquivo do modelo não encontrado: {args.model_path}")
        return

    model_dir = os.path.dirname(args.model_path)
    scaler_input_path = os.path.join(model_dir, 'input_scaler.joblib')
    scaler_target_path = os.path.join(model_dir, 'target_scaler.joblib')

    if not (os.path.exists(scaler_input_path) and os.path.exists(scaler_target_path)):
        print(f"Erro: Arquivos scaler não encontrados em '{model_dir}'")
        return

    model = MLPDynamicsModel(input_dim=5, output_dim=3)
    model.load_state_dict(torch.load(args.model_path))
    input_scaler = joblib.load(scaler_input_path)
    target_scaler = joblib.load(scaler_target_path)
    print("Modelo e scalers carregados com sucesso.")

    # --- Carregar Dataset ---
    print(f"Lendo CSV de: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        print(f"Erro: Arquivo CSV não encontrado: {args.csv_path}")
        return
    
    df = pd.read_csv(args.csv_path)
    df = df.sort_values(["episode", "env", "step"]).reset_index(drop=True)
    
    # Pega o DT do dataset (calculando do timestamp)
    # Pega o passo de tempo mais comum, caso haja algum outlier
    try:
        dt = df['timestamp'].diff().mode()[0]
        if dt <= 0 or dt > 0.1:
            print(f"Aviso: DT detectado ({dt}) parece estranho. Usando 0.02.")
            dt = 0.02
        else:
            print(f"DT detectado do dataset: {dt:.4f}s")
    except Exception:
        print("Aviso: Não foi possível detectar o DT. Usando 0.02s.")
        dt = 0.02


    # --- Processar Episódios ---
    episodes = df["episode"].unique()
    
    # (Opcional: Limite o número de episódios para testar)
    # episodes = episodes[:20] 
    
    print(f"[INFO] Processando {len(episodes)} episódios...")
    os.makedirs(args.output_dir, exist_ok=True)

    for ep in episodes:
        df_ep = df[df["episode"] == ep].copy()

        fig, (ax_traj, ax_error) = plt.subplots(2, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [2, 1]})

        for env_id, df_env in df_ep.groupby("env"):
            if len(df_env) < 2: continue # Pula episódios de 1 passo

            df_env = df_env.sort_values("step")

            # 1. Dados Reais
            x_real = df_env["x"].to_numpy(dtype=float)
            y_real = df_env["y"].to_numpy(dtype=float)
            yaw0   = float(df_env["yaw"].iloc[0])
            v_arr  = df_env["v"].to_numpy(dtype=float)
            w_arr  = df_env["w"].to_numpy(dtype=float)
            x0, y0 = float(x_real[0]), float(y_real[0])
            
            # --- MUDANÇA: Roda as duas simulações ---
            
            # 2. Trajetória PREVISTA (pela MLP)
            x_pred_mlp, y_pred_mlp = rollout_mlp(x0, y0, yaw0, v_arr, w_arr, model, input_scaler, target_scaler)

            # 3. Trajetória PREVISTA (pelo Unicycle)
            x_pred_uni, y_pred_uni = rollout_unicycle(x0, y0, yaw0, v_arr, w_arr, dt)

            # 4. Erros (ambos positivos)
            error_mlp = np.sqrt(np.square(x_real - x_pred_mlp) + np.square(y_real - y_pred_mlp))
            error_uni = np.sqrt(np.square(x_real - x_pred_uni) + np.square(y_real - y_pred_uni))
            steps = np.arange(len(error_mlp))
            
            # --- MUDANÇA: Plota todas as 3 trajetórias ---
            ax_traj.plot(x_real, y_real, 'b-', label=f"env {env_id} — REAL", linewidth=2)
            ax_traj.plot(x_pred_uni, y_pred_uni, 'orange', linestyle="--", label=f"env {env_id} — Unicycle", linewidth=2)
            ax_traj.plot(x_pred_mlp, y_pred_mlp, 'g:', label=f"env {env_id} — MLP", linewidth=2.5) # Verde pontilhado
            
            # --- MUDANÇA: Plota ambos os erros ---
            ax_error.plot(steps, error_uni, 'orange', linestyle="--", label=f"env {env_id} — Erro Unicycle", linewidth=2)
            ax_error.plot(steps, error_mlp, 'g:', label=f"env {env_id} — Erro MLP", linewidth=2.5)

        # --- Configurações dos Plots ---
        ax_traj.set_aspect("equal")
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_xlabel("X (m)")
        ax_traj.set_ylabel("Y (m)")
        ax_traj.set_title("Comparativo de Trajetória: Real vs. Unicycle vs. MLP")
        ax_traj.legend(loc="best")
        
        ax_error.grid(True, alpha=0.3)
        ax_error.set_xlabel("Step (k)")
        ax_error.set_ylabel("Erro Euclidiano (m)")
        ax_error.set_title("Erro de Predição (Distância ao Real)")
        ax_error.legend(loc="best")
        ax_error.set_xlim(left=0, right=len(steps))
        ax_error.set_ylim(bottom=0)

        fig.suptitle(f"Episode {int(ep)} — Comparativo de Modelos", fontsize=16)

        out_path = os.path.join(args.output_dir, f"ep_{int(ep):05d}_comparativo.png")
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        if ep % 20 == 0: # Imprime progresso
             print(f"[OK] Salvo Episódio {ep}...")


# --- 5. Ponto de Entrada do Script ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Testa e compara o modelo MLP e o Unicycle com os dados reais.")
    
    parser.add_argument("--csv_path", type=str, default="/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/datasets/dataset_random.csv", 
                        help="Caminho para o arquivo dataset.csv (para teste).")
    
    # IMPORTANTE: Use o caminho do MELHOR modelo salvo!
    parser.add_argument("--model_path", type=str, default="trained_models_V2/mlp_dynamics_v2.pth",
                        help="Caminho para o modelo treinado (.pth) que você quer testar.")
    
    parser.add_argument("--output_dir", type=str, default="mlp_vs_unicycle_comparacao",
                        help="Pasta para salvar os gráficos de comparação.")
    
    args = parser.parse_args()
    
    main(args)