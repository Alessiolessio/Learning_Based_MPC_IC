#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mlp.py

Este script treina uma Rede Neural MLP (Multilayer Perceptron) para
aprender a dinâmica cinemática de um robô a partir de um dataset.

Objetivo: (System Identification)
  A rede aprende a prever o *próximo estado* (x, y, yaw)
  dado o *estado atual* (x, y, yaw) e a *ação atual* (v, w).

Formato do Modelo:
  (x_{k+1}, y_{k+1}, yaw_{k+1}) = MLP(x_k, y_k, yaw_k, v_k, w_k)

Como executar:
1. Salve este código como 'mlp.py'
2. Certifique-se de ter o dataset (ex: 'dataset_random.csv') na pasta correta.
3. Instale as bibliotecas: pip3 install torch pandas matplotlib
4. Execute no terminal:
   python3 mlp.py --csv_path /home/nexus/VQ_PMCnmpc/VQ_PMC/logs/dataset_random.csv
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. Definição da Rede Neural (MLP) ---

class MLPDynamicsModel(nn.Module):
    """
    Define a arquitetura da nossa Rede Neural (MLP).
    
    'nn.Module' é a classe base do PyTorch para todos os modelos.
    """
    def __init__(self, input_dim, output_dim):
        """
        O 'construtor' da classe. Define as camadas da rede.
        
        - input_dim: Quantos números entram na rede (ex: 5)
        - output_dim: Quantos números saem da rede (ex: 3)
        """
        # Chama o construtor da classe pai (nn.Module)
        super(MLPDynamicsModel, self).__init__()
        
        # 'nn.Sequential' é um contêiner que roda as camadas em ordem.
        # Pense nisso como um pipeline para os dados.
        self.model = nn.Sequential(
            # 1ª Camada: Linear(entradas, neurônios)
            # Pega 'input_dim' (5) e transforma em 64.
            # Esta é a camada de entrada.
            nn.Linear(input_dim, 64),
            
            # Função de Ativação: ReLU (Rectified Linear Unit)
            # Introduz não-linearidade. f(x) = max(0, x).
            # Essencial para a rede aprender padrões complexos.
            nn.ReLU(),
            
            # 2ª Camada: Linear(neurônios_camada_anterior, neurônios)
            # Pega 64 e transforma em 64. Esta é uma "camada oculta".
            nn.Linear(64, 64),
            nn.ReLU(),
            
            # 3ª Camada: Linear(neurônios, saídas)
            # Pega 64 e transforma em 'output_dim' (3).
            # Esta é a camada de saída. Note que não há ReLU no final,
            # pois queremos prever valores contínuos (x, y, yaw).
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        Define a "passagem para frente" (forward pass).
        Isto é chamado automaticamente quando você faz model(entrada).
        
        Simplesmente diz ao PyTorch para passar a entrada 'x' pelo
        modelo sequencial que definimos.
        """
        return self.model(x)


# --- 2. Preparação dos Dados ---

def prepare_data(csv_path):
    """
    Lê o dataset.csv e o transforma em pares de (input, target).
    
    Esta é a parte mais importante e específica para o seu problema.
    A rede precisa aprender:
      TARGET (saída) = f(INPUT)
      (estado_{k+1}) = f(estado_k, acao_k)
    """
    print(f"Lendo e processando o dataset: {csv_path}")
    try:
        # Lê o CSV usando pandas
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo CSV não encontrado em {csv_path}")
        return None, None
        
    # Colunas que usaremos
    # Como decidimos, 5 entradas
    input_cols = ['x', 'y', 'yaw', 'v', 'w']
    # E 3 saídas
    target_cols = ['x', 'y', 'yaw']
    
    all_inputs = []
    all_targets = []

    # É crucial processar cada episódio separadamente
    # para não misturar o fim de um com o começo de outro.
    print(f"Processando {len(df['episode'].unique())} episódios...")
    for episode_id in df['episode'].unique():
        # Filtra o DataFrame apenas para este episódio e ordena pelo 'step'
        df_ep = df[df['episode'] == episode_id].sort_values('step')
        
        # 1. INPUTS (Estado k, Ação k)
        # Pegamos os estados e ações atuais (x_k, y_k, yaw_k, v_k, w_k)
        inputs_df = df_ep[input_cols]
        
        # 2. TARGETS (Estado k+1)
        # Pegamos os estados da *próxima* linha (próximo step)
        # .shift(-1) "puxa" os dados da linha de baixo para cima
        # (x_{k+1}, y_{k+1}, yaw_{k+1})
        targets_df = df_ep[target_cols].shift(-1)
        
        # 3. COMBINAR
        # Juntamos os inputs da linha 'k' com os targets da linha 'k+1'
        # Adicionamos '_next' ao nome das colunas de target por clareza
        combined = pd.concat([inputs_df, targets_df.add_suffix('_next')], axis=1)
        
        # 4. LIMPAR
        # A última linha de cada episódio terá um target "NaN" (Vazio),
        # pois não há um "próximo" step (k+1). Removemos essas linhas.
        combined = combined.dropna()
        
        if not combined.empty:
            all_inputs.append(combined[input_cols])
            all_targets.append(combined[[col + '_next' for col in target_cols]])

    if not all_inputs:
        print("Erro: Nenhum dado válido foi gerado. Verifique o CSV.")
        return None, None

    # Concatena os dados de todos os episódios em um grande DataFrame
    final_inputs_df = pd.concat(all_inputs)
    final_targets_df = pd.concat(all_targets)
    
    # Converte os DataFrames do Pandas para Tensores do PyTorch
    # Tensores são a estrutura de dados principal do PyTorch (como arrays do NumPy)
    inputs_tensor = torch.tensor(final_inputs_df.values, dtype=torch.float32)
    targets_tensor = torch.tensor(final_targets_df.values, dtype=torch.float32)
    
    print(f"Processamento concluído. Total de {len(inputs_tensor)} amostras de treino geradas.")
    
    return inputs_tensor, targets_tensor

# --- 3. Função de Plot ---

def plot_losses(train_losses, val_losses, save_dir="."):
    """
    Plota o gráfico de Perda de Treino vs. Validação.
    Isso nos ajuda a ver se a rede está aprendendo (ambas caindo)
    ou decorando (Train caindo, Val subindo - "overfitting").
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Perda de Treino (Train Loss)")
    plt.plot(val_losses, label="Perda de Validação (Validation Loss)")
    plt.title("MLP: Perda de Treino vs. Validação por Época")
    plt.xlabel("Época (Epoch)")
    plt.ylabel("Perda (MSE Loss)")
    plt.legend()
    plt.grid(True)
    
    # Salva o gráfico em um arquivo PNG
    save_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(save_path)
    print(f"Gráfico de perda salvo em: {save_path}")
    # plt.show() # Descomente se quiser que o gráfico abra na tela
    plt.close()

# --- 4. Loop de Treinamento ---

def main(args):
    """
    Função principal que executa o preparo dos dados,
    o treinamento e a avaliação.
    """
    
    # --- Preparação dos Dados ---
    inputs, targets = prepare_data(args.csv_path)
    if inputs is None:
        return # Sai se a preparação dos dados falhou
        
    # TensorDataset é um container que junta os inputs e targets
    dataset = TensorDataset(inputs, targets)

    # Divide o dataset: 80% para Treino, 20% para Validação
    # Validação: Dados que a rede não usa para treinar, só para
    # checar se ela está aprendendo ("generalizando") ou só
    # decorando ("overfitting").
    val_len = int(0.2 * len(dataset))
    train_len = len(dataset) - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len])

    # DataLoaders: Ferramentas que embaralham e dividem os dados em "lotes"
    # batch_size: Quantos exemplos a rede vê antes de atualizar seus pesos.
    # shuffle=True: Embaralha os dados de treino a cada época (essencial)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    print(f"Iniciando treino com {train_len} amostras de treino e {val_len} de validação.")
    
    # --- Inicialização do Modelo ---
    
    # Decisão de Inputs/Outputs:
    # input_dim = 5 (x, y, yaw, v, w)
    # output_dim = 3 (x_next, y_next, yaw_next)
    input_dim = 5
    output_dim = 3
    model = MLPDynamicsModel(input_dim=input_dim, output_dim=output_dim)
    
    # Otimizador: 'Adam' é um otimizador moderno e robusto.
    # Ele é o "motor" que ajusta os pesos da rede.
    # lr (learning_rate): O "tamanho do passo" que o otimizador dá.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Função de Perda (Loss): 'MSELoss' (Mean Squared Error)
    # Mede o erro: (previsão - alvo)². Perfeito para regressão (prever números).
    criterion = nn.MSELoss()

    # Listas para guardar o histórico de perdas para o gráfico
    train_losses = []
    val_losses = []

    print(f"Treinando por {args.epochs} épocas...")
    # --- Loop de Treinamento ---
    for epoch in range(args.epochs):
        
        # --- Fase de Treino ---
        model.train() # Coloca o modelo em modo de "treino"
        train_loss = 0.0
        
        # Itera sobre os lotes (batches) de dados de treino
        for x_batch, y_batch in train_loader:
            # x_batch: lote de inputs (estado k, ação k)
            # y_batch: lote de targets (estado k+1)
            
            # 1. Fazer a previsão
            pred = model(x_batch)
            
            # 2. Calcular o erro (Loss) entre a previsão (pred) e o alvo (y_batch)
            loss = criterion(pred, y_batch)
            
            # 3. Zerar os gradientes antigos (importante)
            optimizer.zero_grad()
            
            # 4. Calcular os novos gradientes (Backpropagation)
            # (Calcula como cada peso contribuiu para o erro)
            loss.backward()
            
            # 5. Atualizar os pesos da rede
            optimizer.step()
            
            # Acumula a perda do lote
            train_loss += loss.item()
            
        # Calcula a perda média de treino para esta época
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- Fase de Validação ---
        model.eval() # Coloca o modelo em modo de "avaliação" (desliga dropout, etc)
        val_loss = 0.0
        
        # 'with torch.no_grad()' desliga o cálculo de gradientes
        # (economiza memória e tempo, já que não vamos treinar)
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                # 1. Fazer a previsão
                pred = model(x_batch)
                # 2. Calcular o erro (Loss)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                
        # Calcula a perda média de validação para esta época
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Imprime o progresso
        print(f"Época {epoch+1}/{args.epochs}, Perda Treino: {train_loss:.6f}, Perda Validação: {val_loss:.6f}")

    # --- Fim do Treino ---
    print("Treinamento concluído.")

    # Salva o modelo treinado
    if args.model_path:
        # Cria o diretório (ex: "trained_models") se ele não existir
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        # Salva o "dicionário de estados" (os pesos treinados)
        torch.save(model.state_dict(), args.model_path)
        print(f"Modelo salvo em {args.model_path}")

    # Plota os gráficos de perda
    # Salva o gráfico na mesma pasta do modelo
    plot_dir = os.path.dirname(args.model_path) if args.model_path else "."
    plot_losses(train_losses, val_losses, save_dir=plot_dir)


# --- 5. Ponto de Entrada do Script ---
if __name__ == "__main__":

    # Ex: python3 mlp.py --csv_path "meu_dataset.csv" 
    #                    --epochs 50
    #                    --batch_size 32
    #                    --learning_rate 0.0001
    #                    --model_path "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/trained_models/mlp_model.pth"
    #                    
    
    parser = argparse.ArgumentParser(description="Treina um modelo MLP para prever a dinâmica do robô.")
    
    parser.add_argument("--csv_path", type=str, default="/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/dataset_random.csv", 
                        help="Caminho para o arquivo dataset.csv.")
    
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Número de épocas de treinamento.")
    
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Tamanho do lote (batch size) para o treino.")
    
    parser.add_argument("--learning_rate", type=float, default=1e-3, # (0.001)
                        help="Taxa de aprendizado (learning rate) do otimizador.")
    
    parser.add_argument("--model_path", type=str, default="trained_models/mlp_dynamics.pth",
                        help="Caminho para salvar o modelo treinado (arquivo .pth).")
    
    args = parser.parse_args()
    
    # # ------------------------------------------------------------------
    # class MinhasConfiguracoes:
    #     pass
    # args = MinhasConfiguracoes()
    # # ------------------------------------------------------------------
    # args.csv_path = "/home/nexus/VQ_PMCnmpc/VQ_PMC/logs/dataset_random.csv"
    # args.epochs = 100
    # args.batch_size = 64
    # args.learning_rate = 1e-3  # (0.001)
    # args.model_path = "trained_models/mlp_dynamics_fixo.pth"
    # # ------------------------------------------------------------------

    main(args)