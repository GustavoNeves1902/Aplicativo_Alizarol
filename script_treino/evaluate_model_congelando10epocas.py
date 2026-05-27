import os
import argparse
import sys
import torch
import torch.utils.data
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np 
import time # NOVO: Para medir o tempo de avaliação

# --- 1. FUNÇÕES DE CARREGAMENTO E CONFIGURAÇÃO ---

def transform_images_test():
    # IMPORTANTE: deve ser idêntico ao val_transform do train_CNN.py
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return data_transform

def setting_model(cnn_model_name):
    
    if cnn_model_name == 'convnext_large':
        cnn_model = models.convnext_large(weights=None)
        num_features = cnn_model.classifier[-1].in_features
        cnn_model.classifier[-1] = torch.nn.Linear(num_features, 2)
    elif cnn_model_name == 'alexnet':
        cnn_model = models.alexnet(weights=None)
        num_features = cnn_model.classifier[6].in_features
        cnn_model.classifier[6] = torch.nn.Linear(num_features, 2)
    elif cnn_model_name == 'resnet50':
        cnn_model = models.resnet50(weights=None)
        num_features = cnn_model.fc.in_features
        cnn_model.fc = torch.nn.Linear(num_features, 2)
    elif cnn_model_name == 'vgg16':
        cnn_model = models.vgg16(weights=None)
        num_features = cnn_model.classifier[6].in_features
        cnn_model.classifier[6] = torch.nn.Linear(num_features, 2)
    elif cnn_model_name == 'googlenet':
        cnn_model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
        num_features = cnn_model.fc.in_features
        cnn_model.fc = torch.nn.Linear(num_features, 2)
    else:
        print('Invalid model')
        sys.exit(1)
        
    return cnn_model

# --- 2. FUNÇÃO PRINCIPAL DE AVALIAÇÃO ---

def evaluate_model(exp_name, model_name, seed, custom_tag, batch_size=32, threshold=0.5):
    
    start_time = time.time() # Início da avaliação
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. CARREGAR DADOS DE TESTE
    data_transform = transform_images_test()
    test_folder = os.path.join('tests', exp_name)
    test_datasets = datasets.ImageFolder(test_folder, transform=data_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 2. CARREGAR MODELO TREINADO (BUSCA PELO ARQUIVO COM A SEED E TAG)
    tag_suffix = f'_{custom_tag}' if custom_tag else '' # Cria o sufixo da tag
    
    # Nome do arquivo salvo pelo treino: [EXP]_[MODELO]_S[SEED]_[TAG].pth
    weights_filename = f'{exp_name}_{model_name}_S{seed}{tag_suffix}.pth'
    model_path = os.path.abspath(os.path.join('models', exp_name, weights_filename))

    if not os.path.exists(model_path):
        print(f"Erro: Arquivo de pesos não encontrado em {model_path}. Treinamento com Seed={seed} e Tag='{custom_tag}' não concluído.")
        return

    model = setting_model(model_name)
    
    # Carrega os pesos salvos
    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # strict=False: ignora chaves extras (ex: aux1/aux2 do GoogLeNet pré-treinado)
        # que não existem no modelo de avaliação (aux_logits=False)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if unexpected:
            print(f"  [info] Chaves ignoradas no carregamento: {unexpected}")
        model.to(device)

    except Exception as e:
        print(f"Erro ao carregar os pesos do modelo: {e}")
        return
        
    # 3. EXECUTAR AVALIAÇÃO — coleta probabilidades para a Curva ROC
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs_reprovado = []  # Probabilidade da classe Reprovado (índice 1) para a Curva ROC
    print(f"\nIniciando avaliação do modelo {model_name} (Seed={seed}, Tag='{custom_tag}')...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc='Avaliando'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            # Probabilidades via softmax (necessário para a Curva ROC)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            prob_reprovado = probs[:, 1]  # Probabilidade da classe Reprovado (índice 1)

            all_labels.extend(labels.cpu().tolist())
            all_probs_reprovado.extend(prob_reprovado.cpu().tolist())

    # Aplicar o threshold escolhido (substitui as predições por threshold personalizado)
    if threshold != 0.5:
        print(f"\n[info] Aplicando threshold personalizado: {threshold} (padrão seria 0.5)")
    all_predictions = [1 if p >= threshold else 0 for p in all_probs_reprovado]

    # 4. CALCULAR MÉTRICAS E MATRIZ DE CONFUSÃO 
    y_true = all_labels
    y_pred = all_predictions
    y_scores = np.array(all_probs_reprovado)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, VP = cm.ravel()
    
    # Registrar tempo total
    end_time = time.time()
    total_duration = end_time - start_time

    results = {
        'Threshold': threshold,
        'Acurácia': accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Precisão': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensibilidade)': recall_score(y_true, y_pred, zero_division=0),
        'TN (VN - Acerto no Aprovado)': TN,
        'FP (Custo - Erro no Aprovado)': FP,
        'FN (Risco - Erro no Reprovado)': FN,
        'VP (Acerto no Reprovado)': VP,
        'Duration_sec': total_duration
    }

    # 5. EXIBIR E SALVAR RESULTADOS
    print("\n--- Resultados de Classificação Binária ---")
    print(f"Modelo: {model_name} | Seed: {seed} | Tag: {custom_tag}")
    print(f"Conjunto de Teste: {exp_name}")
    print("------------------------------------------")
    
    for metric, value in results.items():
        # Exibe a duração no console
        print(f"{metric:<38}: {value:.4f}") 
    
    # Salvar resultados
    results_df = pd.DataFrame([results])
    results_path = os.path.abspath(f'results/{exp_name}')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # SALVA O CSV COM O NOME DO MODELO, SEED E TAG
    output_file = os.path.join(results_path, f'Evaluation_{model_name}_S{seed}{tag_suffix}_T{threshold}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResultados salvos em: {output_file}")

    # 6. GERAR E SALVAR A CURVA ROC
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend sem interface gráfica
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Threshold de melhor equilíbrio FN/FP: ponto mais próximo do canto superior esquerdo
        distances = np.sqrt((fpr ** 2) + ((1 - tpr) ** 2))
        best_idx = np.argmin(distances)
        best_threshold = thresholds[best_idx]
        best_fpr = fpr[best_idx]
        best_tpr = tpr[best_idx]

        print(f"\n--- Curva ROC ---")
        print(f"AUC                                   : {roc_auc:.4f}")
        print(f"Melhor threshold (equilíbrio FN+FP)   : {best_threshold:.4f}")
        print(f"  → FPR (Taxa FP): {best_fpr:.4f}  |  TPR/Recall: {best_tpr:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#4F86C6', lw=2,
                label=f'Curva ROC (AUC = {roc_auc:.4f})')
        ax.scatter([best_fpr], [best_tpr], color='#E05C5C', zorder=5, s=100,
                   label=f'Melhor threshold = {best_threshold:.2f}\n'
                         f'(FPR={best_fpr:.3f}, TPR={best_tpr:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1,
                label='Classificador aleatório')
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR / Recall)', fontsize=12)
        ax.set_title(f'Curva ROC — {model_name} | Seed {seed}{tag_suffix}', fontsize=13)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        roc_file = os.path.join(results_path, f'ROC_{model_name}_S{seed}{tag_suffix}.png')
        fig.savefig(roc_file, dpi=150)
        plt.close(fig)
        print(f"Curva ROC salva em: {roc_file}")

    except ImportError:
        print("\n[Aviso] matplotlib não instalado. Curva ROC não gerada.")
        print("        Instale com: pip install matplotlib")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Avalia o modelo CNN treinado (Terceira Abordagem).')
    parser.add_argument('-e', '--experiment', type=str, required=True, help='Nome do experimento (ex: Alizarol_Leite)')
    parser.add_argument('-m', '--model', default='resnet50', type=str, help='Modelo CNN (ex: resnet50, convnext_large)')
    parser.add_argument('-s', '--seed', type=int, required=True, help='Seed usada para buscar o modelo correto.')
    parser.add_argument('-t', '--tag', type=str, default='', help='Tag de identificação do experimento.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Limiar de decisão para a classe Reprovado (padrão=0.5). '
                             'Valores menores aumentam a sensibilidade (menos FN, mais FP). '
                             'Veja a Curva ROC para escolher o valor ideal.')
    args = parser.parse_args()

    evaluate_model(args.experiment, args.model, args.seed, args.tag, threshold=args.threshold)