import copy
import os
import sys
import argparse
import time

import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import random_split

import pandas as pd
import numpy as np
import random
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def load_datasets(train_folder, seed):
    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(train_folder)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = CustomSubset(train_subset, transform=train_tf)
    val_dataset   = CustomSubset(val_subset,   transform=val_tf)

    return train_dataset, val_dataset


# ─────────────────────────────────────────────
# 3. MODEL SETUP
# ─────────────────────────────────────────────
def get_backbone_and_classifier_params(model, model_name):
    """
    Separa os parâmetros do backbone dos parâmetros do classifier (cabeça final).
    Retorna (backbone_params, classifier_params).
    """
    if model_name in ('resnet50', 'googlenet'):
        classifier_params = list(model.module.fc.parameters())
        classifier_ids    = set(id(p) for p in classifier_params)
        backbone_params   = [p for p in model.parameters() if id(p) not in classifier_ids]

    elif model_name in ('alexnet', 'vgg16'):
        classifier_params = list(model.module.classifier[-1].parameters())
        classifier_ids    = set(id(p) for p in classifier_params)
        backbone_params   = [p for p in model.parameters() if id(p) not in classifier_ids]

    elif model_name == 'convnext_large':
        classifier_params = list(model.module.classifier[-1].parameters())
        classifier_ids    = set(id(p) for p in classifier_params)
        backbone_params   = [p for p in model.parameters() if id(p) not in classifier_ids]

    else:
        # Fallback: trata tudo como backbone
        backbone_params   = list(model.parameters())
        classifier_params = []

    return backbone_params, classifier_params


def freeze_backbone(model, model_name):
    """Congela todos os parâmetros exceto a cabeça classificadora."""
    backbone_params, _ = get_backbone_and_classifier_params(model, model_name)
    for p in backbone_params:
        p.requires_grad = False
    n_frozen = sum(p.numel() for p in backbone_params)
    print(f"  [Fase 1] Backbone congelado ({n_frozen:,} parâmetros)")


def unfreeze_all(model):
    """Descongela todos os parâmetros do modelo."""
    for p in model.parameters():
        p.requires_grad = True
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  [Fase 2] Todos os parâmetros descongelados ({n_total:,} parâmetros)")


def setting_model(cnn_model_name, pre_trained, device):
    num_classes = 2

    if cnn_model_name == 'convnext_large':
        weights = models.ConvNeXt_Large_Weights.DEFAULT if pre_trained else None
        cnn_model = models.convnext_large(weights=weights)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)
        num_features = cnn_model.module.classifier[-1].in_features
        cnn_model.module.classifier[-1] = torch.nn.Linear(num_features, num_classes).to(device)

    elif cnn_model_name == 'alexnet':
        weights = models.AlexNet_Weights.DEFAULT if pre_trained else None
        cnn_model = models.alexnet(weights=weights)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)
        num_features = cnn_model.module.classifier[6].in_features
        cnn_model.module.classifier[6] = torch.nn.Linear(num_features, num_classes).to(device)

    elif cnn_model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pre_trained else None
        cnn_model = models.resnet50(weights=weights)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)
        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, num_classes).to(device)

    elif cnn_model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if pre_trained else None
        cnn_model = models.vgg16(weights=weights)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)
        num_features = cnn_model.module.classifier[6].in_features
        cnn_model.module.classifier[6] = torch.nn.Linear(num_features, num_classes).to(device)

    elif cnn_model_name == 'googlenet':
        if pre_trained:
            # Com pesos pré-treinados, o torchvision FORÇA aux_logits=True internamente.
            # O _run_epoch já trata o namedtuple com outputs[0] (logits principal).
            cnn_model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        else:
            # Sem pesos pré-treinados podemos desativar as saídas auxiliares
            cnn_model = models.googlenet(weights=None, aux_logits=False)
        cnn_model = torch.nn.DataParallel(cnn_model)
        cnn_model.to(device)
        num_features = cnn_model.module.fc.in_features
        cnn_model.module.fc = torch.nn.Linear(num_features, num_classes).to(device)

    else:
        print(f'Modelo inválido: {cnn_model_name}')
        sys.exit(1)

    return cnn_model


# ─────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────
def train_model(
    model_cnn, epochs, train_loader, val_loader, device, model_name,
    pre_trained,
    # Fase 1: só a cabeça
    lr_head=1e-3,
    warmup_epochs=10,
    # Fase 2: fine-tuning completo
    lr_backbone=1e-5,
    lr_head_ft=1e-4,
    # Early stopping
    patience=15,
):
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model_cnn.state_dict())
    epochs_no_improve = 0

    # ── FASE 1: aquece apenas a cabeça (só para modelos pré-treinados) ──
    if pre_trained and warmup_epochs > 0:
        freeze_backbone(model_cnn, model_name)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_cnn.parameters()),
            lr=lr_head
        )
        phase1_epochs = min(warmup_epochs, epochs)
    else:
        phase1_epochs = 0
        optimizer = None  # definido na Fase 2

    for epoch in range(phase1_epochs):
        _run_epoch(model_cnn, criterion, optimizer, train_loader, val_loader,
                   device, epoch, tag="[Fase1-Warmup]")

    # ── FASE 2: fine-tuning completo ──
    if pre_trained:
        unfreeze_all(model_cnn)

    backbone_params, classifier_params = get_backbone_and_classifier_params(model_cnn, model_name)

    if pre_trained and backbone_params:
        # LR diferenciada: backbone aprendende mais devagar
        optimizer = torch.optim.Adam([
            {'params': backbone_params,   'lr': lr_backbone},
            {'params': classifier_params, 'lr': lr_head_ft},
        ])
    else:
        # Sem pré-treino: usa uma única LR razoável
        optimizer = torch.optim.Adam(model_cnn.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
    )

    remaining_epochs = epochs - phase1_epochs
    for epoch in range(remaining_epochs):
        train_loss, train_acc, val_loss, val_acc = _run_epoch(
            model_cnn, criterion, optimizer, train_loader, val_loader,
            device, epoch + phase1_epochs, tag="[Fase2-FineTune]"
        )
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model_cnn.state_dict())
            epochs_no_improve = 0
            print(f'  🔥 Melhor modelo atualizado na Época {epoch + phase1_epochs} | Val Loss: {val_loss:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'\n⏹  Early stopping disparado após {patience} épocas sem melhora.')
                break

    model_cnn.load_state_dict(best_model_wts)
    return model_cnn


def _run_epoch(model_cnn, criterion, optimizer, train_loader, val_loader, device, epoch, tag=""):
    """Executa uma única época de treino + validação. Retorna (train_loss, train_acc, val_loss, val_acc)."""

    # --- TREINO ---
    model_cnn.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model_cnn(inputs)
        # Segurança extra caso outputs seja tuple/namedtuple
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]

        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_correct += torch.sum(predictions == labels.data).item()
        train_loss    += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    train_acc   = train_correct / len(train_loader.dataset)

    # --- VALIDAÇÃO ---
    model_cnn.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_cnn(inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs[0]

            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_correct += torch.sum(predictions == labels.data).item()
            val_loss    += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc   = val_correct / len(val_loader.dataset)

    print(f'Época {epoch:03d} {tag} | '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    return train_loss, train_acc, val_loss, val_acc


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Usando device: {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment',   type=str,   required=True,  help='Nome do experimento')
    parser.add_argument('-m', '--model',        type=str,   default='resnet50', help='Modelo CNN')
    parser.add_argument('-s', '--seed',         type=int,   default=42,     help='Seed para divisão dos dados')
    parser.add_argument('-t', '--tag',          type=str,   default='',     help='Tag de identificação (ex: FotosLeiteAmarelo)')
    parser.add_argument('--no-pretrained',      action='store_true',        help='Treinar sem pesos pré-treinados')
    parser.add_argument('--warmup-epochs',      type=int,   default=10,     help='Épocas de aquecimento (só cabeça)')
    parser.add_argument('--epochs',             type=int,   default=100,    help='Épocas totais de treinamento')
    parser.add_argument('--batch-size',         type=int,   default=32,     help='Batch size')
    parser.add_argument('--lr-head',            type=float, default=1e-3,   help='LR do classifier no warmup')
    parser.add_argument('--lr-backbone',        type=float, default=1e-5,   help='LR do backbone no fine-tuning')
    parser.add_argument('--lr-head-ft',         type=float, default=1e-4,   help='LR do classifier no fine-tuning')
    parser.add_argument('--patience',           type=int,   default=15,     help='Paciência para early stopping')
    args = parser.parse_args()

    experiment_name = args.experiment
    model_name      = args.model
    seed            = args.seed
    custom_tag      = args.tag
    pre_trained     = not args.no_pretrained
    warmup_epochs   = args.warmup_epochs
    total_epochs    = args.epochs
    batch_size      = args.batch_size
    lr_head         = args.lr_head
    lr_backbone     = args.lr_backbone
    lr_head_ft      = args.lr_head_ft
    patience        = args.patience

    # --- APLICAR A SEED PARA GARANTIR REPRODUTIBILIDADE ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # -------------------------------------------------------

    print('─' * 70)
    print(f'Experimento : {experiment_name}')
    print(f'Modelo      : {model_name}')
    print(f'Pré-treinado: {pre_trained}')
    print(f'Épocas      : {total_epochs}  (warmup={warmup_epochs})')
    print(f'Batch size  : {batch_size}')
    print(f'LR head     : {lr_head}  |  LR backbone: {lr_backbone}  |  LR head ft: {lr_head_ft}')
    print(f'Early stop  : patience={patience}')
    print(f'Seed        : {seed}  |  Tag: {custom_tag}')
    print('─' * 70)

    # 1. DADOS
    train_dataset, val_dataset = load_datasets('trains/' + experiment_name, seed)
    print(f'Treino: {len(train_dataset)} imagens | Validação: {len(val_dataset)} imagens')

    # 2. MODELO
    model = setting_model(model_name, pre_trained, device)

    # 3. DATALOADERS
    num_workers = min(4, os.cpu_count() or 4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # 4. TREINAMENTO
    model = train_model(
        model_cnn=model,
        epochs=total_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_name=model_name,
        pre_trained=pre_trained,
        lr_head=lr_head,
        warmup_epochs=warmup_epochs,
        lr_backbone=lr_backbone,
        lr_head_ft=lr_head_ft,
        patience=patience,
    )

    # 5. SALVAR MODELO
    tag_suffix    = f'_{custom_tag}' if custom_tag else ''
    filename_core = f'{experiment_name}_{model_name}_S{seed}{tag_suffix}'

    model_path = os.path.abspath(os.path.join('models', experiment_name))
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, f'{filename_core}.pth'))
    print(f'\nModelo salvo em: models/{experiment_name}/{filename_core}.pth')

    # 6. SALVAR LOG CSV
    end_time       = time.time()
    total_duration = end_time - start_time

    gs_path = os.path.abspath(os.path.join('gridSearch', experiment_name, model_name))
    os.makedirs(gs_path, exist_ok=True)

    gs_df = pd.DataFrame([{
        'lr_head':       lr_head,
        'lr_backbone':   lr_backbone,
        'lr_head_ft':    lr_head_ft,
        'epochs':        total_epochs,
        'warmup_epochs': warmup_epochs,
        'patience':      patience,
        'pre_trained':   pre_trained,
        'batch_size':    batch_size,
        'seed':          seed,
        'tag':           custom_tag,
        'duration_sec':  total_duration,
    }])
    gs_df.to_csv(os.path.join(gs_path, f'{model_name}_S{seed}{tag_suffix}.csv'), index=False)

    print(f'✅ {model_name} treinado com sucesso para o experimento "{experiment_name}"')
    print(f'⏱  Duração total: {total_duration:.2f} s  ({total_duration/60:.1f} min)')
