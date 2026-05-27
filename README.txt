========================================================================
ALIZAROL APP: CLASSIFICAÇÃO DE QUALIDADE DO LEITE VIA DEEP LEARNING & CV
========================================================================

Este documento descreve detalhadamente o funcionamento técnico, a arquitetura
de código e as funcionalidades atuais do projeto Aplicativo-CNN (Alizarol App).

------------------------------------------------------------------------
1. VISÃO GERAL DO PROJETO
------------------------------------------------------------------------
O Alizarol App é um aplicativo móvel desenvolvido em Flutter (Dart) para
identificar a estabilidade/qualidade do leite através do teste químico de Alizarol.
Ele realiza o processamento digital de imagens e a inferência de Inteligência
Artificial (Deep Learning) LOCALMENTE no dispositivo (offline), focando em
privacidade, rapidez e utilidade prática no campo ou laboratório.

O aplicativo substitui avaliações puramente subjetivas por análises digitais
padronizadas e automatizadas em três etapas consecutivas:
1. Validação de posicionamento em tempo real por Rede Neural (TFLite).
2. Detecção e recorte da placa de Petri por Visão Computacional (OpenCV).
3. Classificação de estabilidade ("APROVADO" ou "REPROVADO") por uma CNN (TFLite).

------------------------------------------------------------------------
2. FLUXO PRINCIPAL DE PROCESSAMENTO (PIPELINE DE ANÁLISE)
------------------------------------------------------------------------

--- FASE 0: Detecção de Posicionamento em Tempo Real (câmera) ---

Antes de qualquer análise, a câmera executa inferência contínua com o modelo
`Alizarol_Leite_resnet50_S46_padrao_foradepadrao.tflite` (ResNet50, ~94 MB)
para avaliar se a amostra está posicionada corretamente.

O modelo retorna logits brutos [logit_classe_0, logit_classe_1] que representam:
  - Classe 0: "fora do padrão" (posicionamento incorreto)
  - Classe 1: "padrão" (posicionamento correto)

O app aplica softmax manualmente nos logits:
  softmax = exp(logit_i - max) / sum(exp(logit_j - max))
  padrao  = softmax[1] >= 0.50  →  captura automática disparada

Enquanto a posição não atinge o limiar, a UI exibe uma barra de confiança
animada em tempo real. Ao atingir >= 50% de confiança em "padrão", a foto
é capturada automaticamente.

Pré-processamento de entrada do modelo:
  - Resize para 224x224 pixels
  - Normalização ImageNet: pixel = (canal/255 - mean) / std
      mean = [0.485, 0.456, 0.406]  (R, G, B)
      std  = [0.229, 0.224, 0.225]  (R, G, B)
  - Tensor de entrada: [1, 224, 224, 3] (float32, NHWC)

Throttle: 1 frame processado a cada 500ms para evitar sobrecarga.

--- FASE 1 (após captura): Detecção da Placa de Petri ---

Quando o usuário seleciona uma foto da galeria ou a câmera captura
automaticamente, o método `_handleProcessedImage` em `lib/main.dart` executa:

Etapa A: Detecção de Círculo (Placa de Petri)
- A imagem é analisada pela função `detectPetriAndCrop` em `lib/detector.dart`.
- Utilizando a biblioteca OpenCV (`opencv_dart`), a imagem é convertida para
  tons de cinza e suavizada por um filtro mediano pesado (`cv.medianBlur`, kernel 11)
  para eliminar texturas indesejadas e ruídos de fundo.
- O algoritmo HoughCircles (`cv.HoughCircles`) localiza a borda externa da
  placa de Petri. Se encontrada, a placa é recortada com uma margem de segurança
  (padding) de 70 pixels para fins de consistência e enquadramento.
- Se nenhum círculo for encontrado, o app interrompe o processamento e
  classifica o resultado como "ALIZAROL NÃO ENCONTRADO".

Etapa B: Validação de Cor (Rosa/Vermelho do Alizarol)
- Caso o círculo exista, a imagem recortada é avaliada pela função `hasAlizarolPink`
  em `lib/detector.dart` antes de ser enviada ao modelo de classificação.
- A imagem é convertida para o espaço de cores HSV e filtrada usando intervalos
  duplos equivalentes aos do script Python original `findind_circles_01.py`:
  * Faixa 1 (vermelho baixo): Hue [0-10], Saturation [30-255], Value [50-255].
  * Faixa 2 (vermelho alto / rosa): Hue [160-180], Saturation [30-255], Value [50-255].
- Ambas as faixas são mescladas. Se a proporção de pixels rosa/vermelho na imagem
  for inferior a 1% (`minPixelRatio = 0.01`), a análise é suspensa e marcada
  como "ALIZAROL NÃO ENCONTRADO" (evitando falsos positivos gerados por fotos de
  objetos circulares alheios ao teste de leite).

Etapa C: Inferência com Inteligência Artificial (CNN — Aprovado/Reprovado)
- Confirmados o formato circular e a presença do tom rosa do reagente, o recorte é
  redimensionado para 224x224 pixels.
- Os pixels RGB são normalizados com a mesma normalização ImageNet descrita acima.
- O interpretador local do TensorFlow Lite (`tflite_flutter`) executa a classificação
  com o modelo neural `assets/models/Alizarol_Leite_resnet50_S43_AprovadoReprovadoSegmentado.tflite`.
- A saída em Logits é convertida em probabilidades por uma função Softmax local.
- O resultado final exibe a classificação ("APROVADO" ou "REPROVADO") acompanhada
  da taxa de confiança estatística (%).

------------------------------------------------------------------------
3. MODELOS DE INTELIGÊNCIA ARTIFICIAL (TFLite)
------------------------------------------------------------------------

Modelo 1: padrao_foradepadrao (detecção de posicionamento)
  Arquivo : assets/models/Alizarol_Leite_resnet50_S46_padrao_foradepadrao.tflite
  Backbone: ResNet50 (treinado com transfer learning sobre ImageNet)
  Tarefa  : Classificação binária — "padrão" vs "fora do padrão"
  Entrada : [1, 224, 224, 3] float32, normalização ImageNet
  Saída   : [1, 2] logits brutos — softmax aplicado no app
  Threshold: softmax[1] >= 0.50 → posição correta → captura automática
  Uso     : Câmera em tempo real (throttle 500ms)

Modelo 2: AprovadoReprovadoSegmentado (classificação de qualidade)
  Arquivo : assets/models/Alizarol_Leite_resnet50_S43_AprovadoReprovadoSegmentado.tflite
  Backbone: ResNet50
  Tarefa  : Classificação binária — "Aprovado" vs "Reprovado"
  Entrada : [1, 224, 224, 3] float32, normalização ImageNet
  Saída   : [1, 2] logits brutos — softmax aplicado no app
  Uso     : Análise da imagem após captura

------------------------------------------------------------------------
4. ESTRUTURA E FUNCIONALIDADE DOS ARQUIVOS DE CÓDIGO
------------------------------------------------------------------------

* lib/main.dart
  - Ponto de entrada do app Flutter (`main()`).
  - Gerencia o Dashboard principal, exibindo um histórico persistente (salvo
    localmente via `shared_preferences`) das análises feitas.
  - Oferece opção de exclusão do histórico e visualizador de imagem em tela cheia
    com controle de zoom dinâmico por pinça (`InteractiveViewer`).
  - Carrega o modelo de classificação e coordena a pipeline principal.

* lib/tflite_classifier.dart (NOVO — Classificador de Posicionamento)
  - Serviço singleton responsável pela inferência em tempo real durante a captura.
  - Carrega o modelo `padrao_foradepadrao.tflite` via `tflite_flutter`.
  - Executa pré-processamento: resize 224x224 + normalização ImageNet.
  - Aplica softmax manualmente nos logits brutos do modelo.
  - Expõe `classifyFrame(Uint8List jpegBytes) → double` retornando a probabilidade
    de "padrão" (valor entre 0.0 e 1.0).
  - Usa 2 threads de inferência (`InterpreterOptions..threads = 2`).

* lib/camera_screen.dart
  - Tela de captura do aplicativo que utiliza inferência TFLite em tempo real.
  - Recebe frames da câmera continuamente (throttle de 500ms) e os envia ao
    `TfliteClassifier` para classificação de posicionamento.
  - Interface visual dinâmica:
    * Barra de confiança animada mostrando a probabilidade de "padrão" em %.
    * Badge "IA" que pisca em âmbar durante a inferência.
    * Feedback de status: "Posicionando..." / "Quase lá... (XX%)" / "Posição correta!".
    * Círculo guia muda de cor (branco → âmbar → verde) conforme a confiança sobe.
  - Captura automática disparada quando o modelo classifica "padrão" (>= 50%).
  - Botão manual de captura disponível como fallback a qualquer momento.

* lib/detector.dart
  - Concentra os algoritmos de processamento digital de imagem do OpenCV.
  - Contém `detectPetriAndCrop` (HoughCircles) e `hasAlizarolPink` (máscara HSV).
  - Inclui conversores eficientes de dados binários do stream da câmera para
    JPEGs redimensionados no formato do Dart Image (`cameraImageToJpeg`).
  - Esta função ainda é usada pela câmera para converter frames antes da inferência.

* lib/camera_screen_test.dart (Experimental — Background Isolate)
  - Desenvolvido especificamente para o teste de viabilidade técnica da classificação
    contínua de imagens rodando de forma fluida (60 FPS na preview) através de um
    Isolate Persistente de Segundo Plano (Background Thread).
  - Utiliza `RootIsolateToken` e `BackgroundIsolateBinaryMessenger` para viabilizar o
    carregamento do modelo `model.tflite` (~94 MB) e a comunicação bidirecional com canais
    nativos a partir do Isolate secundário.
  - Recebe os planos de bytes brutos dos frames (`CameraImage`) da thread principal e
    realiza todo o pré-processamento (downsampling Nearest-Neighbor ultra-rápido para
    224x224, normalização float32 e redimensionamento) dentro do Isolate (< 10ms).
  - Executa a inferência síncrona do TFLite inteiramente na thread secundária, liberando
    completamente a Thread Principal do Flutter de qualquer travamento ou gargalo.
  - Possui um painel flutuante de diagnóstico e telemetria em tempo real mostrando:
    * Tempo de Pré-processamento no Isolate (ms).
    * Tempo de Inferência da rede neural no Isolate (ms).
    * FPS real de atualização das predições.
    * Resultado instantâneo (APROVADO/REPROVADO e confiança %).
  - Controles interativos em tempo real:
    * Chave de Auto-captura (tira a foto automaticamente se classificar "APROVADO").
    * Ajustador de Throttle (intervalo de inferência): Raw/Stress (sem pausas),
      150ms, 300ms ou 600ms, ideal para dosar a carga do Isolate.

* lib/splash_screen.dart
  - Tela inicial de apresentação que carrega a imagem de fundo `alizarol.jpg` e
    conduz o usuário ao dashboard principal ao clicar em "Iniciar".

* script_treino/ (e train_CNN_congelando10epocas.py na raiz)
  - Diretório dedicado à ciência de dados e machine learning:
    * `train_CNN_congelando10epocas.py`: Script PyTorch de treinamento da CNN.
      Utiliza ResNet50 com transfer learning (backbone congelado nas primeiras
      épocas). Normalização: ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
      Entrada: Resize(256,256) → CenterCrop(224) → ToTensor() → Normalize().
    * `evaluate_model_congelando10epocas.py`: Avaliação do modelo gerado.
    * `findind_circles_01.py`: Protótipo em Python (OpenCV) que implementa a mesma
       lógica HSV para recortar o contorno rosa do Alizarol em lote.

------------------------------------------------------------------------
5. ASSETS — MODELOS E IMAGENS
------------------------------------------------------------------------

assets/models/
  - Alizarol_Leite_resnet50_S46_padrao_foradepadrao.tflite  (~94 MB)
      → Detecção de posicionamento em tempo real (câmera)
  - Alizarol_Leite_resnet50_S43_AprovadoReprovadoSegmentado.tflite  (~24 MB)
      → Classificação Aprovado/Reprovado após captura
  - Alizarol_Leite_convnext_large_S42_congelando10epocas.tflite  (~785 MB)
      → Modelo alternativo ConvNeXt-Large (experimental, não ativo)
  - model.tflite  (~94 MB)
      → Cópia do modelo padrao_foradepadrao (uso em experimentos de Isolate)

assets/imagens/
  - logo_coen1.png, logo_VIA.png  → Logos institucionais
  - feijao.png                    → Ícone auxiliar
  - alizarol.jpg                  → Imagem de fundo da splash screen

------------------------------------------------------------------------
6. DEPENDÊNCIAS PRINCIPAIS (pubspec.yaml)
------------------------------------------------------------------------

  flutter          → Framework UI principal
  tflite_flutter   → Inferência TFLite local (modelos ResNet50/ConvNeXt)
  camera           → Acesso ao stream de câmera em tempo real
  opencv_dart      → Processamento de imagem (HoughCircles, HSV, etc.)
  image            → Decodificação/resize de imagens em Dart puro
  image_picker     → Seleção de fotos da galeria
  shared_preferences → Persistência local do histórico de análises

========================================================================
