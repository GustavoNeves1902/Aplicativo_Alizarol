ALIZAROL: Classificação de Qualidade do Leite via Deep Learning

Este repositório contém o código-fonte do aplicativo móvel desenvolvido para o projeto de pesquisa e TCC focado na identificação da estabilidade do leite através do teste de Alizarol, utilizando redes neurais convolucionais (CNN).

O aplicativo permite capturar ou selecionar imagens de amostras de leite misturadas ao reagente Alizarol e prevê automaticamente se a amostra está APROVADA ou REPROVADA, realizando a inferência localmente no dispositivo.

O app integra um modelo TensorFlow Lite (ResNet50) e executa inferências offline, garantindo privacidade e agilidade no campo ou laboratório.

Principais Funcionalidades

Câmera Customizada: Guia visual circular e análise de iluminância em tempo real para garantir capturas padronizadas.

Recorte Inteligente: Processamento de imagem que foca no centro do visor ou detecta automaticamente a região rosada da amostra (Bounding Box).

Inferência Local: Classificação instantânea utilizando a arquitetura ResNet50.

Dashboard de Resultados: Histórico persistente das análises com visualização de data, hora e porcentagem de confiança.

Visualizador de Imagens: Toque em qualquer amostra no dashboard para expandir a imagem com suporte a zoom (InteractiveViewer).

Modelo de Machine Learning

O aplicativo utiliza um modelo de classificação binária treinado em PyTorch e convertido para TFLite.

Atributo	        Especificação
Arquitetura	        ResNet50 (Transfer Learning)
Classes	0:          APROVADO / 1: REPROVADO
Formato de Entrada	RGB (Canais por último - NHWC)
Dimensões do Tensor	[1, 224, 224, 3]
Normalização	    Valores de pixel escalonados para [0.0, 1.0]
Ativação de Saída	Logits processados via Softmax no dispositivo

Tecnologias Utilizadas

Framework: Flutter (Dart)
IA/ML: tflite_flutter (Interpretador de alto desempenho)
Processamento de Imagem: image (Manipulação de buffers e pixels)
Câmera: camera (Acesso ao hardware e stream de vídeo)
Persistência: shared_preferences (Armazenamento local do histórico)
UI/UX: image_picker, google_fonts

Instalação e Execução
    Pré-requisitos

    Flutter SDK (v3.x ou superior)

    Android SDK configurado

Rodar o projeto
    # Instalar dependências
    flutter pub get

    # Executar em modo debug
    flutter run

Gerar APK de Lançamento
flutter clean
flutter pub get
flutter build apk --release

O APK gerado estará disponível em: build/app/outputs/flutter-apk/app-release.apk

Pipeline de Processamento de Imagem
Antes de cada inferência, a imagem passa pelas seguintes etapas:

Captura/Seleção: Recebimento do arquivo de imagem original.

Square Crop: Recorte central de 70% da menor dimensão para isolar o frasco.

Resizing: Redimensionamento bilinear para exatamente 224x224.

Normalização: Conversão dos bytes Uint8 para float32 (divisão por 255.0).

Inferência: Execução no interpretador TFLite.

Pós-processamento: Aplicação da função Softmax nos resultados brutos para gerar a porcentagem de confiança.

Persistência de Dados
O aplicativo utiliza o padrão de persistência em JSON dentro do SharedPreferences.
Dados salvos:

Caminho do arquivo da imagem recortada.

Resultado da classificação (Texto + %).

Data e hora exata da análise.


Toda a inteligência reside no dispositivo; nenhum dado é enviado para servidores externos.

Autor
Gustavo (Adaptado do projeto original de Matheus Henrique C. S. de Souza)
