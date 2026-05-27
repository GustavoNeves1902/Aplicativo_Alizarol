# ALIZAROL: Classificação de Qualidade do Leite via Deep Learning

Este repositório contém o código-fonte do aplicativo móvel desenvolvido para o projeto de pesquisa e TCC focado na identificação da estabilidade do leite através do teste de Alizarol, utilizando redes neurais convolucionais (CNN).

O aplicativo permite capturar ou selecionar imagens de amostras de leite misturadas ao reagente Alizarol e prevê automaticamente se a amostra está **APROVADA** ou **REPROVADA**, realizando a inferência localmente no dispositivo (offline).

## 🚀 Principais Funcionalidades

- **Câmera Inteligente (Auto-Capture)**: Uma Câmera customizada que executa o modelo **ResNet50** em tempo real (via _Isolates_ secundárias) para avaliar se o enquadramento da placa está perfeito. Se a foto for aprovada pelo modelo, ela é tirada automaticamente.
- **Detecção de Placas e Cores (Galeria)**: Para fotos importadas da galeria, o aplicativo utiliza **OpenCV** para localizar automaticamente o círculo (placa de Petri) e aplicar filtros HSV para garantir que o líquido rosa do Alizarol está presente na amostra.
- **Classificação Avançada**: A inferência final da qualidade do leite é feita através da arquitetura pesada **ConvNeXt Large**, garantindo máxima precisão.
- **Recorte Inteligente**: O aplicativo envia para o modelo de IA exatamente a mesma área recortada que é exibida para o usuário, evitando achatamentos ou distorções da parede ao fundo.
- **Dashboard de Resultados**: Histórico persistente das análises com visualização em lista, exibindo as imagens recortadas, data, hora e a porcentagem de confiança da IA.
- **Privacidade Total**: O aplicativo funciona de forma 100% offline, garantindo agilidade e privacidade no laboratório.

## 🧠 Pipeline de Inteligência Artificial

O aplicativo utiliza um fluxo duplo de visão computacional:

1. **Validação de Captura (ResNet50)**:
   - Utilizado apenas na Câmera.
   - Analisa os frames a 30fps para validar o enquadramento.
2. **Classificador Final (ConvNeXt Large)**:
   - Utilizado na predição final de Câmera e Galeria.
   - Recebe a imagem já recortada e centralizada.

**Detalhes do ConvNeXt Large**:
- **Formato de Entrada**: NCHW (`[1, 3, 224, 224]`)
- **Normalização**: Padrão ImageNet (`Mean: [0.485, 0.456, 0.406]`, `Std: [0.229, 0.224, 0.225]`)
- **Ativação**: Logits processados via Softmax no dispositivo, retornando a probabilidade para as classes APROVADO / REPROVADO.

## 🛠️ Tecnologias Utilizadas

- **Framework**: Flutter (Dart)
- **IA/ML**: `tflite_flutter` (Interpretador nativo de TensoFlow Lite)
- **Processamento de Imagem**: `opencv_dart` e `image` (Detecção de bordas, HSV, recortes e conversão de buffers)
- **Hardware**: `camera` (Acesso direto à lente e stream YUV)
- **Persistência**: `shared_preferences` (Armazenamento local do histórico em JSON)

## 📦 Instalação e Execução

### Pré-requisitos
- Flutter SDK (v3.x ou superior)
- Android SDK / Xcode configurado

### Rodar o projeto
```bash
# Instalar dependências
flutter pub get

# Executar no dispositivo conectado
flutter run
```

### Gerar APK (Android)
```bash
flutter clean
flutter pub get
flutter build apk --release
```
*O APK gerado estará disponível em:* `build/app/outputs/flutter-apk/app-release.apk`

---

## 💾 Persistência de Dados
O aplicativo utiliza o padrão de persistência em JSON dentro do SharedPreferences. 
Dados salvos:
- Caminho absoluto local da imagem recortada.
- Resultado da classificação (Texto + %).
- Data e hora exata da análise.

## 👨‍💻 Autor
**Gustavo** (Adaptado do projeto original de Matheus Henrique C. S. de Souza)
