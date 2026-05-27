import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:isolate';

// ─────────────────────────────────────────────────────────────────────────────
// TfliteClassifier — inferência em tempo real com ResNet50 padrão/fora do padrão
// ─────────────────────────────────────────────────────────────────────────────

/// Serviço singleton que carrega o modelo TFLite ResNet50 e classifica frames
/// da câmera como "padrão" ou "fora do padrão".
///
/// O modelo retorna logits brutos [logit_0, logit_1].
/// Aplicamos softmax manualmente e usamos softmax[1] como probabilidade de "padrão".
///
/// Normalização de entrada (ImageNet):
///   mean = [0.485, 0.456, 0.406]
///   std  = [0.229, 0.224, 0.225]
///   tensor = [1, 224, 224, 3] (NHWC)
///
/// Threshold: softmax[1] >= 0.50 → "padrão"
class TfliteClassifier {
  TfliteClassifier._();
  static final TfliteClassifier instance = TfliteClassifier._();

  static const String _modelPath =
      'assets/models/Alizarol_Leite_resnet50_S46_padrao_foradepadrao.tflite';

  // Normalização ImageNet (igual ao treinamento PyTorch)
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  // Threshold de confiança para considerar "padrão"
  static const double threshold = 0.50;

  Interpreter? _interpreter;
  bool _loading = false;

  // ── Inicialização ───────────────────────────────────────────────────────────

  /// Carrega o modelo TFLite. Deve ser chamado uma vez (ex: no initState da câmera).
  Future<void> loadModel() async {
    if (_interpreter != null || _loading) return;
    _loading = true;
    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(_modelPath, options: options);
      print('[TfliteClassifier] Modelo carregado: $_modelPath');
      print(
          '[TfliteClassifier] Input:  ${_interpreter!.getInputTensors().map((t) => t.shape)}');
      print(
          '[TfliteClassifier] Output: ${_interpreter!.getOutputTensors().map((t) => t.shape)}');
    } catch (e) {
      print('[TfliteClassifier] Erro ao carregar modelo: $e');
    } finally {
      _loading = false;
    }
  }

  /// Carrega o modelo TFLite a partir de bytes.
  void loadModelFromBuffer(Uint8List buffer) {
    if (_interpreter != null || _loading) return;
    _loading = true;
    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = Interpreter.fromBuffer(buffer, options: options);
      print('[TfliteClassifier] Modelo carregado de buffer');
      print(
          '[TfliteClassifier] Input:  ${_interpreter!.getInputTensors().map((t) => t.shape)}');
      print(
          '[TfliteClassifier] Output: ${_interpreter!.getOutputTensors().map((t) => t.shape)}');
    } catch (e) {
      print('[TfliteClassifier] Erro ao carregar modelo de buffer: $e');
    } finally {
      _loading = false;
    }
  }

  bool get isReady => _interpreter != null;

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  // ── Inferência ──────────────────────────────────────────────────────────────

  /// Classifica um frame JPEG e retorna a probabilidade de "padrão" (0.0 – 1.0).
  ///
  /// Retorna -1.0 se o modelo não estiver carregado ou ocorrer um erro.
  double classifyFrame(Uint8List jpegBytes) {
    final interpreter = _interpreter;
    if (interpreter == null) return -1.0;

    try {
      // 1. Decodifica e redimensiona para 224×224
      final decoded = img.decodeImage(jpegBytes);
      if (decoded == null) return -1.0;

      final resized = img.copyResize(decoded, width: 224, height: 224);

      // 2. Constrói tensor de entrada [1, 224, 224, 3] com normalização ImageNet
      final input = _buildInputTensor(resized);

      // 3. Prepara tensor de saída [1, 2] (logits)
      final output = List.generate(1, (_) => List<double>.filled(2, 0.0));

      // 4. Executa inferência
      interpreter.run(input, output);

      // 5. Aplica softmax manualmente nos logits brutos
      final logits = output[0];
      final probs = _softmax(logits);

      final probPadrao =
          probs[1]; // índice 1 = "padrão" (ordem alfabética: fora < padrao)
      print(
          '[TfliteClassifier] logits=$logits  softmax=$probs  prob_padrao=${probPadrao.toStringAsFixed(3)}');

      return probPadrao;
    } catch (e) {
      print('[TfliteClassifier] Erro na inferência: $e');
      return -1.0;
    }
  }

  /// Classifica um [img.Image] diretamente sem decodificar JPEG.
  /// Ideal para frames de câmera pré-processados de forma ultra-rápida.
  double classifyImageDirectly(img.Image image) {
    final interpreter = _interpreter;
    if (interpreter == null) return -1.0;

    try {
      img.Image resized = image;
      if (image.width != 224 || image.height != 224) {
        resized = img.copyResize(image, width: 224, height: 224);
      }

      final input = _buildInputTensor(resized);
      final output = List.generate(1, (_) => List<double>.filled(2, 0.0));

      interpreter.run(input, output);

      final logits = output[0];
      final probs = _softmax(logits);
      final probPadrao = probs[1];

      print(
          '[TfliteClassifier] [Direct] logits=$logits  softmax=$probs  prob_padrao=${probPadrao.toStringAsFixed(3)}');
      return probPadrao;
    } catch (e) {
      print('[TfliteClassifier] Erro na inferência direta: $e');
      return -1.0;
    }
  }

  // ── Helpers internos ────────────────────────────────────────────────────────

  /// Converte [img.Image] 224×224 para tensor Float32 [1, 224, 224, 3] (NHWC)
  /// com normalização ImageNet (mean/std por canal RGB).
  List<List<List<List<double>>>> _buildInputTensor(img.Image image) {
    return [
      List.generate(
        224,
        (y) => List.generate(
          224,
          (x) => List.generate(3, (c) {
            final pixel = image.getPixel(x, y);
            final double val = c == 0
                ? pixel.r.toDouble()
                : c == 1
                    ? pixel.g.toDouble()
                    : pixel.b.toDouble();
            return (val / 255.0 - _mean[c]) / _std[c];
          }),
        ),
      )
    ];
  }

  /// Aplica softmax estável numericamente sobre uma lista de logits.
  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce(math.max);
    final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WORKER ISOLATE EM SEGUNDO PLANO
// ─────────────────────────────────────────────────────────────────────────────

/// Classe que encapsula o frame bruto da câmera para ser transferido
/// sem overhead entre Isolates.
class CameraFramePayload {
  final Uint8List bytes;
  final int width;
  final int height;
  final int bytesPerRow;
  final bool isYuv;

  /// Orientação do sensor físico da câmera em graus (0, 90, 180 ou 270).
  /// Em dispositivos Android, o sensor costuma ser montado a 90° (paisagem),
  /// então o frame bruto chega rotacionado 90° em relação ao modo retrato.
  /// Essa informação é usada pelo Isolate para corrigir a orientação da
  /// imagem antes de alimentar o modelo TFLite.
  final int sensorOrientation;

  CameraFramePayload({
    required this.bytes,
    required this.width,
    required this.height,
    required this.bytesPerRow,
    required this.isYuv,
    this.sensorOrientation = 0,
  });
}

/// Gerenciador do Isolate em segundo plano para o classificador ResNet50.
class TfliteIsolateWorker {
  static const String _readyMessage = 'ready';

  Isolate? _isolate;
  SendPort? _toIsolateSendPort;
  final ReceivePort _fromIsolateReceivePort = ReceivePort();

  bool _isReady = false;
  bool get isReady => _isReady;

  // Callback acionado quando uma probabilidade é retornada do Isolate
  void Function(double)? onResult;
  void Function()? onReady;

  TfliteIsolateWorker();

  /// Inicia o Isolate em segundo plano e carrega o modelo lá.
  Future<void> start(Uint8List modelBytes) async {
    if (_isolate != null) return;

    final rootToken = RootIsolateToken.instance;
    if (rootToken == null) {
      print(
          '[TfliteIsolateWorker] Aviso: RootIsolateToken.instance é nulo. O Messenger não será inicializado no Isolate.');
    }

    // Escuta respostas do Isolate secundário
    _fromIsolateReceivePort.listen((message) {
      if (message is SendPort) {
        // Primeiro handshake: o Isolate nos envia o SendPort dele
        _toIsolateSendPort = message;
        print('[TfliteIsolateWorker] Isolate de inferência conectado.');
      } else if (message == _readyMessage) {
        // O modelo já foi carregado dentro do Isolate.
        _isReady = true;
        print('[TfliteIsolateWorker] Modelo carregado no Isolate!');
        onReady?.call();
      } else if (message is double) {
        // Recebeu o resultado da inferência
        onResult?.call(message);
      }
    });

    // Spawna o isolate secundário
    _isolate = await Isolate.spawn(
      _isolateEntry,
      {
        'token': rootToken,
        'sendPort': _fromIsolateReceivePort.sendPort,
        'modelBytes': modelBytes,
      },
    );
  }

  /// Envia um frame para processamento em segundo plano no Isolate.
  /// Retorna false se o Isolate ainda não estiver pronto.
  bool processFrame(CameraFramePayload payload) {
    final sendPort = _toIsolateSendPort;
    if (sendPort == null || !_isReady) return false;
    sendPort.send(payload);
    return true;
  }

  /// Finaliza o Isolate de forma limpa.
  void stop() {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _toIsolateSendPort = null;
    _isReady = false;
    _fromIsolateReceivePort.close();
  }

  /// Ponto de entrada do Isolate em segundo plano (função estática top-level)
  static void _isolateEntry(Map<String, dynamic> initData) async {
    final RootIsolateToken? token = initData['token'];
    final SendPort mainSendPort = initData['sendPort'];
    final Uint8List modelBytes = initData['modelBytes'];

    try {
      // Permite que o Isolate acesse os assets/serviços binários do Flutter se o token estiver presente
      if (token != null) {
        try {
          BackgroundIsolateBinaryMessenger.ensureInitialized(token);
        } catch (e) {
          print(
              '[TfliteIsolateWorker] Erro ao inicializar Messenger no Isolate: $e');
        }
      }

      // Inicializa uma porta para receber comandos/frames da main thread
      final ReceivePort isolateReceivePort = ReceivePort();
      mainSendPort.send(isolateReceivePort.sendPort);

      // Instancia o classificador localmente no Isolate secundário
      final classifier = TfliteClassifier.instance;
      classifier.loadModelFromBuffer(modelBytes);
      if (!classifier.isReady) {
        mainSendPort.send(-2.0);
        return;
      }
      mainSendPort.send(_readyMessage);

      // Loop contínuo recebendo frames e processando
      await for (final message in isolateReceivePort) {
        if (message is CameraFramePayload) {
          try {
            // 1. Processamento e redimensionamento local (com correção de orientação)
            final img = _cameraImageTo224GrayscaleDirect(
              bytes: message.bytes,
              width: message.width,
              height: message.height,
              bytesPerRow: message.bytesPerRow,
              isYuv: message.isYuv,
              sensorOrientation: message.sensorOrientation,
            );

            // 2. Inferência TFLite na thread secundária
            final double prob = classifier.classifyImageDirectly(img);

            // Devolve o resultado para a thread principal
            mainSendPort.send(prob);
          } catch (e) {
            print(
                '[TfliteIsolateWorker] Erro no isolate secundário durante inferência: $e');
            mainSendPort.send(-1.0);
          }
        }
      }
    } catch (e) {
      print(
          '[TfliteIsolateWorker] Erro catastrófico de inicialização do Isolate: $e');
      mainSendPort.send(-2.0); // Sinaliza falha catastrófica de setup
    }
  }

  /// Conversão direta em baixo nível otimizada dentro da thread do Isolate.
  ///
  /// Além de redimensionar para 224×224, aplica a rotação indicada por
  /// [sensorOrientation] (0, 90, 180 ou 270 graus) para que a imagem
  /// entregue à rede neural esteja sempre na orientação correta,
  /// independente do dispositivo (tablet vs celular Android/iOS).
  static img.Image _cameraImageTo224GrayscaleDirect({
    required Uint8List bytes,
    required int width,
    required int height,
    required int bytesPerRow,
    required bool isYuv,
    int sensorOrientation = 0,
  }) {
    // ── Passo 1: determina dimensões lógicas do frame antes da rotação ──────
    // Para rotações de 90° e 270°, largura e altura do sensor são trocadas.
    final bool swapAxes = (sensorOrientation == 90 || sensorOrientation == 270);
    final int logicalW = swapAxes ? height : width;  // largura "como vemos na tela"
    final int logicalH = swapAxes ? width  : height; // altura  "como vemos na tela"

    final double scaleX = logicalW / 224.0;
    final double scaleY = logicalH / 224.0;

    final out = img.Image(width: 224, height: 224, numChannels: 3);

    for (int dy = 0; dy < 224; dy++) {
      for (int dx = 0; dx < 224; dx++) {
        // Coordenadas do pixel desejado no espaço "orientação correta"
        final int lx = (dx * scaleX).toInt().clamp(0, logicalW - 1);
        final int ly = (dy * scaleY).toInt().clamp(0, logicalH - 1);

        // Converte (lx, ly) → (sx, sy) no espaço bruto do sensor
        int sx, sy;
        switch (sensorOrientation) {
          case 90:
            // Sensor girado 90° CW: o topo da tela é o lado direito do sensor
            sx = ly;
            sy = logicalW - 1 - lx;
            break;
          case 180:
            sx = logicalW - 1 - lx;
            sy = logicalH - 1 - ly;
            break;
          case 270:
            // Sensor girado 270° CW (= 90° CCW)
            sx = logicalH - 1 - ly;
            sy = lx;
            break;
          default: // 0° — sem rotação
            sx = lx;
            sy = ly;
        }

        // ── Lê o valor de luma (cinza) do pixel sensor (sx, sy) ─────────────
        int gray;
        if (isYuv) {
          final int rowOffset = sy * bytesPerRow;
          gray = bytes[(rowOffset + sx).clamp(0, bytes.length - 1)];
        } else {
          // BGRA8888 (iOS) — sensor costuma ser 0° no iOS
          final int pixelIndex = sy * bytesPerRow + sx * 4;
          if (pixelIndex + 2 >= bytes.length) {
            gray = 0;
          } else {
            final b = bytes[pixelIndex];
            final g = bytes[pixelIndex + 1];
            final r = bytes[pixelIndex + 2];
            gray = ((r + g + b) / 3).toInt();
          }
        }
        out.setPixelRgb(dx, dy, gray, gray, gray);
      }
    }
    return out;
  }
}
