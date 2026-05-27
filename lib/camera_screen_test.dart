import 'dart:async';
import 'dart:math';
import 'dart:isolate';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// Fração do menor lado usada pelo círculo guia.
const double _kGuideFraction = 0.78;

/// Tela de câmera experimental para testar a classificação contínua de frames
/// rodando de forma 100% fluida em um Isolate de Segundo Plano (Background Thread).
class CameraScreenTest extends StatefulWidget {
  final Function(String) onImageCaptured;
  const CameraScreenTest({Key? key, required this.onImageCaptured})
      : super(key: key);

  @override
  State<CameraScreenTest> createState() => _CameraScreenTestState();
}

class _CameraScreenTestState extends State<CameraScreenTest>
    with SingleTickerProviderStateMixin {
  CameraController? _controller;
  bool _initialized = false;
  bool _capturing = false;

  // Isolate e Comunicação
  Isolate? _backgroundIsolate;
  SendPort? _toIsolateSendPort;
  ReceivePort? _fromIsolateReceivePort;

  bool _modelLoaded = false;
  bool _isProcessingBackground = false;

  // Opções de controle do usuário
  bool _autoCaptureEnabled = false;
  int _throttleMs = 300; // Intervalo padrão entre inferências (em ms)

  // Métricas de telemetria
  int _preprocessTimeMs = 0;
  int _inferenceTimeMs = 0;
  double _fps = 0.0;
  String _predictionResult = 'Sem predição';
  double _confidence = 0.0;

  DateTime? _lastInferenceTime;
  DateTime _lastFrameTime = DateTime.now();

  // Animação de pulso no círculo guia (scanning)
  late AnimationController _pulseController;
  late Animation<double> _pulseAnim;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _pulseAnim = Tween<double>(begin: 0.97, end: 1.03).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _initTest();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _controller?.dispose();
    _backgroundIsolate?.kill(priority: Isolate.beforeNextEvent);
    _fromIsolateReceivePort?.close();
    super.dispose();
  }

  Future<void> _initTest() async {
    await _spawnBackgroundIsolate();
    await _initCamera();
  }

  // ── Inicialização do Isolate de Segundo Plano (Background Thread) ──────────
  Future<void> _spawnBackgroundIsolate() async {
    _fromIsolateReceivePort = ReceivePort();

    // Escuta mensagens vindas da Thread de Segundo Plano
    _fromIsolateReceivePort!.listen((message) {
      if (message is SendPort) {
        // Recebe o SendPort do Isolate para iniciarmos a comunicação
        _toIsolateSendPort = message;
        setState(() {
          _modelLoaded = true;
        });
        print('[CameraTest] Isolate carregado e pronto para comunicação!');
      } else if (message is Map<String, dynamic>) {
        // Recebe os resultados processados em segundo plano
        _handleIsolateResult(message);
      }
    });

    // Passamos o token do Isolate principal para o interpretador TFLite carregar assets no Isolate
    final RootIsolateToken token = RootIsolateToken.instance!;

    _backgroundIsolate = await Isolate.spawn(
      _isolateEntryPoint,
      {
        'sendPort': _fromIsolateReceivePort!.sendPort,
        'token': token,
      },
    );
  }

  // ── Ponto de Entrada da Thread Secundária (Background Isolate) ─────────────
  static void _isolateEntryPoint(dynamic args) async {
    final SendPort mainSendPort = args['sendPort'] as SendPort;
    final RootIsolateToken token = args['token'] as RootIsolateToken;

    // Registra o token no Isolate secundário para habilitar canais nativos (Assets, TFLite)
    BackgroundIsolateBinaryMessenger.ensureInitialized(token);

    final isolateReceivePort = ReceivePort();
    // Devolve o canal de escuta para a Thread Principal
    mainSendPort.send(isolateReceivePort.sendPort);

    // Carrega o modelo TFLite de forma estática (só uma vez em toda a vida do Isolate)
    Interpreter? interpreter;
    try {
      interpreter = await Interpreter.fromAsset('assets/models/model.tflite');
    } catch (e) {
      print('[BackgroundIsolate] Erro crítico ao carregar modelo TFLite: $e');
      return;
    }

    // Loop de recepção e execução contínua
    await for (final message in isolateReceivePort) {
      if (message is Map<String, dynamic>) {
        final planeBytes = message['planeBytes'] as List<Uint8List>;
        final bytesPerRow = message['bytesPerRow'] as List<int>;
        final formatGroupStr = message['formatGroupStr'] as String;
        final width = message['width'] as int;
        final height = message['height'] as int;

        final bool isYUV = formatGroupStr.contains('yuv420');

        final preprocessStart = DateTime.now();

        // 1. Amostragem ultra rápida Nearest-Neighbor direto na thread de segundo plano
        final img.Image resizedImage = _cameraImageTo224ImageInIsolate(
          planeBytes: planeBytes,
          bytesPerRow: bytesPerRow,
          isYUV: isYUV,
          width: width,
          height: height,
        );

        // 2. Normalização dos valores float32
        var input = List.generate(
            1,
            (_) => List.generate(224,
                (_) => List.generate(224, (_) => List.generate(3, (_) => 0.0))));

        for (int y = 0; y < 224; y++) {
          for (int x = 0; x < 224; x++) {
            final pixel = resizedImage.getPixel(x, y);
            input[0][y][x][0] = pixel.r / 255.0;
            input[0][y][x][1] = pixel.g / 255.0;
            input[0][y][x][2] = pixel.b / 255.0;
          }
        }

        final preprocessTimeMs =
            DateTime.now().difference(preprocessStart).inMilliseconds;

        // 3. Inferência síncrona dentro do Isolate (Thread Principal fica totalmente livre!)
        final inferenceStart = DateTime.now();
        var output = List.filled(1, List.filled(2, 0.0));
        interpreter.run(input, output);
        final inferenceTimeMs =
            DateTime.now().difference(inferenceStart).inMilliseconds;

        // 4. Softmax e formatação de saídas
        List<double> rawResults = List<double>.from(output[0]);
        double maxLogit = rawResults.reduce((a, b) => a > b ? a : b);
        List<double> exps = rawResults.map((l) => exp(l - maxLogit)).toList();
        double sumExps = exps.reduce((a, b) => a + b);
        List<double> probabilities = exps.map((e) => e / sumExps).toList();

        double probAprovado = probabilities[0];
        double probReprovado = probabilities[1];

        final String resultadoFinal =
            probAprovado > probReprovado ? "APROVADO" : "REPROVADO";
        final double confidenceVal =
            (probAprovado > probReprovado ? probAprovado : probReprovado) * 100;

        // Devolve os resultados e métricas para atualizar a tela principal
        mainSendPort.send({
          'resultado': resultadoFinal,
          'confianca': confidenceVal,
          'preprocessTime': preprocessTimeMs,
          'inferenceTime': inferenceTimeMs,
        });
      }
    }
  }

  // ── Conversor Ultra Rápido em Segundo Plano ───────────────────────────────
  static img.Image _cameraImageTo224ImageInIsolate({
    required List<Uint8List> planeBytes,
    required List<int> bytesPerRow,
    required bool isYUV,
    required int width,
    required int height,
  }) {
    final image = img.Image(width: 224, height: 224, numChannels: 3);
    final double scaleX = width / 224.0;
    final double scaleY = height / 224.0;

    if (isYUV) {
      final yBytes = planeBytes[0];
      final rowBytes = bytesPerRow[0];

      for (int y = 0; y < 224; y++) {
        final srcY = (y * scaleY).toInt();
        final rowOffset = srcY * rowBytes;
        for (int x = 0; x < 224; x++) {
          final srcX = (x * scaleX).toInt();
          final luma = yBytes[rowOffset + srcX];
          image.setPixelRgb(x, y, luma, luma, luma);
        }
      }
    } else {
      // BGRA8888 (iOS)
      final bytes = planeBytes[0];
      final rowBytes = bytesPerRow[0];

      for (int y = 0; y < 224; y++) {
        final srcY = (y * scaleY).toInt();
        final rowOffset = srcY * rowBytes;
        for (int x = 0; x < 224; x++) {
          final srcX = (x * scaleX).toInt();
          final pixelOffset = rowOffset + srcX * 4;

          final b = bytes[pixelOffset];
          final g = bytes[pixelOffset + 1];
          final r = bytes[pixelOffset + 2];

          image.setPixelRgb(x, y, r, g, b);
        }
      }
    }
    return image;
  }

  // ── Inicialização da Câmera ───────────────────────────────────────────────
  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.medium, // ~720p - ideal para stream móvel
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _initialized = true);

      // Inicia stream de frames da câmera
      await _controller!.startImageStream(_onFrame);
    } catch (e) {
      print('[CameraTest] Erro ao inicializar câmera: $e');
    }
  }

  // ── Recebimento do Stream de Frames da Câmera ─────────────────────────────
  void _onFrame(CameraImage frame) {
    if (!_modelLoaded ||
        _toIsolateSendPort == null ||
        _isProcessingBackground ||
        _capturing) return;

    final now = DateTime.now();
    if (_throttleMs > 0 &&
        now.difference(_lastFrameTime).inMilliseconds < _throttleMs) {
      return;
    }
    _lastFrameTime = now;

    if (mounted) {
      setState(() {
        _isProcessingBackground = true;
      });
    }

    // Mapeia e envia os bytes e dimensões brutos da imagem de forma ultra leve (< 0.1ms)
    _toIsolateSendPort!.send({
      'planeBytes': frame.planes.map((p) => p.bytes).toList(),
      'bytesPerRow': frame.planes.map((p) => p.bytesPerRow).toList(),
      'formatGroupStr': frame.format.group.toString(),
      'width': frame.width,
      'height': frame.height,
    });
  }

  // ── Recepção dos Resultados Processados em Segundo Plano ──────────────────
  void _handleIsolateResult(Map<String, dynamic> result) {
    final String resultadoFinal = result['resultado'] as String;
    final double confidenceVal = result['confianca'] as double;
    final int preprocessTime = result['preprocessTime'] as int;
    final int inferenceTime = result['inferenceTime'] as int;

    // Medição real de FPS
    if (_lastInferenceTime != null) {
      final diff =
          DateTime.now().difference(_lastInferenceTime!).inMilliseconds;
      if (diff > 0) {
        _fps = 1000 / diff;
      }
    }
    _lastInferenceTime = DateTime.now();

    if (mounted) {
      setState(() {
        _predictionResult = resultadoFinal;
        _confidence = confidenceVal;
        _preprocessTimeMs = preprocessTime;
        _inferenceTimeMs = inferenceTime;
        _isProcessingBackground = false;
      });
    }

    // Auto-captura automática se a predição for classificada como APROVADO
    if (_autoCaptureEnabled && resultadoFinal == 'APROVADO' && !_capturing) {
      print('[CameraTest] Disparando captura automática via Isolate secundário...');
      _captureAuto();
    }
  }

  // ── Capturas ──────────────────────────────────────────────────────────────
  Future<void> _captureAuto() async {
    if (_capturing || _controller == null || !_initialized) return;
    setState(() => _capturing = true);
    try {
      await _controller!.stopImageStream();
      final XFile file = await _controller!.takePicture();
      if (mounted) Navigator.pop(context, file.path);
    } catch (e) {
      print('[CameraTest] Erro na captura automática: $e');
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<void> _capture() async {
    if (_capturing || _controller == null || !_initialized) return;
    setState(() => _capturing = true);
    try {
      try {
        await _controller!.stopImageStream();
      } catch (_) {}
      final XFile file = await _controller!.takePicture();
      if (mounted) Navigator.pop(context, file.path);
    } catch (e) {
      print('[CameraTest] Erro ao capturar manual: $e');
      if (mounted) setState(() => _capturing = false);
    }
  }

  // ── UI ─────────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    if (!_initialized || _controller == null) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Colors.white)),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: LayoutBuilder(
        builder: (context, constraints) {
          final double w = constraints.maxWidth;
          final double h = constraints.maxHeight;
          final double guideSize =
              [w, h].reduce((a, b) => a < b ? a : b) * _kGuideFraction;

          return Stack(
            fit: StackFit.expand,
            children: [
              // ── Preview da câmera ──────────────────────────────────────────
              CameraPreview(_controller!),

              // ── Overlay escuro com buraco circular guia ────────────────────
              CustomPaint(
                painter: _DarkMaskPainter(
                  guideRadius: guideSize / 2,
                  center: Offset(w / 2, h / 2),
                ),
              ),

              // ── Borda animada do círculo guia ──────────────────────────────
              Center(
                child: AnimatedBuilder(
                  animation: _pulseAnim,
                  builder: (_, __) => Transform.scale(
                    scale: _capturing ? 1.0 : _pulseAnim.value,
                    child: Container(
                      width: guideSize,
                      height: guideSize,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: _predictionResult == 'APROVADO'
                              ? Colors.greenAccent
                              : Colors.redAccent,
                          width: 3.5,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: (_predictionResult == 'APROVADO'
                                    ? Colors.greenAccent
                                    : Colors.redAccent)
                                .withOpacity(0.35),
                            blurRadius: 16,
                            spreadRadius: 4,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              // ── Painel Superior de Telemetria e Controles (Glassmorphism) ──
              Positioned(
                top: MediaQuery.of(context).padding.top + 10,
                left: 12,
                right: 12,
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    color: Colors.black.withOpacity(0.72),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Row(
                              children: [
                                Icon(Icons.speed,
                                    color: Colors.greenAccent, size: 18),
                                SizedBox(width: 6),
                                Text(
                                  'CLASSIFICAÇÃO PARALELA (ISOLATE ATIVO)',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 11,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ],
                            ),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 8, vertical: 2),
                              decoration: BoxDecoration(
                                color: _modelLoaded
                                    ? Colors.green.withOpacity(0.2)
                                    : Colors.orange.withOpacity(0.2),
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                    color: _modelLoaded
                                        ? Colors.green
                                        : Colors.orange,
                                    width: 0.5),
                              ),
                              child: Text(
                                _modelLoaded ? 'ISOLATE ON' : 'SPAWNING...',
                                style: TextStyle(
                                  color: _modelLoaded
                                      ? Colors.green
                                      : Colors.orange,
                                  fontSize: 8,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            )
                          ],
                        ),
                        const Divider(color: Colors.white12, height: 16),

                        // Telemetrias de tempo (Segundo Plano)
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            _TelemetryItem(
                              label: 'PRÉ-PROC (ISO)',
                              value: '${_preprocessTimeMs}ms',
                              icon: Icons.compress,
                            ),
                            _TelemetryItem(
                              label: 'INFERÊNCIA (ISO)',
                              value: '${_inferenceTimeMs}ms',
                              icon: Icons.psychology,
                            ),
                            _TelemetryItem(
                              label: 'FPS REAL',
                              value: _fps.toStringAsFixed(1),
                              icon: Icons.refresh,
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),

                        // Resultado da predição
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.06),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    'RESULTADO DA INFERÊNCIA PARALELA',
                                    style: TextStyle(
                                      color: Colors.white38,
                                      fontSize: 8,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    '$_predictionResult (${_confidence.toStringAsFixed(1)}%)',
                                    style: TextStyle(
                                      color: _predictionResult == 'APROVADO'
                                          ? Colors.greenAccent
                                          : Colors.redAccent,
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                              if (_isProcessingBackground)
                                const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white70,
                                  ),
                                ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 12),

                        // Controles deslizantes e Interruptores
                        Row(
                          children: [
                            Expanded(
                              child: Row(
                                children: [
                                  const Text('Auto-Captura:',
                                      style: TextStyle(
                                          color: Colors.white70, fontSize: 11)),
                                  const SizedBox(width: 4),
                                  Transform.scale(
                                    scale: 0.8,
                                    child: Switch(
                                      value: _autoCaptureEnabled,
                                      activeColor: Colors.greenAccent,
                                      onChanged: (val) {
                                        setState(() {
                                          _autoCaptureEnabled = val;
                                        });
                                      },
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Expanded(
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.end,
                                children: [
                                  const Text('Throttle:',
                                      style: TextStyle(
                                          color: Colors.white70, fontSize: 11)),
                                  const SizedBox(width: 6),
                                  DropdownButton<int>(
                                    dropdownColor: Colors.black87,
                                    value: _throttleMs,
                                    items: const [
                                      DropdownMenuItem(
                                          value: 0,
                                          child: Text('Raw (Stress)',
                                              style: TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 11))),
                                      DropdownMenuItem(
                                          value: 150,
                                          child: Text('150ms',
                                              style: TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 11))),
                                      DropdownMenuItem(
                                          value: 300,
                                          child: Text('300ms',
                                              style: TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 11))),
                                      DropdownMenuItem(
                                          value: 600,
                                          child: Text('600ms',
                                              style: TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 11))),
                                    ],
                                    onChanged: (val) {
                                      if (val != null) {
                                        setState(() {
                                          _throttleMs = val;
                                        });
                                      }
                                    },
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),

              // ── Botão de Voltar ────────────────────────────────────────────
              Positioned(
                bottom: 40,
                left: 24,
                child: CircleAvatar(
                  radius: 25,
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    icon: const Icon(Icons.arrow_back,
                        color: Colors.white, size: 24),
                    onPressed: () => Navigator.pop(context),
                  ),
                ),
              ),

              // ── Botão de Captura Manual ────────────────────────────────────
              Positioned(
                bottom: 30,
                left: 0,
                right: 0,
                child: Center(
                  child: GestureDetector(
                    onTap: _capturing ? null : _capture,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 150),
                      width: 76,
                      height: 76,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _capturing ? Colors.grey.shade700 : Colors.white,
                        border: Border.all(color: Colors.white70, width: 4),
                        boxShadow: _capturing
                            ? []
                            : [
                                BoxShadow(
                                  color: Colors.white.withOpacity(0.3),
                                  blurRadius: 10,
                                  spreadRadius: 1,
                                ),
                              ],
                      ),
                      child: _capturing
                          ? const Padding(
                              padding: EdgeInsets.all(18),
                              child: CircularProgressIndicator(
                                  color: Colors.white, strokeWidth: 3),
                            )
                          : const Icon(Icons.camera_alt,
                              color: Colors.black87, size: 32),
                    ),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers de UI auxiliares
// ─────────────────────────────────────────────────────────────────────────────

class _TelemetryItem extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _TelemetryItem({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white30, size: 10),
            const SizedBox(width: 4),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white30,
                fontSize: 8,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}

class _DarkMaskPainter extends CustomPainter {
  final double guideRadius;
  final Offset center;

  const _DarkMaskPainter({required this.guideRadius, required this.center});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.black.withOpacity(0.55);

    final fullRect = Rect.fromLTWH(0, 0, size.width, size.height);
    final path = Path()
      ..addRect(fullRect)
      ..addOval(Rect.fromCircle(center: center, radius: guideRadius))
      ..fillType = PathFillType.evenOdd;

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _DarkMaskPainter old) =>
      old.guideRadius != guideRadius || old.center != center;
}
