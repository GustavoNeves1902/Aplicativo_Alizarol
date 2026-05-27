import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'tflite_classifier.dart';

/// Throttle de inferência: processa no máximo 1 frame a cada N milissegundos.
const int _kFrameThrottleMs = 500;

/// Fração do menor lado usada pelo círculo guia.
const double _kGuideFraction = 0.78;

// ─────────────────────────────────────────────────────────────────────────────
// Widget
// ─────────────────────────────────────────────────────────────────────────────

/// Tela de câmera com auto-captura por inferência TFLite (ResNet50).
///
/// - A câmera analisa frames continuamente via [TfliteClassifier].
/// - Quando o modelo classifica o frame como "padrão" (prob ≥ 0.50), a foto
///   é tirada automaticamente.
/// - O botão manual de captura permanece disponível como fallback.
class CameraScreen extends StatefulWidget {
  final Function(String) onImageCaptured;
  const CameraScreen({super.key, required this.onImageCaptured});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

class _CameraScreenState extends State<CameraScreen>
    with SingleTickerProviderStateMixin {
  CameraController? _controller;
  bool _capturing = false;
  bool _initialized = false;
  bool _inferring =
      false; // true enquanto o classificador está rodando no Isolate
  bool _positionOk = false; // true após frame classificado como "padrão"
  double _confidence = 0.0; // probabilidade atual de "padrão" (0.0–1.0)

  /// true quando o modelo falha ao carregar (erro no Isolate ou timeout).
  bool _modelLoadFailed = false;

  /// Timer de timeout: se o modelo não carregar em 40s, marca como falha.
  Timer? _loadingTimeout;

  /// Orientação física do sensor da câmera (0, 90, 180 ou 270 graus).
  int _sensorOrientation = 0;

  int _framesSent = 0;
  int _framesReceived = 0;
  String _lastDebugStatus = 'Inicializando...';

  DateTime _lastFrameCheck = DateTime.now();
  late TfliteIsolateWorker _worker;

  // Animação de pulso no círculo guia (scanning)
  late AnimationController _pulseController;
  late Animation<double> _pulseAnim;

  Future<void> _loadModelAndStartWorker() async {
    if (!mounted) return;
    setState(() {
      _modelLoadFailed = false;
      _lastDebugStatus = 'Carregando bytes do modelo...';
    });

    // Timeout: se o modelo não responder em 40 segundos, exibe erro.
    _loadingTimeout?.cancel();
    _loadingTimeout = Timer(const Duration(seconds: 40), () {
      if (mounted && !_worker.isReady) {
        setState(() => _modelLoadFailed = true);
      }
    });

    try {
      final ByteData data = await DefaultAssetBundle.of(context).load(
        'assets/models/Alizarol_Leite_resnet50_S46_padrao_foradepadrao.tflite',
      );
      // IMPORTANTE: usar offsetInBytes + lengthInBytes para pegar apenas
      // os bytes do asset, evitando o erro "No subgraph in the model".
      final Uint8List modelBytes =
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      if (mounted) setState(() => _lastDebugStatus = 'Iniciando Isolate...');
      await _worker.start(modelBytes);
      if (mounted) setState(() => _lastDebugStatus = 'Carregando modelo no Isolate...');
    } catch (e) {
      print('[CameraScreen] Erro ao carregar modelo: $e');
      if (mounted) setState(() => _modelLoadFailed = true);
    }
  }

  /// Reinicia todo o worker do Isolate e tenta carregar o modelo novamente.
  void _retryModelLoad() {
    _worker.stop();
    _worker = TfliteIsolateWorker()
      ..onResult = _onWorkerResult
      ..onReady = _onWorkerReady;
    _loadModelAndStartWorker();
  }

  @override
  void initState() {
    super.initState();

    // Inicializa o worker do Isolate em segundo plano
    _worker = TfliteIsolateWorker()
      ..onResult = _onWorkerResult
      ..onReady = _onWorkerReady;
    _loadModelAndStartWorker();

    _initCamera();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _pulseAnim = Tween<double>(begin: 0.97, end: 1.03).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _loadingTimeout?.cancel();
    _worker.stop();
    _pulseController.dispose();
    _controller?.dispose();
    super.dispose();
  }

  // ── Câmera ─────────────────────────────────────────────────────────────────

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final camera = cameras.first;

    _controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;

      // Salva a orientação do sensor físico para corrigir os frames da IA.
      // Em Android, costuma ser 90°; em iOS, 0° ou 90°.
      _sensorOrientation = camera.sensorOrientation;
      print('[Camera] sensorOrientation=$_sensorOrientation');

      setState(() => _initialized = true);

      // Inicia stream contínuo de frames para inferência
      await _controller!.startImageStream(_onFrame);
    } catch (e) {
      print('[Camera] Erro ao inicializar: $e');
    }
  }

  // ── Processamento de frame (Isolate) ───────────────────────────────────────

  void _onWorkerReady() {
    if (!mounted) return;
    _loadingTimeout?.cancel();
    setState(() {
      _modelLoadFailed = false;
      _lastDebugStatus = 'Modelo pronto';
    });
  }

  /// Callback a cada frame da câmera. Aplica throttle e ignora frames se o Isolate
  /// estiver ocupado ou se a captura já tiver sido iniciada.
  void _onFrame(CameraImage frame) {
    if (_inferring || _positionOk || _capturing || !_worker.isReady) return;

    final now = DateTime.now();
    if (now.difference(_lastFrameCheck).inMilliseconds < _kFrameThrottleMs)
      return;
    _lastFrameCheck = now;

    _classifyFrame(frame);
  }

  /// Extrai os bytes brutos da câmera e envia de forma assíncrona ao Isolate secundário.
  void _classifyFrame(CameraImage frame) {
    if (!mounted) return;
    if (frame.planes.isEmpty) return;

    try {
      // Extrai os bytes do primeiro plano de forma instantânea (0.1ms)
      final yPlane = frame.planes[0];
      final bytes =
          Uint8List.fromList(yPlane.bytes); // Cópia segura para cruzar isolates

      final payload = CameraFramePayload(
        bytes: bytes,
        width: frame.width,
        height: frame.height,
        bytesPerRow: yPlane.bytesPerRow,
        isYuv: frame.planes.length > 1,
        // Repassa a orientação do sensor para o Isolate corrigir a imagem
        sensorOrientation: _sensorOrientation,
      );

      // Envia os dados para a thread de segundo plano
      final sent = _worker.processFrame(payload);
      if (sent) {
        setState(() {
          _inferring = true;
          _framesSent++;
          _lastDebugStatus = 'Frame $_framesSent enviado';
        });
      } else {
        setState(() {
          _lastDebugStatus = 'Worker ocupado ou não pronto';
        });
      }
    } catch (e) {
      print('[CameraScreen] Erro ao enviar frame para Isolate: $e');
      setState(() => _lastDebugStatus = 'Erro envio: $e');
    }
  }

  /// Callback acionado quando a predição é calculada na thread de segundo plano.
  void _onWorkerResult(double prob) async {
    if (!mounted || _capturing || _positionOk) return;

    if (prob < 0) {
      setState(() {
        _inferring = false;
        _lastDebugStatus = prob == -2.0
            ? 'Falha ao carregar modelo no Isolate'
            : 'Erro na inferência';
        // Marca falha visível para o usuário quando o Isolate reporta erro
        if (prob == -2.0) _modelLoadFailed = true;
      });
      return;
    }

    setState(() {
      _confidence = prob;
      _inferring = false;
      _framesReceived++;
      _lastDebugStatus =
          'Rec: prob=${prob.toStringAsFixed(3)} (#$_framesReceived)';
    });

    // Se o modelo diz "padrão" → captura automática
    if (prob >= TfliteClassifier.threshold && !_positionOk) {
      setState(() => _positionOk = true);
      print(
          '[AutoCapture] Posição correta pelo Isolate (prob=${prob.toStringAsFixed(3)}) — capturando...');

      // Pequeno delay para o usuário ver o feedback verde
      await Future.delayed(const Duration(milliseconds: 300));
      if (mounted && !_capturing) await _captureAuto();
    }
  }

  /// Captura automática: para o stream e tira a foto.
  Future<void> _captureAuto() async {
    if (_capturing || _controller == null || !_initialized) return;
    setState(() => _capturing = true);
    try {
      await _controller!.stopImageStream();
      final XFile file = await _controller!.takePicture();
      if (mounted) Navigator.pop(context, file.path);
    } catch (e) {
      print('[AutoCapture] Erro na captura automática: $e');
      if (mounted)
        setState(() {
          _capturing = false;
          _positionOk = false;
        });
    }
  }

  // ── Captura Manual (botão) ─────────────────────────────────────────────────

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
      print('[Camera] Erro ao capturar: $e');
      if (mounted) setState(() => _capturing = false);
    }
  }

  // ── Helpers de estado visual ───────────────────────────────────────────────

  bool get _hasModelReading => _framesReceived > 0 && _confidence >= 0;

  int get _confidencePercent => (_confidence.clamp(0.0, 1.0) * 100).round();

  Color get _guideColor {
    if (_positionOk || _capturing) return Colors.greenAccent;
    if (!_worker.isReady || !_hasModelReading) return Colors.white;
    if (_confidence >= TfliteClassifier.threshold) return Colors.greenAccent;
    if (_confidence >= 0.35) return Colors.amber;
    if (_confidence >= 0.18) return Colors.deepOrangeAccent;
    return Colors.redAccent;
  }

  String get _statusLabel {
    if (_capturing) return 'Capturando...';
    if (_positionOk) return 'Posição correta!';
    if (_modelLoadFailed) return 'Falha ao carregar IA';
    if (!_worker.isReady) return 'Carregando modelo...';
    if (!_hasModelReading) return 'Analisando posicionamento...';
    if (_confidence >= TfliteClassifier.threshold) return 'Posição correta!';
    if (_confidence >= 0.35) return 'Quase lá - $_confidencePercent%';
    if (_confidence >= 0.18) return 'Ajuste a posição - $_confidencePercent%';
    return 'Fora da posição - $_confidencePercent%';
  }

  String get _positionHint {
    if (_capturing) return 'Mantendo captura';
    if (_positionOk) return 'Captura automática iniciada';
    if (!_worker.isReady) return 'Preparando IA';
    if (!_hasModelReading) return 'Aguardando leitura';
    if (_confidence >= TfliteClassifier.threshold) return 'PADRAO detectado';
    if (_confidence >= 0.35) return 'Mantenha e refine';
    if (_confidence >= 0.18) return 'Centralize e aproxime';
    return 'Reposicione a amostra';
  }

  Color get _statusBg {
    if (_positionOk || _capturing) return Colors.green.withOpacity(0.85);
    if (_modelLoadFailed) return Colors.red.shade900.withOpacity(0.88);
    if (!_worker.isReady || !_hasModelReading) return Colors.black54;
    if (_confidence >= 0.35) return Colors.amber.shade900.withOpacity(0.88);
    if (_confidence >= 0.18) return Colors.deepOrange.withOpacity(0.88);
    return Colors.red.shade900.withOpacity(0.88);
  }

  Color get _progressColor {
    if (_positionOk || _capturing) return Colors.greenAccent;
    if (!_worker.isReady || !_hasModelReading) return Colors.white70;
    if (_confidence >= TfliteClassifier.threshold) return Colors.greenAccent;
    if (_confidence >= 0.35) return Colors.amber;
    if (_confidence >= 0.18) return Colors.deepOrangeAccent;
    return Colors.redAccent;
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
              // ── Preview da câmera (preenche a tela, mantendo proporção) ───
              // FittedBox com BoxFit.cover faz o preview cobrir toda a tela sem
              // distorção — comportamento igual ao da câmera nativa / Instagram.
              // previewSize reporta em orientação landscape, por isso w/h são trocados.
              SizedBox.expand(
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _controller!.value.previewSize?.height ?? w,
                    height: _controller!.value.previewSize?.width ?? h,
                    child: CameraPreview(_controller!),
                  ),
                ),
              ),

              // ── Overlay escuro com "buraco" circular ─────────────────────
              CustomPaint(
                painter: _DarkMaskPainter(
                  guideRadius: guideSize / 2,
                  center: Offset(w / 2, h / 2),
                ),
              ),

              // ── Borda animada do círculo guia ─────────────────────────────
              Center(
                child: AnimatedBuilder(
                  animation: _pulseAnim,
                  builder: (_, __) => Transform.scale(
                    scale: (_capturing || _positionOk) ? 1.0 : _pulseAnim.value,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      width: guideSize,
                      height: guideSize,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: _guideColor,
                          width: _positionOk ? 5 : 3,
                        ),
                        boxShadow: _positionOk
                            ? [
                                BoxShadow(
                                  color: Colors.greenAccent.withOpacity(0.45),
                                  blurRadius: 24,
                                  spreadRadius: 8,
                                ),
                              ]
                            : _capturing
                                ? []
                                : [
                                    BoxShadow(
                                      color: _guideColor.withOpacity(0.34),
                                      blurRadius: 18,
                                      spreadRadius: 5,
                                    ),
                                  ],
                      ),
                    ),
                  ),
                ),
              ),

              // ── Barra de confiança do modelo ──────────────────────────────
              Positioned(
                bottom: h / 2 - guideSize / 2 - 90,
                left: w * 0.1,
                right: w * 0.1,
                child: Column(
                  children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.58),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: _progressColor.withOpacity(0.65),
                          width: 1,
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            _positionOk
                                ? Icons.check_circle_rounded
                                : Icons.adjust_rounded,
                            color: _progressColor,
                            size: 18,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            _hasModelReading
                                ? 'PADRAO $_confidencePercent%'
                                : _positionHint,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 15,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 8),
                    // Barra de progresso animada
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: SizedBox(
                        height: 8,
                        width: double.infinity,
                        child: TweenAnimationBuilder<double>(
                          tween: Tween<double>(
                            begin: 0,
                            end: _confidence.clamp(0.0, 1.0),
                          ),
                          duration: const Duration(milliseconds: 450),
                          curve: Curves.easeOutCubic,
                          builder: (_, value, __) {
                            final bool waitingForFirstReading =
                                _worker.isReady && !_hasModelReading;
                            return LinearProgressIndicator(
                              value: waitingForFirstReading ? null : value,
                              backgroundColor: Colors.white24,
                              valueColor:
                                  AlwaysStoppedAnimation<Color>(_progressColor),
                            );
                          },
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // ── Label de status (com animação de troca) ───────────────────
              Positioned(
                bottom: h / 2 - guideSize / 2 - 48,
                left: 0,
                right: 0,
                child: Center(
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    child: Container(
                      key: ValueKey(_statusLabel),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 6),
                      decoration: BoxDecoration(
                        color: _statusBg,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        _statusLabel,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ),
                ),
              ),

              // ── Botão "Tentar novamente" (só aparece em caso de falha) ────
              if (_modelLoadFailed)
                Positioned(
                  bottom: h / 2 - guideSize / 2 - 10,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: TextButton.icon(
                      onPressed: _retryModelLoad,
                      icon: const Icon(Icons.refresh, color: Colors.white, size: 16),
                      label: const Text(
                        'Tentar novamente',
                        style: TextStyle(color: Colors.white, fontSize: 13),
                      ),
                      style: TextButton.styleFrom(
                        backgroundColor: Colors.white24,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 18, vertical: 8),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                      ),
                    ),
                  ),
                ),


              // ── Badge "AI" — indica se o classificador está rodando ───────
              Positioned(
                top: MediaQuery.of(context).padding.top + 12,
                right: 12,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _inferring
                        ? Colors.amber.withOpacity(0.9)
                        : Colors.black54,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: _inferring ? Colors.amber : Colors.white30,
                      width: 1,
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        _inferring ? Icons.memory : Icons.auto_awesome,
                        color: Colors.white,
                        size: 12,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        _inferring ? 'IA...' : 'IA',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1,
                        ),
                      ),
                    ],
                  ),
                ),
              ),

              // ── Botão de voltar ───────────────────────────────────────────
              Positioned(
                top: MediaQuery.of(context).padding.top + 12,
                left: 12,
                child: CircleAvatar(
                  backgroundColor: Colors.black54,
                  child: IconButton(
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                    onPressed: () => Navigator.pop(context),
                  ),
                ),
              ),

              // Debug console removido (era usado apenas durante desenvolvimento)

              // ── Botão de captura manual ───────────────────────────────────
              Positioned(
                bottom: 40,
                left: 0,
                right: 0,
                child: Center(
                  child: GestureDetector(
                    onTap: _capturing ? null : _capture,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 150),
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _capturing ? Colors.grey.shade700 : Colors.white,
                        border: Border.all(color: Colors.white70, width: 4),
                        boxShadow: _capturing
                            ? []
                            : [
                                BoxShadow(
                                  color: Colors.white.withOpacity(0.4),
                                  blurRadius: 12,
                                  spreadRadius: 2,
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
                              color: Colors.black87, size: 36),
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
// CustomPainter — máscara escura com buraco circular
// ─────────────────────────────────────────────────────────────────────────────

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
