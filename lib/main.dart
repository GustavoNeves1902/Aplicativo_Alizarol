import 'dart:io';
import 'dart:math';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show compute;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'splash_screen.dart';
import 'camera_screen.dart';
import 'detector.dart';

class _PreparedAnalysisImage {
  final Uint8List displayBytes;
  final Uint8List? croppedBytes;
  final bool circleFound;
  final bool hasPink;

  const _PreparedAnalysisImage({
    required this.displayBytes,
    required this.croppedBytes,
    required this.circleFound,
    required this.hasPink,
  });
}

/// Prepara imagens da Câmera: faz um recorte matemático exato da área do círculo guia,
/// garantindo que a foto no dashboard seja apenas a placa, sem usar o OpenCV.
_PreparedAnalysisImage _prepareCameraImage(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw Exception('Não foi possível decodificar a imagem.');
  }

  // Na UI da câmera, o círculo tem diâmetro de 78% do menor lado da tela.
  // Vamos aplicar a mesma proporção para recortar a foto original.
  final minDim = min(decoded.width, decoded.height);
  final cropSize = (minDim * 0.78).toInt();
  
  final x = (decoded.width - cropSize) ~/ 2;
  final y = (decoded.height - cropSize) ~/ 2;
  
  // Adiciona um leve padding ao redor do recorte para segurança (ex: 5%)
  final padding = (minDim * 0.05).toInt();
  final xMin = max(0, x - padding);
  final yMin = max(0, y - padding);
  final xMax = min(decoded.width, x + cropSize + padding);
  final yMax = min(decoded.height, y + cropSize + padding);
  
  final croppedImage = img.copyCrop(
    decoded,
    x: xMin,
    y: yMin,
    width: xMax - xMin,
    height: yMax - yMin,
  );

  final displayImg = croppedImage.width > 800
      ? img.copyResize(croppedImage, width: 800)
      : croppedImage;

  return _PreparedAnalysisImage(
    displayBytes: Uint8List.fromList(img.encodeJpg(displayImg, quality: 90)),
    croppedBytes: Uint8List.fromList(img.encodeJpg(croppedImage, quality: 95)),
    circleFound: true, // Força true, pois o usuário centralizou pelo guia
    hasPink: true,     // Ignorado na câmera
  );
}

_PreparedAnalysisImage _prepareAnalysisImage(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw Exception('Não foi possível decodificar a imagem.');
  }

  final detection = detectPetriAndCrop(decoded);
  final croppedImage = detection.image;
  final displayImg = croppedImage.width > 800
      ? img.copyResize(croppedImage, width: 800)
      : croppedImage;

  return _PreparedAnalysisImage(
    displayBytes: Uint8List.fromList(img.encodeJpg(displayImg, quality: 90)),
    croppedBytes: detection.circleFound
        ? Uint8List.fromList(img.encodeJpg(croppedImage, quality: 95))
        : null,
    circleFound: detection.circleFound,
    hasPink: detection.circleFound && hasAlizarolPink(croppedImage),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Alizarol',
      theme: ThemeData.dark().copyWith(
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFFE8735A),
          secondary: Color(0xFFFFAB91),
        ),
        scaffoldBackgroundColor: const Color(0xFF12080A),
      ),
      home: const SplashScreen(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Home Screen — Dashboard + FAB
// ─────────────────────────────────────────────────────────────────────────────

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // ── Estado ─────────────────────────────────────────────────────────────────
  final List<Map<String, dynamic>> _results = [];
  Interpreter? _interpreter;
  bool _isProcessing = false;

  // ImageNet normalization (mesmos valores usados no treinamento PyTorch)
  static const List<double> _mean = [0.485, 0.456, 0.406];
  static const List<double> _std = [0.229, 0.224, 0.225];

  @override
  void initState() {
    super.initState();
    _loadModel();
    _loadSavedResults();
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  // ── Modelo ConvNeXt Large ──────────────────────────────────────────────────

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/Alizarol_Leite_convnext_large_S42_congelando10epocas.tflite',
      );
      // Necessário para modelos grandes como ConvNeXt: garante que todos os
      // tensores intermediários sejam alocados antes da primeira inferência.
      _interpreter!.allocateTensors();
      print('[ConvNeXt] Input:  ${_interpreter!.getInputTensor(0).shape}');
      print('[ConvNeXt] Output: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      print('[ConvNeXt] Erro ao carregar modelo: $e');
    }
  }

  // ── Persistência ───────────────────────────────────────────────────────────

  Future<void> _loadSavedResults() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString('results_v2');
    if (raw == null) return;
    try {
      final list = List<Map<String, dynamic>>.from(
        (jsonDecode(raw) as List).map((e) => Map<String, dynamic>.from(e)),
      );
      if (mounted) setState(() => _results.addAll(list));
    } catch (_) {}
  }

  Future<void> _saveResults() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      'results_v2',
      jsonEncode(_results
          .map((e) => {
                'path': e['path'],
                'resultado': e['resultado'],
                'timestamp': e['timestamp'],
              })
          .toList()),
    );
  }

  // ── Câmera: ResNet50 valida posicionamento e auto-captura ──────────────────

  Future<void> _openCamera() async {
    if (_isProcessing) return;
    final String? imagePath = await Navigator.push<String>(
      context,
      MaterialPageRoute(
        builder: (_) => CameraScreen(onImageCaptured: (path) => path),
      ),
    );
    if (imagePath != null && mounted) {
      await _processImage(File(imagePath), isFromCamera: true);
    }
  }

  // ── Galeria: direto para ConvNeXt Large ────────────────────────────────────

  Future<void> _pickFromGallery() async {
    if (_isProcessing) return;
    final picked = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (picked == null || !mounted) return;
    await _processImage(File(picked.path), isFromCamera: false);
  }

  // ── Pipeline de classificação ──────────────────────────────────────────────
  //
  // IMPORTANTE: NÃO converter o canal aqui — o _imgToMat dentro do detector
  // já lida com a conversão via encodeJpg/imdecode internamente.
  // Converter antes quebraria a detecção de cor HSV.

  Future<void> _processImage(File imageFile, {bool isFromCamera = false}) async {
    if (!mounted) return;
    setState(() => _isProcessing = true);

    try {
      // Dá uma chance para o overlay ser desenhado antes do trabalho pesado.
      await Future<void>.delayed(const Duration(milliseconds: 80));

      final bytes = await imageFile.readAsBytes();
      
      final prepared = await compute(
        isFromCamera ? _prepareCameraImage : _prepareAnalysisImage,
        bytes,
      );

      // 2. Salva imagem recortada para exibição no dashboard
      final tempDir = Directory.systemTemp;
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final displayFile = File('${tempDir.path}/display_$timestamp.jpg');
      await displayFile.writeAsBytes(prepared.displayBytes);

      // 3. Decide resultado
      String resultado;
      if (isFromCamera) {
        // Câmera: ignora a verificação de cor rosa e de detecção estrita.
        // Tenta usar o recorte se achou o prato, senão usa a imagem original redimensionada.
        final bytesToDecode = prepared.croppedBytes ?? prepared.displayBytes;
        final imageToClassify = img.decodeImage(bytesToDecode);
        if (imageToClassify == null) {
          throw Exception('Não foi possível preparar a imagem da câmera.');
        }
        await Future<void>.delayed(const Duration(milliseconds: 16));
        resultado = await _classifyWithConvNeXt(imageToClassify);
      } else {
        // Galeria: exige a detecção do prato E a cor rosa do Alizarol
        if (!prepared.circleFound || prepared.croppedBytes == null) {
          resultado = 'ALIZAROL NÃO ENCONTRADO';
        } else if (!prepared.hasPink) {
          print('[Main] Tom rosa não detectado na imagem da galeria.');
          resultado = 'ALIZAROL NÃO ENCONTRADO';
        } else {
          final croppedImage = img.decodeImage(prepared.croppedBytes!);
          if (croppedImage == null) {
            throw Exception('Não foi possível preparar a imagem da galeria.');
          }
          await Future<void>.delayed(const Duration(milliseconds: 16));
          resultado = await _classifyWithConvNeXt(croppedImage);
        }
      }

      if (!mounted) return;

      // 4. Adiciona ao dashboard e persiste
      setState(() {
        _results.insert(0, {
          'path': displayFile.path,
          'resultado': resultado,
          'timestamp': DateTime.now().toIso8601String(),
        });
        _isProcessing = false;
      });
      await _saveResults();
    } catch (e) {
      print('[Main] Erro ao processar imagem: $e');
      if (mounted) {
        setState(() => _isProcessing = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Erro: $e'),
            backgroundColor: Colors.red.shade800,
          ),
        );
      }
    }
  }

  // ── Inferência ConvNeXt Large ──────────────────────────────────────────────

  Future<String> _classifyWithConvNeXt(img.Image croppedImage) async {
    final interpreter = _interpreter;
    if (interpreter == null) return 'Modelo não carregado';

    try {
      // Garante RGB apenas para construir o tensor (não afeta o detector)
      img.Image toClassify = croppedImage;
      if (toClassify.numChannels != 3) {
        toClassify = toClassify.convert(numChannels: 3);
      }

      final resized = img.copyResize(toClassify, width: 224, height: 224);

      // Tensor NCHW [1, 3, 224, 224] como lista aninhada.
      // Float32List plana é tratada como 1D pelo tflite_flutter → TRANSPOSE falha.
      // Lista aninhada permite ao runtime inferir o shape 4D corretamente.
      final input = [
        List.generate(
          3,
          (c) => List.generate(
            224,
            (y) => List.generate(224, (x) {
              final pixel = resized.getPixel(x, y);
              final raw = c == 0
                  ? pixel.r / 255.0
                  : c == 1
                      ? pixel.g / 255.0
                      : pixel.b / 255.0;
              return (raw - _mean[c]) / _std[c];
            }),
          ),
        ),
      ];

      final output = [List<double>.filled(2, 0.0)];
      interpreter.allocateTensors();
      interpreter.run(input, output);

      final logits = output[0];
      final probs = _softmax(logits);
      final probAprovado = probs[0];
      final probReprovado = probs[1];
      final confianca =
          (max(probAprovado, probReprovado) * 100).toStringAsFixed(1);

      print('[ConvNeXt] logits=$logits  probs=$probs');
      return probAprovado > probReprovado
          ? 'APROVADO ($confianca%)'
          : 'REPROVADO ($confianca%)';
    } catch (e) {
      print('[ConvNeXt] Erro na inferência: $e');
      return 'Erro na classificação';
    }
  }

  List<double> _softmax(List<double> logits) {
    final maxL = logits.reduce(max);
    final exps = logits.map((l) => exp(l - maxL)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  // ── Dashboard helpers ──────────────────────────────────────────────────────

  void _deleteResult(int index) {
    setState(() => _results.removeAt(index));
    _saveResults();
  }

  void _showFullImage(File imageFile, String resultado) {
    showDialog(
      context: context,
      builder: (_) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.all(12),
        child: Stack(
          alignment: Alignment.center,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: InteractiveViewer(
                child: Image.file(imageFile, fit: BoxFit.contain),
              ),
            ),
            Positioned(
              top: 8,
              right: 8,
              child: CircleAvatar(
                backgroundColor: Colors.black54,
                child: IconButton(
                  icon: const Icon(Icons.close, color: Colors.white),
                  onPressed: () => Navigator.pop(context),
                ),
              ),
            ),
            Positioned(
              bottom: 20,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  resultado,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── UI ─────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF12080A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E0D0A),
        elevation: 0,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: const Color(0xFFE8735A).withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.science_outlined,
                color: Color(0xFFE8735A),
                size: 20,
              ),
            ),
            const SizedBox(width: 10),
            const Text(
              'ALIZAROL',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                color: Color(0xFFE8735A),
                letterSpacing: 2.5,
              ),
            ),
          ],
        ),
        actions: [
          if (_results.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Center(
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFFE8735A).withOpacity(0.12),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: const Color(0xFFE8735A).withOpacity(0.3),
                      width: 1,
                    ),
                  ),
                  child: Text(
                    '${_results.length} análise${_results.length > 1 ? 's' : ''}',
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.white.withOpacity(0.7),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
      body: Stack(
        children: [
          // ── Dashboard / Empty State
          _results.isEmpty ? _buildEmptyState() : _buildDashboard(),

          // ── Overlay de carregamento
          if (_isProcessing)
            Container(
              color: Colors.black.withOpacity(0.78),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const SizedBox(
                      width: 52,
                      height: 52,
                      child: CircularProgressIndicator(
                        color: Color(0xFFE8735A),
                        strokeWidth: 3,
                      ),
                    ),
                    const SizedBox(height: 20),
                    const Text(
                      'Analisando imagem...',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 17,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Detectando e classificando a amostra',
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.5),
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),

      // ── FAB: câmera / galeria
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isProcessing ? null : _showSourcePicker,
        backgroundColor: const Color(0xFFE8735A),
        foregroundColor: Colors.white,
        icon: const Icon(Icons.add_a_photo_rounded),
        label: const Text(
          'Nova Análise',
          style: TextStyle(fontWeight: FontWeight.w700),
        ),
      ),
    );
  }

  // ── Empty State ────────────────────────────────────────────────────────────

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                color: const Color(0xFFE8735A).withOpacity(0.08),
                shape: BoxShape.circle,
                border: Border.all(
                  color: const Color(0xFFE8735A).withOpacity(0.2),
                  width: 1.5,
                ),
              ),
              child: Icon(
                Icons.science_outlined,
                size: 46,
                color: const Color(0xFFE8735A).withOpacity(0.5),
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Nenhuma análise ainda',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white.withOpacity(0.85),
              ),
            ),
            const SizedBox(height: 10),
            Text(
              'Use o botão abaixo para capturar\numa foto ou selecionar da galeria.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.white.withOpacity(0.4),
                height: 1.6,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Dashboard ──────────────────────────────────────────────────────────────

  Widget _buildDashboard() {
    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
      itemCount: _results.length,
      itemBuilder: (context, index) {
        final item = _results[index];
        final resultado = item['resultado'] as String;
        final imageFile = File(item['path'] as String);
        final timestamp = item['timestamp'] as String;

        final bool isAprovado = resultado.contains('APROVADO');
        final bool isReprovado = resultado.contains('REPROVADO');

        final Color resultColor = isAprovado
            ? const Color(0xFF4CAF50)
            : isReprovado
                ? const Color(0xFFEF5350)
                : const Color(0xFFE8735A);

        final IconData resultIcon = isAprovado
            ? Icons.check_circle_rounded
            : isReprovado
                ? Icons.cancel_rounded
                : Icons.search_off_rounded;

        final String dateLabel = _formatTimestamp(timestamp);

        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: InkWell(
            onTap: imageFile.existsSync()
                ? () => _showFullImage(imageFile, resultado)
                : null,
            borderRadius: BorderRadius.circular(18),
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(18),
                color: Colors.white.withOpacity(0.04),
                border: Border.all(
                  color: resultColor.withOpacity(0.2),
                  width: 1,
                ),
              ),
              padding: const EdgeInsets.all(14),
              child: Row(
                children: [
                  // Thumbnail
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: SizedBox(
                      width: 68,
                      height: 68,
                      child: imageFile.existsSync()
                          ? Image.file(
                              imageFile,
                              fit: BoxFit.cover,
                              errorBuilder: (_, __, ___) => _imagePlaceholder(),
                            )
                          : _imagePlaceholder(),
                    ),
                  ),
                  const SizedBox(width: 14),

                  // Texto
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(resultIcon, color: resultColor, size: 18),
                            const SizedBox(width: 6),
                            Expanded(
                              child: Text(
                                resultado,
                                style: TextStyle(
                                  color: resultColor,
                                  fontWeight: FontWeight.w700,
                                  fontSize: 13.5,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 5),
                        Text(
                          dateLabel,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withOpacity(0.35),
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Botão deletar
                  IconButton(
                    icon: Icon(
                      Icons.delete_outline_rounded,
                      color: Colors.white.withOpacity(0.3),
                      size: 20,
                    ),
                    onPressed: () => _deleteResult(index),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _imagePlaceholder() => Container(
        color: Colors.white.withOpacity(0.04),
        child: Icon(
          Icons.image_outlined,
          color: Colors.white.withOpacity(0.2),
        ),
      );

  String _formatTimestamp(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final pad = (int n) => n.toString().padLeft(2, '0');
      return '${pad(dt.day)}/${pad(dt.month)}/${dt.year} '
          '${pad(dt.hour)}:${pad(dt.minute)}';
    } catch (_) {
      return iso.split('T').first;
    }
  }

  // ── Bottom Sheet: escolha de fonte ────────────────────────────────────────

  void _showSourcePicker() {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1E0D0A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Handle
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Como deseja analisar?',
                style: TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                  color: Colors.white.withOpacity(0.9),
                ),
              ),
              const SizedBox(height: 16),

              // Câmera
              _SourceOption(
                icon: Icons.camera_alt_rounded,
                title: 'Câmera',
                subtitle: 'IA detecta o posicionamento ideal da amostra',
                accentColor: const Color(0xFFE8735A),
                onTap: () {
                  Navigator.pop(ctx);
                  _openCamera();
                },
              ),
              const SizedBox(height: 12),

              // Galeria
              _SourceOption(
                icon: Icons.photo_library_rounded,
                title: 'Galeria',
                subtitle: 'Selecione uma foto existente no dispositivo',
                accentColor: const Color(0xFF7B8EE8),
                onTap: () {
                  Navigator.pop(ctx);
                  _pickFromGallery();
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source Option Widget
// ─────────────────────────────────────────────────────────────────────────────

class _SourceOption extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color accentColor;
  final VoidCallback onTap;

  const _SourceOption({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.accentColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          color: accentColor.withOpacity(0.07),
          border: Border.all(
            color: accentColor.withOpacity(0.2),
            width: 1,
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: accentColor.withOpacity(0.12),
                borderRadius: BorderRadius.circular(13),
              ),
              child: Icon(icon, color: accentColor, size: 26),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.white.withOpacity(0.45),
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios_rounded,
              color: accentColor.withOpacity(0.5),
              size: 15,
            ),
          ],
        ),
      ),
    );
  }
}
