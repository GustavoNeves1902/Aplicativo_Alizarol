import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';
import 'dart:typed_data';
import 'dart:math';

// ─────────────────────────────────────────────────────────────────────────────
// DETECÇÃO DE PLACA + VALIDAÇÃO DA COR DO ALIZAROL
// ─────────────────────────────────────────────────────────────────────────────

/// Detecta a placa de Petri com HoughCircles e recorta com padding de 70px.
/// Pipeline idêntico ao script Python fornecido.
///
/// Retorna um record `({img.Image image, bool circleFound})`:
///  - `image`       → imagem recortada (ou original se nenhum círculo foi detectado).
///  - `circleFound` → `true` se HoughCircles achou um círculo; `false` caso contrário.
({img.Image image, bool circleFound}) detectPetriAndCrop(img.Image image) {
  cv.Mat? mat;
  try {
    mat = _imgToMat(image);
    final result = _houghPipeline(mat);
    if (result.image != null) {
      final out = _matToImg(result.image!);
      result.image!.dispose();
      return (image: out, circleFound: true);
    }
  } catch (e) {
    print('[Detector] Erro HoughCircles: $e');
  } finally {
    mat?.dispose();
  }
  print('[Detector] Círculo não encontrado — usando imagem original.');
  return (image: image, circleFound: false);
}

// ─────────────────────────────────────────────────────────────────────────────
// VALIDAÇÃO DA COR ROSA/VERMELHO (ALIZAROL)
// ─────────────────────────────────────────────────────────────────────────────

/// Verifica se a imagem (já recortada sobre o círculo) contém o tom rosa/vermelho
/// característico do Alizarol.
///
/// Usa os mesmos intervalos HSV do script Python `findind_circles_01.py`:
///   • Faixa 1: H[0,10]   S[30,255]  V[50,255]  (vermelho baixo)
///   • Faixa 2: H[160,180] S[30,255]  V[50,255]  (vermelho alto / rosa)
///
/// [minPixelRatio] — fração mínima de pixels rosa para considerar positivo.
/// Valor padrão: 0.01 (1% da imagem recortada), ajuste conforme necessário.
bool hasAlizarolPink(img.Image croppedImage, {double minPixelRatio = 0.01}) {
  cv.Mat? mat;
  cv.Mat? hsv;
  cv.Mat? mask1;
  cv.Mat? mask2;
  cv.Mat? mask;
  try {
    mat = _imgToMat(croppedImage);

    // Converte BGR → HSV (mesmo espaço de cor do script Python)
    hsv = cv.cvtColor(mat, cv.COLOR_BGR2HSV);

    // Faixa 1: vermelho baixo (H 0–10)
    mask1 = cv.inRange(
      hsv,
      cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [0,   30,  50]),
      cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [10,  255, 255]),
    );

    // Faixa 2: rosa/vermelho alto (H 160–180)
    mask2 = cv.inRange(
      hsv,
      cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [160, 30,  50]),
      cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [180, 255, 255]),
    );

    // União das duas faixas
    mask = cv.Mat.zeros(mat.rows, mat.cols, cv.MatType.CV_8UC1);
    cv.bitwiseOR(mask1, mask2, dst: mask);

    // Conta pixels rosa e calcula a proporção
    final pinkPixels = cv.countNonZero(mask);
    final totalPixels = mat.rows * mat.cols;
    final ratio = pinkPixels / totalPixels;

    print('[Detector] Pixels rosa: $pinkPixels / $totalPixels  (${(ratio * 100).toStringAsFixed(2)}%)');
    return ratio >= minPixelRatio;
  } catch (e) {
    print('[Detector] Erro hasAlizarolPink: $e');
    return false;
  } finally {
    mat?.dispose();
    hsv?.dispose();
    mask1?.dispose();
    mask2?.dispose();
    mask?.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTO-CAPTURE: CONVERSÃO DE FRAME + DETECÇÃO EM ISOLATE
// ─────────────────────────────────────────────────────────────────────────────

/// Converte um [CameraImage] do stream para bytes JPEG prontos para o Isolate.
///
/// - **Android (YUV420):** usa apenas o plano Y (luma) como imagem em tons de cinza.
///   Isso é 3× mais rápido que a conversão YUV→RGB completa e suficiente para o
///   HoughCircles, que opera internamente sobre tons de cinza.
/// - **iOS (BGRA8888):**  usa [img.Image.fromBytes] sem loop de pixels.
///
/// A imagem é reduzida para 640 px de largura antes da codificação JPEG (quality 75)
/// para minimizar o tempo de processamento no Isolate.
Uint8List cameraImageToJpeg(CameraImage frame) {
  img.Image image;

  if (frame.format.group == ImageFormatGroup.yuv420) {
    image = _yPlaneToGrayscale(frame);
  } else {
    // BGRA8888 (iOS)
    image = _bgraToImage(frame);
  }

  // Reduz para 640 px de largura — círculos são fortes e detectados mesmo em baixa res
  final scaled = img.copyResize(image, width: 640);
  return Uint8List.fromList(img.encodeJpg(scaled, quality: 75));
}

/// Converte um [CameraImage] de alta resolução diretamente para um [img.Image]
/// em tons de cinza de 224x224, sem alocações pesadas ou conversões para JPEG.
/// É extremamente rápido (menos de 2ms) pois faz subamostragem direta na memória.
img.Image cameraImageTo224Grayscale(CameraImage frame) {
  final width = frame.width;
  final height = frame.height;
  
  final out = img.Image(width: 224, height: 224, numChannels: 1);
  final double scaleX = width / 224.0;
  final double scaleY = height / 224.0;

  if (frame.format.group == ImageFormatGroup.yuv420) {
    final yPlane = frame.planes[0];
    final yBytes = yPlane.bytes;
    final bytesPerRow = yPlane.bytesPerRow;

    for (int y = 0; y < 224; y++) {
      final int srcY = (y * scaleY).toInt();
      final int rowOffset = srcY * bytesPerRow;
      for (int x = 0; x < 224; x++) {
        final int srcX = (x * scaleX).toInt();
        final int gray = yBytes[rowOffset + srcX];
        out.setPixelRgb(x, y, gray, gray, gray);
      }
    }
  } else {
    // BGRA8888 (iOS)
    final plane = frame.planes[0];
    final bytes = plane.bytes;
    final bytesPerRow = plane.bytesPerRow;

    for (int y = 0; y < 224; y++) {
      final int srcY = (y * scaleY).toInt();
      final int rowOffset = srcY * bytesPerRow;
      for (int x = 0; x < 224; x++) {
        final int srcX = (x * scaleX).toInt();
        final int pixelIndex = rowOffset + srcX * 4;
        
        final b = bytes[pixelIndex];
        final g = bytes[pixelIndex + 1];
        final r = bytes[pixelIndex + 2];
        
        final gray = ((r + g + b) / 3).toInt();
        out.setPixelRgb(x, y, gray, gray, gray);
      }
    }
  }

  return out;
}

/// Extrai o plano Y (luma) de um frame YUV420 como imagem em tons de cinza.
/// Usa [img.Image.fromBytes] (sem loop de pixels) sempre que não há padding de linha.
img.Image _yPlaneToGrayscale(CameraImage frame) {
  final width = frame.width;
  final height = frame.height;
  final yPlane = frame.planes[0];

  final Uint8List yBytes;
  if (yPlane.bytesPerRow == width) {
    // Sem padding — usa o buffer diretamente
    yBytes = yPlane.bytes;
  } else {
    // Remove o padding linha a linha (operação de cópia de bytes, muito rápida)
    yBytes = Uint8List(width * height);
    for (int row = 0; row < height; row++) {
      yBytes.setRange(
        row * width,
        row * width + width,
        yPlane.bytes,
        row * yPlane.bytesPerRow,
      );
    }
  }

  return img.Image.fromBytes(
    width: width,
    height: height,
    bytes: yBytes.buffer,
    numChannels: 1, // tons de cinza
  );
}

/// Converte um frame BGRA8888 (iOS) para [img.Image] sem loop de pixels.
img.Image _bgraToImage(CameraImage frame) {
  final plane = frame.planes[0];
  return img.Image.fromBytes(
    width: frame.width,
    height: frame.height,
    bytes: plane.bytes.buffer,
    numChannels: 4,
    order: img.ChannelOrder.bgra,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DETECÇÃO DE CÍRCULO EM ISOLATE (top-level para compute())
// ─────────────────────────────────────────────────────────────────────────────

/// Função top-level compatível com [compute] (Flutter Isolates).
///
/// Recebe [jpegBytes] de um frame da câmera (já convertido por [cameraImageToJpeg])
/// e retorna um Record contendo os bytes JPEG do recorte (se detectado e válido)
/// e uma String com a instrução para o usuário.
({Uint8List? jpegBytes, String instruction}) detectCircleOnFrame(Uint8List jpegBytes) {
  cv.Mat? mat;
  cv.Mat? result;
  try {
    mat = cv.imdecode(jpegBytes, cv.IMREAD_COLOR);
    final pipeResult = _houghPipeline(mat);
    result = pipeResult.image;
    
    if (result == null) {
      return (jpegBytes: null, instruction: pipeResult.instruction);
    }
    
    // Para captura automática em tempo real: exige alinhamento perfeito ('OK')
    if (pipeResult.instruction != 'OK') {
      result.dispose();
      return (jpegBytes: null, instruction: pipeResult.instruction);
    }
    
    final (success, encoded) = cv.imencode('.jpg', result);
    return (jpegBytes: success ? encoded : null, instruction: pipeResult.instruction);
  } catch (e) {
    print('[Detector] detectCircleOnFrame erro: $e');
    return (jpegBytes: null, instruction: 'Buscando placa...');
  } finally {
    mat?.dispose();
    result?.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE HOUGHCIRCLES (interno)
// ─────────────────────────────────────────────────────────────────────────────

({cv.Mat? image, String instruction}) _houghPipeline(cv.Mat mat) {
  // 1. Tons de Cinza
  final gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY);

  // 2. Median Blur pesado (kernel 11).
  // O filtro mediano é incrivelmente eficaz para matar pequenas texturas
  // (como veios da folha, amassados de papel ou ruídos no fundo)
  // enquanto preserva bordas geométricas fortes (como a borda da placa).
  final blurred = cv.medianBlur(gray, 11);
  gray.dispose();

  // 3. Parâmetros dinâmicos do HoughCircles baseados no tamanho da imagem
  final minDim = min(mat.rows, mat.cols);
  final minDist = max(100.0, minDim * 0.15); // Permite círculos um pouco mais juntos/descentralizados
  
  // ERRO ANTERIOR: maxR estava travado em no máximo 800px.
  // Em fotos de galeria (12 Megapixels), o raio do prato pode ser muito maior
  // que 800px. Removemos a limitação "min(800, ...)" para liberar o raio gigante.
  final minR = max(20, (minDim * 0.10).toInt()); // Pelo menos 10% da imagem
  final maxR = (minDim * 0.80).toInt();          // Pode ocupar até 80% da imagem

  // Ajuste sutil do acumulador
  final double param2 = minDim > 1500 ? 50.0 : 40.0;

  final circles = cv.HoughCircles(
    blurred,
    cv.HOUGH_GRADIENT,
    1.2, // dp (resolução do acumulador pouco menor q a imagem para limpar ruídos)
    minDist,
    param1: 100,     // Limiar alto do edge detector (bordas fortes)
    param2: param2,  // Exigência de ser "muito perfeitamente circular"
    minRadius: minR,
    maxRadius: maxR,
  );
  blurred.dispose();

  if (circles.isEmpty || circles.cols == 0) {
    circles.dispose();
    return (image: null, instruction: 'Buscando placa...');
  }

  // 4. Extrai o círculo com mais "votos"
  final c = circles.at<cv.Vec3f>(0, 0);
  circles.dispose();

  final x = c.val1;
  final y = c.val2;
  final r = c.val3;

  print('[Detector] Círculo: (${x.toInt()}, ${y.toInt()}), r=${r.toInt()}');

  final bx = (x - r).toInt();
  final by = (y - r).toInt();
  final bw = (2 * r).toInt();
  final bh = (2 * r).toInt();

  final xMin = max(0, bx - 70);
  final yMin = max(0, by - 70);
  final xMax = min(mat.cols, bx + bw + 70);
  final yMax = min(mat.rows, by + bh + 70);

  if (xMax <= xMin || yMax <= yMin) return (image: null, instruction: 'Buscando placa...');

  String instruction = 'OK';
  
  if (r < minDim * 0.28) {
    instruction = 'Aproxime a câmera';
  } else if (r > minDim * 0.48) {
    instruction = 'Afaste a câmera';
  } else {
    final dx = x - mat.cols / 2;
    final dy = y - mat.rows / 2;
    final dist = sqrt(dx * dx + dy * dy);
    if (dist > minDim * 0.15) {
      instruction = 'Centralize a placa';
    }
  }

  // Retorna a imagem recortada mesmo se o alinhamento da câmera não for perfeito
  // (essencial para análise de imagens estáticas da galeria ou capturas manuais),
  // mas envia a instrução calculada para guiar o feedback visual da câmera.
  final cropped = mat.region(cv.Rect(xMin, yMin, xMax - xMin, yMax - yMin)).clone();
  return (image: cropped, instruction: instruction);
}

// ─────────────────────────────────────────────────────────────────────────────
// CONVERSÃO img.Image ↔ OpenCV Mat
// ─────────────────────────────────────────────────────────────────────────────

cv.Mat _imgToMat(img.Image image) {
  final bytes = img.encodeJpg(image, quality: 95);
  return cv.imdecode(Uint8List.fromList(bytes), cv.IMREAD_COLOR);
}

img.Image _matToImg(cv.Mat mat) {
  final (success, bytes) = cv.imencode('.jpg', mat);
  if (!success || bytes.isEmpty) throw Exception('imencode falhou');
  return img.decodeImage(bytes) ?? (throw Exception('decodeImage falhou'));
}