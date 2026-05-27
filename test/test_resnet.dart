import 'package:flutter_test/flutter_test.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  test('Check ResNet50 model shape', () async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final interpreter = await Interpreter.fromAsset(
      'assets/models/Alizarol_Leite_resnet50_S46_padrao_foradepadrao.tflite',
    );
    print('[ResNet50] Input tensor shape: ${interpreter.getInputTensors().map((t) => t.shape).toList()}');
    print('[ResNet50] Output tensor shape: ${interpreter.getOutputTensors().map((t) => t.shape).toList()}');
    interpreter.close();
  });
}
