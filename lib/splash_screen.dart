import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:Alizarol_app/main.dart';

/// Splash screen exibida ao iniciar o app.
class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // ── Imagem de fundo
          Positioned.fill(
            child: Image.asset(
              'assets/imagens/alizarol.jpg',
              fit: BoxFit.cover,
            ),
          ),

          // ── Overlay com blur
          Positioned.fill(
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
              child: Container(
                color: const Color(0xFF1A0A08).withOpacity(0.72),
              ),
            ),
          ),

          // ── Conteúdo centralizado
          SafeArea(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Logo
                  Image.asset(
                    'assets/imagens/logo_VIA.png',
                    width: 140,
                    height: 140,
                  ),

                  const SizedBox(height: 28),

                  // Título
                  const Text(
                    'ALIZAROL',
                    style: TextStyle(
                      fontSize: 30,
                      fontWeight: FontWeight.w800,
                      color: Color(0xFFE8735A),
                      letterSpacing: 4,
                    ),
                  ),

                  const SizedBox(height: 6),

                  Text(
                    'Análise de Leite por Inteligência Artificial',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.white.withOpacity(0.55),
                      letterSpacing: 0.5,
                    ),
                  ),

                  const SizedBox(height: 56),

                  // Botão iniciar
                  ElevatedButton.icon(
                    onPressed: () {
                      Navigator.pushReplacement(
                        context,
                        MaterialPageRoute(
                          builder: (_) => const HomeScreen(),
                        ),
                      );
                    },
                    icon: const Icon(Icons.arrow_forward_rounded),
                    label: const Text('Iniciar'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFE8735A),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 40, vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                      textStyle: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
