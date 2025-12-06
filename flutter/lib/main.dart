import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medical Image Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const ClassifierPage(),
    );
  }
}

class ClassifierPage extends StatefulWidget {
  const ClassifierPage({super.key});

  @override
  State<ClassifierPage> createState() => _ClassifierPageState();
}

class _ClassifierPageState extends State<ClassifierPage> {
  File? _image;
  Uint8List? _imageBytes; // For web support
  String? _imagePath; // Store path for API call
  bool _isLoading = false;
  Map<String, dynamic>? _prediction;
  final ImagePicker _picker = ImagePicker();
  
  // API endpoint - change this if your API is on different host/port
  final String apiUrl = 'http://localhost:3000/predict';

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (pickedFile != null) {
        final bytes = await pickedFile.readAsBytes();
        setState(() {
          _imagePath = pickedFile.path;
          _imageBytes = bytes;
          if (!kIsWeb) {
            _image = File(pickedFile.path);
          }
          _prediction = null; // Clear previous prediction
        });
      }
    } catch (e) {
      _showError('Failed to pick image: $e');
    }
  }

  Future<void> _classifyImage() async {
    if (_imageBytes == null) return;

    setState(() {
      _isLoading = true;
      _prediction = null;
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      
      // Use bytes for both web and mobile
      request.files.add(http.MultipartFile.fromBytes(
        'image',
        _imageBytes!,
        filename: 'image.png',
      ));

      var response = await request.send();
      var responseData = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        setState(() {
          _prediction = json.decode(responseData);
          _isLoading = false;
        });
      } else {
        throw Exception('Server returned ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showError('Classification failed: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  String _getClassEmoji(String className) {
    switch (className) {
      case 'health':
        return '✅';
      case 'sick':
        return '🤒';
      case 'tb':
        return '🦠';
      default:
        return '❓';
    }
  }

  Color _getConfidenceColor(String level) {
    switch (level) {
      case 'HIGH':
        return Colors.green;
      case 'MEDIUM':
        return Colors.orange;
      case 'LOW':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Medical Image Classifier'),
        centerTitle: true,
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview Card
            Card(
              elevation: 4,
              child: Container(
                height: 300,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                ),
                child: _imageBytes == null
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.image, size: 80, color: Colors.grey[400]),
                            const SizedBox(height: 16),
                            Text(
                              'No image selected',
                              style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                            ),
                          ],
                        ),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: kIsWeb
                            ? Image.memory(_imageBytes!, fit: BoxFit.contain)
                            : Image.file(_image!, fit: BoxFit.contain),
                      ),
              ),
            ),

            const SizedBox(height: 20),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            // Classify Button
            FilledButton.icon(
              onPressed: _imageBytes == null || _isLoading ? null : _classifyImage,
              icon: _isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.analytics),
              label: Text(_isLoading ? 'Analyzing...' : 'Classify Image'),
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),

            const SizedBox(height: 24),

            // Results Section
            if (_prediction != null) ...[
              _buildResultsCard(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultsCard() {
    final prediction = _prediction!['prediction'];
    final className = prediction['class'];
    final confidence = prediction['confidence_percentage'];
    final confidenceLevel = prediction['confidence_level'];
    final recommendation = prediction['recommendation'];
    final probabilities = prediction['probabilities'];
    final emoji = _getClassEmoji(className);

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Main Result
            Center(
              child: Column(
                children: [
                  Text(
                    emoji,
                    style: const TextStyle(fontSize: 64),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    className.toUpperCase(),
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: _getConfidenceColor(confidenceLevel).withOpacity(0.2),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: _getConfidenceColor(confidenceLevel),
                        width: 2,
                      ),
                    ),
                    child: Text(
                      '$confidence • $confidenceLevel',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: _getConfidenceColor(confidenceLevel),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),
            const Divider(),
            const SizedBox(height: 16),

            // Recommendation
            const Text(
              'Recommendation',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                recommendation,
                style: const TextStyle(fontSize: 14),
              ),
            ),

            const SizedBox(height: 20),

            // Probabilities
            const Text(
              'Class Probabilities',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            ...probabilities.entries.map((entry) => _buildProbabilityBar(
                  entry.key,
                  (entry.value as num).toDouble(),
                )),

            const SizedBox(height: 16),

            // Additional Info
            if (prediction['entropy'] != null) ...[
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Entropy:',
                    style: TextStyle(fontWeight: FontWeight.w500),
                  ),
                  Text(
                    prediction['entropy'].toStringAsFixed(4),
                    style: const TextStyle(fontFamily: 'monospace'),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildProbabilityBar(String className, double probability) {
    final emoji = _getClassEmoji(className);
    final percentage = (probability * 100).toStringAsFixed(2);

    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('$emoji $className'),
              Text('$percentage%', style: const TextStyle(fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: probability,
              minHeight: 8,
              backgroundColor: Colors.grey[300],
            ),
          ),
        ],
      ),
    );
  }
}
