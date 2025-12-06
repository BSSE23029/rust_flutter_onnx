# Medical Image Classifier (Rust)

Fast, minimal Rust CLI for X-ray image classification using ONNX.

## Features

- ✅ **Zero preprocessing code** - Model handles everything
- ✅ **7 flashy outputs** - Confidence levels, entropy, top-k predictions
- ✅ **Beautiful CLI** - Color-coded results with emojis
- ✅ **Fast** - ONNX Runtime optimized inference
- ✅ **Simple** - Just load image and predict

## Build

```bash
cargo build --release
```

## Usage

### Basic Usage

```bash
cargo run --release -- ../input/h4615.png
```

### With Custom Model Path

```bash
cargo run --release -- ../input/tb1075.png --model ../model/unified_model.onnx
```

### Verbose Mode (show model info)

```bash
cargo run --release -- ../input/s4615.png --verbose
```

### Built Binary (Optimized Release)

**Option 1: Using the wrapper script (easiest)**
```bash
./classify.sh ../input/test.jpg
```

**Option 2: Manual library path**
```bash
DYLD_LIBRARY_PATH=./target/release ./target/release/image_classifier ../input/test.jpg
```

> **Note:** The release binary requires the ONNX Runtime library to be in the library path. The `classify.sh` script handles this automatically.

## Output

The CLI displays:
- 🎯 **Primary prediction** with confidence level
- 📊 **Top 2 predictions** (1st and 2nd choice)
- 📈 **All class probabilities** with visual bars
- 📋 **Interpretation** (confidence level, uncertainty)
- ✅/⚠️/🚨 **Clinical recommendation**

## Example

```bash
$ cargo run --release -- ../input/h4615.png

🔄 Loading ONNX model...
✅ Model loaded successfully

🔄 Processing image: ../input/h4615.png
   Original size: 512x512

======================================================================
🏥 MEDICAL IMAGE CLASSIFICATION RESULT
======================================================================

----------------------------------------------------------------------
🎯 PRIMARY PREDICTION
----------------------------------------------------------------------
  ✅  Class: HEALTH
  📊  Confidence: 100.00%
  🎚️   Confidence Level: 🟢 HIGH
  📉  Uncertainty (Entropy): -0.000000

----------------------------------------------------------------------
📊 TOP 2 PREDICTIONS
----------------------------------------------------------------------
  🥇 ✅  HEALTH: 100.00%
  🥈 ⚠️  SICK: 0.00%

----------------------------------------------------------------------
📈 ALL CLASS PROBABILITIES
----------------------------------------------------------------------
  ✅  HEALTH   ████████████████████████████████████████ 100.00%
  ⚠️  SICK     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.00%
  🚨  TB       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.00%

----------------------------------------------------------------------
📋 INTERPRETATION
----------------------------------------------------------------------
  ✅ High confidence prediction - Result is reliable
  ✅ Very low uncertainty - Model is very certain

======================================================================
✅ No abnormalities detected. Continue regular monitoring.
======================================================================
```

## Model

- **Location:** `../model/unified_model.onnx`
- **Size:** 68 MB
- **Input:** RGB uint8 image (any size)
- **Output:** 7 values (probabilities, predicted class, confidence, top-2, confidence level, entropy)
- **Classes:** `health`, `sick`, `tb`

## Dependencies

- `ort` - ONNX Runtime bindings
- `image` - Image loading
- `ndarray` - Tensor operations
- `clap` - CLI argument parsing
- `colored` - Terminal colors

## Code Structure

- `main.rs` - Complete CLI application (~300 lines)
  - Command-line parsing
  - Model loading
  - Image preprocessing
  - Inference
  - Flashy result display

## Next Steps

After CLI works, we'll add:
- 🌐 REST API server
- 📁 Batch processing
- 🔄 Watch folder mode
- 📊 Results export (JSON/CSV)
