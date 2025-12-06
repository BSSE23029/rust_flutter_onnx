// Library module for shared functionality between CLI and API

use image::{DynamicImage, GenericImageView};
use ndarray::{Array, IxDyn};
use ort::{Session, Value};
use serde::{Deserialize, Serialize};
use std::error::Error;

// Class names matching the model training
pub const CLASS_NAMES: [&str; 3] = ["health", "sick", "tb"];
pub const CLASS_EMOJIS: [&str; 3] = ["✅", "⚠️ ", "🚨"];

/// Model outputs structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutputs {
    pub probabilities: Vec<f32>,
    pub predicted_class: usize,
    pub confidence: f32,
    pub top_2_classes: Vec<usize>,
    pub top_2_scores: Vec<f32>,
    pub confidence_level: i32,
    pub entropy: f32,
}

impl ModelOutputs {
    /// Get the class name for the predicted class
    pub fn class_name(&self) -> &str {
        CLASS_NAMES[self.predicted_class]
    }

    /// Get the emoji for the predicted class
    pub fn class_emoji(&self) -> &str {
        CLASS_EMOJIS[self.predicted_class]
    }

    /// Get confidence level as text
    pub fn confidence_level_text(&self) -> &str {
        match self.confidence_level {
            2 => "HIGH",
            1 => "MEDIUM",
            _ => "LOW",
        }
    }

    /// Get confidence level emoji
    pub fn confidence_level_emoji(&self) -> &str {
        match self.confidence_level {
            2 => "🟢",
            1 => "🟡",
            _ => "🔴",
        }
    }

    /// Get clinical recommendation
    pub fn recommendation(&self) -> &str {
        match self.predicted_class {
            0 => "No abnormalities detected. Continue regular monitoring.",
            1 => "Sickness indicators detected. Recommend medical consultation.",
            2 => "Tuberculosis indicators present. Immediate medical attention required.",
            _ => "Unknown classification.",
        }
    }
}

/// Run inference on an image
pub fn predict(session: &mut Session, img: &DynamicImage) -> Result<ModelOutputs, Box<dyn Error>> {
    let img_rgb = img.to_rgb8();
    let (width, height) = img.dimensions();

    // Create input tensor as dynamic array (batch=1, height, width, channels=3)
    // No preprocessing needed - model handles everything!
    let shape = vec![1, height as usize, width as usize, 3];
    let mut array_data = vec![0u8; shape.iter().product()];

    for (x, y, pixel) in img_rgb.enumerate_pixels() {
        let idx = (y as usize * width as usize + x as usize) * 3;
        array_data[idx] = pixel[0];
        array_data[idx + 1] = pixel[1];
        array_data[idx + 2] = pixel[2];
    }

    let array = Array::from_shape_vec(IxDyn(&shape), array_data)?;

    // Create input value - convert to CowArray
    let cow_array = ndarray::CowArray::from(&array);
    let input_tensor_value = Value::from_array(session.allocator(), &cow_array)?;

    // Run inference - get all 7 outputs!
    let outputs: Vec<Value> = session.run(vec![input_tensor_value])?;

    // The outputs are in alphabetical order by name:
    // 0: confidence, 1: confidence_level, 2: entropy, 3: predicted_class,
    // 4: probabilities, 5: top_2_classes, 6: top_2_scores

    // Extract outputs by index
    // Note: ONNX conversion may output int32 or int64 for argmax operations
    let confidence = outputs[0].try_extract::<f32>()?.view()[0];
    let confidence_level = outputs[1].try_extract::<f32>()?.view()[0] as i32;
    let entropy = outputs[2].try_extract::<f32>()?.view()[0];

    // Try int64 first, fall back to int32 if needed
    let predicted_class = if let Ok(tensor) = outputs[3].try_extract::<i64>() {
        tensor.view()[0] as usize
    } else {
        outputs[3].try_extract::<i32>()?.view()[0] as usize
    };

    let probabilities = outputs[4]
        .try_extract::<f32>()?
        .view()
        .to_slice()
        .unwrap()
        .to_vec();

    let top_2_classes: Vec<usize> = if let Ok(tensor) = outputs[5].try_extract::<i64>() {
        tensor.view().iter().map(|&x| x as usize).collect()
    } else {
        outputs[5]
            .try_extract::<i32>()?
            .view()
            .iter()
            .map(|&x| x as usize)
            .collect()
    };

    let top_2_scores: Vec<f32> = outputs[6]
        .try_extract::<f32>()?
        .view()
        .to_slice()
        .unwrap()
        .to_vec();

    Ok(ModelOutputs {
        probabilities,
        predicted_class,
        confidence,
        top_2_classes,
        top_2_scores,
        confidence_level,
        entropy,
    })
}
