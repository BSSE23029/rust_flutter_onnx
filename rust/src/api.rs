// REST API module

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use image::ImageReader;
use ort::{session::SessionBuilder, Environment, GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::{io::Cursor, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::info;

use image_classifier::{predict, CLASS_NAMES};

/// Application state shared across handlers
pub struct AppState {
    pub session: Arc<Mutex<Session>>,
}

/// JSON response for prediction endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub success: bool,
    pub prediction: PredictionResult,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionResult {
    pub class: String,
    pub class_index: usize,
    pub confidence: f32,
    pub confidence_percentage: String,
    pub confidence_level: String,
    pub entropy: f32,
    pub probabilities: ClassProbabilities,
    pub top_2_predictions: Vec<TopPrediction>,
    pub recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClassProbabilities {
    pub health: f32,
    pub sick: f32,
    pub tb: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TopPrediction {
    pub rank: u8,
    pub class: String,
    pub class_index: usize,
    pub confidence: f32,
    pub confidence_percentage: String,
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "medical-image-classifier",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Prediction endpoint - accepts image upload
async fn predict_endpoint(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<PredictionResponse>, (StatusCode, String)> {
    // Extract image from multipart form
    let mut image_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read field: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();

        if name == "image" || name == "file" {
            let data = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read bytes: {}", e)))?;
            image_data = Some(data.to_vec());
            break;
        }
    }

    let image_bytes = image_data
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "No image field found".to_string()))?;

    // Load image
    let img = ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid image format: {}", e)))?
        .decode()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to decode image: {}", e)))?;

    info!(
        "Received image for prediction: {}x{}",
        img.width(),
        img.height()
    );

    // Run prediction
    let mut session = state.session.lock().await;
    let outputs = predict(&mut session, &img)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Prediction failed: {}", e)))?;

    // Convert to response format
    let response = PredictionResponse {
        success: true,
        prediction: PredictionResult {
            class: outputs.class_name().to_string(),
            class_index: outputs.predicted_class,
            confidence: outputs.confidence,
            confidence_percentage: format!("{:.2}%", outputs.confidence * 100.0),
            confidence_level: outputs.confidence_level_text().to_string(),
            entropy: outputs.entropy,
            probabilities: ClassProbabilities {
                health: outputs.probabilities[0],
                sick: outputs.probabilities[1],
                tb: outputs.probabilities[2],
            },
            top_2_predictions: outputs
                .top_2_classes
                .iter()
                .zip(outputs.top_2_scores.iter())
                .enumerate()
                .map(|(i, (&class_idx, &score))| TopPrediction {
                    rank: (i + 1) as u8,
                    class: CLASS_NAMES[class_idx].to_string(),
                    class_index: class_idx,
                    confidence: score,
                    confidence_percentage: format!("{:.2}%", score * 100.0),
                })
                .collect(),
            recommendation: outputs.recommendation().to_string(),
        },
    };

    Ok(Json(response))
}

/// Create and configure the API router
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/predict", post(predict_endpoint))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the API server
pub async fn serve(model_path: PathBuf, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("🔄 Loading ONNX model from: {}", model_path.display());

    // Initialize ONNX Runtime environment
    let environment = Environment::builder()
        .with_name("medical_classifier_api")
        .build()?
        .into_arc();

    // Load the model
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(&model_path)?;

    info!("✅ Model loaded successfully");

    // Print model info
    info!("📋 Model inputs: {}", session.inputs.len());
    info!("📋 Model outputs: {}", session.outputs.len());

    // Create application state
    let state = Arc::new(AppState {
        session: Arc::new(Mutex::new(session)),
    });

    // Build router
    let app = create_router(state);

    // Create address
    let addr = format!("0.0.0.0:{}", port);
    info!("🚀 Starting server on http://{}", addr);
    info!("📡 Health check: http://localhost:{}/health", port);
    info!("📡 Prediction endpoint: POST http://localhost:{}/predict", port);

    // Start server
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
