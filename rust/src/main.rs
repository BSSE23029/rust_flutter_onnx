mod api;

use clap::{Parser, Subcommand};
use colored::*;
use image::GenericImageView;
use image_classifier::{predict, ModelOutputs, CLASS_EMOJIS, CLASS_NAMES};
use ort::{session::SessionBuilder, Environment, GraphOptimizationLevel, Session};
use std::error::Error;
use std::path::PathBuf;

/// Medical Image Classifier - X-ray classification using ONNX
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run CLI prediction on a single image
    Predict {
        /// Path to the image file to classify
        #[arg(value_name = "IMAGE")]
        image_path: PathBuf,

        /// Path to the ONNX model file
        #[arg(short, long, default_value = "../model/unified_model.onnx")]
        model: PathBuf,

        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Start the REST API server
    Serve {
        /// Path to the ONNX model file
        #[arg(short, long, default_value = "../model/unified_model.onnx")]
        model: PathBuf,

        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match args.command {
        Commands::Predict {
            image_path,
            model,
            verbose,
        } => run_cli(image_path, model, verbose).await,
        Commands::Serve { model, port } => {
            api::serve(model, port).await?;
            Ok(())
        }
    }
}

async fn run_cli(
    image_path: PathBuf,
    model_path: PathBuf,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    // Initialize ONNX Runtime environment
    let environment = Environment::builder()
        .with_name("medical_classifier")
        .build()?
        .into_arc();

    // Load the model
    println!("\n{}", "🔄 Loading ONNX model...".cyan());
    let mut session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(&model_path)?;
    println!("{}", "✅ Model loaded successfully".green());

    if verbose {
        print_model_info(&session);
    }

    // Load and process image
    println!(
        "\n{}",
        format!("🔄 Processing image: {}", image_path.display()).cyan()
    );
    let img = image::open(&image_path)?;
    let (width, height) = img.dimensions();
    println!(
        "{}",
        format!("   Original size: {}x{}", width, height).bright_black()
    );

    // Run inference
    let outputs = predict(&mut session, &img)?;

    // Display results
    display_results(&outputs);

    Ok(())
}

/// Print model information
fn print_model_info(session: &Session) {
    println!("\n{}", "📋 Model Information:".cyan());
    println!(
        "   Inputs ({} total):",
        session.inputs.len().to_string().bold()
    );
    for input in &session.inputs {
        println!(
            "     - {}: {:?}",
            input.name.bright_white().bold(),
            input.dimensions
        );
    }
    println!(
        "   Outputs ({} total):",
        session.outputs.len().to_string().bold()
    );
    for output in &session.outputs {
        println!(
            "     - {}: {:?}",
            output.name.bright_white().bold(),
            output.dimensions
        );
    }
}

/// Display flashy results
fn display_results(outputs: &ModelOutputs) {
    let separator = "=".repeat(70);
    let sub_separator = "-".repeat(70);

    println!("\n{}", separator.bright_black());
    println!(
        "{}",
        "🏥 MEDICAL IMAGE CLASSIFICATION RESULT"
            .bold()
            .bright_cyan()
    );
    println!("{}", separator.bright_black());

    // Primary prediction
    println!("\n{}", sub_separator.bright_black());
    println!("{}", "🎯 PRIMARY PREDICTION".bold().yellow());
    println!("{}", sub_separator.bright_black());

    let emoji = outputs.class_emoji();
    let class_name = outputs.class_name().to_uppercase();
    let confidence_color = if outputs.confidence > 0.9 {
        "green"
    } else if outputs.confidence > 0.7 {
        "yellow"
    } else {
        "red"
    };

    println!(
        "  {}  Class: {}",
        emoji,
        class_name.color(confidence_color).bold()
    );
    println!(
        "  📊  Confidence: {}",
        format!("{:.2}%", outputs.confidence * 100.0)
            .color(confidence_color)
            .bold()
    );

    let level_emoji = outputs.confidence_level_emoji();
    let level_text = outputs.confidence_level_text();
    println!(
        "  🎚️   Confidence Level: {} {}",
        level_emoji,
        level_text.color(confidence_color).bold()
    );

    println!("  📉  Uncertainty (Entropy): {:.6}", outputs.entropy);

    // Top 2 predictions
    println!("\n{}", sub_separator.bright_black());
    println!("{}", "📊 TOP 2 PREDICTIONS".bold().yellow());
    println!("{}", sub_separator.bright_black());

    for (i, (&class_idx, &score)) in outputs
        .top_2_classes
        .iter()
        .zip(outputs.top_2_scores.iter())
        .enumerate()
    {
        let rank = if i == 0 { "🥇" } else { "🥈" };
        let emoji = CLASS_EMOJIS[class_idx];
        let class_name = CLASS_NAMES[class_idx].to_uppercase();
        println!(
            "  {} {}  {}: {:.2}%",
            rank,
            emoji,
            class_name,
            score * 100.0
        );
    }

    // All probabilities
    println!("\n{}", sub_separator.bright_black());
    println!("{}", "📈 ALL CLASS PROBABILITIES".bold().yellow());
    println!("{}", sub_separator.bright_black());

    for (idx, &prob) in outputs.probabilities.iter().enumerate() {
        let emoji = CLASS_EMOJIS[idx];
        let class_name = CLASS_NAMES[idx].to_uppercase();
        let bar_length = (prob * 40.0) as usize;
        let bar = "█".repeat(bar_length);
        let empty = "░".repeat(40 - bar_length);
        println!(
            "  {}  {:8} {} {:>6.2}%",
            emoji,
            class_name,
            format!("{}{}", bar, empty).bright_black(),
            prob * 100.0
        );
    }

    // Interpretation
    println!("\n{}", sub_separator.bright_black());
    println!("{}", "📋 INTERPRETATION".bold().yellow());
    println!("{}", sub_separator.bright_black());

    match outputs.confidence_level {
        2 => println!(
            "  {} High confidence prediction - Result is reliable",
            "✅".green()
        ),
        1 => println!(
            "  {} Medium confidence - Consider additional verification",
            "⚠️ ".yellow()
        ),
        _ => println!("  {} Low confidence - Result may be unreliable", "🔴".red()),
    }

    if outputs.entropy < 0.1 {
        println!(
            "  {} Very low uncertainty - Model is very certain",
            "✅".green()
        );
    } else if outputs.entropy < 0.5 {
        println!(
            "  {} Low uncertainty - Model is reasonably certain",
            "✅".green()
        );
    } else {
        println!(
            "  {} High uncertainty - Model is less certain",
            "⚠️ ".yellow()
        );
    }

    // Final recommendation
    println!("\n{}", separator.bright_black());
    let rec_color = match outputs.predicted_class {
        0 => "green",
        1 => "yellow",
        2 => "red",
        _ => "white",
    };
    let rec_emoji = CLASS_EMOJIS[outputs.predicted_class];
    println!(
        "{} {}",
        rec_emoji,
        outputs.recommendation().color(rec_color).bold()
    );
    println!("{}", separator.bright_black());
}
