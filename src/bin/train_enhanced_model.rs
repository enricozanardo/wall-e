use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::collections::HashMap;
use std::time::{Instant, Duration};
use wall_e1::tokenizer::{WordPieceBPETokenizer, Tokenizer};
use wall_e1::training::curriculum::{CurriculumScheduler, DifficultyLevel};
use wall_e1::training::generation::TextGenerator;
use wall_e1::training::enhanced_trainer::EnhancedTrainer;
use ndarray;
use rand::prelude::*;
use serde_json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <training_data_path> [options]", args[0]);
        println!("Options:");
        println!("  --model-dim <dim>      Model dimension (default: 192)");
        println!("  --ff-dim <dim>         Feed forward dimension (default: 768)");
        println!("  --heads <num>          Number of attention heads (default: 6)");
        println!("  --layers <num>         Number of layers (default: 4)");
        println!("  --dropout <rate>       Dropout rate (default: 0.1)");
        println!("  --learning-rate <rate> Learning rate (default: 0.0005)");
        println!("  --epochs <num>         Number of epochs (default: 10)");
        println!("  --no-curriculum        Disable curriculum learning");
        println!("  --vocab-size <size>    Vocabulary size (default: 5000)");
        println!("  --min-freq <freq>      Min token frequency (default: 2)");
        println!("  --save-path <path>     Model save path (default: model.json)");
        println!("  --enable-skip          Enable skip connections (residual)");
        println!("  --strong-anti-rep      Enable stronger anti-repetition");
        println!("  --json-format          Process input as TinyStories JSON format");
        println!("  --stories <num>        Maximum number of stories to use from JSON (default: 8000)");
        return Ok(());
    }
    
    // Read configuration from command line
    let training_data_path = &args[1];
    
    // Default parameters
    let mut model_dim = 192;
    let mut ff_dim = 768;
    let mut num_heads = 6;
    let mut num_layers = 4;
    let mut dropout_rate = 0.1;
    let mut learning_rate = 0.0005;
    let mut epochs = 10;
    let mut use_curriculum = true;
    let mut vocab_size = 5000;
    let mut min_frequency = 2;
    let mut save_path = "model.json".to_string();
    let mut enable_skip = false;
    let mut strong_anti_rep = false;
    let mut json_format = false;
    let mut max_stories = 8000;
    
    // Parse additional arguments
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dim" => {
                if i + 1 < args.len() {
                    model_dim = args[i + 1].parse().unwrap_or(192);
                    i += 2;
                } else { i += 1; }
            },
            "--ff-dim" => {
                if i + 1 < args.len() {
                    ff_dim = args[i + 1].parse().unwrap_or(768);
                    i += 2;
                } else { i += 1; }
            },
            "--heads" => {
                if i + 1 < args.len() {
                    num_heads = args[i + 1].parse().unwrap_or(6);
                    i += 2;
                } else { i += 1; }
            },
            "--layers" => {
                if i + 1 < args.len() {
                    num_layers = args[i + 1].parse().unwrap_or(4);
                    i += 2;
                } else { i += 1; }
            },
            "--dropout" => {
                if i + 1 < args.len() {
                    dropout_rate = args[i + 1].parse().unwrap_or(0.1);
                    i += 2;
                } else { i += 1; }
            },
            "--learning-rate" => {
                if i + 1 < args.len() {
                    learning_rate = args[i + 1].parse().unwrap_or(0.0005);
                    i += 2;
                } else { i += 1; }
            },
            "--epochs" => {
                if i + 1 < args.len() {
                    epochs = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                } else { i += 1; }
            },
            "--no-curriculum" => {
                use_curriculum = false;
                i += 1;
            },
            "--vocab-size" => {
                if i + 1 < args.len() {
                    vocab_size = args[i + 1].parse().unwrap_or(5000);
                    i += 2;
                } else { i += 1; }
            },
            "--min-freq" => {
                if i + 1 < args.len() {
                    min_frequency = args[i + 1].parse().unwrap_or(2);
                    i += 2;
                } else { i += 1; }
            },
            "--save-path" => {
                if i + 1 < args.len() {
                    save_path = args[i + 1].clone();
                    i += 2;
                } else { i += 1; }
            },
            "--enable-skip" => {
                enable_skip = true;
                i += 1;
            },
            "--strong-anti-rep" => {
                strong_anti_rep = true;
                i += 1;
            },
            "--json-format" => {
                json_format = true;
                i += 1;
            },
            "--stories" => {
                if i + 1 < args.len() {
                    max_stories = args[i + 1].parse().unwrap_or(8000);
                    i += 2;
                } else { i += 1; }
            },
            _ => {
                println!("Unknown option: {}", args[i]);
                i += 1;
            }
        }
    }
    
    // Display configuration
    println!("Training Configuration:");
    println!("  Training data: {}", training_data_path);
    println!("  Data format: {}", if json_format { "TinyStories JSON" } else { "Plain text" });
    if json_format {
        println!("  Max stories: {}", max_stories);
    }
    println!("  Model dimension: {}", model_dim);
    println!("  FF dimension: {}", ff_dim);
    println!("  Attention heads: {}", num_heads);
    println!("  Layers: {}", num_layers);
    println!("  Dropout rate: {}", dropout_rate);
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", epochs);
    println!("  Curriculum learning: {}", if use_curriculum { "enabled" } else { "disabled" });
    println!("  Vocabulary size: {}", vocab_size);
    println!("  Min token frequency: {}", min_frequency);
    println!("  Skip connections: {}", if enable_skip { "enabled" } else { "disabled" });
    println!("  Strong anti-repetition: {}", if strong_anti_rep { "enabled" } else { "disabled" });
    println!("  Save path: {}", save_path);
    
    // Read training data
    println!("Reading training data...");
    let training_text = if json_format {
        // Process TinyStories JSON format
        process_json_data(training_data_path, max_stories)?
    } else {
        // Process plain text format
        let mut file = File::open(training_data_path)?;
        let mut training_text = String::new();
        file.read_to_string(&mut training_text)?;
        training_text
    };
    
    // Create enhanced trainer
    println!("Creating trainer...");
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    ).with_curriculum_learning(use_curriculum)
     .with_dynamic_learning_rate(true)
     .with_gradient_clipping(Some(1.0));
    
    // Configure anti-repetition
    if strong_anti_rep {
        println!("Configuring strong anti-repetition mechanisms...");
        trainer.configure_advanced_anti_repetition(1.3, 0.7, 0.7, 0.8);
    } else {
        trainer.configure_anti_repetition(1.1, 0.2, 0.3);
    }
    
    // Enable skip connections if requested
    if enable_skip {
        println!("Enabling skip connections...");
        if let Err(e) = trainer.enable_skip_connections("residual") {
            println!("Warning: Failed to enable skip connections: {}", e);
        }
    }
    
    // Learn tokenizer vocabulary
    println!("Learning tokenizer vocabulary...");
    trainer.learn_tokenizer_from_text(&training_text, vocab_size, min_frequency);
    
    // Split data for training and validation (90/10 split)
    let total_length = training_text.len();
    let train_length = (total_length as f64 * 0.9) as usize;
    let training_text_subset = &training_text[..train_length];
    let validation_text = &training_text[train_length..];
    
    // Create curriculum scheduler with training data
    println!("Setting up curriculum learning...");
    let tokenizer = trainer.get_tokenizer();
    
    // Tokenize training and validation data
    let training_tokens = tokenizer.encode(training_text_subset);
    let validation_tokens = tokenizer.encode(validation_text);
    
    // Set up evaluation prompts
    let eval_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a world where",
        "The most important thing",
        "I would like to",
    ];
    
    // Prepare validation data
    let mut validation_inputs = Vec::new();
    let mut validation_targets = Vec::new();
    prepare_validation_data(&validation_tokens, &mut validation_inputs, &mut validation_targets);
    
    // Start training
    println!("Starting training for {} epochs...", epochs);
    let start_time = Instant::now();
    
    let mut metrics_history: Vec<HashMap<String, f32>> = Vec::new();
    
    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch + 1, epochs);
        let epoch_start = Instant::now();
        
        // Train on the tokenized data
        let loss = train_epoch(&mut trainer, &training_tokens, epoch);
        let epoch_duration = epoch_start.elapsed();
        
        // Evaluate the model
        println!("Evaluating model...");
        let metrics = trainer.evaluate_model(&validation_inputs, &validation_targets, &eval_prompts);
        metrics_history.push(metrics.clone());
        
        println!("Epoch {}/{} completed in {:?}", epoch + 1, epochs, epoch_duration);
        println!("  Loss: {:.6}", loss);
        println!("  Perplexity: {:.2}", metrics.get("perplexity").unwrap_or(&f32::INFINITY));
        println!("  Accuracy: {:.2}%", metrics.get("accuracy").unwrap_or(&0.0));
        println!("  Repetition score: {:.2}", metrics.get("repetition_score").unwrap_or(&0.0));
        println!("  Fluency score: {:.2}", metrics.get("fluency_score").unwrap_or(&0.0));
        println!("  Quality score: {:.2}", metrics.get("quality_score").unwrap_or(&0.0));
        
        // Generate sample text
        let prompt = "The";
        let generated = trainer.generate_text(prompt, Some(50));
        println!("\nSample generation:");
        println!("Prompt: \"{}\"", prompt);
        println!("Generated: \"{}\"", generated);
        println!();
        
        // Check for repetition patterns
        let (has_repetition, patterns) = trainer.detect_repetition_patterns(&generated);
        if has_repetition {
            println!("Detected repetition patterns:");
            for pattern in patterns {
                println!("  - \"{}\"", pattern);
            }
        }
        
        // Save model checkpoint
        if epoch % 2 == 0 || epoch == epochs - 1 {
            let checkpoint_path = format!("{}.epoch{}", save_path, epoch + 1);
            println!("Saving checkpoint to {}", checkpoint_path);
            match trainer.save_model(&checkpoint_path) {
                Ok(_) => println!("Checkpoint saved successfully"),
                Err(e) => println!("Failed to save checkpoint: {}", e),
            }
        }
    }
    
    // Training completed
    let total_duration = start_time.elapsed();
    println!("Training completed in {:?}", total_duration);
    
    // Save final model
    println!("Saving final model to {}", save_path);
    trainer.save_model(&save_path)?;
    
    // Print final metrics
    if !metrics_history.is_empty() {
        let final_metrics = &metrics_history[metrics_history.len() - 1];
        println!("Final model metrics:");
        println!("  Perplexity: {:.2}", final_metrics.get("perplexity").unwrap_or(&f32::INFINITY));
        println!("  Accuracy: {:.2}%", final_metrics.get("accuracy").unwrap_or(&0.0));
        println!("  Repetition score: {:.2}", final_metrics.get("repetition_score").unwrap_or(&0.0));
        println!("  Fluency score: {:.2}", final_metrics.get("fluency_score").unwrap_or(&0.0));
        println!("  Quality score: {:.2}", final_metrics.get("quality_score").unwrap_or(&0.0));
    }
    
    Ok(())
}

// Process TinyStories JSON format data
fn process_json_data(file_path: &str, max_stories: usize) -> Result<String, Box<dyn std::error::Error>> {
    // Read the file
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    // Parse JSON
    let data: serde_json::Value = serde_json::from_str(&contents)?;
    
    // Extract stories array
    let stories = match &data["stories"] {
        serde_json::Value::Array(arr) => arr,
        _ => {
            println!("Warning: Expected 'stories' array in JSON, not found. Trying direct parse...");
            match &data {
                serde_json::Value::Array(arr) => arr,
                _ => return Err("Could not find stories array in JSON file".into()),
            }
        }
    };
    
    println!("Found {} stories in JSON file", stories.len());
    let num_stories = std::cmp::min(stories.len(), max_stories);
    println!("Will use {} stories for training", num_stories);
    
    // Randomly sample stories if we have more than we need
    let selected_indices: Vec<usize> = if stories.len() > max_stories {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..stories.len()).collect();
        indices.shuffle(&mut rng);
        indices.into_iter().take(max_stories).collect()
    } else {
        (0..stories.len()).collect()
    };
    
    // Combine stories into a single text
    let mut combined_text = String::new();
    for &idx in &selected_indices {
        let story = match &stories[idx] {
            serde_json::Value::String(s) => s.clone(),
            _ => {
                // Try to extract text from a structured story object
                match &stories[idx]["text"] {
                    serde_json::Value::String(s) => s.clone(),
                    _ => match &stories[idx]["story"] {
                        serde_json::Value::String(s) => s.clone(),
                        _ => {
                            println!("Warning: Could not extract text from story at index {}", idx);
                            continue;
                        }
                    }
                }
            }
        };
        
        combined_text.push_str(&story);
        combined_text.push_str("\n\n"); // Add some separation between stories
    }
    
    println!("Processed {} stories, total text length: {} characters", selected_indices.len(), combined_text.len());
    
    Ok(combined_text)
}

// Train for a single epoch on tokenized data
fn train_epoch(trainer: &mut EnhancedTrainer, tokens: &Vec<usize>, epoch: usize) -> f32 {
    // Create sliding windows of input/target pairs
    let max_sequence_length = trainer.get_max_seq_len();
    let stride = max_sequence_length / 2; // 50% overlap between windows
    
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for i in (0..tokens.len().saturating_sub(max_sequence_length)).step_by(stride) {
        if i + max_sequence_length <= tokens.len() {
            let input = tokens[i..i + max_sequence_length].to_vec();
            inputs.push(input);
            
            // For each sequence, we create a corresponding target sequence
            // The target is the input shifted one position to the right (next token prediction)
            let mut target = Vec::with_capacity(max_sequence_length);
            
            // We use tokens from i+1 to i+max_sequence_length+1 as targets
            // If we reach the end of the tokens, we wrap around to the beginning
            for j in 0..max_sequence_length {
                let target_idx = (i + j + 1) % tokens.len();
                target.push(tokens[target_idx]);
            }
            
            targets.push(target);
        }
    }
    
    println!("Created {} input/target pairs for training", inputs.len());
    
    // Process in batches
    let batch_size = trainer.get_batch_size();
    let mut batched_inputs = Vec::new();
    let mut batched_targets = Vec::new();
    
    for batch_start in (0..inputs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(inputs.len());
        let batch_inputs = inputs[batch_start..batch_end].to_vec();
        let batch_targets = targets[batch_start..batch_end].to_vec();
        
        // Convert batch targets to ndarray
        let mut targets_array = ndarray::Array2::zeros((batch_targets.len(), max_sequence_length));
        for (i, target) in batch_targets.iter().enumerate() {
            for (j, &token) in target.iter().enumerate() {
                targets_array[[i, j]] = token;
            }
        }
        
        batched_inputs.push(batch_inputs);
        batched_targets.push(targets_array);
    }
    
    // Train on batches
    trainer.train_epoch(&batched_inputs, &batched_targets)
}

// Prepare validation data for model evaluation
fn prepare_validation_data(tokens: &Vec<usize>, inputs: &mut Vec<Vec<usize>>, targets: &mut Vec<Vec<usize>>) {
    // Create context windows of varying lengths for validation
    for window_size in [16, 32, 64].iter() {
        for i in (0..tokens.len().saturating_sub(*window_size)).step_by(*window_size) {
            if i + *window_size + 1 <= tokens.len() {
                // Input: tokens[i..i+window_size]
                let input = tokens[i..i + *window_size].to_vec();
                // Target: tokens[i+1..i+window_size+1] (shifted by 1)
                let target = tokens[i + 1..i + *window_size + 1].to_vec();
                
                inputs.push(input);
                targets.push(target);
            }
        }
    }
} 