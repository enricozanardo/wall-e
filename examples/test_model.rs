use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use wall_e1::training::enhanced_trainer::EnhancedTrainer;
use wall_e1::tokenizer::{WordPieceBPETokenizer, Tokenizer};
use serde_json::Value;
use ndarray::Array2;
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;

// Try to save the model to allow loading later
fn add_load_method_to_trainer() -> Result<(), Box<dyn Error>> {
    println!("Loading model is not directly supported in the current version");
    println!("We'll build a model with similar structure instead");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("===== TESTING WALL-E1 MODEL PERFORMANCE =====");
    
    // This script uses a freshly initialized model with the same structure 
    // as our trained model, since model loading is not fully implemented
    println!("NOTE: Using a new model with the same structure as the trained model");
    println!("This demonstrates basic functionality but not trained performance");
    
    // Initialize default parameters - these should match the trained model
    let model_dim = 256;
    let ff_dim = 512;
    let num_heads = 8;
    let num_layers = 6;
    
    // Create base trainer
    println!("\nInitializing model structure...");
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        0.0, // No dropout for evaluation
        0.0  // Learning rate doesn't matter for evaluation
    );
    
    // Load dataset for tokenizer training
    let dataset_path = "data/tiny_stories_sample_updated.json";
    if !Path::new(dataset_path).exists() {
        eprintln!("Error: Dataset not found!");
        return Err("Dataset not found".into());
    }
    
    // Read the file
    let mut file = File::open(dataset_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    // Parse JSON
    let data: Value = serde_json::from_str(&contents)?;
    
    // Extract stories
    let stories = &data["stories"];
    let stories_array = stories.as_array().unwrap();
    
    // Create a corpus for the tokenizer
    println!("Building corpus for tokenizer...");
    let mut corpus = String::new();
    for story in stories_array.iter().take(500) {
        corpus.push_str(story.as_str().unwrap());
        corpus.push(' ');
    }
    
    // Train the tokenizer
    println!("Learning tokenizer vocabulary...");
    trainer.learn_tokenizer_from_text(&corpus, 6000, 2);
    println!("Tokenizer vocabulary built with {} tokens", trainer.get_tokenizer().get_vocab().len());
    
    // Configure for text generation with strong anti-repetition settings
    trainer.configure_anti_repetition(1.8, 0.5, 0.5);
    
    // Test prompts for generation
    let test_prompts = [
        "Once upon a time,",
        "The little dog",
        "In the garden,",
        "Today I will",
        "The princess",
        "My favorite toy",
    ];
    
    // Generate text for each prompt
    println!("\n===== TEXT GENERATION TESTS =====");
    for prompt in &test_prompts {
        println!("\nPrompt: '{}'", prompt);
        
        let generation_start = Instant::now();
        let generated = trainer.generate_text(prompt, Some(50));
        let generation_time = generation_start.elapsed();
        
        println!("Generated: '{}'", generated);
        println!("Generation time: {:?}", generation_time);
        
        // Check content quality
        if generated.starts_with(prompt) {
            let prompt_words = prompt.split_whitespace().count();
            let generated_words = generated.split_whitespace().count();
            let additional_words = generated_words.saturating_sub(prompt_words);
            
            println!("Added {} new words", additional_words);
            
            // Count repeated words as a simple quality metric
            let words: Vec<&str> = generated.split_whitespace().collect();
            let mut repeated_words = 0;
            
            if words.len() > 1 {
                for i in 1..words.len() {
                    if words[i] == words[i-1] {
                        repeated_words += 1;
                    }
                }
            }
            
            println!("Repeated words: {}", repeated_words);
            
            // Rough quality score (higher is better)
            let quality_score = if additional_words == 0 {
                0.0
            } else {
                let uniqueness = 1.0 - (repeated_words as f32 / additional_words as f32).min(1.0);
                let length_factor = (additional_words as f32 / 50.0).min(1.0);
                uniqueness * length_factor * 100.0
            };
            
            println!("Quality score: {:.1}%", quality_score);
        } else {
            println!("Generated text doesn't start with prompt!");
        }
    }
    
    println!("\n===== TESTING COMPLETE =====");
    println!("This test used an untrained model. For real performance:");
    println!("1. Run ./run_full_training.sh to train a full model");
    println!("2. Edit this script to properly load model weights");
    
    Ok(())
} 