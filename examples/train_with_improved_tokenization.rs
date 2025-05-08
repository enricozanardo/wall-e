use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use wall_e1::training::enhanced_trainer::EnhancedTrainer;
use wall_e1::training::curriculum::CurriculumScheduler;
use wall_e1::training::generation::TextGenerator;
use wall_e1::tokenizer::{WordPieceBPETokenizer, Tokenizer};
use serde_json::Value;
use ndarray::Array2;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::env;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    println!("===== IMPROVED TRAINING WITH TOKEN-LEVEL PROCESSING =====");
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mut fast_mode = false;
    let mut max_stories = 5000;
    let mut epochs = 10;
    let mut use_curriculum = true;
    let mut vocab_size = 5000;
    let mut model_dim = 128;
    let mut ff_dim = 256;
    let mut num_heads = 8;
    let mut num_layers = 6;
    
    // Process command line arguments
    for i in 1..args.len() {
        match args[i].as_str() {
            "--fast" => {
                fast_mode = true;
                max_stories = 5000;
                epochs = 3;
                vocab_size = 2000;
                model_dim = 128;
                ff_dim = 128;
                num_heads = 4;
                num_layers = 3;
                println!("Fast mode enabled: using reduced parameters for quick testing");
            },
            "--no-curriculum" => {
                use_curriculum = false;
                println!("Curriculum learning disabled");
            },
            "--stories" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        max_stories = val;
                    }
                }
            },
            "--epochs" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        epochs = val;
                    }
                }
            },
            "--vocab-size" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        vocab_size = val;
                    }
                }
            },
            _ => {}
        }
    }
    
    // Display parameters
    println!("Training parameters:");
    println!("  - Fast mode: {}", fast_mode);
    println!("  - Max stories: {}", max_stories);
    println!("  - Epochs: {}", epochs);
    println!("  - Vocabulary size: {}", vocab_size);
    println!("  - Model dimension: {}", model_dim);
    println!("  - Feed-forward dimension: {}", ff_dim);
    println!("  - Num heads: {}", num_heads);
    println!("  - Num layers: {}", num_layers);
    println!("  - Curriculum learning: {}", use_curriculum);
    
    // Load the dataset
    println!("\nLoading dataset from 'data/tiny_stories_sample_updated.json'...");
    let dataset_path = "data/tiny_stories_sample_updated.json";
    
    if !Path::new(dataset_path).exists() {
        eprintln!("Error: Dataset not found!");
        println!("Please run update_tiny_stories.py first to prepare the dataset.");
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
    
    // Setup progress bar for loading stories
    let progress_bar = ProgressBar::new(stories_array.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    progress_bar.set_message("Loading stories");
    
    // Select stories (randomized subset)
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..stories_array.len()).collect();
    indices.shuffle(&mut rng);
    
    let max_stories = std::cmp::min(stories_array.len(), max_stories);
    let selected_indices = indices.into_iter().take(max_stories).collect::<Vec<_>>();
    
    // Load stories
    let mut training_texts = Vec::new();
    for &idx in &selected_indices {
        let story = stories_array[idx].as_str().unwrap().to_string();
        training_texts.push(story);
        progress_bar.inc(1);
    }
    progress_bar.finish_with_message(format!("Loaded {} stories", training_texts.len()));
    
    // Create enhanced trainer
    println!("\nInitializing enhanced trainer...");
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        0.15, // dropout
        0.0015 // learning rate
    );
    
    // Configure the trainer
    trainer = trainer
        .with_curriculum_learning(use_curriculum)
        .with_dynamic_learning_rate(true)
        .with_gradient_clipping(Some(1.0));

    // Configure text generator with stronger anti-repetition settings
    trainer.configure_anti_repetition(1.5, 0.3, 0.3);
    
    // Configure text generator for better text generation
    let generator = TextGenerator::new()
        .with_repetition_penalty(1.3)
        .with_presence_penalty(0.2)
        .with_frequency_penalty(0.2)
        .with_temperature(0.8)
        .with_dynamic_temperature(true)
        .with_entropy_threshold(1.0);
    
    trainer = trainer.with_text_generator(generator);
    
    // Create the curriculum scheduler
    if use_curriculum {
        let scheduler = CurriculumScheduler::new()
            .with_epochs_per_level(2)
            .with_harder_examples_ratio(0.2)
            .with_easier_examples_ratio(0.1);
        
        trainer = trainer.with_curriculum_scheduler(scheduler);
    }
    
    // Create the corpus for tokenizer training
    println!("\nBuilding tokenizer from text corpus...");
    let corpus = {
        progress_bar.set_length(training_texts.len() as u64);
        progress_bar.set_position(0);
        progress_bar.set_message("Processing corpus");
        
        // Join all texts with space separation
        let corpus_chunks: Vec<String> = training_texts.iter()
            .map(|text| {
                progress_bar.inc(1);
                text.clone() + " "
            })
            .collect();
        
        corpus_chunks.join("")
    };
    progress_bar.finish_with_message("Corpus built");
    
    // Train the tokenizer
    println!("Learning tokenizer vocabulary (size: {})...", vocab_size);
    trainer.learn_tokenizer_from_text(&corpus, vocab_size, 2);
    println!("Tokenizer vocabulary built with {} tokens", trainer.get_tokenizer().get_vocab().len());
    
    // Debug tokenization of a few sample texts
    println!("\nTesting tokenization with the learned vocabulary:");
    trainer.debug_tokenize("Hello, world! How are you?");
    trainer.debug_tokenize("Once upon a time, there was a little dog.");
    trainer.debug_tokenize("The quick brown fox jumps over the lazy dog.");
    
    // Prepare training examples
    println!("\nPreparing training examples...");
    let examples_result = prepare_training_examples(&training_texts, trainer.get_tokenizer(), 64);
    let (training_examples, training_targets) = examples_result;
    
    println!("Generated {} training examples", training_examples.len());
    
    // Create batches
    println!("Creating batches...");
    let (batched_examples, batched_targets) = create_batches(&training_examples, &training_targets, 32);
    println!("Created {} batches", batched_examples.len());
    
    // Add examples to curriculum
    if use_curriculum {
        println!("Organizing examples for curriculum learning...");
        for (batch, targets) in batched_examples.iter().zip(batched_targets.iter()) {
            trainer.add_examples_to_curriculum(batch, targets);
        }
    }
    
    // Start training
    println!("\n===== STARTING TRAINING =====");
    let training_start = Instant::now();
    
    // Create example prompts for text generation
    let prompts = [
        "Once upon a time,",
        "The little dog",
        "In the garden,",
        "Today I will",
    ];
    
    // Train for specified number of epochs
    for epoch in 1..=epochs {
        println!("\nEpoch {}/{}", epoch, epochs);
        let epoch_start = Instant::now();
        
        // Train for one epoch
        let avg_loss = if use_curriculum {
            // Use curriculum learning
            trainer.train_epoch(&[], &[])
        } else {
            // Use standard training
            trainer.train_epoch(&batched_examples, &batched_targets)
        };
        
        let epoch_duration = epoch_start.elapsed();
        println!("Epoch completed in {:?}", epoch_duration);
        println!("Average loss: {:.6}", avg_loss);
        
        // Generate sample text
        println!("\nGenerating sample text (Epoch {}):", epoch);
        for prompt in &prompts {
            let generated = trainer.generate_text(prompt, Some(50));
            println!("Prompt: '{}'", prompt);
            println!("Generated: '{}'", generated);
        }
        
        // Save checkpoint
        let checkpoint_path = format!("models/improved_model_epoch_{}.bin", epoch);
        println!("Saving checkpoint to '{}'...", checkpoint_path);
        trainer.save_model(&checkpoint_path)?;
    }
    
    // Save final model
    println!("\nTraining completed in {:?}", training_start.elapsed());
    let model_path = "models/improved_model.bin";
    println!("Saving final model to '{}'...", model_path);
    trainer.save_model(model_path)?;
    
    println!("\n===== TRAINING COMPLETE =====");
    println!("Final model saved to: {}", model_path);
    
    Ok(())
}

// Function to prepare training examples from stories
fn prepare_training_examples(
    stories: &[String],
    tokenizer: &WordPieceBPETokenizer,
    max_seq_len: usize
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    let progress_bar = ProgressBar::new(stories.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    progress_bar.set_message("Processing stories");
    
    // Process each story
    for story in stories {
        let tokens = tokenizer.encode(story);
        
        // Skip very short sequences
        if tokens.len() < 5 {
            progress_bar.inc(1);
            continue;
        }
        
        // Process overlapping windows of tokens
        let window_size = std::cmp::min(max_seq_len, 64);
        let stride = 16; // Overlap between windows
        
        let mut pos = 0;
        while pos + 2 <= tokens.len() {
            let end = std::cmp::min(pos + window_size, tokens.len());
            if end - pos < 3 {
                break; // Too short to be useful
            }
            
            // Input: all tokens except the last one
            let input = tokens[pos..end - 1].to_vec();
            
            // Target: all tokens except the first one
            let target = tokens[pos + 1..end].to_vec();
            
            inputs.push(input);
            targets.push(target);
            
            pos += stride;
        }
        
        progress_bar.inc(1);
    }
    
    progress_bar.finish_with_message("Examples prepared");
    
    (inputs, targets)
}

// Function to create batches
fn create_batches(
    inputs: &[Vec<usize>],
    targets: &[Vec<usize>],
    batch_size: usize
) -> (Vec<Vec<Vec<usize>>>, Vec<Array2<usize>>) {
    let mut batched_inputs = Vec::new();
    let mut batched_targets = Vec::new();
    
    // Shuffle the inputs and targets together
    let mut indices: Vec<usize> = (0..inputs.len()).collect();
    indices.shuffle(&mut rand::thread_rng());
    
    // Create batches
    for chunk in indices.chunks(batch_size) {
        let mut batch_inputs = Vec::new();
        
        // Find max length in this batch for padding
        let mut max_target_len = 0;
        for &idx in chunk {
            max_target_len = std::cmp::max(max_target_len, targets[idx].len());
        }
        
        // Create batch input and target arrays
        let batch_size = chunk.len();
        let mut batch_targets = Array2::zeros((batch_size, max_target_len));
        
        // Add each example to batch
        for (i, &idx) in chunk.iter().enumerate() {
            batch_inputs.push(inputs[idx].clone());
            
            // Set target tokens
            for (j, &token) in targets[idx].iter().enumerate() {
                batch_targets[[i, j]] = token;
            }
        }
        
        // Add batch to results
        batched_inputs.push(batch_inputs);
        batched_targets.push(batch_targets);
    }
    
    (batched_inputs, batched_targets)
} 