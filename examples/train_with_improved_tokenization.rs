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
    let mut rep_penalty = 1.8; // Default repetition penalty
    let mut pres_penalty = 0.4; // Default presence penalty
    let mut freq_penalty = 0.4; // Default frequency penalty
    let mut curriculum_step = 2; // Default epochs per curriculum level
    let mut custom_batch_size = 0; // Default to auto-determined batch size
    
    // Process command line arguments
    for i in 1..args.len() {
        match args[i].as_str() {
            "--fast" => {
                fast_mode = true;
                max_stories = 5000;
                epochs = 5;
                vocab_size = 4000;
                model_dim = 128;
                ff_dim = 256;
                num_heads = 8;
                num_layers = 4;
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
            "--model-dim" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        model_dim = val;
                    }
                }
            },
            "--ff-dim" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        ff_dim = val;
                    }
                }
            },
            "--heads" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        num_heads = val;
                    }
                }
            },
            "--layers" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        num_layers = val;
                    }
                }
            },
            "--rep-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        rep_penalty = val;
                    }
                }
            },
            "--pres-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        pres_penalty = val;
                    }
                }
            },
            "--freq-penalty" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        freq_penalty = val;
                    }
                }
            },
            "--curriculum-step" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        curriculum_step = val;
                    }
                }
            },
            "--batch-size" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        custom_batch_size = val;
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
    println!("  - Repetition penalty: {}", rep_penalty);
    println!("  - Presence penalty: {}", pres_penalty);
    println!("  - Frequency penalty: {}", freq_penalty);
    println!("  - Curriculum step: {}", curriculum_step);
    println!("  - Custom batch size: {}", if custom_batch_size > 0 { custom_batch_size.to_string() } else { "auto".to_string() });
    
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
        0.1,  // Reduced dropout for better training
        0.001 // Adjusted learning rate for better convergence
    );
    
    // Configure the trainer
    trainer = trainer
        .with_curriculum_learning(use_curriculum)
        .with_dynamic_learning_rate(true)
        .with_gradient_clipping(Some(1.0));

    // Configure text generator with stronger anti-repetition settings
    trainer.configure_anti_repetition(rep_penalty, pres_penalty, freq_penalty);
    
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
            .with_epochs_per_level(curriculum_step)
            .with_harder_examples_ratio(0.25)  // More exposure to harder examples
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
    
    // Determine appropriate batch size based on model dimension or user preference
    let batch_size = if custom_batch_size > 0 {
        custom_batch_size
    } else if model_dim <= 64 {
        32
    } else if model_dim <= 128 {
        24
    } else if model_dim <= 256 {
        16
    } else {
        8
    };
    
    // Prepare training examples
    println!("\nPreparing training examples...");
    let examples_result = prepare_training_examples(&training_texts, trainer.get_tokenizer(), 64);
    let (training_examples, training_targets) = examples_result;
    
    println!("Generated {} training examples", training_examples.len());
    
    // Create batches
    println!("Creating batches...");
    let (batched_examples, batched_targets) = create_batches(&training_examples, &training_targets, batch_size);
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
    
    // Set aside some examples for validation
    let validation_size = training_examples.len() / 10; // 10% for validation
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..training_examples.len()).collect();
    indices.shuffle(&mut rng);

    let validation_indices = indices.iter().take(validation_size).cloned().collect::<Vec<_>>();
    let training_indices = indices.iter().skip(validation_size).cloned().collect::<Vec<_>>();

    // Create validation sets
    let mut validation_inputs = Vec::new();
    let mut validation_targets = Vec::new();

    for &idx in &validation_indices {
        validation_inputs.push(training_examples[idx].clone());
        validation_targets.push(training_targets[idx].clone());
    }

    println!("Reserved {} examples for validation", validation_inputs.len());

    // Create filtered training examples
    let mut filtered_training_examples = Vec::new();
    let mut filtered_training_targets = Vec::new();

    for &idx in &training_indices {
        filtered_training_examples.push(training_examples[idx].clone());
        filtered_training_targets.push(training_targets[idx].clone());
    }

    println!("Using {} examples for training", filtered_training_examples.len());

    // Track metrics over time
    let mut epoch_metrics = Vec::new();

    // Train for specified number of epochs
    for epoch in 1..=epochs {
        println!("\nEpoch {}/{}", epoch, epochs);
        let epoch_start = Instant::now();
        
        // Train for one epoch
        let avg_loss = if use_curriculum {
            // Use curriculum learning
            trainer.train_epoch(&[], &[])
        } else {
            // Use standard training with batched examples
            trainer.train_epoch(&batched_examples, &batched_targets)
        };
        
        // Calculate validation metrics
        let perplexity = trainer.calculate_perplexity(&validation_inputs, &validation_targets);
        let accuracy = trainer.calculate_accuracy(&validation_inputs, &validation_targets);
        
        // Store metrics
        epoch_metrics.push((epoch, avg_loss, perplexity, accuracy));
        
        let epoch_duration = epoch_start.elapsed();
        println!("Epoch completed in {:?}", epoch_duration);
        println!("Average training loss: {:.6}", avg_loss);
        println!("Validation perplexity: {:.2}", perplexity);
        println!("Validation accuracy: {:.2}%", accuracy);
        
        // Generate sample text
        println!("\nGenerating sample text (Epoch {}):", epoch);
        for prompt in &prompts {
            let generated = trainer.generate_text(prompt, Some(50));
            println!("Prompt: '{}'", prompt);
            println!("Generated: '{}'", generated);
            
            // Calculate similarity to prompt (to check for coherence)
            if generated.starts_with(prompt) {
                let prompt_words = prompt.split_whitespace().count();
                let generated_words = generated.split_whitespace().count();
                let additional_words = generated_words.saturating_sub(prompt_words);
                println!("Added {} new words", additional_words);
            }
        }
        
        // Save checkpoint
        let checkpoint_path = format!("models/improved_model_epoch_{}.bin", epoch);
        println!("Saving checkpoint to '{}'...", checkpoint_path);
        trainer.save_model(&checkpoint_path)?;
    }

    // Print final metrics report
    println!("\n===== TRAINING METRICS =====");
    println!("Epoch | Loss      | Perplexity | Accuracy(%)");
    println!("------------------------------------------");
    for (epoch, loss, perplexity, accuracy) in &epoch_metrics {
        println!("{:5} | {:9.6} | {:10.2} | {:10.2}", epoch, loss, perplexity, accuracy);
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
        if tokens.len() < 10 {
            progress_bar.inc(1);
            continue;
        }
        
        // Process overlapping windows of tokens
        let window_size = std::cmp::min(max_seq_len, 64);
        let stride = 16; // Overlap between windows
        
        let mut pos = 0;
        while pos + 2 <= tokens.len() {
            let end = std::cmp::min(pos + window_size, tokens.len());
            if end - pos < 10 {
                break; // Too short to be useful
            }
            
            // Input: all tokens except the last one
            let input = tokens[pos..end - 1].to_vec();
            
            // Target: all tokens except the first one
            let target = tokens[pos + 1..end].to_vec();
            
            // Only add good quality examples (ones with proper text, not just special tokens)
            let normal_token_count = input.iter().filter(|&&id| {
                if let Some(token) = tokenizer.get_vocab().id_to_token(id) {
                    !token.starts_with('[') && !token.ends_with(']')
                } else {
                    false
                }
            }).count();
            
            // Only include examples where at least 30% are normal word tokens
            // Lowered from 70% to ensure we have enough examples
            if normal_token_count as f32 / input.len() as f32 >= 0.3 && input.len() >= 8 {
                inputs.push(input);
                targets.push(target);
            }
            
            pos += stride;
        }
        
        progress_bar.inc(1);
    }
    
    progress_bar.finish_with_message(format!("Examples prepared: {}", inputs.len()));
    
    (inputs, targets)
}

// Function to create batches with tighter length controls
fn create_batches(
    inputs: &[Vec<usize>],
    targets: &[Vec<usize>],
    batch_size: usize
) -> (Vec<Vec<Vec<usize>>>, Vec<Array2<usize>>) {
    let mut batched_inputs = Vec::new();
    let mut batched_targets = Vec::new();
    
    // Group examples by similar lengths to avoid padding issues
    let mut length_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    
    // Group by length ranges (every 8 tokens)
    for (idx, input) in inputs.iter().enumerate() {
        let length_bucket = (input.len() / 8) * 8;  // Round down to nearest multiple of 8
        length_groups.entry(length_bucket).or_default().push(idx);
    }
    
    // Process each length group separately
    for (length, indices) in length_groups.into_iter() {
        let mut group_indices = indices;
        group_indices.shuffle(&mut rand::thread_rng());
        
        // Create batches from this length group
        for chunk in group_indices.chunks(batch_size) {
            if chunk.is_empty() {
                continue;
            }
            
            // Ensure all examples in the batch have the same length to avoid dimension errors
            let min_input_len = chunk.iter()
                .map(|&idx| inputs[idx].len())
                .min()
                .unwrap_or(0);
            
            let min_target_len = chunk.iter()
                .map(|&idx| targets[idx].len())
                .min()
                .unwrap_or(0);
            
            // Skip if any sequence is too short
            if min_input_len < 8 || min_target_len < 8 {
                continue;
            }
            
            let mut batch_inputs = Vec::new();
            let batch_size = chunk.len();
            let mut batch_targets = Array2::zeros((batch_size, min_target_len));
            
            // Add each example to batch, truncating to min length in batch
            for (i, &idx) in chunk.iter().enumerate() {
                // Truncate input and target to consistent lengths
                let input = inputs[idx][..min_input_len].to_vec();
                batch_inputs.push(input);
                
                // Fill target array with truncated values
                for j in 0..min_target_len {
                    if j < targets[idx].len() {
                        batch_targets[[i, j]] = targets[idx][j];
                    }
                }
            }
            
            // Add batch to results if not empty
            if !batch_inputs.is_empty() {
                batched_inputs.push(batch_inputs);
                batched_targets.push(batch_targets);
            }
        }
    }
    
    (batched_inputs, batched_targets)
} 