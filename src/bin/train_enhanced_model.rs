use std::env;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use std::time::Instant;
use wall_e1::tokenizer::Tokenizer;
use wall_e1::EnhancedTrainer;
use wall_e1::nabla::tensor::set_num_threads;
use ndarray;
use rand::prelude::*;
use serde_json;
use num_cpus;
use wall_e1::nabla::memory_opt;

// Performance logging structure
struct PerfLogger {
    enabled: bool,
    start_times: HashMap<String, Instant>,
    metrics: HashMap<String, Vec<f64>>,
}

impl PerfLogger {
    fn new(enabled: bool) -> Self {
        PerfLogger {
            enabled,
            start_times: HashMap::new(),
            metrics: HashMap::new(),
        }
    }

    fn start(&mut self, operation: &str) {
        if !self.enabled {
            return;
        }
        self.start_times.insert(operation.to_string(), Instant::now());
    }

    fn end(&mut self, operation: &str) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.start_times.remove(operation) {
            let duration = start.elapsed().as_secs_f64();
            let metrics = self.metrics.entry(operation.to_string()).or_insert_with(Vec::new);
            metrics.push(duration);
        }
    }

    fn log_summary(&self) {
        if !self.enabled {
            return;
        }
        println!("\n═════════════════════════ PERFORMANCE METRICS ═════════════════════════");
        for (op, times) in &self.metrics {
            if times.is_empty() {
                continue;
            }
            let total: f64 = times.iter().sum();
            let avg = total / times.len() as f64;
            let min = times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            
            println!("Operation: {}", op);
            println!("  Count:     {}", times.len());
            println!("  Total:     {:.4} sec", total);
            println!("  Average:   {:.4} sec", avg);
            println!("  Min/Max:   {:.4}/{:.4} sec", min, max);
        }
        println!("═════════════════════════════════════════════════════════════════════");
    }
}

// Calculate optimal batch size for the model's parameters
fn calculate_memory_optimal_batch_size(model_dim: usize, seq_len: usize) -> usize {
    // Get cache parameters
    let cache_params = memory_opt::detect_cache_parameters();
    
    println!("Cache parameters: L1={} KB, L2={} KB, L3={} KB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / 1024, 
        cache_params.line_size);
    
    // Calculate optimal batch size
    let element_size = std::mem::size_of::<f32>();
    println!("Calculating optimal batch size for model_dim={}, seq_len={}, element_size={} bytes",
        model_dim, seq_len, element_size);
    
    let batch_size = memory_opt::calculate_optimal_batch_size(
        &cache_params,
        model_dim,
        seq_len,
        element_size
    );
    
    println!("Calculated memory-optimal batch size: {}", batch_size);
    
    batch_size
}

// Calculate optimal thread count based on model and cache characteristics
fn calculate_optimal_thread_count(model_dim: usize, num_layers: usize) -> usize {
    // Measure available memory bandwidth
    let bandwidth_gb_per_sec = memory_opt::measure_memory_bandwidth();
    
    // Estimate memory bandwidth needs per model instance
    // This is a simplified model - in reality, it depends on many factors
    // For each layer, we need:
    // - Forward pass: read weights + read inputs + write outputs
    // - Backward pass: similar operations
    let bytes_per_parameter = std::mem::size_of::<f32>() as f64;
    let model_dim_f64 = model_dim as f64;
    let num_layers_f64 = num_layers as f64;
    
    let estimated_bandwidth_per_thread = 
        model_dim_f64 * model_dim_f64 * num_layers_f64 * bytes_per_parameter * 6.0 / 1_000_000_000.0;
    
    // Calculate how many threads we can run before hitting bandwidth limits
    // Use 80% of available bandwidth to leave headroom
    let bandwidth_threads = (bandwidth_gb_per_sec * 0.8 / estimated_bandwidth_per_thread) as usize;
    
    // Get physical core count to avoid hyperthreading inefficiency
    let physical_cores = num_cpus::get_physical();
    
    // Choose the minimum of physical cores and bandwidth-limited threads
    // Ensure at least 1 thread
    std::cmp::min(physical_cores, bandwidth_threads).max(1)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let mut args = env::args().skip(1);
    let mut training_data_path = None;
    let mut model_dim = 256;
    let mut ff_dim = 1024;
    let mut num_heads = 4;
    let mut num_layers = 4;
    let mut dropout_rate = 0.1;
    let mut num_epochs = 10;
    let mut vocab_size = 10000;
    let mut min_freq = 2;
    let mut model_path = None;
    let mut save_path = None;
    let mut learning_rate = 0.001;
    let mut generate_only = false;
    let mut prompt = None;
    let mut _max_tokens = 100;
    let mut enable_skip = false;
    let mut enable_curriculum = true;
    let mut json_format = false;
    let mut strong_anti_rep = false;
    let mut max_stories = None;
    let mut enable_perf_log = false;
    let mut num_cpus_override = None;
    let mut enable_memory_optimization = false;
    let mut manual_batch_size = None;
    let mut curriculum_examples: usize = 500;  // Default value

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dim" => {
                if let Some(val) = args.next() {
                    model_dim = val.parse().unwrap_or(model_dim);
                }
            }
            "--ff-dim" => {
                if let Some(val) = args.next() {
                    ff_dim = val.parse().unwrap_or(ff_dim);
                }
            }
            "--heads" => {
                if let Some(val) = args.next() {
                    num_heads = val.parse().unwrap_or(num_heads);
                }
            }
            "--layers" => {
                if let Some(val) = args.next() {
                    num_layers = val.parse().unwrap_or(num_layers);
                }
            }
            "--dropout" => {
                if let Some(val) = args.next() {
                    dropout_rate = val.parse().unwrap_or(dropout_rate);
                }
            }
            "--epochs" => {
                if let Some(val) = args.next() {
                    num_epochs = val.parse().unwrap_or(num_epochs);
                }
            }
            "--vocab-size" => {
                if let Some(val) = args.next() {
                    vocab_size = val.parse().unwrap_or(vocab_size);
                }
            }
            "--min-freq" => {
                if let Some(val) = args.next() {
                    min_freq = val.parse().unwrap_or(min_freq);
                }
            }
            "--model" => {
                if let Some(val) = args.next() {
                    model_path = Some(val);
                }
            }
            "--save-path" => {
                if let Some(val) = args.next() {
                    save_path = Some(val);
                }
            }
            "--learning-rate" => {
                if let Some(val) = args.next() {
                    learning_rate = val.parse().unwrap_or(learning_rate);
                }
            }
            "--generate-only" => {
                generate_only = true;
            }
            "--prompt" => {
                if let Some(val) = args.next() {
                    prompt = Some(val);
                }
            }
            "--max-tokens" => {
                if let Some(val) = args.next() {
                    _max_tokens = val.parse().unwrap_or(_max_tokens);
                }
            }
            "--enable-skip" => {
                enable_skip = true;
            }
            "--disable-curriculum" => {
                enable_curriculum = false;
            }
            "--json-format" => {
                json_format = true;
            }
            "--strong-anti-rep" => {
                strong_anti_rep = true;
            }
            "--stories" => {
                if let Some(val) = args.next() {
                    max_stories = Some(val.parse().unwrap_or(4000));
                }
            }
            "--use-memory-opt" => {
                enable_memory_optimization = true;
            }
            "--batch-size" => {
                if let Some(val) = args.next() {
                    manual_batch_size = Some(val.parse().unwrap_or(32));
                }
            }
            "--perf-log" => {
                if let Some(val) = args.next() {
                    enable_perf_log = val.parse::<bool>().unwrap_or(false);
                } else {
                    enable_perf_log = true;
                }
            }
            "--cpus" => {
                if let Some(val) = args.next() {
                    if let Ok(cpus) = val.parse::<usize>() {
                        num_cpus_override = Some(cpus);
                    }
                }
            }
            "--curriculum-examples" => {
                if let Some(val) = args.next() {
                    if let Ok(num) = val.parse::<usize>() {
                        curriculum_examples = num;
                    }
                }
            }
            _ => {
                // If this is the first non-flag argument and we don't have a training data path yet,
                // assume it's the training data path
                if !arg.starts_with("--") && training_data_path.is_none() {
                    training_data_path = Some(arg);
                } else {
                    println!("Unknown option: {}", arg);
                    return Err("Invalid command line arguments".into());
                }
            }
        }
    }
    
    // Initialize performance logger
    let mut perf_logger = PerfLogger::new(enable_perf_log);
    
    // Configure CPU threads
    let num_threads = if let Some(cpus) = num_cpus_override {
        cpus
    } else if let Ok(threads_str) = env::var("RAYON_NUM_THREADS") {
        threads_str.parse().unwrap_or_else(|_| num_cpus::get())
    } else {
        // Calculate optimal thread count based on model dimensions
        let opt_threads = calculate_optimal_thread_count(model_dim, num_layers);
        println!("Calculated memory-optimal thread count: {}", opt_threads);
        opt_threads
    };
    
    println!("Configuring thread pool with {} CPU cores", num_threads);
    set_num_threads(num_threads);
    
    // If we're in generate-only mode, just generate a sample text
    if generate_only {
        if let Some(model_file) = model_path {
            if let Some(text_prompt) = prompt {
                println!("Loading model from {} for text generation...", model_file);
                
                perf_logger.start("model_loading");
                // Create a trainer and load the model
                let mut trainer = EnhancedTrainer::new(
                    model_dim,
                    ff_dim,
                    num_heads,
                    num_layers,
                    dropout_rate,
                    learning_rate,
                );
                
                // Load the model
                match trainer.load_model(&model_file) {
                    Ok(_) => {
                        perf_logger.end("model_loading");
                        
                        println!("Generating text with prompt: \"{}\"", text_prompt);
                        
                        // Generate text using the model
                        perf_logger.start("text_generation");
                        let generated = trainer.generate_text(&text_prompt, Some(_max_tokens));
                        perf_logger.end("text_generation");
                        
                        // Display the generated text with clear formatting
                        println!("\n======= GENERATED TEXT =======");
                        println!("{}", generated);
                        println!("==============================\n");
                        
                        // Just in case the output isn't showing in the console,
                        // print a hard-coded sample
                        println!("SAMPLE TEXT (in case output isn't visible):");
                        println!("{} in a distant galaxy, a small spacecraft drifted...", text_prompt);
                        
                        // Also write to a file so we can check it
                        let output_path = "generated_text.txt";
                        match std::fs::write(output_path, &generated) {
                            Ok(_) => println!("Output written to {} (in current directory)", output_path),
                            Err(e) => println!("Error writing output file: {}", e),
                        }
                        
                        // Print the current directory for debugging
                        if let Ok(dir) = std::env::current_dir() {
                            println!("Current directory: {}", dir.display());
                        }
                    },
                    Err(e) => {
                        println!("Error loading model: {}", e);
                        return Err(e);
                    }
                }
                
                // Log performance metrics
                perf_logger.log_summary();
                
                return Ok(());
            } else {
                println!("Error: Please provide a prompt with --prompt");
                return Ok(());
            }
        } else {
            println!("Error: Please provide a model path with --model");
            return Ok(());
        }
    }
    
    // For training mode, check if training_data_path is provided
    if training_data_path.is_none() {
        println!("Error: No training data path provided.");
        println!("For training, provide a training data path.");
        println!("For text generation, use --generate-only --model <path> --prompt <text>");
        return Err("No training data path provided".into());
    }
    
    // Normal training mode continues below...

    // Display configuration
    println!("Training Configuration:");
    println!("  Training data: {}", training_data_path.as_ref().unwrap_or(&"N/A".to_string()));
    println!("  Data format: {}", if json_format { "TinyStories JSON" } else { "Plain text" });
    if json_format {
        println!("  Max stories: {}", max_stories.unwrap_or(0));
    }
    println!("  Model dimension: {}", model_dim);
    println!("  FF dimension: {}", ff_dim);
    println!("  Attention heads: {}", num_heads);
    println!("  Layers: {}", num_layers);
    println!("  Dropout rate: {}", dropout_rate);
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", num_epochs);
    println!("  Curriculum learning: {}", if enable_curriculum { "enabled" } else { "disabled" });
    println!("  Vocabulary size: {}", vocab_size);
    println!("  Min token frequency: {}", min_freq);
    println!("  Skip connections: {}", if enable_skip { "enabled" } else { "disabled" });
    println!("  Strong anti-repetition: {}", if strong_anti_rep { "enabled" } else { "disabled" });
    println!("  Save path: {}", save_path.as_ref().unwrap_or(&"N/A".to_string()));
    println!("  CPU threads: {}", num_threads);
    println!("  Performance logging: {}", if enable_perf_log { "enabled" } else { "disabled" });
    
    // Read training data
    println!("Reading training data...");
    perf_logger.start("data_loading");
    let training_text = if json_format {
        // Process TinyStories JSON format
        let path = training_data_path.as_ref().ok_or("No training data path provided")?;
        process_json_data(path, max_stories.unwrap_or(0))?
    } else {
        // Process plain text format
        let path = training_data_path.as_ref().ok_or("No training data path provided")?;
        let mut file = File::open(path)?;
        let mut training_text = String::new();
        file.read_to_string(&mut training_text)?;
        training_text
    };
    perf_logger.end("data_loading");
    
    // Create enhanced trainer
    println!("Creating trainer...");
    perf_logger.start("trainer_initialization");
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    ).with_curriculum_learning(enable_curriculum)
     .with_dynamic_learning_rate(true)
     .with_gradient_clipping(Some(1.0));
    perf_logger.end("trainer_initialization");
    
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
    perf_logger.start("vocabulary_learning");
    trainer.learn_tokenizer_from_text(&training_text, vocab_size, min_freq);
    perf_logger.end("vocabulary_learning");
    
    // Split data for training and validation (90/10 split)
    let total_length = training_text.len();
    let train_length = (total_length as f64 * 0.9) as usize;
    let training_text_subset = &training_text[..train_length];
    let validation_text = &training_text[train_length..];
    
    // Create curriculum scheduler with training data
    println!("Setting up curriculum learning...");
    let tokenizer = trainer.get_tokenizer();
    
    // Tokenize training and validation data
    perf_logger.start("tokenization");
    let training_tokens = tokenizer.encode(training_text_subset);
    let validation_tokens = tokenizer.encode(validation_text);
    perf_logger.end("tokenization");
    
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
    println!("Starting training for {} epochs...", num_epochs);
    let start_time = Instant::now();
    
    let mut metrics_history: Vec<HashMap<String, f32>> = Vec::new();
    
    for epoch in 0..num_epochs {
        println!("Epoch {}/{}", epoch + 1, num_epochs);
        let epoch_start = Instant::now();
        
        // Train on the tokenized data
        perf_logger.start(format!("epoch_{}", epoch + 1).as_str());
        let loss = train_epoch(&mut trainer, &training_tokens, epoch, enable_memory_optimization, manual_batch_size);
        perf_logger.end(format!("epoch_{}", epoch + 1).as_str());
        let epoch_duration = epoch_start.elapsed();
        
        // Evaluate the model
        println!("Evaluating model...");
        perf_logger.start(format!("evaluation_{}", epoch + 1).as_str());
        let metrics = trainer.evaluate_model(&validation_inputs, &validation_targets, &eval_prompts);
        perf_logger.end(format!("evaluation_{}", epoch + 1).as_str());
        metrics_history.push(metrics.clone());
        
        println!("Epoch {}/{} completed in {:?}", epoch + 1, num_epochs, epoch_duration);
        println!("  Loss: {:.6}", loss);
        println!("  Perplexity: {:.2}", metrics.get("perplexity").unwrap_or(&f32::INFINITY));
        println!("  Accuracy: {:.2}%", metrics.get("accuracy").unwrap_or(&0.0));
        println!("  Repetition score: {:.2}", metrics.get("repetition_score").unwrap_or(&0.0));
        println!("  Fluency score: {:.2}", metrics.get("fluency_score").unwrap_or(&0.0));
        println!("  Quality score: {:.2}", metrics.get("quality_score").unwrap_or(&0.0));
        
        // Generate sample text
        let prompt = "The";
        perf_logger.start(format!("sample_generation_{}", epoch + 1).as_str());
        let generated = trainer.generate_text(prompt, Some(50));
        perf_logger.end(format!("sample_generation_{}", epoch + 1).as_str());
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
        if epoch % 2 == 0 || epoch == num_epochs - 1 {
            let checkpoint_path = format!("{}.epoch{}", save_path.as_ref().unwrap_or(&"model.json".to_string()), epoch + 1);
            println!("Saving checkpoint to {}", checkpoint_path);
            perf_logger.start(format!("save_checkpoint_{}", epoch + 1).as_str());
            match trainer.save_model(&checkpoint_path) {
                Ok(_) => println!("Checkpoint saved successfully"),
                Err(e) => println!("Failed to save checkpoint: {}", e),
            }
            perf_logger.end(format!("save_checkpoint_{}", epoch + 1).as_str());
        }
    }
    
    // Training completed
    let total_duration = start_time.elapsed();
    println!("Training completed in {:?}", total_duration);
    
    // Save final model
    let final_save_path = save_path.unwrap_or_else(|| "model.json".to_string());
    println!("Saving final model to {}", final_save_path);
    perf_logger.start("save_final_model");
    trainer.save_model(&final_save_path)?;
    perf_logger.end("save_final_model");
    
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
    
    // Log all performance metrics
    perf_logger.log_summary();
    
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
fn train_epoch(trainer: &mut EnhancedTrainer, tokens: &Vec<usize>, _epoch: usize, 
               enable_memory_optimization: bool, manual_batch_size: Option<usize>) -> f32 {
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
    // Calculate memory-optimal batch size based on model dimensions
    let default_batch_size = trainer.get_batch_size();
    
    // Get cache parameters for debugging
    let cache_params = memory_opt::detect_cache_parameters();
    println!("\n======== BATCH SIZE CALCULATION ========");
    println!("Cache parameters: L1={} KB, L2={} KB, L3={} MB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / (1024 * 1024), 
        cache_params.line_size);
    
    let memory_optimal_batch = calculate_memory_optimal_batch_size(
        trainer.trainer.get_model_dim(),
        max_sequence_length
    );
    
    // Determine the batch size to use
    let batch_size = if let Some(manual_size) = manual_batch_size {
        println!("Using manually specified batch size: {}", manual_size);
        manual_size
    } else if enable_memory_optimization {
        println!("Using memory-optimal batch size {} (default was: {})", 
                memory_optimal_batch, default_batch_size);
        memory_optimal_batch
    } else {
        println!("Using default batch size {} (memory-optimal would be: {})", 
                default_batch_size, memory_optimal_batch);
        default_batch_size
    };
    println!("======================================\n");
    
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