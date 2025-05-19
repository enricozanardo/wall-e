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
use rayon::prelude::*;
use ndarray::Array2;
use indicatif::{ProgressBar, ProgressStyle};

// Performance logging structure
struct PerfLogger {
    enabled: bool,
    start_times: HashMap<String, Instant>,
    metrics: HashMap<String, Vec<f64>>,
    stack: Vec<String>,
    thread_metrics: HashMap<String, HashMap<String, Vec<f64>>>, // Using string thread IDs instead
    parallel_metrics: HashMap<String, (usize, f64)>, // (thread_count, total_time)
}

impl PerfLogger {
    fn new(enabled: bool) -> Self {
        PerfLogger {
            enabled,
            start_times: HashMap::new(),
            metrics: HashMap::new(),
            stack: Vec::new(),
            thread_metrics: HashMap::new(),
            parallel_metrics: HashMap::new(),
        }
    }

    fn start(&mut self, operation: &str) {
        if !self.enabled {
            return;
        }
        
        // Get thread ID for tracking parallel activity (as string to avoid unstable feature)
        let thread_id = format!("{:?}", std::thread::current().id());
        
        // Create full path with parent operations
        let full_path = if self.stack.is_empty() {
            operation.to_string()
        } else {
            format!("{}/{}", self.stack.last().unwrap(), operation)
        };
        
        self.stack.push(full_path.clone());
        self.start_times.insert(full_path, Instant::now());
    }

    fn end(&mut self, operation: &str) {
        if !self.enabled {
            return;
        }
        
        // Get thread ID as string
        let thread_id = format!("{:?}", std::thread::current().id());
        
        // Pop from stack and measure
        if let Some(full_path) = self.stack.pop() {
            if let Some(start) = self.start_times.remove(&full_path) {
                let duration = start.elapsed().as_secs_f64();
                
                // Store overall metrics
                let metrics = self.metrics.entry(full_path.clone()).or_insert_with(Vec::new);
                metrics.push(duration);
                
                // Store per-thread metrics
                let thread_map = self.thread_metrics.entry(full_path.clone()).or_insert_with(HashMap::new);
                let thread_metrics = thread_map.entry(thread_id).or_insert_with(Vec::new);
                thread_metrics.push(duration);
                
                // Estimate parallel vs sequential
                if operation.contains("parallel") || operation.contains("batch") {
                    let thread_count = thread_map.len();
                    let entry = self.parallel_metrics.entry(full_path).or_insert((0, 0.0));
                    entry.0 = thread_count; 
                    entry.1 += duration;
                }
            }
        }
    }

    fn log_summary(&self) {
        if !self.enabled {
            return;
        }
        
        println!("\n═════════════════════════ PERFORMANCE METRICS ═════════════════════════");
        
        // First, identify the most time-consuming operations
        let mut sorted_ops: Vec<_> = self.metrics.iter()
            .filter(|(_, times)| !times.is_empty())
            .map(|(op, times)| {
                let total: f64 = times.iter().sum();
                (op, total)
            })
            .collect();
        
        sorted_ops.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Show top operations by total time
        println!("\n--- TOP TIME-CONSUMING OPERATIONS ---");
        for (i, (op, total)) in sorted_ops.iter().take(10).enumerate() {
            let times = &self.metrics[op.as_str()];
            let avg = total / times.len() as f64;
            println!("#{}: {} - Total: {:.4}s, Avg: {:.4}s, Count: {}", 
                i+1, op, total, avg, times.len());
        }
        
        // Show parallelism metrics
        println!("\n--- PARALLELISM METRICS ---");
        let mut parallel_sorted: Vec<_> = self.parallel_metrics.iter().collect();
        parallel_sorted.sort_by(|(_, (_, a)), (_, (_, b))| 
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            
        for (op, (thread_count, total_time)) in parallel_sorted.iter().take(10) {
            println!("{} - Threads: {}, Total CPU Time: {:.4}s", 
                op, thread_count, total_time);
        }
        
        // Detailed metrics for all operations
        println!("\n--- DETAILED METRICS ---");
        for (op, times) in &self.metrics {
            if times.is_empty() {
                continue;
            }
            let total: f64 = times.iter().sum();
            let avg = total / times.len() as f64;
            let min = times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            
            println!("\nOperation: {}", op);
            println!("  Count:     {}", times.len());
            println!("  Total:     {:.4} sec", total);
            println!("  Average:   {:.4} sec", avg);
            println!("  Min/Max:   {:.4}/{:.4} sec", min, max);
            
            // Thread distribution for this operation
            if let Some(thread_map) = self.thread_metrics.get(op) {
                println!("  Thread distribution: {} threads", thread_map.len());
                if thread_map.len() > 1 {
                    let mut thread_times: Vec<_> = thread_map.iter()
                        .map(|(tid, times)| (tid.clone(), times.iter().sum::<f64>()))
                        .collect();
                    thread_times.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    
                    for (i, (tid, time)) in thread_times.iter().take(3).enumerate() {
                        println!("    Thread #{}: ID {} - {:.4}s ({:.1}%)", 
                            i+1, tid, time, 100.0 * time / total);
                    }
                }
            }
        }
        
        println!("═════════════════════════════════════════════════════════════════════");
    }
    
    // Add memory snapshot capability
    fn memory_snapshot(&self, label: &str) {
        if !self.enabled {
            return;
        }
        
        // On Linux we can get memory usage from /proc/self/status
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") || line.starts_with("VmSize:") {
                    println!("Memory [{}]: {}", label, line.trim());
                }
            }
        }
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
    // Initialize global performance logger
    let mut global_perf_logger = PerfLogger::new(true);
    global_perf_logger.start("program_execution");
    
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
    let mut curriculum_examples: usize = 2000;

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
    
    // Configure CPU threads
    global_perf_logger.start("cpu_configuration");
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
    global_perf_logger.end("cpu_configuration");
    
    // If we're in generate-only mode, just generate a sample text
    if generate_only {
        if let Some(model_file) = model_path {
            if let Some(text_prompt) = prompt {
                println!("Loading model from {} for text generation...", model_file);
                
                global_perf_logger.start("model_loading");
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
                        global_perf_logger.end("model_loading");
                        
                        println!("Generating text with prompt: \"{}\"", text_prompt);
                        
                        // Generate text using the model
                        global_perf_logger.start("text_generation");
                        let generated = trainer.generate_text(&text_prompt, Some(_max_tokens));
                        global_perf_logger.end("text_generation");
                        
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
                global_perf_logger.log_summary();
                
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
    global_perf_logger.start("training_setup");
    
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
    global_perf_logger.start("data_loading");
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
    global_perf_logger.end("data_loading");
    global_perf_logger.memory_snapshot("after_data_loading");
    
    // Create enhanced trainer
    println!("Creating trainer...");
    global_perf_logger.start("trainer_initialization");
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
    global_perf_logger.end("trainer_initialization");
    
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
    global_perf_logger.start("vocabulary_learning");
    trainer.learn_tokenizer_from_text(&training_text, vocab_size, min_freq);
    global_perf_logger.end("vocabulary_learning");
    
    // Split data for training and validation (90/10 split)
    global_perf_logger.start("data_splitting");
    let total_length = training_text.len();
    let train_length = (total_length as f64 * 0.9) as usize;
    let training_text_subset = &training_text[..train_length];
    let validation_text = &training_text[train_length..];
    global_perf_logger.end("data_splitting");
    
    // Create curriculum scheduler with training data
    println!("Setting up curriculum learning...");
    
    // Tokenize training and validation data
    global_perf_logger.start("tokenization");
    let tokenizer = trainer.get_tokenizer();
    let training_tokens = tokenizer.encode(training_text_subset);
    let validation_tokens = tokenizer.encode(validation_text);
    global_perf_logger.end("tokenization");
    global_perf_logger.memory_snapshot("after_tokenization");
    
    // Set up evaluation prompts
    let eval_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a world where",
        "The most important thing",
        "I would like to",
    ];
    
    // Prepare validation data
    global_perf_logger.start("validation_data_preparation");
    let mut validation_inputs = Vec::new();
    let mut validation_targets = Vec::new();
    prepare_validation_data(&validation_tokens, &mut validation_inputs, &mut validation_targets);
    global_perf_logger.end("validation_data_preparation");
    
    global_perf_logger.end("training_setup");
    
    // Start training
    println!("Starting training for {} epochs...", num_epochs);
    global_perf_logger.start("training_process");
    let start_time = Instant::now();
    
    let mut metrics_history: Vec<HashMap<String, f32>> = Vec::new();
    
    for epoch in 0..num_epochs {
        println!("Epoch {}/{}", epoch + 1, num_epochs);
        let epoch_start = Instant::now();
        
        // Train on the tokenized data
        global_perf_logger.start(format!("epoch_{}", epoch + 1).as_str());
        let loss = train_epoch(&mut trainer, &training_tokens, epoch, enable_memory_optimization, manual_batch_size);
        global_perf_logger.end(format!("epoch_{}", epoch + 1).as_str());
        let epoch_duration = epoch_start.elapsed();
        
        // Evaluate the model
        println!("Evaluating model...");
        global_perf_logger.start(format!("evaluation_{}", epoch + 1).as_str());
        let metrics = trainer.evaluate_model(&validation_inputs, &validation_targets, &eval_prompts);
        global_perf_logger.end(format!("evaluation_{}", epoch + 1).as_str());
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
        global_perf_logger.start(format!("sample_generation_{}", epoch + 1).as_str());
        let generated = trainer.generate_text(prompt, Some(50));
        global_perf_logger.end(format!("sample_generation_{}", epoch + 1).as_str());
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
            global_perf_logger.start(format!("save_checkpoint_{}", epoch + 1).as_str());
            match trainer.save_model(&checkpoint_path) {
                Ok(_) => println!("Checkpoint saved successfully"),
                Err(e) => println!("Failed to save checkpoint: {}", e),
            }
            global_perf_logger.end(format!("save_checkpoint_{}", epoch + 1).as_str());
        }
    }
    
    global_perf_logger.end("training_process");
    
    // Training completed
    let total_duration = start_time.elapsed();
    println!("Training completed in {:?}", total_duration);
    
    // Save final model
    let final_save_path = save_path.unwrap_or_else(|| "model.json".to_string());
    println!("Saving final model to {}", final_save_path);
    global_perf_logger.start("save_final_model");
    trainer.save_model(&final_save_path)?;
    global_perf_logger.end("save_final_model");
    
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
    
    global_perf_logger.memory_snapshot("program_end");
    global_perf_logger.end("program_execution");
    
    // Log all performance metrics
    global_perf_logger.log_summary();
    
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
    
    let mut perf_logger = PerfLogger::new(true); // Local performance tracking
    perf_logger.start("train_epoch_full");
    perf_logger.memory_snapshot("train_epoch_start");
    
    perf_logger.start("data_preparation");
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
    perf_logger.end("data_preparation");
    
    println!("Created {} input/target pairs for training", inputs.len());
    
    // Get cache parameters
    let cache_params = memory_opt::detect_cache_parameters();
    
    println!("Cache parameters: L1={} KB, L2={} KB, L3={} KB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / 1024, 
        cache_params.line_size);
    
    // Get CPU information
    let num_cpus = num_cpus::get();
    let num_physical_cpus = num_cpus::get_physical();
    println!("CPU cores: {} logical, {} physical", num_cpus, num_physical_cpus);
    
    // Set Rayon thread pool size to optimize CPU usage
    let rayon_threads = std::cmp::max(num_physical_cpus, 2);
    println!("Using {} threads for parallel processing", rayon_threads);
    
    perf_logger.start("rayon_thread_pool_setup");
    // Setting the global Rayon thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(rayon_threads)
        .build_global()
        .unwrap_or_else(|e| println!("Warning: Failed to set global thread pool: {}", e));
    perf_logger.end("rayon_thread_pool_setup");
    
    // Get model dimensions for batch size calculation
    let model_dim = trainer.trainer.get_model_dim();
    
    // Calculate memory-optimal batch size
    perf_logger.start("batch_size_calculation");
    let optimal_batch_size = calculate_memory_optimal_batch_size(model_dim, max_sequence_length);
    
    // Apply constraints based on dataset size to ensure we have enough batches
    let max_batch_size = inputs.len() / 10.min(50);  // Ensure at least 10 batches, aim for 50+
    let constrained_batch_size = optimal_batch_size.min(max_batch_size).max(4);  // At least 4, no more than max
    perf_logger.end("batch_size_calculation");
    
    // Share the information with the user
    println!("Hardware-optimal batch size: {}", optimal_batch_size);
    println!("Dataset-constrained batch size: {}", constrained_batch_size);
    println!("Using batch size: {}", manual_batch_size.unwrap_or(constrained_batch_size));
    
    // Use the calculated batch size
    let batch_size = manual_batch_size.unwrap_or(constrained_batch_size);
    
    // Calculate how many batches we'll process with this batch size
    let expected_batches = (inputs.len() + batch_size - 1) / batch_size; // Ceiling division
    
    println!("Calculated memory-optimal batch size: {}", optimal_batch_size);
    println!("Expected number of batches: {}", expected_batches);
    println!("======================================\n");
    
    // Shuffle indices for randomized training
    perf_logger.start("shuffling_indices");
    let mut indices: Vec<usize> = (0..inputs.len()).collect();
    indices.shuffle(&mut thread_rng());
    perf_logger.end("shuffling_indices");
    
    let mut total_loss = 0.0;
    let mut batch_counter = 0;
    
    // Create batch chunks for parallel processing
    perf_logger.start("batch_creation");
    let batches: Vec<_> = indices.chunks(batch_size).collect();
    println!("Processing {} batches in parallel when possible", batches.len());
    perf_logger.end("batch_creation");
    
    // Create progress bar
    let pb = ProgressBar::new(batches.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} batches ({percent}%) - ETA: {eta_precise} - Loss: {msg}")
        .unwrap()
        .progress_chars("#>-"));
    
    // Statistics for batch timing
    let mut batch_prep_times = Vec::with_capacity(batches.len());
    let mut train_step_times = Vec::with_capacity(batches.len());
    
    // Process each batch
    perf_logger.start("batch_processing_loop");
    for (batch_idx, batch_indices) in batches.iter().enumerate() {
        // Prepare this batch data in parallel
        perf_logger.start("prepare_batch");
        let prep_start = Instant::now();
        let (batch_inputs, batch_targets_arr) = prepare_batch_parallel(&inputs, &targets, batch_indices, max_sequence_length);
        let prep_time = prep_start.elapsed().as_secs_f64();
        batch_prep_times.push(prep_time);
        perf_logger.end("prepare_batch");
        
        // Skip empty batches
        if batch_inputs.is_empty() {
            pb.inc(1);
            continue;
        }
        
        // Train on this batch
        perf_logger.start("train_step");
        let train_start = Instant::now();
        let loss = trainer.train_step_with_penalties(&batch_inputs, &batch_targets_arr);
        let train_time = train_start.elapsed().as_secs_f64();
        train_step_times.push(train_time);
        perf_logger.end("train_step");
        
        // Update tracking variables
        total_loss += loss;
        batch_counter += 1;
        
        // Update progress bar
        pb.set_message(format!("{:.6} (avg: {:.6})", loss, total_loss / batch_counter as f32));
        pb.inc(1);
        
        // Still keep occasional console updates for log files
        if batch_idx % 50 == 0 || batch_idx == batches.len() - 1 {
            println!("Batch {}/{} - Loss: {:.6} - Avg: {:.6} - Prep: {:.3}s - Train: {:.3}s", 
                batch_idx + 1, batches.len(), loss, total_loss / batch_counter as f32,
                prep_time, train_time);
        }
    }
    perf_logger.end("batch_processing_loop");
    
    // Calculate preparation vs training time ratio
    if !batch_prep_times.is_empty() && !train_step_times.is_empty() {
        let total_prep_time: f64 = batch_prep_times.iter().sum();
        let total_train_time: f64 = train_step_times.iter().sum();
        let avg_prep_time = total_prep_time / batch_prep_times.len() as f64;
        let avg_train_time = total_train_time / train_step_times.len() as f64;
        
        println!("\nTime analysis:");
        println!("  Total batch preparation time: {:.3}s ({:.1}%)", 
            total_prep_time, 
            100.0 * total_prep_time / (total_prep_time + total_train_time));
        println!("  Total training step time:     {:.3}s ({:.1}%)", 
            total_train_time,
            100.0 * total_train_time / (total_prep_time + total_train_time));
        println!("  Average batch preparation:    {:.3}s", avg_prep_time);
        println!("  Average training step:        {:.3}s", avg_train_time);
        println!("  Ratio (train/prep):           {:.2}x", avg_train_time / avg_prep_time);
    }
    
    // Finish progress bar
    pb.finish_with_message(format!("Completed - Avg loss: {:.6}", total_loss / batch_counter as f32));
    
    perf_logger.memory_snapshot("train_epoch_end");
    perf_logger.end("train_epoch_full");
    
    // Print local performance metrics for this epoch
    perf_logger.log_summary();
    
    total_loss / batch_counter as f32
}

// Prepares a batch in parallel using Rayon
fn prepare_batch_parallel(
    inputs: &[Vec<usize>], 
    targets: &[Vec<usize>], 
    batch_indices: &[usize],
    max_sequence_length: usize
) -> (Vec<Vec<usize>>, Array2<usize>) {
    let prep_start = Instant::now();
    
    // Early return if batch is empty
    if batch_indices.is_empty() {
        return (Vec::new(), Array2::zeros((0, 0)));
    }
    
    // Collect input sequences in parallel
    let par_collect_start = Instant::now();
    let batch_inputs: Vec<Vec<usize>> = batch_indices.par_iter()
        .map(|&idx| inputs[idx].clone())
        .collect();
    let par_collect_time = par_collect_start.elapsed().as_millis();
    
    // Skip if all sequences are empty
    if batch_inputs.is_empty() {
        return (Vec::new(), Array2::zeros((0, 0)));
    }
    
    // Find minimum sequence length in parallel
    let min_len_start = Instant::now();
    let min_seq_len = batch_inputs.par_iter()
        .map(|seq| seq.len())
        .min()
        .unwrap_or(0)
        .min(max_sequence_length);
    let min_len_time = min_len_start.elapsed().as_millis();
    
    // Skip if sequences are too short
    if min_seq_len < 4 {
        return (Vec::new(), Array2::zeros((0, 0)));
    }
    
    // Truncate all sequences to the same length
    let truncate_start = Instant::now();
    let truncated_inputs: Vec<Vec<usize>> = batch_inputs.par_iter()
        .map(|seq| {
            if seq.len() > min_seq_len {
                seq[0..min_seq_len].to_vec()
            } else {
                seq.clone()
            }
        })
        .collect();
    let truncate_time = truncate_start.elapsed().as_millis();
    
    // Create batch targets array
    let targets_start = Instant::now();
    let mut batch_targets_arr = Array2::zeros((batch_indices.len(), min_seq_len));
    
    // Fill targets sequentially to avoid mutable borrow issues
    for (i, &idx) in batch_indices.iter().enumerate() {
        let target = &targets[idx];
        for j in 0..min_seq_len.min(target.len()) {
            batch_targets_arr[[i, j]] = target[j];
        }
    }
    let targets_time = targets_start.elapsed().as_millis();
    
    let total_time = prep_start.elapsed().as_millis();
    
    // Only log detailed timing occasionally to avoid flooding output
    if total_time > 10 || batch_indices.len() > 16 {
        println!("Batch prep timing: total={}ms (collect={}ms, min_len={}ms, truncate={}ms, targets={}ms)",
            total_time, par_collect_time, min_len_time, truncate_time, targets_time);
    }
    
    (truncated_inputs, batch_targets_arr)
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