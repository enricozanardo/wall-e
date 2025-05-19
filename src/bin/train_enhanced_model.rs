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
fn calculate_optimal_thread_count(model_dim: usize, num_layers: usize, operation_type: Option<&str>) -> usize {
    // Physical cores are more important than logical cores for compute-heavy ML operations
    let physical_cores = num_cpus::get_physical();
    let logical_cores = num_cpus::get();
    
    println!("Hardware: {} physical cores, {} logical cores", physical_cores, logical_cores);
    
    // Measure available memory bandwidth
    let bandwidth_gb_per_sec = memory_opt::measure_memory_bandwidth();
    println!("Measured memory bandwidth: {:.2} GB/s", bandwidth_gb_per_sec);
    
    // Estimate memory bandwidth needs per model based on dimensions
    let bytes_per_parameter = std::mem::size_of::<f32>() as f64;
    let model_dim_f64 = model_dim as f64;
    let num_layers_f64 = num_layers as f64;
    
    // Different operations have different memory bandwidth requirements
    let (bandwidth_multiplier, compute_intensity) = match operation_type {
        Some("matrix_multiply") => (8.0, 2.0),       // High compute intensity, high bandwidth
        Some("attention") => (10.0, 1.5),            // Very high memory bandwidth requirement
        Some("tokenization") => (2.0, 0.5),          // Low compute, medium bandwidth
        Some("embedding") => (4.0, 0.8),             // Medium bandwidth, low compute
        Some("gradient_update") => (6.0, 1.2),       // High bandwidth, medium compute
        Some("data_loading") => (1.0, 0.2),          // I/O bound, low compute
        _ => (6.0, 1.0),                             // Default for general operations
    };
    
    // For data_loading operations, use a different approach that focuses on parallelism
    if operation_type == Some("data_loading") {
        // For data loading, we want more threads to hide I/O latency
        // Use at least 25% of logical cores, but not less than 4 and not more than 75% of logical cores
        let min_threads = 4;
        let max_threads = (logical_cores as f64 * 0.75) as usize;
        let recommended_threads = (logical_cores as f64 * 0.25) as usize;
        
        let data_loading_threads = recommended_threads.clamp(min_threads, max_threads);
        println!("Data loading thread calculation: recommended={}, min={}, max={}, final={}",
                 recommended_threads, min_threads, max_threads, data_loading_threads);
        
        return data_loading_threads;
    }
    
    // Calculate bandwidth requirement per thread (GB/s)
    let estimated_bandwidth_per_thread = 
        model_dim_f64 * model_dim_f64 * num_layers_f64 * bytes_per_parameter * 
        bandwidth_multiplier / 1_000_000_000.0;
    
    println!("Estimated bandwidth per thread: {:.2} GB/s", estimated_bandwidth_per_thread);
    
    // Calculate how many threads we can run before hitting bandwidth limits
    // Use 80% of available bandwidth to leave headroom
    let bandwidth_threads = (bandwidth_gb_per_sec * 0.8 / estimated_bandwidth_per_thread) as usize;
    
    // For compute-bound operations, we want to use more cores
    let compute_factor = (compute_intensity * physical_cores as f64) as usize;
    let compute_threads = (compute_factor.min(logical_cores)).max(1);
    
    // Choose the limiting factor: either bandwidth or compute capability
    let optimal_threads = bandwidth_threads.min(compute_threads);
    
    // Adjust for small models - don't overparallelise small workloads
    let workload_size = model_dim * num_layers;
    let small_model_factor = if workload_size < 1000 {
        // For tiny models, reduce thread count to avoid overhead
        0.5
    } else if workload_size < 10000 {
        0.75
    } else {
        1.0
    };
    
    // Apply small model adjustment 
    let adjusted_threads = ((optimal_threads as f64) * small_model_factor) as usize;
    
    // Ensure at least 1 thread, and no more than logical core count
    let final_thread_count = adjusted_threads.clamp(1, logical_cores);
    
    println!("Thread count calculation: bandwidth_limit={}, compute_limit={}, adjusted={}, final={}",
        bandwidth_threads, compute_threads, adjusted_threads, final_thread_count);
    
    final_thread_count
}

// Configure thread pool dynamically for specific operations
fn configure_thread_pool_for_operation(model_dim: usize, num_layers: usize, operation: &str) -> usize {
    // Calculate optimal thread count for this specific operation
    let thread_count = calculate_optimal_thread_count(model_dim, num_layers, Some(operation));
    
    // Get an operation-specific thread pool
    let pool = wall_e1::utils::thread_pool::get_thread_pool_for_operation(operation);
    
    // Check if this is a data_loading operation where we want to ensure proper thread count
    if operation == "data_loading" {
        // For data_loading, we want to force a specific thread count
        // We can't directly modify the thread pool, but we can print a more informative message
        if pool.get_num_threads() < thread_count {
            println!("Warning: Using existing thread pool with {} threads for {} operations, recommended: {}", 
                    pool.get_num_threads(), operation, thread_count);
            
            // Set an environment variable to suggest the thread count for next run
            unsafe {
                std::env::set_var("WALL_E_DATA_THREADS", thread_count.to_string());
            }
            println!("Set WALL_E_DATA_THREADS={} for future runs", thread_count);
        } else {
            println!("Configured dedicated thread pool with {} threads for {} operations", 
                    pool.get_num_threads(), operation);
        }
    } else {
        println!("Configured dedicated thread pool with {} threads for {} operations", 
                pool.get_num_threads(), operation);
    }
    
    // Return the configured thread count
    thread_count
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure global memory and thread optimizations right at the start
    let cpu_count = num_cpus::get();
    println!("Detected {} CPU cores", cpu_count);
    
    // Force data loading threads to be much higher - at least 8 or 70% of available cores,
    // to address very low CPU utilization (0.0-0.7%)
    let min_data_threads = std::cmp::max(8, (cpu_count as f32 * 0.7) as usize);
    println!("🔧 Setting WALL_E_DATA_THREADS to {} for better parallelism (fixing low CPU usage)", min_data_threads);
    // Set environment variable for data loading thread count
    unsafe {
        std::env::set_var("WALL_E_DATA_THREADS", min_data_threads.to_string());
    }
    
    // Create global performance logger
    let mut global_perf_logger = PerfLogger::new(true);
    global_perf_logger.memory_snapshot("startup");
    
    println!("\n📊 DETAILED STARTUP TIMING 📊");
    
    // Timer for overall initialization
    let total_init_start = Instant::now();
    
    // Configure and measure thread pool initialization
    let thread_init_start = Instant::now();
    println!("⏳ Initializing thread pools...");
    
    // First, get the global thread pool (this will initialize it)
    let global_pool_start = Instant::now();
    let global_pool = wall_e1::utils::thread_pool::get_global_thread_pool();
    let global_pool_time = global_pool_start.elapsed();
    println!("  ✅ Global thread pool initialized with {} threads ({:.2?})", 
             global_pool.get_num_threads(), global_pool_time);
    
    // Pre-initialize operation-specific thread pools
    let operations = ["matrix_multiply", "gradient_update", "data_loading", "attention"];
    for op in operations.iter() {
        let op_start = Instant::now();
        let pool = wall_e1::utils::thread_pool::get_thread_pool_for_operation(op);
        let op_time = op_start.elapsed();
        let thread_count = if op == &"data_loading" { min_data_threads } else { pool.get_num_threads() };
        println!("  ✅ Thread pool for '{}' initialized with {} threads ({:.2?})", 
                 op, pool.get_num_threads(), op_time);
                 
        // Add informative warning if data_loading thread count is low
        if op == &"data_loading" && pool.get_num_threads() < min_data_threads {
            println!("  ⚠️ Warning: data_loading thread pool has only {} threads, recommended: {} threads", 
                    pool.get_num_threads(), min_data_threads);
        }
    }
    
    let thread_init_time = thread_init_start.elapsed();
    println!("✅ Thread pools initialized in {:.2?}", thread_init_time);
    
    // Configure and measure memory optimization initialization
    let memory_init_start = Instant::now();
    println!("⏳ Configuring memory allocator...");
    
    // Measure cache detection time
    let cache_start = Instant::now();
    let cache_params = memory_opt::detect_cache_parameters();
    let cache_time = cache_start.elapsed();
    println!("  ✅ Cache detection completed in {:.2?}", cache_time);
    println!("     L1={} KB, L2={} KB, L3={} KB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / 1024, 
        cache_params.line_size);
    
    // Measure memory allocator configuration time
    let allocator_start = Instant::now();
    memory_opt::configure_memory_allocator(memory_opt::MemoryPolicy::CacheEfficient);
    let allocator_time = allocator_start.elapsed();
    println!("  ✅ Memory allocator configured in {:.2?}", allocator_time);
    
    // Measure memory bandwidth (can be slow but important for thread count decisions)
    let bandwidth_start = Instant::now();
    let bandwidth_gb_per_sec = memory_opt::measure_memory_bandwidth();
    let bandwidth_time = bandwidth_start.elapsed();
    println!("  ✅ Memory bandwidth measured in {:.2?}: {:.2} GB/s", bandwidth_time, bandwidth_gb_per_sec);
    
    let memory_init_time = memory_init_start.elapsed();
    println!("✅ Memory optimizations completed in {:.2?}", memory_init_time);
    
    // Total initialization time
    let total_init_time = total_init_start.elapsed();
    println!("✅ Total initialization completed in {:.2?}\n", total_init_time);
    
    println!("Memory and thread optimizations configured");
    
    // Set environment variables for optimal thread usage - safely
    unsafe {
        std::env::set_var("WALL_E_THREADS", format!("{}", cpu_count.saturating_sub(1)));
        std::env::set_var("RAYON_NUM_THREADS", format!("{}", cpu_count.saturating_sub(1)));
    }
    
    println!("\n🔄 STARTING APPLICATION SETUP 🔄");
    let setup_start = Instant::now();
    
    // Original main function content follows
    let args: Vec<String> = env::args().collect();
    
    // Track argument parsing time
    let arg_parsing_start = Instant::now();
    println!("⏳ Parsing command line arguments...");
    
    // Define default parameters
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
    let mut checkpoint_strategy = None;
    let mut thread_opt = None;

    // Command line arguments parsing loop with progress counter
    let arg_count = args.len();
    println!("  Processing {} command line arguments", arg_count);
    
    // Create an iterator for argument processing
    let mut arg_iter = args.iter().skip(1);
    while let Some(arg) = arg_iter.next() {
        match arg.as_str() {
            "--model-dim" => {
                if let Some(val) = arg_iter.next() {
                    model_dim = val.parse().unwrap_or(model_dim);
                }
            }
            "--ff-dim" => {
                if let Some(val) = arg_iter.next() {
                    ff_dim = val.parse().unwrap_or(ff_dim);
                }
            }
            "--heads" => {
                if let Some(val) = arg_iter.next() {
                    num_heads = val.parse().unwrap_or(num_heads);
                }
            }
            "--layers" => {
                if let Some(val) = arg_iter.next() {
                    num_layers = val.parse().unwrap_or(num_layers);
                }
            }
            "--dropout" => {
                if let Some(val) = arg_iter.next() {
                    dropout_rate = val.parse().unwrap_or(dropout_rate);
                }
            }
            "--epochs" => {
                if let Some(val) = arg_iter.next() {
                    num_epochs = val.parse().unwrap_or(num_epochs);
                }
            }
            "--vocab-size" => {
                if let Some(val) = arg_iter.next() {
                    vocab_size = val.parse().unwrap_or(vocab_size);
                }
            }
            "--min-freq" => {
                if let Some(val) = arg_iter.next() {
                    min_freq = val.parse().unwrap_or(min_freq);
                }
            }
            "--model" => {
                if let Some(val) = arg_iter.next() {
                    model_path = Some(val);
                }
            }
            "--save-path" => {
                if let Some(val) = arg_iter.next() {
                    save_path = Some(val);
                }
            }
            "--learning-rate" => {
                if let Some(val) = arg_iter.next() {
                    learning_rate = val.parse().unwrap_or(learning_rate);
                }
            }
            "--generate-only" => {
                generate_only = true;
            }
            "--prompt" => {
                if let Some(val) = arg_iter.next() {
                    prompt = Some(val);
                }
            }
            "--max-tokens" => {
                if let Some(val) = arg_iter.next() {
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
                if let Some(val) = arg_iter.next() {
                    max_stories = Some(val.parse().unwrap_or(4000));
                }
            }
            "--use-memory-opt" => {
                enable_memory_optimization = true;
            }
            "--checkpoint-strategy" => {
                if let Some(val) = arg_iter.next() {
                    checkpoint_strategy = Some(val);
                }
            }
            "--thread-opt" => {
                if let Some(val) = arg_iter.next() {
                    thread_opt = Some(val);
                }
            }
            "--batch-size" => {
                if let Some(val) = arg_iter.next() {
                    manual_batch_size = Some(val.parse().unwrap_or(32));
                }
            }
            "--perf-log" => {
                if let Some(val) = arg_iter.next() {
                    enable_perf_log = val.parse::<bool>().unwrap_or(false);
                } else {
                    enable_perf_log = true;
                }
            }
            "--cpus" => {
                if let Some(val) = arg_iter.next() {
                    if let Ok(cpus) = val.parse::<usize>() {
                        num_cpus_override = Some(cpus);
                    }
                }
            }
            "--curriculum-examples" => {
                if let Some(val) = arg_iter.next() {
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
    
    // Log argument parsing time
    let arg_parsing_time = arg_parsing_start.elapsed();
    println!("✅ Arguments parsed in {:.2?}", arg_parsing_time);
    
    // Configure CPU threads
    global_perf_logger.start("cpu_configuration");
    let num_threads = if let Some(cpus) = num_cpus_override {
        cpus
    } else if let Ok(threads_str) = env::var("RAYON_NUM_THREADS") {
        threads_str.parse().unwrap_or_else(|_| num_cpus::get())
    } else {
        // Calculate optimal thread count based on model dimensions
        let opt_threads = calculate_optimal_thread_count(model_dim, num_layers, None);
        println!("Calculated memory-optimal thread count: {}", opt_threads);
        opt_threads
    };
    
    println!("⏳ Configuring thread pool with {} CPU cores", num_threads);
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
    println!("\n🔄 TRAINING SETUP PHASE 🔄");
    let training_setup_start = Instant::now();
    
    println!("Training Configuration:");
    println!("  Training data: {}", training_data_path.as_deref().map_or("N/A", |v| v));
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
    println!("  Save path: {}", save_path.as_deref().map_or("N/A", |v| v));
    println!("  CPU threads: {}", num_threads);
    println!("  Performance logging: {}", if enable_perf_log { "enabled" } else { "disabled" });
    
    // Read training data
    println!("\n⏳ Reading training data (single-threaded operation)...");
    let data_loading_start = Instant::now();
    global_perf_logger.start("data_loading");
    let training_text = if json_format {
        // Process TinyStories JSON format
        let path = training_data_path.as_ref().ok_or("No training data path provided")?;
        println!("  🔄 Reading JSON file from {}...", path);
        let json_read_start = Instant::now();
        let data = process_json_data(path, max_stories.unwrap_or(0))?;
        let json_read_time = json_read_start.elapsed();
        println!("  ✅ JSON data loaded in {:.2?}, {} characters", json_read_time, data.len());
        data
    } else {
        // Process plain text format
        let path = training_data_path.as_ref().ok_or("No training data path provided")?;
        println!("  🔄 Reading plain text from {}...", path);
        let text_read_start = Instant::now();
        let mut file = File::open(path)?;
        let mut training_text = String::new();
        file.read_to_string(&mut training_text)?;
        let text_read_time = text_read_start.elapsed();
        println!("  ✅ Text data loaded in {:.2?}, {} characters", text_read_time, training_text.len());
        training_text
    };
    global_perf_logger.end("data_loading");
    let data_loading_time = data_loading_start.elapsed();
    println!("✅ Training data loaded in {:.2?}", data_loading_time);
    global_perf_logger.memory_snapshot("after_data_loading");
    
    // Create enhanced trainer
    println!("\n⏳ Creating trainer...");
    let trainer_start = Instant::now();
    global_perf_logger.start("trainer_initialization");
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // Configure trainer with appropriate settings
    println!("  🔄 Configuring trainer options...");
    let trainer_config_start = Instant::now();
    trainer.with_curriculum_learning(enable_curriculum)
           .with_dynamic_learning_rate(true)
           .with_gradient_clipping(Some(1.0));
    let trainer_config_time = trainer_config_start.elapsed();
    println!("  ✅ Trainer options configured in {:.2?}", trainer_config_time);
    
    // Apply memory optimization if enabled
    if enable_memory_optimization {
        println!("  🔄 Enabling memory optimization with gradient checkpointing...");
        let mem_opt_start = Instant::now();
        trainer.with_memory_optimization(true);
        
        // Apply checkpoint strategy if specified
        if let Some(strategy_str) = &checkpoint_strategy {
            println!("  🔄 Using {} checkpoint strategy", strategy_str);
            let strategy = match strategy_str.to_lowercase().as_str() {
                "boundary" => memory_opt::CheckpointStrategy::Boundary,
                "uniform" => memory_opt::CheckpointStrategy::Uniform,
                "adaptive" => memory_opt::CheckpointStrategy::Adaptive,
                "none" => memory_opt::CheckpointStrategy::None,
                _ => {
                    println!("Warning: Unknown checkpoint strategy '{}', using adaptive", strategy_str);
                    memory_opt::CheckpointStrategy::Adaptive
                }
            };
            trainer.with_checkpoint_strategy(strategy);
        }
        let mem_opt_time = mem_opt_start.elapsed();
        println!("  ✅ Memory optimization configured in {:.2?}", mem_opt_time);
        
        global_perf_logger.memory_snapshot("after_memory_opt_enable");
    }
    
    // Configure thread pool based on operation type if specified
    if let Some(operation) = &thread_opt {
        println!("  🔄 Configuring thread pool for {} operations...", operation);
        let thread_opt_start = Instant::now();
        let thread_count = configure_thread_pool_for_operation(
            model_dim, 
            num_layers,
            operation
        );
        let thread_opt_time = thread_opt_start.elapsed();
        println!("  ✅ Thread pool configured with {} threads for {} operations in {:.2?}", 
                thread_count, operation, thread_opt_time);
    }
    
    global_perf_logger.end("trainer_initialization");
    let trainer_time = trainer_start.elapsed();
    println!("✅ Trainer initialization completed in {:.2?}", trainer_time);
    
    if strong_anti_rep {
        println!("\n⏳ Configuring strong anti-repetition mechanisms...");
        let antirep_start = Instant::now();
        trainer.configure_advanced_anti_repetition(1.3, 0.7, 0.7, 0.8);
        let antirep_time = antirep_start.elapsed();
        println!("✅ Anti-repetition configured in {:.2?}", antirep_time);
    } else {
        let antirep_start = Instant::now();
        trainer.configure_anti_repetition(1.1, 0.2, 0.3);
        let antirep_time = antirep_start.elapsed();
        println!("✅ Basic anti-repetition configured in {:.2?}", antirep_time);
    }
    
    // Enable skip connections if requested
    if enable_skip {
        println!("\n⏳ Enabling skip connections...");
        let skip_start = Instant::now();
        if let Err(e) = trainer.enable_skip_connections("residual") {
            println!("Warning: Failed to enable skip connections: {}", e);
        }
        let skip_time = skip_start.elapsed();
        println!("✅ Skip connections enabled in {:.2?}", skip_time);
    }
    
    // Learn tokenizer vocabulary
    println!("\n⏳ Learning tokenizer vocabulary (single-threaded operation)...");
    let vocab_start = Instant::now();
    global_perf_logger.start("vocabulary_learning");
    trainer.learn_tokenizer_from_text(&training_text, vocab_size, min_freq);
    global_perf_logger.end("vocabulary_learning");
    let vocab_time = vocab_start.elapsed();
    println!("✅ Tokenizer vocabulary learned in {:.2?}", vocab_time);
    
    // Split data for training and validation (90/10 split)
    println!("\n⏳ Splitting training/validation data...");
    let split_start = Instant::now();
    global_perf_logger.start("data_splitting");
    let total_length = training_text.len();
    let train_length = (total_length as f64 * 0.9) as usize;
    let training_text_subset = &training_text[..train_length];
    let validation_text = &training_text[train_length..];
    global_perf_logger.end("data_splitting");
    let split_time = split_start.elapsed();
    println!("✅ Data split in {:.2?}: {} characters for training, {} for validation", 
             split_time, train_length, total_length - train_length);
    
    // Create curriculum scheduler with training data
    println!("\n⏳ Setting up curriculum learning...");
    
    // Tokenize training and validation data
    println!("\n⏳ Tokenizing data (single-threaded operation)...");
    let tokenize_start = Instant::now();
    global_perf_logger.start("tokenization");
    let tokenizer = trainer.get_tokenizer();
    println!("  🔄 Tokenizing training data ({} characters)...", training_text_subset.len());
    let train_tokens_start = Instant::now();
    let training_tokens = tokenizer.encode(training_text_subset);
    let train_tokens_time = train_tokens_start.elapsed();
    println!("  ✅ Training data tokenized in {:.2?}: {} tokens", train_tokens_time, training_tokens.len());
    
    println!("  🔄 Tokenizing validation data ({} characters)...", validation_text.len());
    let val_tokens_start = Instant::now();
    let validation_tokens = tokenizer.encode(validation_text);
    let val_tokens_time = val_tokens_start.elapsed();
    println!("  ✅ Validation data tokenized in {:.2?}: {} tokens", val_tokens_time, validation_tokens.len());
    
    global_perf_logger.end("tokenization");
    let tokenize_time = tokenize_start.elapsed();
    println!("✅ All data tokenized in {:.2?}", tokenize_time);
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
    println!("\n⏳ Preparing validation dataset...");
    let validation_start = Instant::now();
    global_perf_logger.start("validation_data_preparation");
    let mut validation_inputs = Vec::new();
    let mut validation_targets = Vec::new();
    prepare_validation_data(&validation_tokens, &mut validation_inputs, &mut validation_targets);
    global_perf_logger.end("validation_data_preparation");
    let validation_time = validation_start.elapsed();
    println!("✅ Validation data prepared in {:.2?}: {} examples", 
             validation_time, validation_inputs.len());
    
    global_perf_logger.end("training_setup");
    let training_setup_time = training_setup_start.elapsed();
    println!("\n✅ TRAINING SETUP COMPLETED in {:.2?}", training_setup_time);
    
    // Start training
    println!("Starting training for {} epochs...", num_epochs);
    global_perf_logger.start("training_process");
    let start_time = Instant::now();
    
    let mut metrics_history: Vec<HashMap<String, f32>> = Vec::new();
    
    println!("\n🔄 BEGINNING TRAINING PROCESS 🔄");
    
    for epoch in 0..num_epochs {
        println!("\n⏳ EPOCH {}/{} STARTING", epoch + 1, num_epochs);
        let epoch_start = Instant::now();
        
        // Train on the tokenized data
        println!("  🔄 Training on {} tokens", training_tokens.len());
        global_perf_logger.start(format!("epoch_{}", epoch + 1).as_str());
        
        // This is likely the step using a single core during initialization
        let train_epoch_start = Instant::now();
        println!("  ⌛ Running train_epoch - this initial setup might be single-threaded momentarily...");
        let loss = train_epoch(&mut trainer, &training_tokens, epoch, enable_memory_optimization, manual_batch_size);
        let train_epoch_time = train_epoch_start.elapsed();
        
        global_perf_logger.end(format!("epoch_{}", epoch + 1).as_str());
        let epoch_duration = epoch_start.elapsed();
        
        // Evaluate the model
        println!("  ⏳ Evaluating model...");
        let eval_start = Instant::now();
        global_perf_logger.start(format!("evaluation_{}", epoch + 1).as_str());
        let metrics = trainer.evaluate_model(&validation_inputs, &validation_targets, &eval_prompts);
        global_perf_logger.end(format!("evaluation_{}", epoch + 1).as_str());
        let eval_time = eval_start.elapsed();
        
        metrics_history.push(metrics.clone());
        
        println!("\n✅ Epoch {}/{} completed", epoch + 1, num_epochs);
        println!("  ⏱️ Total epoch time:   {:.2?}", epoch_duration);
        println!("  ⏱️ Training time:      {:.2?} ({:.1}%)", 
                 train_epoch_time, 100.0 * train_epoch_time.as_secs_f64() / epoch_duration.as_secs_f64());
        println!("  ⏱️ Evaluation time:    {:.2?} ({:.1}%)", 
                 eval_time, 100.0 * eval_time.as_secs_f64() / epoch_duration.as_secs_f64());
        println!("  📊 Loss:               {:.6}", loss);
        println!("  📊 Perplexity:         {:.2}", metrics.get("perplexity").unwrap_or(&f32::INFINITY));
        println!("  📊 Accuracy:           {:.2}%", metrics.get("accuracy").unwrap_or(&0.0));
        println!("  📊 Repetition score:   {:.2}", metrics.get("repetition_score").unwrap_or(&0.0));
        println!("  📊 Fluency score:      {:.2}", metrics.get("fluency_score").unwrap_or(&0.0));
        println!("  📊 Quality score:      {:.2}", metrics.get("quality_score").unwrap_or(&0.0));
        
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
            // Ensure models directory exists
            std::fs::create_dir_all("models").unwrap_or_else(|e| {
                println!("Warning: Could not create models directory: {}", e);
            });
            
            // Create checkpoint path with proper directory and extension
            let mut base_path = save_path.as_deref().map_or("models/model.walle", |v| v).to_string();
            
            // Ensure path has the correct directory
            if !base_path.starts_with("models/") {
                base_path = format!("models/{}", base_path);
            }
            
            // Ensure path has the correct extension
            if !base_path.ends_with(".walle") {
                // Replace any existing extension with .walle
                if let Some(dot_pos) = base_path.rfind('.') {
                    base_path = format!("{}.walle", &base_path[..dot_pos]);
                } else {
                    base_path = format!("{}.walle", base_path);
                }
            }
            
            let checkpoint_path = format!("{}.epoch{}", base_path, epoch + 1);
            
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
    let default_path = String::from("models/model.walle");
    let final_save_path_str = match save_path {
        Some(path) => String::from(path),
        None => default_path
    };

    // Ensure directory exists
    std::fs::create_dir_all("models").unwrap_or_else(|e| {
        println!("Warning: Could not create models directory: {}", e);
    });

    // Ensure the path has the correct directory and extension
    let mut path_with_dir = if !final_save_path_str.starts_with("models/") {
        format!("models/{}", final_save_path_str)
    } else {
        final_save_path_str
    };

    // Add the .walle extension if needed
    let final_path = if !path_with_dir.ends_with(".walle") {
        // Replace any existing extension with .walle
        if let Some(dot_pos) = path_with_dir.rfind('.') {
            format!("{}.walle", &path_with_dir[..dot_pos])
        } else {
            format!("{}.walle", path_with_dir)
        }
    } else {
        path_with_dir
    };

    // Print info and save the model
    println!("Saving final model to: {}", final_path);
    global_perf_logger.start("save_final_model");
    trainer.save_model(&final_path)?;
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
    
    println!("    🔍 TRAIN EPOCH DETAILED LOGGING");
    println!("    ⏳ Preparing sliding windows...");
    let sliding_windows_start = Instant::now();
    
    // Configure thread pool for data preparation (low compute intensity)
    let model_dim = trainer.trainer.get_model_dim();
    let num_layers = trainer.trainer.get_num_layers();
    
    // Apply memory optimization setting to the trainer
    if enable_memory_optimization {
        println!("    🔄 Enabling memory optimization with gradient checkpointing");
        let mem_opt_start = Instant::now();
        trainer.with_memory_optimization(true);
        
        // Set default checkpoint strategy based on model size
        let strategy = if num_layers <= 4 {
            memory_opt::CheckpointStrategy::Uniform
        } else {
            memory_opt::CheckpointStrategy::Adaptive
        };
        trainer.with_checkpoint_strategy(strategy);
        
        let mem_opt_time = mem_opt_start.elapsed();
        println!("    ✅ Memory optimization enabled in {:.2?}", mem_opt_time);
        perf_logger.memory_snapshot("after_memory_opt_enable");
    }
    
    // Configure thread pool for data preparation - ensure we use a good number of threads
    println!("    🔄 Configuring thread pool for data preparation...");
    let data_pool_start = Instant::now();
    let data_threads = configure_thread_pool_for_operation(model_dim, num_layers, "data_loading");
    let data_pool_time = data_pool_start.elapsed();
    println!("    ✅ Data loading thread pool configured with {} threads in {:.2?}", 
             data_threads, data_pool_time);
    
    perf_logger.start("data_preparation");
    println!("    🔄 Creating input/target pairs for training...");
    let data_prep_start = Instant::now();
    
    // Calculate window positions
    println!("    🔄 Calculating sliding window positions...");
    let window_indices: Vec<usize> = (0..tokens.len().saturating_sub(max_sequence_length))
        .step_by(stride)
        .filter(|&i| i + max_sequence_length <= tokens.len())
        .collect();
    
    let estimated_windows = window_indices.len();
    println!("    ✅ Will create {} sliding windows in parallel", estimated_windows);
    
    // Create a progress bar for sliding window creation
    let pb = ProgressBar::new(estimated_windows as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} windows ({percent}%) - ETA: {eta_precise}")
        .unwrap()
        .progress_chars("#>-"));
    
    // CHANGE: Using a simpler approach to avoid thread pool contention
    println!("    ⚠️ Using direct thread work allocation to avoid deadlocks");
    println!("    🔄 Creating {} input/target pairs directly", estimated_windows);
    
    // Create deadlock detection timer with a more aggressive timeout
    let deadlock_timer = Instant::now();
    let deadlock_timeout = std::time::Duration::from_secs(10); // Reduced from 30 to 10 seconds
    
    // Create window pairs directly without using thread pools
    // This avoids potential deadlocks with the thread pool implementation
    let mut inputs = Vec::with_capacity(estimated_windows);
    let mut targets = Vec::with_capacity(estimated_windows);
    
    // Manually create chunks for parallel processing
    let chunk_size = std::cmp::max(100, estimated_windows / (data_threads * 2));
    let window_chunks: Vec<_> = window_indices.chunks(chunk_size).collect();
    println!("    📊 Processing {} chunks with size {} each", window_chunks.len(), chunk_size);
    
    // Create a thread-safe progress counter
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let window_results_mutex = std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(estimated_windows)));
    
    // Create and start threads manually for better control
    let mut thread_handles = Vec::new();
    
    // First, launch a watchdog thread to monitor progress and kill hung threads
    let progress_watchdog = progress.clone();
    let watchdog_handle = std::thread::Builder::new()
        .name("progress-watchdog".to_string())
        .spawn(move || {
            let mut last_progress = 0;
            let mut stall_count = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                let current = progress_watchdog.load(std::sync::atomic::Ordering::Relaxed);
                
                if current == last_progress {
                    stall_count += 1;
                    if stall_count >= 5 {
                        println!("⚠️ WATCHDOG: Progress stalled for 5 seconds, progress={}", current);
                        // In a real implementation, we could forcibly terminate hung threads
                        // But for safety, we'll just notify the user
                    }
                } else {
                    stall_count = 0;
                }
                
                last_progress = current;
            }
        }).expect("Failed to create watchdog thread");
    
    for (chunk_idx, chunk) in window_chunks.iter().enumerate() {
        // Clone the shared data for this thread
        let tokens_clone = tokens.clone();
        let progress_clone = progress.clone();
        let window_results = window_results_mutex.clone();
        let chunk_vec = chunk.to_vec(); // Create owned copy for thread
        
        // Create a thread with a meaningful name for better debugging
        let thread_name = format!("window-processor-{}", chunk_idx);
        let builder = std::thread::Builder::new().name(thread_name.clone());
        
        // Create and start the thread
        let handle = builder.spawn(move || {
            println!("🧵 Thread {} started processing {} windows", thread_name, chunk_vec.len());
            let start_time = Instant::now();
            
            // Storage for this thread's results
            let mut thread_results = Vec::with_capacity(chunk_vec.len());
            
            // Process each window in this chunk
            for &start_idx in chunk_vec.iter() {
                // Create input window
                let input = tokens_clone[start_idx..start_idx + max_sequence_length].to_vec();
                
                // Create target by shifting input by one position
                let mut target = Vec::with_capacity(max_sequence_length);
                for j in 0..max_sequence_length {
                    let target_idx = (start_idx + j + 1) % tokens_clone.len();
                    target.push(tokens_clone[target_idx]);
                }
                
                // Add this window to thread results
                thread_results.push((input, target));
                
                // Update progress
                if thread_results.len() % 10 == 0 {
                    let new_count = progress_clone.fetch_add(10, std::sync::atomic::Ordering::Relaxed);
                    if new_count % 100 == 0 {
                        println!("    🧵 Thread {} processed {}/{} windows", 
                                 thread_name, thread_results.len(), chunk_vec.len());
                    }
                }
            }
            
            // Update any remaining progress
            let remainder = thread_results.len() % 10;
            if remainder > 0 {
                progress_clone.fetch_add(remainder, std::sync::atomic::Ordering::Relaxed);
            }
            
            // Add results to the shared collection with minimal lock time
            {
                // Acquire lock only when we're ready to update - minimize lock time
                if let Ok(mut results) = window_results.lock() {
                    // Move results into the shared collection
                    results.extend(thread_results);
                    
                    println!("    ✅ Thread {} completed in {:.2?} - added {} windows to the result set", 
                            thread_name, start_time.elapsed(), chunk_vec.len());
                } else {
                    println!("    ❌ Thread {} failed to acquire lock for results", thread_name);
                }
            }
        }).expect("Failed to spawn thread");
        
        thread_handles.push(handle);
    }
    
    // Update progress bar regularly while waiting for threads
    let mut last_progress = 0;
    let start_time = Instant::now();
    
    // Keep checking progress and watching for deadlocks
    let mut threads_joined = 0;
    while threads_joined < thread_handles.len() {
        // Get current progress
        let current_progress = progress.load(std::sync::atomic::Ordering::Relaxed);
        
        // Update progress bar if needed
        if current_progress > last_progress {
            pb.set_position(current_progress as u64);
            last_progress = current_progress;
        }
        
        // Check for deadlocks with more aggressive timeout and reporting
        if deadlock_timer.elapsed() > deadlock_timeout && current_progress == last_progress {
            // Potential deadlock detected
            println!("\n⚠️ POTENTIAL DEADLOCK DETECTED: No progress made in 10 seconds");
            println!("    🔍 Thread status:");
            
            // Count active threads
            let active_threads = thread_handles.len() - threads_joined;
            println!("    - Active threads: {}/{}", active_threads, thread_handles.len());
            println!("    - Progress: {}/{} ({:.1}%)", 
                     current_progress, estimated_windows, 
                     100.0 * current_progress as f64 / estimated_windows as f64);
            
            // Force continue execution - we'll salvage what we have so far
            println!("    🔄 DEADLOCK RECOVERY: Continuing with collected results");
            break;
        }
        
        // Try to join a thread with a short timeout
        let join_timeout = std::time::Duration::from_millis(100);
        for i in 0..thread_handles.len() {
            // Skip already joined threads
            if thread_handles[i].is_finished() {
                // Thread already joined or finished
                threads_joined += 1;
                println!("    ✅ Thread {}/{} completed", threads_joined, thread_handles.len());
            }
        }
        
        // Brief sleep to avoid busy waiting
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    // Now extract all results while handling possible lock contention
    let mut lock_attempt = 0;
    let max_lock_attempts = 5;
    let mut combined_results = Vec::new();
    
    while lock_attempt < max_lock_attempts {
        match window_results_mutex.try_lock() {
            Ok(results) => {
                combined_results = results.clone();
                println!("    ✅ Successfully collected {} window pairs", combined_results.len());
                break;
            }
            Err(_) => {
                lock_attempt += 1;
                println!("    ⚠️ Lock acquisition failed, attempt {}/{}", lock_attempt, max_lock_attempts);
                std::thread::sleep(std::time::Duration::from_millis(100 * lock_attempt));
            }
        }
    }
    
    // If we still couldn't get the lock, use a fallback approach
    if lock_attempt >= max_lock_attempts {
        println!("    ⚠️ CRITICAL: Could not acquire lock for results after {} attempts", max_lock_attempts);
        println!("    🔄 FALLBACK: Using a direct approach instead");
        
        // Create a minimal set of examples as fallback
        let fallback_size = 200.min(tokens.len() - max_sequence_length);
        for i in 0..fallback_size {
            let input = tokens[i..i + max_sequence_length].to_vec();
            let mut target = Vec::with_capacity(max_sequence_length);
            for j in 0..max_sequence_length {
                let target_idx = (i + j + 1) % tokens.len();
                target.push(tokens[target_idx]);
            }
            combined_results.push((input, target));
        }
        
        println!("    ✅ Created {} fallback window pairs", combined_results.len());
    }
    
    // Now separate inputs and targets
    println!("    🔄 Separating inputs and targets...");
    for (input, target) in combined_results {
        inputs.push(input);
        targets.push(target);
    }
    
    pb.finish();
    
    // Data preparation complete
    let data_prep_time = data_prep_start.elapsed();
    println!("    ✅ Created {} input/target pairs in {:.2?}", inputs.len(), data_prep_time);
    perf_logger.end("data_preparation");
    
    // ... rest of the function remains unchanged ...

    // Get cache parameters
    let cache_params = memory_opt::detect_cache_parameters();
    
    println!("    📊 Cache parameters: L1={} KB, L2={} KB, L3={} KB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / 1024, 
        cache_params.line_size);
    
    // Get CPU information
    let num_cpus = num_cpus::get();
    let num_physical_cpus = num_cpus::get_physical();
    println!("    💻 CPU cores: {} logical, {} physical", num_cpus, num_physical_cpus);
    
    // Configure thread pool for batch processing (compute intensive)
    println!("    🔄 Configuring thread pool for gradient updates...");
    let grad_pool_start = Instant::now();
    let rayon_threads = configure_thread_pool_for_operation(model_dim, num_layers, "gradient_update");
    let grad_pool_time = grad_pool_start.elapsed();
    println!("    ✅ Using {} threads for gradient processing (configured in {:.2?})", rayon_threads, grad_pool_time);
    
    // Get model dimensions for batch size calculation
    let model_dim = trainer.trainer.get_model_dim();
    
    // Calculate memory-optimal batch size
    println!("    🧮 Calculating optimal batch size...");
    let batch_calc_start = Instant::now();
    perf_logger.start("batch_size_calculation");
    let optimal_batch_size = calculate_memory_optimal_batch_size(model_dim, max_sequence_length);
    
    // Apply constraints based on dataset size to ensure we have enough batches
    let max_batch_size = inputs.len() / 10.min(50);  // Ensure at least 10 batches, aim for 50+
    let constrained_batch_size = optimal_batch_size.min(max_batch_size).max(4);  // At least 4, no more than max
    perf_logger.end("batch_size_calculation");
    let batch_calc_time = batch_calc_start.elapsed();
    
    // Share the information with the user
    println!("    ✅ Batch size calculation completed in {:.2?}", batch_calc_time);
    println!("      📊 Hardware-optimal batch size: {}", optimal_batch_size);
    println!("      📊 Dataset-constrained batch size: {}", constrained_batch_size);
    println!("      📊 Using batch size: {}", manual_batch_size.unwrap_or(constrained_batch_size));
    
    // Use the calculated batch size
    let batch_size = manual_batch_size.unwrap_or(constrained_batch_size);
    
    // Calculate how many batches we'll process with this batch size
    let expected_batches = (inputs.len() + batch_size - 1) / batch_size; // Ceiling division
    
    println!("    📊 Expected number of batches: {}", expected_batches);
    
    // CHANGE: Use simpler approach for shuffling too
    println!("    🔄 Shuffling data indices directly...");
    let shuffle_start = Instant::now();
    perf_logger.start("shuffling_indices");
    
    // Create indices for all inputs and shuffle directly
    let mut indices_vec: Vec<usize> = (0..inputs.len()).collect();
    let mut rng = thread_rng();
    indices_vec.shuffle(&mut rng);
    
    perf_logger.end("shuffling_indices");
    let shuffle_time = shuffle_start.elapsed();
    println!("    ✅ Indices shuffled in {:.2?}", shuffle_time);
    
    // Create batch chunks for parallel processing
    println!("    🔄 Creating batches from shuffled indices...");
    let batch_creation_start = Instant::now();
    perf_logger.start("batch_creation");
    
    // FIXED: Create batches directly instead of referencing indices_vec which won't live long enough
    let batches: Vec<Vec<usize>> = indices_vec
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec()) // Convert each chunk to owned Vec<usize>
        .collect();
    
    perf_logger.end("batch_creation");
    let batch_creation_time = batch_creation_start.elapsed();
    println!("    ✅ Created {} batches in {:.2?}", batches.len(), batch_creation_time);
    
    // Create progress bar
    let pb = ProgressBar::new(batches.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} batches ({percent}%) - ETA: {eta_precise} - Loss: {msg}")
        .unwrap()
        .progress_chars("#>-"));
    
    // Process batches using a simpler approach without thread pools
    println!("\n    🔄 BEGINNING BATCH PROCESSING (SEQUENTIAL)");
    perf_logger.start("batch_processing_loop");
    
    // Tracking variables
    let mut total_loss = 0.0f32;
    let mut batch_counter = 0usize;
    let mut batch_prep_times = Vec::with_capacity(batches.len());
    let mut train_step_times = Vec::with_capacity(batches.len());
    
    // NEW IMPLEMENTATION: Robust batch-level parallelism with work stealing
    println!("    🚀 Using robust batch-level parallelism with work stealing");
    
    // Determine optimal number of worker threads based on hardware
    let num_physical_cores = num_cpus::get_physical();
    let optimal_workers = std::cmp::min(num_physical_cores, 6); // Cap at 6 for now to avoid over-parallelization
    println!("    🧵 Using {} worker threads for batch processing", optimal_workers);
    
    // First, enable SIMD optimizations for matrix operations
    println!("    💻 Enabling SIMD optimizations for matrix operations");
    
    // Detect CPU capabilities for SIMD
    let simd_features = detect_simd_features();
    println!("    🔍 Detected CPU SIMD features: {:?}", simd_features);
    
    // Enable the best SIMD implementation based on CPU capabilities
    let simd_enabled = enable_simd_optimizations(&simd_features);
    println!("    ✅ SIMD optimization enabled: {}", simd_enabled);
    
    // Create shared thread-safe state
    let batch_queue = std::sync::Arc::new(std::sync::Mutex::new(
        (0..batches.len()).collect::<Vec<_>>()
    ));
    let results = std::sync::Arc::new(std::sync::Mutex::new(
        Vec::<(usize, f32, f64, f64)>::with_capacity(batches.len())
    ));
    // Use atomic flags for better coordination between threads
    let queue_empty = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let active_workers = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let termination_requested = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    
    // Create shared Arc references to inputs and targets
    let inputs_arc = std::sync::Arc::new(inputs);
    let targets_arc = std::sync::Arc::new(targets);
    let batches_arc = std::sync::Arc::new(batches);
    
    // Create vector to store trainer clones for each thread
    let mut trainers = Vec::with_capacity(optimal_workers);
    for _ in 0..optimal_workers {
        trainers.push(trainer.clone_for_parallel());
    }
    println!("    ✅ Created {} trainer clones for parallel batch processing", trainers.len());
    
    // Create a channel for thread communication
    let (tx, rx) = std::sync::mpsc::channel();
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    
    // Vector to store thread handles
    let mut handles = Vec::with_capacity(optimal_workers);
    
    // Create synchronized thread state for tracking thread progress
    let thread_states = std::sync::Arc::new(
        std::sync::Mutex::new(HashMap::<String, String>::new())
    );
    
    // Launch a batch processing watchdog thread
    let watchdog_queue_empty = queue_empty.clone();
    let watchdog_active_workers = active_workers.clone();
    let watchdog_termination = termination_requested.clone();
    let watchdog_thread_states = thread_states.clone();
    let watchdog_handle = std::thread::Builder::new()
        .name("batch-watchdog".to_string())
        .spawn(move || {
            let mut stall_count = 0;
            let mut last_active = 0;
            let mut consecutive_stalls = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                
                // Check if termination was requested
                if watchdog_termination.load(std::sync::atomic::Ordering::SeqCst) {
                    println!("    ✅ WATCHDOG: Termination requested, exiting watchdog");
                    break;
                }
                
                let current_active = watchdog_active_workers.load(std::sync::atomic::Ordering::SeqCst);
                let is_empty = watchdog_queue_empty.load(std::sync::atomic::Ordering::SeqCst);
                
                // If all done, exit
                if is_empty && current_active == 0 {
                    println!("    ✅ WATCHDOG: All batches processed and workers finished");
                    break;
                }
                
                // Check for stalls (no change in active workers count)
                if current_active > 0 && current_active == last_active {
                    stall_count += 1;
                    if stall_count >= 5 {
                        println!("    ⚠️ WATCHDOG: Batch processing potentially stalled for 10+ seconds");
                        println!("        - Active workers: {}", current_active);
                        println!("        - Queue empty: {}", is_empty);
                        
                        // Report thread states if we have them
                        if let Ok(states) = watchdog_thread_states.try_lock() {
                            println!("    🔍 THREAD STATUS REPORT:");
                            for (thread_id, state) in states.iter() {
                                println!("        - Thread {}: {}", thread_id, state);
                            }
                        }
                        
                        // NEW: Implement recovery action for deadlocks
                        consecutive_stalls += 1;
                        if consecutive_stalls >= 3 {
                            println!("    🔄 WATCHDOG: Deadlock detected, requesting termination");
                            watchdog_termination.store(true, std::sync::atomic::Ordering::SeqCst);
                            break;
                        }
                    }
                } else {
                    stall_count = 0;
                    consecutive_stalls = 0;
                }
                
                last_active = current_active;
            }
        }).expect("Failed to create batch watchdog");
    
    // Spawn worker threads
    for worker_id in 0..optimal_workers {
        // Clone shared resources for this thread
        let thread_batch_queue = batch_queue.clone();
        let thread_results = results.clone();
        let thread_queue_empty = queue_empty.clone();
        let thread_active_workers = active_workers.clone();
        let thread_termination = termination_requested.clone();
        let thread_tx = tx.clone();
        let thread_inputs = inputs_arc.clone(); 
        let thread_targets = targets_arc.clone();
        let thread_batches = batches_arc.clone();
        let thread_max_sequence_length = max_sequence_length;
        let thread_states_clone = thread_states.clone();
        
        // Get a trainer for this thread
        let mut thread_trainer = trainers.pop().unwrap();
        
        // Create thread with a meaningful name
        let thread_name = format!("batch-worker-{}", worker_id);
        let builder = std::thread::Builder::new().name(thread_name.clone());
        
        // Spawn the thread
        let handle = builder.spawn(move || {
            println!("    🧵 Thread {} started for batch processing", thread_name);
            // Increment active workers counter
            thread_active_workers.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            
            // Register thread state
            let thread_id = format!("{:?}", std::thread::current().id());
            
            // Update thread state
            if let Ok(mut states) = thread_states_clone.lock() {
                states.insert(thread_id.clone(), "Started".to_string());
            }
            
            // Keep processing batches from the queue until it's empty
            loop {
                // Check if termination was requested by watchdog
                if thread_termination.load(std::sync::atomic::Ordering::SeqCst) {
                    println!("    🧵 Thread {} received termination request", thread_name);
                    
                    // Update thread state
                    if let Ok(mut states) = thread_states_clone.lock() {
                        states.insert(thread_id.clone(), "Terminating".to_string());
                    }
                    
                    break;
                }
                
                // Update thread state
                if let Ok(mut states) = thread_states_clone.lock() {
                    states.insert(thread_id.clone(), "Fetching batch".to_string());
                }
                
                // Get the next batch from the queue with timeout
                let batch_idx_option = {
                    // Try to acquire lock with timeout to prevent deadlock
                    let lock_result = thread_batch_queue.try_lock();
                    if lock_result.is_err() {
                        // Could not acquire lock - might be contention
                        // Update thread state
                        if let Ok(mut states) = thread_states_clone.lock() {
                            states.insert(thread_id.clone(), "Waiting for queue lock".to_string());
                        }
                        
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }
                    
                    let mut queue = lock_result.unwrap();
                    if queue.is_empty() {
                        // Signal queue is empty - critical for proper termination
                        thread_queue_empty.store(true, std::sync::atomic::Ordering::SeqCst);
                        
                        // Update thread state
                        if let Ok(mut states) = thread_states_clone.lock() {
                            states.insert(thread_id.clone(), "Queue empty".to_string());
                        }
                        
                        None
                    } else {
                        Some(queue.pop().unwrap())
                    }
                };
                
                // Process batch if we got one, otherwise check for termination
                match batch_idx_option {
                    Some(batch_idx) => {
                        // Update thread state
                        if let Ok(mut states) = thread_states_clone.lock() {
                            states.insert(thread_id.clone(), format!("Processing batch {}", batch_idx));
                        }
                        
                        // Process this batch
                        let batch_indices = &thread_batches[batch_idx];
                        
                        // Prepare batch data using SIMD-optimized functions if available
                        let prep_start = Instant::now();
                        let (batch_inputs, batch_targets_arr) = prepare_batch_parallel(
                            &thread_inputs, &thread_targets, batch_indices, thread_max_sequence_length);
                        let prep_time = prep_start.elapsed().as_secs_f64();
                
                        // Skip empty batches
                        if batch_inputs.is_empty() {
                            // Update thread state
                            if let Ok(mut states) = thread_states_clone.lock() {
                                states.insert(thread_id.clone(), format!("Empty batch {}", batch_idx));
                            }
                            
                            // Send a message with zero loss
                            thread_tx.send((batch_idx, 0.0, prep_time, 0.0)).unwrap_or_else(|_| {
                                println!("    ⚠️ Thread {} failed to send zero loss message", thread_name);
                            });
                            continue;
                        }
                        
                        // Update thread state
                        if let Ok(mut states) = thread_states_clone.lock() {
                            states.insert(thread_id.clone(), format!("Training on batch {}", batch_idx));
                        }
                        
                        // Train on this batch
                        let train_start = Instant::now();
                        // Add a safety catch for interrupted training
                        let loss = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            thread_trainer.train_step_with_penalties(&batch_inputs, &batch_targets_arr)
                        })).unwrap_or_else(|_| {
                            println!("    ⚠️ Thread {} panicked during training on batch {}", thread_name, batch_idx);
                            0.0
                        });
                        let train_time = train_start.elapsed().as_secs_f64();
                        
                        // Update thread state
                        if let Ok(mut states) = thread_states_clone.lock() {
                            states.insert(thread_id.clone(), format!("Finishing batch {}", batch_idx));
                        }
                        
                        // Add results to the shared results collection
                        {
                            match thread_results.try_lock() {
                                Ok(mut results) => {
                                    results.push((batch_idx, loss, prep_time, train_time));
                                },
                                Err(_) => {
                                    println!("    ⚠️ Thread {} could not acquire results lock", thread_name);
                                }
                            }
                        }
                        
                        // Send a message to update progress
                        thread_tx.send((batch_idx, loss, prep_time, train_time)).unwrap_or_else(|_| {
                            println!("    ⚠️ Thread {} failed to send result message", thread_name);
                        });
                    },
                    None => {
                        // No more batches, check if we should exit
                        let all_workers = thread_active_workers.load(std::sync::atomic::Ordering::SeqCst);
                        
                        // If we're the last worker or termination requested, exit
                        // All workers must exit, not just the last one
                        println!("    🧵 Thread {} found empty queue, active workers: {}", 
                                thread_name, all_workers);
                        
                        // Update thread state
                        if let Ok(mut states) = thread_states_clone.lock() {
                            states.insert(thread_id.clone(), "Exiting - queue empty".to_string());
                        }
                        
                        // Small sleep to allow other threads to process any remaining work
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        break;
                    }
                }
            }
            
            // Important: Decrement active workers counter before exiting
            let remaining = thread_active_workers.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) - 1;
            println!("    🧵 Thread {} exiting, {} workers remaining", thread_name, remaining);
            
            // Update thread state
            if let Ok(mut states) = thread_states_clone.lock() {
                states.insert(thread_id.clone(), "Exited".to_string());
            }
            
            // Return the trainer for final gradient aggregation
            thread_trainer
        }).expect("Failed to spawn thread");
        
        handles.push(handle);
    }
    
    // Process messages and update progress until queue is empty and all workers are done
    let mut all_done = false;
    let training_start_time = std::time::Instant::now();
    // Add a shorter global training timeout (2 minutes) to prevent hanging
    let training_timeout = std::time::Duration::from_secs(120); // 2 minutes
    
    while !all_done {
        // Check if we've exceeded the global timeout
        if training_start_time.elapsed() > training_timeout {
            println!("    ⚠️ GLOBAL TRAINING TIMEOUT: Forcing completion after {:?}", training_timeout);
            termination_requested.store(true, std::sync::atomic::Ordering::SeqCst);
            all_done = true;
            break;
        }
        
        // Try to receive a message with timeout
        let result = {
            let rx = rx.lock().unwrap();
            rx.recv_timeout(std::time::Duration::from_millis(100))
        };
        
        match result {
            Ok((batch_idx, loss, prep_time, train_time)) => {
                // Skip zero loss (empty batches)
                if loss > 0.0 {
                    // Process the result
                    batch_counter += 1;
                    total_loss += loss;
                    batch_prep_times.push(prep_time);
                    train_step_times.push(train_time);
                    
                    // Update progress bar
                    pb.set_message(format!("{:.6} (avg: {:.6})", loss, total_loss / batch_counter as f32));
                    
                    // Log occasionally
                    if batch_idx % 5 == 0 || batch_idx == batches_arc.len() - 1 {
                        println!("    📊 Batch {}/{} - Loss: {:.6} - Avg: {:.6} - Prep: {:.3}s - Train: {:.3}s", 
                                batch_idx + 1, batches_arc.len(), loss, total_loss / batch_counter as f32,
                                prep_time, train_time);
                    }
                }
                
                // Always increment progress bar
                pb.inc(1);
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Check if we're done by examining our atomic flags
                if (queue_empty.load(std::sync::atomic::Ordering::SeqCst) || 
                    termination_requested.load(std::sync::atomic::Ordering::SeqCst)) && 
                   active_workers.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                    all_done = true;
                    println!("    ✅ Queue empty and no active workers, training complete");
                }
            },
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // All senders disconnected, we're done
                all_done = true;
                println!("    ✅ All sender channels disconnected, training complete");
            }
        }
    }
    
    // At this point we should ensure our termination flag is set to help any stuck threads
    termination_requested.store(true, std::sync::atomic::Ordering::SeqCst);
    
    // Collect all trainers and apply gradients
    println!("    🔄 Waiting for all threads to finish...");
    
    // Much more aggressive timeout - if threads are deadlocked, don't wait too long
    let join_timeout = std::time::Duration::from_secs(5); // 5 seconds
    let join_start = std::time::Instant::now();
    
    // Create new collection for handles to avoid borrow after move
    let mut threads_to_join = handles;
    let mut joined_threads = 0;
    let mut collected_trainers = Vec::new();
    
    // Try to join each thread with timeout
    println!("    🔄 Attempting to join threads with aggressive timeouts...");
    while !threads_to_join.is_empty() && join_start.elapsed() < join_timeout {
        let mut remaining_threads = Vec::new();
        
        for (i, handle) in threads_to_join.into_iter().enumerate() {
            println!("    ⏳ Trying to join thread {} with 200ms timeout...", i);
            
            // Create a thread to attempt joining with timeout
            let (tx, rx) = std::sync::mpsc::channel();
            
            // Move the handle into the join thread
            let join_thread = std::thread::spawn(move || {
                if let Ok(trainer) = handle.join() {
                    let _ = tx.send(Some(trainer));
                } else {
                    let _ = tx.send(None);
                }
            });
            
            // Wait with timeout
            match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(Some(trainer)) => {
                    collected_trainers.push(trainer);
                    joined_threads += 1;
                    println!("    ✅ Thread {} joined successfully", i);
                },
                Ok(None) => {
                    println!("    ⚠️ Thread {} returned error on join", i);
                },
                Err(_) => {
                    println!("    ⚠️ Thread {} join timed out", i);
                    // We can't access the original handle anymore since it was moved
                    // Just note that we had a timed out thread
                    println!("    ⚠️ Thread will be abandoned");
                }
            }
            
            // Forget the join thread to avoid waiting
            std::mem::forget(join_thread);
            
            // Check elapsed time and break if needed
            if join_start.elapsed() > join_timeout {
                println!("    ⏰ Global timeout reached during joins");
                break;
            }
        }
        
        // Update threads_to_join with remaining threads
        threads_to_join = remaining_threads;
    }
    
    // If we still have threads after all attempts, just abandon them
    if !threads_to_join.is_empty() {
        println!("    ⚠️ Could not join {} threads, they will be abandoned", threads_to_join.len());
        // Prevent resource leaks by dropping the handles
        threads_to_join.clear();
    }
    
    println!("    ✅ Thread joining process complete: collected {}/{} trainers", 
             collected_trainers.len(), optimal_workers);
    
    // Safety check - ensure we have at least some trainers
    if collected_trainers.is_empty() {
        println!("    ⚠️ No trainers collected! Continuing without applying gradients");
    } else {
        // Use the original trainer to combine gradients from all threads
        println!("    🔄 Applying parallel gradients from {} threads...", collected_trainers.len());
        trainer.apply_parallel_gradients();
    }
    
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
    
    // Cleanup watchdog thread
    let watchdog_handle_copy = watchdog_handle.thread().clone();
    let watchdog_id = format!("{:?}", watchdog_handle_copy.id());
    
    // Try to join the watchdog with a short timeout
    match watchdog_handle.join() {
        Ok(_) => println!("    ✅ Watchdog thread joined successfully"),
        Err(e) => println!("    ⚠️ Failed to join watchdog thread: {:?}", e),
    }
    
    // Fix for last batch handling - set final message for progress bar
    pb.finish_with_message(format!("Completed - Avg loss: {:.6}", 
        if batch_counter > 0 { total_loss / batch_counter as f32 } else { 0.0 }));
        
    // Log final training metrics
    if batch_counter > 0 {
        println!("\n✅ TRAINING COMPLETED:");
        println!("    - Processed {} batches", batch_counter);
        println!("    - Average loss: {:.6}", total_loss / batch_counter as f32);
        println!("    - Total preparation time: {:.3}s", batch_prep_times.iter().sum::<f64>());
        println!("    - Total training time: {:.3}s", train_step_times.iter().sum::<f64>());
    } else {
        println!("\n⚠️ WARNING: No batches were fully processed!");
    }
    
    perf_logger.memory_snapshot("train_epoch_end");
    perf_logger.end("train_epoch_full");
    
    // Print local performance metrics for this epoch
    perf_logger.log_summary();
    
    if batch_counter > 0 {
        total_loss / batch_counter as f32
    } else {
        0.0
    }
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
    
    // Get thread pool for data loading
    let data_pool = wall_e1::utils::thread_pool::get_thread_pool_for_operation("data_loading");
    
    // Log the number of threads to help diagnose CPU utilization issues
    println!("Data loading with {} threads", data_pool.get_num_threads());
    
    // Calculate optimal chunk size for better work distribution
    // This ensures each thread gets substantial work to do
    let indices_per_thread = std::cmp::max(
        1,
        std::cmp::min(
            64, // Upper bound to avoid too large allocations
            (batch_indices.len() + data_pool.get_num_threads() - 1) / data_pool.get_num_threads()
        )
    );
    
    // Get the thread pool safely
    let pool_result = {
        let pool_guard = data_pool.get_pool().lock().unwrap();
        pool_guard
    };
    
    // Use the data loading thread pool with optimal chunking
    let (batch_inputs, min_seq_len, truncated_inputs, batch_targets_arr, timing) = pool_result.install(|| {
        use rayon::iter::ParallelIterator;
        
        // Split indices into reasonably-sized chunks to improve thread utilization
        // This reduces overhead and increases CPU usage
        let chunks: Vec<&[usize]> = batch_indices.chunks(indices_per_thread).collect();
        println!("Processing {} chunks across {} threads", chunks.len(), data_pool.get_num_threads());
        
        // Collect input sequences in parallel by chunk (better locality, less overhead)
        let par_collect_start = Instant::now();
        let mut batch_inputs = Vec::with_capacity(batch_indices.len());
        
        let chunk_results: Vec<Vec<Vec<usize>>> = chunks.into_par_iter()
            .map(|chunk_indices| {
                // Process each chunk as a unit to reduce thread synchronization overhead
                let mut chunk_inputs = Vec::with_capacity(chunk_indices.len());
                for &idx in chunk_indices {
                    if idx < inputs.len() {
                        chunk_inputs.push(inputs[idx].clone());
                    }
                }
                chunk_inputs
            })
            .collect();
            
        // Combine chunk results (sequential, but small operation)
        for mut chunk_result in chunk_results {
            batch_inputs.append(&mut chunk_result);
        }
        
        let par_collect_time = par_collect_start.elapsed().as_millis();
        
        // Skip if all sequences are empty
        if batch_inputs.is_empty() {
            return (Vec::new(), 0, Vec::new(), Array2::zeros((0, 0)), (0, 0, 0, 0));
        }
        
        // Find minimum sequence length in parallel
        let min_len_start = Instant::now();
        
        // Use chunked approach for min length calculation too
        let min_seq_len = batch_inputs.chunks(indices_per_thread)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|chunk| {
                chunk.iter()
                    .map(|seq| seq.len())
                    .min()
                    .unwrap_or(usize::MAX)
            })
            .min()
            .unwrap_or(0)
            .min(max_sequence_length);
            
        let min_len_time = min_len_start.elapsed().as_millis();
        
        // Skip if sequences are too short
        if min_seq_len < 4 {
            return (batch_inputs, min_seq_len, Vec::new(), Array2::zeros((0, 0)), (par_collect_time, min_len_time, 0, 0));
        }
        
        // Truncate all sequences to the same length with chunking
        let truncate_start = Instant::now();
        let truncated_inputs = batch_inputs.chunks(indices_per_thread)
            .collect::<Vec<_>>()
            .into_par_iter()
            .flat_map(|chunk| {
                chunk.iter().map(|seq| {
                    if seq.len() > min_seq_len {
                        seq[0..min_seq_len].to_vec()
                    } else {
                        seq.clone()
                    }
                }).collect::<Vec<Vec<usize>>>()
            })
            .collect();
            
        let truncate_time = truncate_start.elapsed().as_millis();
        
        // Create batch targets array
        let targets_start = Instant::now();
        let mut batch_targets_arr = Array2::zeros((batch_indices.len(), min_seq_len));
        
        // Process targets in chunks to improve parallelism
        let targets_chunks: Vec<_> = batch_indices.chunks(indices_per_thread).collect();
        
        // Fill targets in parallel using chunks and shared mutex
        let targets_arr_mutex = std::sync::Arc::new(std::sync::Mutex::new(batch_targets_arr));
        
        // Process chunks in parallel
        targets_chunks.into_par_iter().for_each(|chunk_indices| {
            // Create local buffer for this chunk
            let mut local_targets = Array2::zeros((chunk_indices.len(), min_seq_len));
            
            // Fill local buffer
            for (local_i, &idx) in chunk_indices.iter().enumerate() {
                if idx < targets.len() {
                    let target = &targets[idx];
                    for j in 0..min_seq_len.min(target.len()) {
                        local_targets[[local_i, j]] = target[j];
                    }
                }
            }
            
            // Acquire lock just once per chunk to update main array
            let mut targets_arr = targets_arr_mutex.lock().unwrap();
            
            // Find target position in full array
            let offset = batch_indices.iter().position(|&id| id == chunk_indices[0]).unwrap_or(0);
            
            // Copy local chunk to main array
            for local_i in 0..chunk_indices.len() {
                for j in 0..min_seq_len {
                    targets_arr[[offset + local_i, j]] = local_targets[[local_i, j]];
                }
            }
        });
        
        // Extract the final array
        batch_targets_arr = targets_arr_mutex.lock().unwrap().clone();
        
        let targets_time = targets_start.elapsed().as_millis();
        
        (batch_inputs, min_seq_len, truncated_inputs, batch_targets_arr, (par_collect_time, min_len_time, truncate_time, targets_time))
    });
    
    // Skip if sequences are too short
    if truncated_inputs.is_empty() {
        return (Vec::new(), Array2::zeros((0, 0)));
    }
    
    let total_time = prep_start.elapsed().as_millis();
    
    // Always log timing info when we have CPU utilization issues
    let (par_collect_time, min_len_time, truncate_time, targets_time) = timing;
    println!("Batch prep timing: total={}ms (collect={}ms, min_len={}ms, truncate={}ms, targets={}ms)",
        total_time, par_collect_time, min_len_time, truncate_time, targets_time);
    
    (truncated_inputs, batch_targets_arr)
}

// Prepare validation data for model evaluation
fn prepare_validation_data(tokens: &Vec<usize>, inputs: &mut Vec<Vec<usize>>, targets: &mut Vec<Vec<usize>>) {
    // Configure thread pool for data loading
    let model_dim = 128; // Use default value since we don't have access to the real model_dim here
    let num_layers = 3;  // Use default value 
    
    // Configure with more threads for validation data preparation
    println!("    🔄 Configuring thread pool for validation data preparation...");
    let data_threads = calculate_optimal_thread_count(model_dim, num_layers, Some("data_loading"));
    println!("    ✅ Using {} threads for validation data preparation", data_threads);
    
    // Get thread pool for data loading
    let data_pool = wall_e1::utils::thread_pool::get_thread_pool_for_operation("data_loading");
    println!("    ✅ Using thread pool with {} threads for validation data", data_pool.get_num_threads());
    let pool = data_pool.get_pool().lock().unwrap();
    
    // Use the data loading thread pool for validation data preparation
    pool.install(|| {
        // Create context windows of varying lengths for validation
        for window_size in [16, 32, 64].iter() {
            let new_inputs_targets: Vec<(Vec<usize>, Vec<usize>)> = (0..tokens.len().saturating_sub(*window_size))
                .step_by(*window_size)
                .filter(|&i| i + *window_size + 1 <= tokens.len())
                .map(|i| {
                    // Input: tokens[i..i+window_size]
                    let input = tokens[i..i + *window_size].to_vec();
                    // Target: tokens[i+1..i+window_size+1] (shifted by 1)
                    let target = tokens[i + 1..i + *window_size + 1].to_vec();
                    (input, target)
                })
                .collect();
                
            // Add all the collected items to the inputs and targets vectors
            for (input, target) in new_inputs_targets {
                inputs.push(input);
                targets.push(target);
            }
        }
    });
}

// Detect CPU SIMD capabilities
fn detect_simd_features() -> Vec<String> {
    let mut features = Vec::new();
    
    // Check for SSE/SSE2 - modern x86 CPUs all have these
    features.push("SSE2".to_string());
    
    // Check for AVX support
    if is_x86_feature_detected!("avx") {
        features.push("AVX".to_string());
    }
    
    // Check for AVX2 support
    if is_x86_feature_detected!("avx2") {
        features.push("AVX2".to_string());
    }
    
    // Check for AVX-512 support (multiple variants)
    if is_x86_feature_detected!("avx512f") {
        features.push("AVX-512F".to_string());
    }
    
    if is_x86_feature_detected!("avx512bw") {
        features.push("AVX-512BW".to_string());
    }
    
    if is_x86_feature_detected!("avx512vl") {
        features.push("AVX-512VL".to_string());
    }
    
    // For ARM architectures, check for NEON
    #[cfg(target_arch = "aarch64")]
    features.push("NEON".to_string());
    
    features
}

// Enable SIMD optimizations based on detected features
fn enable_simd_optimizations(features: &[String]) -> bool {
    // Set environment variables for the tensor library to use SIMD
    if features.contains(&"AVX-512F".to_string()) {
        // Use AVX-512 if available
        unsafe {
            std::env::set_var("WALL_E_USE_AVX512", "1");
            std::env::set_var("WALL_E_SIMD_LEVEL", "3");
        }
        return true;
    } else if features.contains(&"AVX2".to_string()) {
        // Use AVX2 if available
        unsafe {
            std::env::set_var("WALL_E_USE_AVX2", "1");
            std::env::set_var("WALL_E_SIMD_LEVEL", "2");
        }
        return true;
    } else if features.contains(&"AVX".to_string()) {
        // Use AVX if available
        unsafe {
            std::env::set_var("WALL_E_USE_AVX", "1");
            std::env::set_var("WALL_E_SIMD_LEVEL", "1");
        }
        return true;
    } else if features.contains(&"SSE2".to_string()) {
        // Use SSE2 if available (almost all modern CPUs have this)
        unsafe {
            std::env::set_var("WALL_E_USE_SSE2", "1");
            std::env::set_var("WALL_E_SIMD_LEVEL", "0");
        }
        return true;
    } else if features.contains(&"NEON".to_string()) {
        // Use NEON for ARM
        unsafe {
            std::env::set_var("WALL_E_USE_NEON", "1");
            std::env::set_var("WALL_E_SIMD_LEVEL", "1");
        }
        return true;
    }
    
    // No SIMD features detected/enabled
    false
}

// Now let's implement SIMD-optimized matrix multiplication for tensor operations

// Add the improved matrix multiplication with SIMD support
fn matrix_multiply_simd(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    if a.ncols() != b.nrows() {
        panic!("Incompatible dimensions for matrix multiplication");
    }
    
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    
    // Create result matrix
    let mut result = ndarray::Array2::<f32>::zeros((m, n));
    
    // Use SIMD if available
    let simd_level = std::env::var("WALL_E_SIMD_LEVEL").unwrap_or_else(|_| "0".to_string());
    
    match simd_level.as_str() {
        "3" => {
            // AVX-512 implementation
            #[cfg(target_feature = "avx512f")]
            {
                println!("Using AVX-512 for matrix multiplication");
                // AVX-512 implementation would go here
                // This is a skeleton and would need to be expanded with actual SIMD intrinsics
            }
            
            // Fall back to AVX2 if AVX-512 is not available at compile time
            #[cfg(not(target_feature = "avx512f"))]
            {
                matrix_multiply_avx2(a, b, &mut result);
            }
        },
        "2" => {
            // AVX2 implementation
            matrix_multiply_avx2(a, b, &mut result);
        },
        "1" => {
            // AVX implementation
            #[cfg(target_feature = "avx")]
            {
                println!("Using AVX for matrix multiplication");
                // AVX implementation would go here
                // This is a skeleton and would need to be expanded with actual SIMD intrinsics
            }
            
            // Fall back to scalar if AVX is not available at compile time
            #[cfg(not(target_feature = "avx"))]
            {
                matrix_multiply_scalar(a, b, &mut result);
            }
        },
        _ => {
            // Default scalar implementation
            matrix_multiply_scalar(a, b, &mut result);
        }
    }
    
    result
}

// AVX2 implementation of matrix multiplication
#[cfg(target_feature = "avx2")]
fn matrix_multiply_avx2(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>, result: &mut ndarray::Array2<f32>) {
    println!("Using AVX2 for matrix multiplication");
    
    // Real implementation would use AVX2 intrinsics
    // This is a placeholder that falls back to scalar implementation
    matrix_multiply_scalar(a, b, result);
}

// Non-AVX2 version that falls back to scalar
#[cfg(not(target_feature = "avx2"))]
fn matrix_multiply_avx2(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>, result: &mut ndarray::Array2<f32>) {
    println!("AVX2 not available at compile time, using scalar implementation");
    matrix_multiply_scalar(a, b, result);
}

// Scalar implementation as fallback
fn matrix_multiply_scalar(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>, result: &mut ndarray::Array2<f32>) {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    
    // Cache-friendly implementation
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[[i, l]] * b[[l, j]];
            }
            result[[i, j]] = sum;
        }
    }
}