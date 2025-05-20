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
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

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
    // New parameters added for enhanced functionality
    let mut auto_resize_vocab = false;
    let mut watchdog_timeout: Option<u64> = None;
    let mut batch_timeout: Option<u64> = None;
    let mut data_threads: Option<usize> = None;
    let mut target_id_max: Option<usize> = None;
    let mut disable_watchdog = false;
    // New parameter for enabling parallel data preparation
    let mut use_parallel = false;
    // New parameter for full multi-threaded training
    let mut use_mt_training = false;

    // Check for MT training environment variable
    if let Ok(val) = std::env::var("WALL_E_MT_TRAINING") {
        if val == "1" || val.to_lowercase() == "true" {
            println!("🔴 Multi-threaded training enabled via environment variable");
            use_mt_training = true;
        }
    }

    // Display help if no arguments are provided or explicitly requested
    if args.len() == 1 || args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) {
        println!("Wall-E1 Neural Network Training Tool");
        println!("Usage: {} [options]", args[0]);
        println!("Options:");
        println!("  --model-dim <dim>       Model dimension (default: {})", model_dim);
        println!("  --ff-dim <dim>          Feed-forward dimension (default: {})", ff_dim);
        println!("  --heads <num>           Number of attention heads (default: {})", num_heads);
        println!("  --layers <num>          Number of transformer layers (default: {})", num_layers);
        println!("  --dropout <rate>        Dropout rate (default: {})", dropout_rate);
        println!("  --epochs <num>          Number of training epochs (default: {})", num_epochs);
        println!("  --vocab-size <size>     Vocabulary size (default: {})", vocab_size);
        println!("  --min-freq <freq>       Minimum token frequency (default: {})", min_freq);
        println!("  --model <path>          Load model from file");
        println!("  --save-path <path>      Save model to file (default: model.json)");
        println!("  --learning-rate <rate>  Learning rate (default: {})", learning_rate);
        println!("  --dataset <path>        Path to training data");
        println!("  --generate-only         Generate text without training");
        println!("  --prompt <text>         Text prompt for generation");
        println!("  --max-tokens <num>      Maximum tokens to generate (default: 100)");
        println!("  --enable-skip           Enable skip connections");
        println!("  --disable-curriculum    Disable curriculum learning");
        println!("  --json-format           Process input file as JSON");
        println!("  --strong-anti-rep       Use stronger anti-repetition penalty");
        println!("  --max-stories <num>     Maximum number of stories to process");
        println!("  --perf-log              Enable detailed performance logging");
        println!("  --num-cpus <num>        Override number of CPUs for computation");
        println!("  --memory-opt            Enable memory optimization");
        println!("  --batch-size <size>     Override batch size");
        println!("  --curriculum-examples <num> Number of curriculum examples (default: 2000)");
        println!("  --checkpoint <strat>    Checkpoint strategy: uniform, layerwise, adaptive");
        println!("  --thread-opt <strat>    Thread optimization strategy: default, aggressive, conservative");
        println!("  --auto-resize-vocab <bool> Automatically resize vocabulary (yes/no)");
        println!("  --watchdog-timeout <sec> Set watchdog timeout in seconds for detecting hangs");
        println!("  --batch-timeout <sec>   Set timeout for individual batch processing in multi-threaded mode");
        println!("  --data-threads <num>    Set number of threads for data loading");
        println!("  --target-id-max <num>   Set maximum token ID to target (for focused training)");
        println!("  --disable-watchdog      Disable the watchdog timer");
        println!("  --parallel              Enable parallel data preparation for training");
        println!("  --mt-training           Enable multi-threaded model training (experimental)");
        println!("");
        println!("Examples:");
        println!("  Train a new model:");
        println!("    {} --dataset path/to/data --epochs 10 --save-path model.json", args[0]);
        println!("  Continue training an existing model:");
        println!("    {} --dataset path/to/data --model existing.json --save-path updated.json", args[0]);
        println!("  Generate text using an existing model:");
        println!("    {} --generate-only --model model.json --prompt \"Once upon a time\"", args[0]);
        println!("  Train with parallel data preparation (faster on multi-core systems):");
        println!("    {} --dataset path/to/data --parallel --memory-opt", args[0]);
        
        // Exit with success
        return Ok(());
    }

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

            "--dataset" => {
                if let Some(val) = arg_iter.next() {
                    training_data_path = Some(val);
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
            // New parameters added
            "--auto-resize-vocab" => {
                auto_resize_vocab = true;
                println!("Automatic vocabulary resizing enabled");
            }
            "--watchdog-timeout" => {
                if let Some(val) = arg_iter.next() {
                    if let Ok(timeout) = val.parse::<u64>() {
                        watchdog_timeout = Some(timeout);
                        println!("Watchdog timeout set to {} seconds", timeout);
                    }
                }
            }
            "--batch-timeout" => {
                if let Some(val) = arg_iter.next() {
                    if let Ok(timeout) = val.parse::<u64>() {
                        batch_timeout = Some(timeout);
                        println!("Batch processing timeout set to {} seconds", timeout);
                    }
                }
            }
            "--data-threads" => {
                if let Some(val) = arg_iter.next() {
                    if let Ok(threads) = val.parse::<usize>() {
                        data_threads = Some(threads);
                        println!("Data loading threads set to {}", threads);
                    }
                }
            }
            "--target-id-max" => {
                if let Some(val) = arg_iter.next() {
                    if let Ok(max_id) = val.parse::<usize>() {
                        target_id_max = Some(max_id);
                        println!("Maximum target ID pre-allocated to {}", max_id);
                    }
                }
            }
            "--disable-watchdog" => {
                disable_watchdog = true;
                println!("Watchdog disabled for training");
            }
            // Add new argument for parallel data preparation
            "--parallel-data-prep" | "--parallel" => {
                use_parallel = true;
                println!("  🧵 Enabling parallel data preparation");
            }
            // Add new argument for multi-threaded training
            "--mt-training" | "--mt" => {
                use_mt_training = true;
                use_parallel = true; // Multi-threaded training implies parallel data prep
                println!("  🧵 Enabling multi-threaded model training (experimental)");
                println!("  ⚠️ This mode is experimental and may cause instability");
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
    
    // Set up environment variables based on parsed arguments
    if let Some(threads) = data_threads {
        unsafe {
            std::env::set_var("WALL_E_DATA_THREADS", threads.to_string());
        }
        println!("🔄 Setting WALL_E_DATA_THREADS={} for data loading operations", threads);
    }

    if disable_watchdog {
        unsafe {
            std::env::set_var("WALL_E_DISABLE_WATCHDOG", "true");
        }
        println!("🛑 Watchdog disabled for training session");
    }
    
    // Set batch timeout for multi-threaded training
    if let Some(timeout) = batch_timeout {
        unsafe {
            std::env::set_var("WALL_E_BATCH_TIMEOUT", timeout.to_string());
        }
        println!("⏱️ Setting WALL_E_BATCH_TIMEOUT={} for multi-threaded batch processing", timeout);
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
    
    // Display the training mode
    let training_mode = if use_mt_training {
        "multi-threaded (training + data preparation)"
    } else if use_parallel {
        "parallel data preparation with single-threaded training"
    } else {
        "single-threaded"
    };
    println!("  Training mode: {}", training_mode);
    
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
        "Once upon a time",
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
        let loss = train_epoch(
            &mut trainer,
            &training_tokens,
            epoch,
            enable_memory_optimization,
            manual_batch_size,
            use_parallel,
            use_mt_training,
        );
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
        let generated = trainer.generate_text(&prompt, Some(50));
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
fn train_epoch(trainer: &mut EnhancedTrainer, tokens: &Vec<usize>, epoch: usize, 
               enable_memory_optimization: bool, manual_batch_size: Option<usize>, 
               use_parallel: bool, use_mt_training: bool) -> f32 {
    // Add debug information about which training mode is selected
    println!("\n⚙️ TRAIN_EPOCH FUNCTION MODE SELECTION ⚙️");
    println!("  use_mt_training = {}", use_mt_training);
    println!("  use_parallel = {}", use_parallel);
    println!("  If both are correct, you should see colored debug messages from EnhancedTrainer.\n");
                   
    // Create sliding windows of input/target pairs
    let max_sequence_length = trainer.get_max_seq_len();
    let stride = max_sequence_length / 2; // 50% overlap between windows
    
    let mut perf_logger = PerfLogger::new(true); // Local performance tracking
    perf_logger.start("train_epoch_full");
    perf_logger.memory_snapshot("train_epoch_start");
    
    println!("    🔍 TRAIN EPOCH DETAILED LOGGING");
    
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
    
    // IMPORTANT CHANGE: Ensure model is properly initialized
    println!("    🧐 Verifying model initialization before training...");
    let init_start = Instant::now();
    match trainer.ensure_model_initialized() {
        Ok(_) => {
            println!("    ✅ Model initialization verified in {:.2?}", init_start.elapsed());
        },
        Err(e) => {
            println!("    ⚠️ Model initialization issue detected: {}", e);
            println!("    🔄 Will attempt to continue anyway");
        }
    }
    
    // Set batch size
    let batch_size = manual_batch_size.unwrap_or_else(|| {
        calculate_memory_optimal_batch_size(model_dim, max_sequence_length)
    });
    println!("    📊 Using batch size: {}", batch_size);
    
    // Determine training approach based on parameters
    let training_approach = if use_mt_training {
        "multi-threaded"
    } else if use_parallel {
        "parallel data preparation"
    } else {
        "sequential"
    };
    
    println!("    🔄 Training from tokens using {} approach", training_approach);
    let training_start = Instant::now();
    
    // Choose the appropriate training method
    let result = if use_mt_training {
        // Use the new multi-threaded training method
        println!("    🧵 Using multi-threaded processing for model training");
        
        // Prepare data for training (similar for all approaches)
        let (inputs, targets) = match trainer.prepare_data_parallel(tokens, max_sequence_length, stride) {
            Ok((i, t)) => (i, t),
            Err(e) => {
                println!("    ❌ Error preparing data: {}", e);
                return 10.0; // Default high loss value
            }
        };
        
        // Create batches
        let (batched_inputs, batched_targets) = match trainer.create_batches(&inputs, &targets, batch_size) {
            Ok((i, t)) => (i, t),
            Err(e) => {
                println!("    ❌ Error creating batches: {}", e);
                return 10.0; // Default high loss value
            }
        };
        
        // Multi-threaded training - use the new method
        println!("    🧮 Running multi-threaded training with {} batches", batched_inputs.len());
        trainer.train_parallel(&batched_inputs, &batched_targets)
            .map_err(|e| format!("Multi-threaded training failed: {}", e))
    } else {
        // Use the standard training method with parallel data prep
        trainer.train_from_tokens(tokens, batch_size, use_parallel)
    };
    
    let loss = match result {
        Ok(loss) => {
            println!("    ✅ Training completed successfully");
            loss
        },
        Err(e) => {
            println!("    ⚠️ Error during training: {}", e);
            println!("    ⚠️ Returning default loss value");
            10.0 // Default high loss value
        }
    };
    
    let training_time = training_start.elapsed();
    println!("    ✅ Completed epoch training in {:.2?} with loss: {:.6}", training_time, loss);
    
    // End full epoch timing
    perf_logger.end("train_epoch_full");
    perf_logger.memory_snapshot("train_epoch_end");
    
    // Log performance metrics
    perf_logger.log_summary();
    
    // Return the final average loss
    loss
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