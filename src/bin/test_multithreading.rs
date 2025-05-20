use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wall_e1::training::{EnhancedTrainer, TextGenerator};
use wall_e1::tokenizer::WordPieceBPETokenizer;
use ndarray::Array2;
use rand::Rng;

// CPU usage monitoring
struct CpuUsageMonitor {
    thread_handle: Option<thread::JoinHandle<()>>,
    running: Arc<AtomicUsize>,
    cpu_usage: Arc<Mutex<Vec<f32>>>,
}

impl CpuUsageMonitor {
    fn new() -> Self {
        let running = Arc::new(AtomicUsize::new(1));
        let cpu_usage = Arc::new(Mutex::new(Vec::new()));
        
        let running_clone = Arc::clone(&running);
        let cpu_usage_clone = Arc::clone(&cpu_usage);
        
        let handle = thread::spawn(move || {
            // Get initial CPU info
            let mut prev_total = 0;
            let mut prev_idle = 0;
            
            if let Ok(stats) = std::fs::read_to_string("/proc/stat") {
                if let Some(line) = stats.lines().next() {
                    let values: Vec<&str> = line.split_whitespace().collect();
                    if values.len() > 4 {
                        prev_idle = values[4].parse::<u64>().unwrap_or(0);
                        prev_total = values[1..].iter()
                            .filter_map(|v| v.parse::<u64>().ok())
                            .sum();
                    }
                }
            }
            
            while running_clone.load(Ordering::SeqCst) > 0 {
                // Sleep for a bit
                thread::sleep(Duration::from_secs(1));
                
                // Get current CPU info
                if let Ok(stats) = std::fs::read_to_string("/proc/stat") {
                    if let Some(line) = stats.lines().next() {
                        let values: Vec<&str> = line.split_whitespace().collect();
                        if values.len() > 4 {
                            let idle = values[4].parse::<u64>().unwrap_or(0);
                            let total: u64 = values[1..].iter()
                                .filter_map(|v| v.parse::<u64>().ok())
                                .sum();
                            
                            // Calculate CPU usage
                            let idle_delta = idle - prev_idle;
                            let total_delta = total - prev_total;
                            
                            if total_delta > 0 {
                                let usage = 100.0 * (1.0 - (idle_delta as f32) / (total_delta as f32));
                                
                                // Store CPU usage
                                if let Ok(mut cpu_usage) = cpu_usage_clone.lock() {
                                    cpu_usage.push(usage);
                                }
                                
                                // Update previous values
                                prev_idle = idle;
                                prev_total = total;
                            }
                        }
                    }
                }
            }
        });
        
        Self {
            thread_handle: Some(handle),
            running,
            cpu_usage,
        }
    }
    
    fn stop(&mut self) -> Vec<f32> {
        self.running.store(0, Ordering::SeqCst);
        
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        
        let cpu_usage = self.cpu_usage.lock().unwrap().clone();
        cpu_usage
    }
}

fn create_synthetic_dataset(size: usize, seq_len: usize, vocab_size: usize) -> (Vec<Vec<usize>>, Array2<usize>) {
    let mut rng = rand::thread_rng();
    
    // Create inputs
    let mut inputs = Vec::with_capacity(size);
    for _ in 0..size {
        let mut sequence = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            sequence.push(rng.gen_range(0..vocab_size));
        }
        inputs.push(sequence);
    }
    
    // Create targets
    let mut targets = Array2::zeros((size, seq_len));
    for i in 0..size {
        for j in 0..seq_len {
            targets[[i, j]] = rng.gen_range(0..vocab_size);
        }
    }
    
    (inputs, targets)
}

fn test_multithreaded_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing multi-threaded training...");
    
    // Start CPU usage monitoring
    let mut cpu_monitor = CpuUsageMonitor::new();
    
    // Create synthetic dataset
    let batch_size = 16;
    let seq_len = 32;
    let vocab_size = 1000;
    let num_batches = 10;
    
    println!("Creating synthetic dataset with {} batches, batch_size={}, seq_len={}, vocab_size={}",
            num_batches, batch_size, seq_len, vocab_size);
    
    let mut all_inputs = Vec::with_capacity(num_batches);
    let mut all_targets = Vec::with_capacity(num_batches);
    
    for _ in 0..num_batches {
        let (inputs, targets) = create_synthetic_dataset(batch_size, seq_len, vocab_size);
        all_inputs.push(inputs);
        all_targets.push(targets);
    }
    
    // Create tokenizer
    let mut tokenizer = WordPieceBPETokenizer::new();
    tokenizer.update_vocab_size(vocab_size);
    
    // Create trainer
    let model_dim = 64;
    let ff_dim = 256;
    let num_heads = 8;
    let num_layers = 2;
    let dropout_rate = 0.1;
    let learning_rate = 0.001;
    
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // Set timeout
    unsafe {
        std::env::set_var("WALL_E_BATCH_TIMEOUT", "10");
    }
    
    // Train with multi-threading
    println!("Starting multi-threaded training...");
    let start_time = Instant::now();
    
    // Important: First test with multi-threading disabled
    println!("\n === Training with single-thread (baseline) ===\n");
    let single_thread_start = Instant::now();
    let single_thread_result = trainer.train_reliable(&all_inputs, &all_targets, false);
    let single_thread_time = single_thread_start.elapsed();
    println!("Single-threaded training complete in {:?} with loss: {}", 
             single_thread_time, single_thread_result);
    
    // Then test with parallel data preparation
    println!("\n === Training with parallel data preparation ===\n");
    let parallel_data_start = Instant::now();
    let parallel_data_result = trainer.train_reliable(&all_inputs, &all_targets, true);
    let parallel_data_time = parallel_data_start.elapsed();
    println!("Parallel data training complete in {:?} with loss: {}", 
             parallel_data_time, parallel_data_result);
    
    // Finally test with multi-threaded training
    println!("\n === Training with full multi-threading ===\n");
    let mt_training_start = Instant::now();
    let mt_training_result = match trainer.train_parallel(&all_inputs, &all_targets) {
        Ok(loss) => {
            println!("Multi-threaded training complete in {:?} with loss: {}", 
                    mt_training_start.elapsed(), loss);
            loss
        },
        Err(e) => {
            println!("Multi-threaded training failed: {}", e);
            std::f32::NAN
        }
    };
    let mt_training_time = mt_training_start.elapsed();
    
    // Stop CPU monitoring
    let cpu_usage = cpu_monitor.stop();
    
    // Calculate average CPU usage
    let avg_cpu_usage = if !cpu_usage.is_empty() {
        cpu_usage.iter().sum::<f32>() / cpu_usage.len() as f32
    } else {
        0.0
    };
    
    // Calculate peak CPU usage
    let peak_cpu_usage = cpu_usage.iter().fold(0.0_f32, |a: f32, &b| if a > b { a } else { b });
    
    // Calculate speedup factors
    let parallel_data_speedup = single_thread_time.as_secs_f32() / parallel_data_time.as_secs_f32();
    let mt_training_speedup = single_thread_time.as_secs_f32() / mt_training_time.as_secs_f32();
    
    // Print results
    println!("\n=== Performance Results ===");
    println!("Single-threaded:      {:?} (baseline)", single_thread_time);
    println!("Parallel data:        {:?} ({:.2}x speedup)", parallel_data_time, parallel_data_speedup);
    println!("Multi-threaded:       {:?} ({:.2}x speedup)", mt_training_time, mt_training_speedup);
    println!("Avg CPU Usage:        {:.2}%", avg_cpu_usage);
    println!("Peak CPU Usage:       {:.2}%", peak_cpu_usage);
    println!("Total runtime:        {:?}", start_time.elapsed());
    
    // Verify that multi-threaded training actually gave a reasonable result
    if !mt_training_result.is_nan() {
        println!("\n✅ Multi-threaded training completed successfully!");
        println!("Single-thread loss: {:.6}", single_thread_result);
        println!("Parallel data loss: {:.6}", parallel_data_result);
        println!("Multi-thread loss:  {:.6}", mt_training_result);
    } else {
        println!("\n❌ Multi-threaded training failed!");
    }
    
    // Report CPU count
    let num_cpus = num_cpus::get();
    println!("\nSystem has {} logical CPU cores", num_cpus);
    println!("Peak CPU utilization: {:.1}%", peak_cpu_usage);
    if peak_cpu_usage > 100.0 {
        println!("✅ Multi-core utilization confirmed!");
    } else {
        println!("⚠️ Limited to single-core performance");
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    test_multithreaded_training()
}
