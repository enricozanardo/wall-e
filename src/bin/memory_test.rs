use std::env;
use std::time::Instant;
use wall_e1::nabla::memory_opt;
use wall_e1::nabla::tensor::Tensor;
use wall_e1::utils::thread_pool::get_global_thread_pool;
use ndarray::Array2;

// Helper function to measure memory usage on Linux
fn measure_memory_usage() -> Option<(usize, usize)> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            let mut rss = None;
            let mut vm_size = None;
            
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    rss = line.split_whitespace().nth(1)
                        .and_then(|s| s.parse::<usize>().ok());
                } else if line.starts_with("VmSize:") {
                    vm_size = line.split_whitespace().nth(1)
                        .and_then(|s| s.parse::<usize>().ok());
                }
                
                if rss.is_some() && vm_size.is_some() {
                    break;
                }
            }
            
            if let (Some(rss), Some(vm_size)) = (rss, vm_size) {
                return Some((rss, vm_size));
            }
        }
    }
    
    None
}

// Function to log memory usage
fn log_memory(label: &str) {
    if let Some((rss, vm_size)) = measure_memory_usage() {
        println!("Memory [{}]: RSS = {} KB, VM = {} KB", label, rss, vm_size);
    } else {
        println!("Memory [{}]: Unable to measure on this platform", label);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Wall-E Memory Optimization Test");
    
    // Initialize thread pool
    let thread_pool = get_global_thread_pool();
    println!("Global thread pool initialized with {} threads", thread_pool.get_num_threads());
    
    log_memory("startup");
    
    // Set environment variables for memory optimization using the safe method
    unsafe {
        env::set_var("MALLOC_ARENA_MAX", "2"); // For better memory locality
        env::set_var("RAYON_NUM_THREADS", thread_pool.get_num_threads().to_string());
    }
    
    // Configure tensor operations to be cache-efficient
    let cache_params = memory_opt::detect_cache_parameters();
    println!("Cache parameters: L1={} KB, L2={} KB, L3={} KB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / 1024, 
        cache_params.line_size);
    
    log_memory("after-config");
    
    // Test memory bandwidth
    println!("\nMeasuring memory bandwidth...");
    let bandwidth = memory_opt::measure_memory_bandwidth();
    println!("Memory bandwidth: {:.2} GB/s", bandwidth);
    
    // Create tensors of increasing sizes to test memory behavior
    println!("\nTesting memory behavior with increasing tensor sizes...");
    
    let sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    
    for &size in &sizes {
        println!("\nTesting with tensor size {}x{}", size, size);
        log_memory(&format!("before-tensor-{}", size));
        
        let start = Instant::now();
        
        // Create matrix and test matmul
        let a = Array2::<f32>::zeros((size, size));
        let b = Array2::<f32>::zeros((size, size));
        
        log_memory(&format!("after-alloc-{}", size));
        
        let c = memory_opt::cache_blocked_matmul(&a, &b);
        
        let duration = start.elapsed();
        println!("Computation time: {:?}", duration);
        
        log_memory(&format!("after-compute-{}", size));
        
        // Force drop to measure deallocation
        drop(a);
        drop(b);
        drop(c);
        
        log_memory(&format!("after-drop-{}", size));
    }
    
    // Test with batch processing to simulate training
    println!("\nTesting batch processing memory behavior...");
    
    let model_dim = 256;
    let seq_len = 128;
    
    let optimal_batch_size = memory_opt::calculate_optimal_batch_size(
        &cache_params, model_dim, seq_len, std::mem::size_of::<f32>());
    
    println!("Optimal batch size for model_dim={}, seq_len={}: {}", 
             model_dim, seq_len, optimal_batch_size);
    
    println!("\nTest completed successfully");
    Ok(())
} 