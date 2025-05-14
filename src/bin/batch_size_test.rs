use std::env;
use wall_e1::nabla::memory_opt;

fn main() {
    // Default values
    let mut model_dim = 128;
    let mut seq_len = 256;
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    for i in 1..args.len() {
        if args[i] == "--model-dim" && i + 1 < args.len() {
            model_dim = args[i + 1].parse().unwrap_or(128);
        }
        
        if args[i] == "--seq-len" && i + 1 < args.len() {
            seq_len = args[i + 1].parse().unwrap_or(256);
        }
    }
    
    // Get cache parameters
    let cache_params = memory_opt::detect_cache_parameters();
    
    println!("Cache parameters: L1={} KB, L2={} KB, L3={} MB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / (1024 * 1024), 
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
    
    // Also calculate for different model dimensions to see the trend
    println!("\nBatch sizes for different model dimensions:");
    for dim in [64, 128, 256, 512, 1024].iter() {
        let size = memory_opt::calculate_optimal_batch_size(
            &cache_params,
            *dim,
            seq_len,
            element_size
        );
        println!("  model_dim={}, batch_size={}", dim, size);
    }
    
    // Calculate for different sequence lengths
    println!("\nBatch sizes for different sequence lengths:");
    for len in [64, 128, 256, 512, 1024].iter() {
        let size = memory_opt::calculate_optimal_batch_size(
            &cache_params,
            model_dim,
            *len,
            element_size
        );
        println!("  seq_len={}, batch_size={}", len, size);
    }
} 