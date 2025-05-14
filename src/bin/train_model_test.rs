use std::fs::File;
use std::io::Read;
use std::time::Instant;
use wall_e1::nabla::memory_opt;
use wall_e1::training::enhanced_trainer::EnhancedTrainer;
use wall_e1::nabla::tensor::set_num_threads;
use wall_e1::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default values
    let model_dim = 128;
    let ff_dim = 512;
    let num_heads = 4;
    let num_layers = 3;
    let dropout_rate = 0.1;
    let learning_rate = 0.001;
    let vocab_size = 5000;
    let max_sequence_length = 256;
    
    // Configure threads
    let num_threads = num_cpus::get();
    println!("Using {} CPU threads", num_threads);
    set_num_threads(num_threads);
    
    // Create a trainer
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // Load some sample text
    let mut file = File::open("data/tiny_stories_sample.json")?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    
    // Learn a vocabulary
    println!("Learning vocabulary...");
    trainer.learn_tokenizer_from_text(&text, vocab_size, 2);
    
    // Tokenize text
    println!("Tokenizing text...");
    let tokenizer = trainer.get_tokenizer();
    let tokens = tokenizer.encode(&text);
    
    // Create some training examples
    println!("Creating training examples...");
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..tokens.len().saturating_sub(max_sequence_length) {
        if i + max_sequence_length <= tokens.len() {
            let input = tokens[i..i + max_sequence_length].to_vec();
            inputs.push(input);
            
            // Target is next token prediction
            let mut target = Vec::with_capacity(max_sequence_length);
            for j in 0..max_sequence_length {
                let target_idx = (i + j + 1) % tokens.len();
                target.push(tokens[target_idx]);
            }
            
            targets.push(target);
        }
    }
    
    println!("Created {} input/target pairs", inputs.len());
    
    // Calculate memory-optimal batch size
    let cache_params = memory_opt::detect_cache_parameters();
    println!("\n======== BATCH SIZE CALCULATION ========");
    println!("Cache parameters: L1={} KB, L2={} KB, L3={} MB, Line size={} bytes",
        cache_params.l1_size / 1024, 
        cache_params.l2_size / 1024, 
        cache_params.l3_size / (1024 * 1024), 
        cache_params.line_size);
    
    let element_size = std::mem::size_of::<f32>();
    println!("Calculating for model_dim={}, seq_len={}, element_size={} bytes",
        model_dim, max_sequence_length, element_size);
    
    let memory_optimal_batch = memory_opt::calculate_optimal_batch_size(
        &cache_params,
        model_dim,
        max_sequence_length,
        element_size
    );
    
    // Use the default batch size for comparison
    let default_batch_size = 32;
    
    println!("Memory-optimal batch size: {}", memory_optimal_batch);
    println!("Default batch size: {}", default_batch_size);
    println!("========================================\n");
    
    // Train with default batch size
    println!("Training with DEFAULT batch size ({})...", default_batch_size);
    let start_time_default = Instant::now();
    
    let mut default_batched_inputs = Vec::new();
    let mut default_batched_targets = Vec::new();
    
    for batch_start in (0..inputs.len()).step_by(default_batch_size) {
        let batch_end = (batch_start + default_batch_size).min(inputs.len());
        let batch_inputs = inputs[batch_start..batch_end].to_vec();
        let batch_targets = targets[batch_start..batch_end].to_vec();
        
        // Convert batch targets to ndarray
        let mut targets_array = ndarray::Array2::zeros((batch_targets.len(), max_sequence_length));
        for (i, target) in batch_targets.iter().enumerate() {
            for (j, &token) in target.iter().enumerate() {
                targets_array[[i, j]] = token;
            }
        }
        
        default_batched_inputs.push(batch_inputs);
        default_batched_targets.push(targets_array);
    }
    
    // Train on a single batch for timing comparison
    trainer.train_epoch(&default_batched_inputs[0..1].to_vec(), &default_batched_targets[0..1].to_vec());
    let default_duration = start_time_default.elapsed();
    println!("Default batch size training time: {:?}", default_duration);
    
    // Reset trainer for fair comparison
    let mut trainer = EnhancedTrainer::new(
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    trainer.learn_tokenizer_from_text(&text, vocab_size, 2);
    
    // Train with optimal batch size
    println!("Training with OPTIMAL batch size ({})...", memory_optimal_batch);
    let start_time_optimal = Instant::now();
    
    let mut optimal_batched_inputs = Vec::new();
    let mut optimal_batched_targets = Vec::new();
    
    for batch_start in (0..inputs.len()).step_by(memory_optimal_batch) {
        let batch_end = (batch_start + memory_optimal_batch).min(inputs.len());
        let batch_inputs = inputs[batch_start..batch_end].to_vec();
        let batch_targets = targets[batch_start..batch_end].to_vec();
        
        // Convert batch targets to ndarray
        let mut targets_array = ndarray::Array2::zeros((batch_targets.len(), max_sequence_length));
        for (i, target) in batch_targets.iter().enumerate() {
            for (j, &token) in target.iter().enumerate() {
                targets_array[[i, j]] = token;
            }
        }
        
        optimal_batched_inputs.push(batch_inputs);
        optimal_batched_targets.push(targets_array);
    }
    
    // Train on a single batch for timing comparison
    if !optimal_batched_inputs.is_empty() {
        trainer.train_epoch(&optimal_batched_inputs[0..1].to_vec(), &optimal_batched_targets[0..1].to_vec());
        let optimal_duration = start_time_optimal.elapsed();
        println!("Optimal batch size training time: {:?}", optimal_duration);
        
        // Calculate improvement
        let improvement_ratio = default_duration.as_secs_f64() / optimal_duration.as_secs_f64();
        println!("\nImprovement with optimal batch size: {:.2}x faster", improvement_ratio);
    } else {
        println!("Not enough data to create optimal batches");
    }
    
    Ok(())
} 