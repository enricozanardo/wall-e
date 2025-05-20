use std::env;
use wall_e1::EnhancedTrainer;
use wall_e1::tokenizer::WordPieceBPETokenizer;
use ndarray::Array2;

fn main() {
    println!("Testing multi-threaded training...");
    
    // Simple data for testing
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    // Create some fake batches (just for testing the code path)
    for _i in 0..10 {
        // Each batch has 4 sequences
        let mut batch = Vec::new();
        for j in 0..4 {
            batch.push(vec![j, j+1, j+2, j+3, j+4, j+5, j+6, j+7]);
        }
        inputs.push(batch);
        
        // Create a target tensor for this batch
        let target_data: Vec<usize> = (0..32).map(|x| x % 8).collect();
        targets.push(Array2::from_shape_vec((4, 8), target_data).unwrap());
    }
    
    // Create a trainer
    let mut trainer = EnhancedTrainer::new(32, 64, 2, 2, 0.1, 0.001);
    
    // Flag for multi-threaded training
    let use_mt_training = true;
    
    println!("Starting training test with use_mt_training = {}", use_mt_training);
    
    // Test the multi-threaded training
    if use_mt_training {
        println!("Using train_parallel method...");
        match trainer.train_parallel(&inputs, &targets) {
            Ok(loss) => println!("Training completed successfully with loss: {}", loss),
            Err(e) => println!("Error during multi-threaded training: {}", e),
        }
    } else {
        println!("Using train_reliable method...");
        let loss = trainer.train_reliable(&inputs, &targets, true);
        println!("Training completed successfully with loss: {}", loss);
    }
    
    println!("Test completed!");
}
