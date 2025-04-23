mod nabla;
mod tokenizer;
mod embedding;

use ndarray::Array;
use nabla::tensor::Tensor;
use tokenizer::{Tokenizer, basic_tokenizer::BasicTokenizer, vocab::Vocab};
use embedding::TransformerEmbedding;


fn main() {
    println!("Hello, world!");
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
      
    // Esempio di training di un modello semplice
    // train_simple_model();
    
    // Esempio di utilizzo di tokenizer e embedding
    embedding_example();
}

// Esempio che mostra l'uso degli embedding
fn embedding_example() {
    println!("\n--- Embedding Example ---");
    
    // 1. Crea un tokenizer semplice
    let mut tokenizer = BasicTokenizer::new();
    
    // 2. Costruisci un vocabolario minimo
    let text = "hello world transformer models are amazing for natural language processing tasks";
    tokenizer.build_vocab(text, 1);
    
    println!("Vocabolario costruito con {} token", tokenizer.get_vocab().len());
    
    // 3. Tokenizza una frase di esempio
    let example = "hello transformer models";
    let tokens = tokenizer.tokenize(example);
    let token_ids = tokenizer.encode(example);
    
    println!("Frase: '{}'", example);
    println!("Token: {:?}", tokens);
    println!("Token IDs: {:?}", token_ids);
    
    // 4. Crea un embedding
    let embedding = TransformerEmbedding::new(
        tokenizer.get_vocab().len(),  // vocab_size
        64,                           // embedding_dim
        100,                          // max_seq_len
        0.1,                          // dropout_rate
    );
    
    // 5. Passa i token attraverso l'embedding
    let embedded = embedding.forward(&token_ids);
    
    println!("Dimensione output embedding: {:?}", embedded.data.shape());
    println!("Primo token embedding generato con successo");
    
    // 6. Esempio con batch di sequenze
    let example2 = "natural language processing";
    let token_ids2 = tokenizer.encode(example2);
    
    let batch_token_ids = vec![token_ids.clone(), token_ids2];
    
    println!("\nEsempio di batch processing:");
    println!("Batch sequenze: [{}, {}]", example, example2);
    
    // Forward pass con batch
    let batch_embedded = embedding.forward_batch(&batch_token_ids);
    
    println!("Dimensione output batch embedding: {:?}", batch_embedded.data.shape());
    println!("Batch embedding generato con successo");
}

// Esempio di training di un modello semplice
fn train_simple_model() {
    println!("\n--- Training a simple model ---");
    
    // Create a model with parameters to be learned
    let x = Tensor::new(Array::from_shape_fn((1, 2), |_| 0.5));
    let mut w = Tensor::new(Array::from_shape_fn((2, 1), |_| 1.0));
    
    // Target: what we want our model to produce
    let target = Tensor::new(Array::from_elem((1, 1), 0.8));
    
    // Learning rate - use a smaller value to prevent overshooting
    let lr = 0.02;
    
    // Train for several steps
    println!("\nTraining for 20 steps:");
    for i in 0..20 {
        // Forward pass
        let out = Tensor::matmul(&x, &w);
        
        // Compute loss
        let diff_data = Array::from_shape_vec((1, 1), vec![out.data.as_slice().unwrap()[0] - target.data.as_slice().unwrap()[0]]).unwrap();
        let diff = Tensor::new(diff_data);
        let squared = Tensor::square(&diff);
        let loss = Tensor::sum(&squared);
        
        // Backward pass
        loss.backward(None);
        
        // Get gradient and update weights - unwrap safely with default
        let w_grad = w.grad.lock().unwrap().clone().unwrap_or_else(|| Array::zeros(w.data.raw_dim()));
        
        // Convert to Array2 for calculations
        let w_data = w.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let w_grad_2d = w_grad.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        
        // Create new weights by subtracting gradient * learning rate
        let new_weights = &w_data - &(&w_grad_2d * lr);
        
        // Print progress
        println!("Step {}: loss = {:.6}, prediction = {:.6}, w = [{:.6}, {:.6}]", 
                i, 
                loss.data.as_slice().unwrap()[0], 
                out.data.as_slice().unwrap()[0],
                w_data[[0, 0]],
                w_data[[1, 0]]);
        
        // Re-create weight tensor with new values and reset grad
        w = Tensor::new(new_weights);
    }
    
    // Final forward pass to check result
    let final_out = Tensor::matmul(&x, &w);
    let w_data = w.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    
    println!("\nFinal prediction: {:.6} (target: 0.8)", final_out.data.as_slice().unwrap()[0]);
    println!("Final weights: w = [{:.6}, {:.6}]", w_data[[0, 0]], w_data[[1, 0]]);
    
    // For our simple network, both weights should converge to 0.8 / 2 = 0.4
    // since x = [0.5, 0.5] and we want out = 0.8
    println!("Expected optimal weights = [0.8, 0.8] for single input or [0.4, 0.4] for both inputs summing to 0.8");
}

