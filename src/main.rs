mod nabla;
mod tokenizer;
mod embedding;
mod attention;

use ndarray::{Array, Array3};
use nabla::tensor::Tensor;
use tokenizer::{Tokenizer, basic_tokenizer::BasicTokenizer, vocab::Vocab};
use embedding::TransformerEmbedding;
use attention::{Attention, SelfAttention, MultiHeadAttention, FeedForward};


fn main() {
    println!("Hello, world!");
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
      
    // Esempio di training di un modello semplice
    // train_simple_model();
    
    // Esempio di utilizzo di tokenizer e embedding
    embedding_example();
    
    // Esempio completo di transformer pipeline
    transformer_pipeline_example();
}

// Esempio che dimostra la pipeline completa da una frase all'output del transformer
fn transformer_pipeline_example() {
    println!("\n--- Transformer Pipeline Example ---");
    
    // 1. Input
    let input_text = "Io sono un robot";
    println!("Input text: '{}'", input_text);
    
    // 2. Tokenizzazione
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab(input_text, 1); // Costruisce il vocabolario
    
    // Aggiungiamo altre parole per arricchire il vocabolario
    tokenizer.build_vocab("Ciao mondo come stai intelligenza artificiale", 1);
    
    let token_ids = tokenizer.encode(input_text);
    println!("Token IDs: {:?}", token_ids);
    
    // 3. Configurazione dei parametri del modello
    let vocab_size = tokenizer.get_vocab().len();
    let d_model = 64;      // Dimensione dell'embedding
    let max_seq_len = 128; // Lunghezza massima delle sequenze
    let num_heads = 4;     // Numero di teste per multi-head attention
    let ff_dim = 128;      // Dimensione interna del feed-forward network
    let num_layers = 2;    // Numero di layer nell'encoder stack
    let dropout_rate = 0.1;
    
    println!("Model config: d_model={}, num_heads={}, ff_dim={}, num_layers={}, vocab_size={}", 
             d_model, num_heads, ff_dim, num_layers, vocab_size);
    
    // 4. Creare l'embedding layer
    let embedding = TransformerEmbedding::new(
        vocab_size,
        d_model,
        max_seq_len,
        dropout_rate
    );
    
    // 5. Forward pass attraverso l'embedding layer
    let embedded = embedding.forward(&token_ids);
    println!("Embedding output shape: {:?}", embedded.data.shape());
    
    // Convertiamo l'output dell'embedding in formato adatto per self-attention
    // Reshaping da [seq_len, d_model] a [batch=1, seq_len, d_model]
    let seq_len = token_ids.len();
    let mut embedded_3d = Array3::<f32>::zeros((1, seq_len, d_model));
    
    for i in 0..seq_len {
        for j in 0..d_model {
            embedded_3d[[0, i, j]] = embedded.data[[i, j]];
        }
    }
    
    let embedded_tensor = Tensor::new_3d(embedded_3d);
    
    // 6. Self-Attention
    println!("\nProcessing through Self-Attention...");
    let self_attention = SelfAttention::new(d_model, 0.1);
    
    // Self-attention richiede Q, K, V identici per il self-attention puro
    let attention_output = self_attention.forward(
        &embedded_tensor, 
        &embedded_tensor, 
        &embedded_tensor, 
        None  // No mask
    );
    
    println!("Self-Attention output shape: {:?}", attention_output.data.shape());
    
    // 7. Multi-Head Attention
    println!("\nProcessing through Multi-Head Attention...");
    let multi_head = MultiHeadAttention::new(d_model, num_heads, 0.1);
    
    let mha_output = multi_head.forward(
        &embedded_tensor,
        &embedded_tensor,
        &embedded_tensor,
        None  // No mask
    );
    
    println!("Multi-Head Attention output shape: {:?}", mha_output.data.shape());
    
    // 8. Feed-Forward Network
    println!("\nProcessing through Feed-Forward Network...");
    
    // Reshape da [batch, seq_len, d_model] a [batch*seq_len, d_model] per FFN
    let batch_size = mha_output.data.shape()[0];
    let seq_len = mha_output.data.shape()[1];
    let d_model = mha_output.data.shape()[2];
    
    let mha_output_2d = mha_output.data.clone()
        .into_dimensionality::<ndarray::Ix3>().unwrap()
        .into_shape((batch_size * seq_len, d_model)).unwrap();
    
    let mha_output_2d_tensor = Tensor::new(mha_output_2d.into_dimensionality::<ndarray::Ix2>().unwrap());
    
    // Feed-Forward
    let feed_forward = FeedForward::new(d_model, Some(ff_dim));
    let ff_output = feed_forward.forward(&mha_output_2d_tensor);
    
    println!("Feed-Forward output shape: {:?}", ff_output.data.shape());
    
    // 9. Encoder Stack completo
    println!("\nProcessing through complete Encoder Stack...");
    use attention::EncoderStack;
    
    // Crea l'encoder stack
    let encoder_stack = EncoderStack::new(d_model, ff_dim, num_heads, num_layers, dropout_rate);
    
    // Forward pass attraverso l'encoder stack
    let encoder_output = encoder_stack.forward(&embedded_tensor, None);
    
    println!("Encoder Stack output shape: {:?}", encoder_output.data.shape());
    
    // 10. Visualizzare alcuni valori di output per conferma
    println!("\nChecking output values...");
    let encoder_data = encoder_output.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
    
    // Mostra i primi 5 valori della prima posizione
    println!("First 5 values of first position from Encoder Stack:");
    for i in 0..5.min(d_model) {
        println!("  [0, 0, {}] = {:.6}", i, encoder_data[[0, 0, i]]);
    }
    
    // 11. Creazione di una maschera causale per dimostrare il funzionamento
    println!("\nDimostrazione con maschera causale:");
    use attention::create_causal_mask;
    
    // Creiamo una maschera causale
    let causal_mask = create_causal_mask(seq_len);
    
    // Forward pass con maschera causale
    let masked_output = encoder_stack.forward(&embedded_tensor, Some(&causal_mask));
    
    println!("Encoder Stack output with causal mask shape: {:?}", masked_output.data.shape());
    
    // Verifica se ci sono differenze tra output con e senza maschera
    let masked_data = masked_output.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
    
    let mut diff_count = 0;
    let mut max_diff = 0.0;
    
    for ((b, i, j), &v1) in encoder_data.indexed_iter() {
        let v2 = masked_data[[b, i, j]];
        let diff = (v1 - v2).abs();
        if diff > 1e-5 {
            diff_count += 1;
            max_diff = f32::max(max_diff, diff);
        }
    }
    
    println!("Differenze trovate tra output con e senza maschera: {}", diff_count);
    println!("Differenza massima: {:.6}", max_diff);
    
    println!("\nPipeline completa eseguita con successo!");
    println!("La frase '{}' è stata elaborata attraverso:", input_text);
    println!("1. Tokenizzazione → {:?}", token_ids);
    println!("2. Embedding → shape {:?}", embedded.data.shape());
    println!("3. Self-Attention → shape {:?}", attention_output.data.shape());
    println!("4. Multi-Head Attention → shape {:?}", mha_output.data.shape());
    println!("5. Feed-Forward Network → shape {:?}", ff_output.data.shape());
    println!("6. Encoder Stack → shape {:?}", encoder_output.data.shape());
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

