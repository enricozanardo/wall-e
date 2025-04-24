mod nabla;
mod tokenizer;
mod embedding;
mod attention;
mod training;

use ndarray::{Array, Array3};
use nabla::tensor::Tensor;
use tokenizer::{Tokenizer, basic_tokenizer::BasicTokenizer};
use embedding::TransformerEmbedding;
use attention::{Attention, SelfAttention, MultiHeadAttention, FeedForward};
use training::Trainer;
use std::env;

/// Funzione principale
fn main() {
    println!("Hello, world!");
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
    
    // Ottieni gli argomenti dalla riga di comando
    let args: Vec<String> = env::args().collect();
      
    // Esempi di utilizzo
    // transformer_pipeline_example();
    // embedding_example();
    dataset_example();
    
    // Se viene passato un argomento da riga di comando, usalo come percorso al file di configurazione
    if args.len() > 1 {
        let config_path = &args[1];
        
        // Se il percorso contiene "qa", usa l'esempio di question answering
        if config_path.contains("qa") {
            question_answering_example(Some(config_path));
        } else {
            training_example(Some(config_path));
        }
    } else {
        // Altrimenti usa il file predefinito
        training_example(None);
    }
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

// Esempio di training di un modello transformer
fn training_example(config_path: Option<&str>) {
    println!("\n--- Training Example ---");
    
    // 1. Caricamento dei dati da file esterni
    println!("Caricamento dei dati di training e test da file...");
    
    // Controlla se è stato fornito un percorso personalizzato
    let default_json_path = "data/dataset.json";
    let default_train_path = "data/story.txt";
    let default_test_path = "data/test.txt";
    
    // Possiamo caricare da file di testo o da JSON
    let use_json = true; // Imposta su true per usare il file JSON, false per i file di testo
    
    let train_text: String;
    let test_text: String;
    let mut custom_prompts: Vec<String> = vec![];
    
    // Parametri del modello con valori predefiniti
    let mut d_model = 64;      // Dimensione dell'embedding
    let mut max_seq_len = 128; // Lunghezza massima delle sequenze
    let mut num_heads = 4;     // Numero di teste per multi-head attention
    let mut ff_dim = 128;      // Dimensione interna del feed-forward network
    let mut num_layers = 2;    // Numero di layer nell'encoder stack
    let mut dropout_rate = 0.1;
    let mut learning_rate = 0.001;
    
    if use_json {
        // Usa il percorso fornito da riga di comando o quello predefinito
        let json_file_path = config_path.unwrap_or(default_json_path);
        println!("Usando il file JSON: {}", json_file_path);
        
        // Carica i dati direttamente con serde_json
        let file = match std::fs::File::open(json_file_path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Errore nell'apertura del file JSON {}: {}", json_file_path, e);
                return;
            }
        };
        
        let reader = std::io::BufReader::new(file);
        let json_data: serde_json::Value = match serde_json::from_reader(reader) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Errore nella decodifica del JSON da {}: {}", json_file_path, e);
                return;
            }
        };
        
        // Estrai i testi dal JSON
        train_text = match json_data.get("train_text") {
            Some(text) => match text.as_str() {
                Some(s) => s.to_string(),
                None => {
                    eprintln!("Errore: il campo 'train_text' nel JSON non è una stringa valida");
                    return;
                }
            },
            None => {
                eprintln!("Errore: campo 'train_text' non trovato nel file JSON");
                return;
            }
        };
        
        test_text = match json_data.get("test_text") {
            Some(text) => match text.as_str() {
                Some(s) => s.to_string(),
                None => {
                    eprintln!("Errore: il campo 'test_text' nel JSON non è una stringa valida");
                    return;
                }
            },
            None => {
                eprintln!("Errore: campo 'test_text' non trovato nel file JSON");
                return;
            }
        };
        
        // Estrai i prompt personalizzati se presenti
        if let Some(prompts_json) = json_data.get("prompts") {
            if let Some(prompts_arr) = prompts_json.as_array() {
                custom_prompts = prompts_arr.iter()
                    .filter_map(|p| p.as_str().map(String::from))
                    .collect();
            }
        }
        
        // Estrai i parametri del modello se presenti
        if let Some(model_params) = json_data.get("model_params") {
            // Estrai i parametri uno per uno, usando il valore predefinito se non trovato
            if let Some(val) = model_params.get("d_model") {
                if let Some(val) = val.as_u64() {
                    d_model = val as usize;
                }
            }
            
            if let Some(val) = model_params.get("max_seq_len") {
                if let Some(val) = val.as_u64() {
                    max_seq_len = val as usize;
                }
            }
            
            if let Some(val) = model_params.get("num_heads") {
                if let Some(val) = val.as_u64() {
                    num_heads = val as usize;
                }
            }
            
            if let Some(val) = model_params.get("ff_dim") {
                if let Some(val) = val.as_u64() {
                    ff_dim = val as usize;
                }
            }
            
            if let Some(val) = model_params.get("num_layers") {
                if let Some(val) = val.as_u64() {
                    num_layers = val as usize;
                }
            }
            
            if let Some(val) = model_params.get("dropout_rate") {
                if let Some(val) = val.as_f64() {
                    dropout_rate = val as f32;
                }
            }
            
            if let Some(val) = model_params.get("learning_rate") {
                if let Some(val) = val.as_f64() {
                    learning_rate = val as f32;
                }
            }
        }
        
        println!("Parametri del modello caricati da JSON:");
        println!("- d_model: {}", d_model);
        println!("- max_seq_len: {}", max_seq_len);
        println!("- num_heads: {}", num_heads);
        println!("- ff_dim: {}", ff_dim);
        println!("- num_layers: {}", num_layers);
        println!("- dropout_rate: {}", dropout_rate);
        println!("- learning_rate: {}", learning_rate);
    } else {
        // Caricamento da file di testo separati
        let train_file_path = default_train_path;
        let test_file_path = default_test_path;
        
        println!("Usando i file di testo:");
        println!("- Training: {}", train_file_path);
        println!("- Test: {}", test_file_path);
        
        // Utilizzo delle funzioni del modulo dataset per caricare i file
        train_text = match wall_e1::dataset::loader::load_text(train_file_path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Errore nel caricamento del file di training {}: {}", train_file_path, e);
                return;
            }
        };
        
        test_text = match wall_e1::dataset::loader::load_text(test_file_path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Errore nel caricamento del file di test {}: {}", test_file_path, e);
                return;
            }
        };
    }
    
    println!("Dati caricati con successo:");
    println!("- Testo di training: {} caratteri", train_text.len());
    println!("- Testo di test: {} caratteri", test_text.len());

    // 2. Tokenizzazione
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab(&train_text, 1);
    tokenizer.build_vocab(&test_text, 1);

    println!("Vocabolario costruito con {} token", tokenizer.get_vocab().len());

    // 3. Preparazione dei dati di training
    let train_tokens = tokenizer.encode(&train_text);
    let test_tokens = tokenizer.encode(&test_text);

    // 4. Configurazione del modello
    let vocab_size = tokenizer.get_vocab().len();
    
    // Questi parametri ora sono impostati tramite variabili caricate dal JSON o valori predefiniti
    // let d_model = 64;      // Dimensione dell'embedding
    // let max_seq_len = 128; // Lunghezza massima delle sequenze
    // let num_heads = 4;     // Numero di teste per multi-head attention
    // let ff_dim = 128;      // Dimensione interna del feed-forward network
    // let num_layers = 2;    // Numero di layer nell'encoder stack
    // let dropout_rate = 0.1;
    // let learning_rate = 0.001;

    // 5. Creazione del trainer, ottimizzatore e loss function
    let mut trainer = Trainer::new(
        Box::new(tokenizer.clone()) as Box<dyn Tokenizer>,
        d_model,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );

    // 6. Training loop
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        // Prepara gli input e i target
        // Input: tutti i token tranne l'ultimo
        // Target: tutti i token tranne il primo (predizione del token successivo)
        let input_tokens = train_tokens[0..train_tokens.len()-1].to_vec();
        let target_tokens = train_tokens[1..train_tokens.len()].to_vec();
        
        // Converti target_tokens in Array2
        let mut targets = ndarray::Array2::zeros((1, target_tokens.len()));
        for (j, &token_id) in target_tokens.iter().enumerate() {
            targets[[0, j]] = token_id;
        }
        
        // Forward pass per calcolare l'accuracy prima dell'aggiornamento
        let output = trainer.forward(&[input_tokens.clone()], Some(&targets));
        
        // Calcola l'accuracy
        let mut correct = 0;
        let total = target_tokens.len();
        
        // Estrai i logits dell'output
        let logits_data = output.logits.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        
        // Per ogni posizione, trova il token con la probabilità più alta
        for j in 0..total {
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            
            for v in 0..logits_data.shape()[2] {
                let val = logits_data[[0, j, v]];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }
            
            // Confronta con il target
            if max_idx == targets[[0, j]] as usize {
                correct += 1;
            }
        }
        
        let accuracy = (correct as f32) / (total as f32) * 100.0;
        
        // Backward pass e aggiornamento dei parametri
        let loss = trainer.train_step(
            &[input_tokens], 
            &targets
        );
        
        println!("Epoca {}/{}: loss = {:.6}, accuracy = {:.2}%", epoch + 1, num_epochs, loss, accuracy);
    }

    // 7. Generazione di testo
    let prompt = "C'era una volta";
    
    // Generazione con temperatura bassa (output più deterministico)
    let temperature_low = 0.2;
    let generated_text_low_temp = trainer.generate(
        prompt,
        15,              // max_tokens
        temperature_low, // temperatura bassa
        None             // top_k (None = usa tutti i token)
    );

    println!("Prompt: '{}'", prompt);
    println!("Testo generato (temperatura bassa): '{}'", generated_text_low_temp);

    // Generazione con temperatura alta (output più creativo)
    let temperature_high = 1.5;
    let generated_text_high_temp = trainer.generate(
        prompt,
        15,               // max_tokens
        temperature_high, // temperatura alta
        None              // top_k (None = usa tutti i token)
    );

    println!("Testo generato (temperatura alta): '{}'", generated_text_high_temp);

    // Prepara i dati di valutazione usando un closure per funzione forward
    let forward_fn = |model: &Trainer, token_ids: &Vec<Vec<usize>>, _: Option<ndarray::Array2<f32>>| {
        let mut targets = ndarray::Array2::zeros((token_ids.len(), token_ids[0].len()));
        for (i, seq) in token_ids.iter().enumerate() {
            for (j, &token_id) in seq.iter().enumerate() {
                targets[[i, j]] = token_id;
            }
        }
        model.forward(token_ids, Some(&targets))
    };

    // Prepara i dati di test
    let test_input = test_tokens[0..test_tokens.len()-1].to_vec();
    let test_dataset = vec![vec![test_input]];

    // Padding token ID (assumiamo 0 per [PAD])
    let padding_token_id = 0;

    // Valutazione
    let (avg_loss, accuracy) = training::evaluate::evaluate(
        &trainer,
        forward_fn,
        &test_dataset,
        &(Box::new(tokenizer.clone()) as Box<dyn Tokenizer>),
        Some(padding_token_id)
    );

    let perplexity = (avg_loss as f64).exp();

    println!("Loss: {:.4}", avg_loss);
    println!("Perplexity: {:.4}", perplexity);
    println!("Accuratezza: {:.2}%", accuracy * 100.0);

    // Generazione di campioni di testo
    let prompts_str: Vec<&str> = if !custom_prompts.is_empty() {
        // Usa i prompt dal file JSON se disponibili
        custom_prompts.iter().map(|s| s.as_str()).collect()
    } else {
        // Altrimenti usa i prompt predefiniti
        vec![
            "Stella Marie",
            "Ma la bambina", 
            "La luce si"
        ]
    };

    training::evaluate::print_generated_samples(
        &trainer,
        forward_fn,
        &prompts_str,
        &(Box::new(tokenizer.clone()) as Box<dyn Tokenizer>),
        20,   // max_new_tokens
        0.8   // temperature
    );
}

/// Esempio di utilizzo del modulo di dataset
fn dataset_example() {
    println!("\n--- Esempio di utilizzo del modulo dataset ---");
    
    // Dati di esempio
    let data: Vec<i32> = (0..100).collect();
    println!("Dataset originale: {} elementi", data.len());
    
    // Divisione in train, validation e test
    match wall_e1::dataset::split_dataset(&data, 0.7, 0.15, 0.15, true) {
        Ok(split) => {
            println!("Split del dataset:");
            println!("- Training: {} elementi", split.train.len());
            println!("- Validation: {} elementi", split.validation.len());
            println!("- Test: {} elementi", split.test.len());
        },
        Err(e) => {
            println!("Errore nella divisione del dataset: {}", e);
        }
    }
    
    // Divisione K-fold
    let k = 5;
    let folds = wall_e1::dataset::k_fold_split(&data, k, true);
    println!("\nK-fold cross validation (k={})", k);
    for (i, (train, val)) in folds.iter().enumerate() {
        println!("Fold {}: train={}, validation={}", i+1, train.len(), val.len());
    }
    
    println!("\nFine dell'esempio dataset");
}

/// Esempio di fine-tuning di un modello per question answering
fn question_answering_example(config_path: Option<&str>) {
    println!("\n--- Question Answering Example ---");
    
    // 1. Caricamento dei dati da file JSON
    println!("Caricamento del dataset di domande e risposte...");
    
    // Usa il percorso fornito o quello predefinito
    let default_json_path = "data/qa_dataset.json";
    let json_file_path = config_path.unwrap_or(default_json_path);
    println!("Usando il file JSON: {}", json_file_path);
    
    // Carica i dati JSON
    let file = match std::fs::File::open(json_file_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Errore nell'apertura del file JSON {}: {}", json_file_path, e);
            return;
        }
    };
    
    let reader = std::io::BufReader::new(file);
    let json_data: serde_json::Value = match serde_json::from_reader(reader) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Errore nella decodifica del JSON da {}: {}", json_file_path, e);
            return;
        }
    };
    
    // 2. Prepara i dati di training e test
    let mut train_contexts = Vec::new();
    let mut train_questions = Vec::new();
    let mut train_answers = Vec::new();
    
    // Estrai i dati di training dal JSON
    if let Some(train_data) = json_data.get("train_data").and_then(|v| v.as_array()) {
        for item in train_data {
            if let Some(context) = item.get("context").and_then(|v| v.as_str()) {
                if let Some(questions) = item.get("questions").and_then(|v| v.as_array()) {
                    for q in questions {
                        if let (Some(question), Some(answer)) = (
                            q.get("question").and_then(|v| v.as_str()),
                            q.get("answer").and_then(|v| v.as_str()),
                        ) {
                            train_contexts.push(context.to_string());
                            train_questions.push(question.to_string());
                            train_answers.push(answer.to_string());
                        }
                    }
                }
            }
        }
    }
    
    let mut test_contexts = Vec::new();
    let mut test_questions = Vec::new();
    let mut test_answers = Vec::new();
    
    // Estrai i dati di test dal JSON
    if let Some(test_data) = json_data.get("test_data").and_then(|v| v.as_array()) {
        for item in test_data {
            if let Some(context) = item.get("context").and_then(|v| v.as_str()) {
                if let Some(questions) = item.get("questions").and_then(|v| v.as_array()) {
                    for q in questions {
                        if let (Some(question), Some(answer)) = (
                            q.get("question").and_then(|v| v.as_str()),
                            q.get("answer").and_then(|v| v.as_str()),
                        ) {
                            test_contexts.push(context.to_string());
                            test_questions.push(question.to_string());
                            test_answers.push(answer.to_string());
                        }
                    }
                }
            }
        }
    }
    
    println!("Dataset caricato con successo:");
    println!("- Esempi di training: {}", train_questions.len());
    println!("- Esempi di test: {}", test_questions.len());
    
    // 3. Estrai i parametri del modello dal JSON
    let mut d_model = 64;      // Dimensione dell'embedding
    let mut max_seq_len = 256; // Lunghezza massima delle sequenze
    let mut num_heads = 4;     // Numero di teste per multi-head attention
    let mut ff_dim = 128;      // Dimensione interna del feed-forward network
    let mut num_layers = 2;    // Numero di layer nell'encoder stack
    let mut dropout_rate = 0.1;
    let mut learning_rate = 0.001;
    
    // Estrai i parametri se presenti
    if let Some(model_params) = json_data.get("model_params") {
        if let Some(val) = model_params.get("d_model").and_then(|v| v.as_u64()) {
            d_model = val as usize;
        }
        
        if let Some(val) = model_params.get("max_seq_len").and_then(|v| v.as_u64()) {
            max_seq_len = val as usize;
        }
        
        if let Some(val) = model_params.get("num_heads").and_then(|v| v.as_u64()) {
            num_heads = val as usize;
        }
        
        if let Some(val) = model_params.get("ff_dim").and_then(|v| v.as_u64()) {
            ff_dim = val as usize;
        }
        
        if let Some(val) = model_params.get("num_layers").and_then(|v| v.as_u64()) {
            num_layers = val as usize;
        }
        
        if let Some(val) = model_params.get("dropout_rate").and_then(|v| v.as_f64()) {
            dropout_rate = val as f32;
        }
        
        if let Some(val) = model_params.get("learning_rate").and_then(|v| v.as_f64()) {
            learning_rate = val as f32;
        }
    }
    
    println!("Parametri del modello:");
    println!("- d_model: {}", d_model);
    println!("- max_seq_len: {}", max_seq_len);
    println!("- num_heads: {}", num_heads);
    println!("- ff_dim: {}", ff_dim);
    println!("- num_layers: {}", num_layers);
    println!("- dropout_rate: {}", dropout_rate);
    println!("- learning_rate: {}", learning_rate);
    
    // 4. Formatta i dati di training nel formato corretto per QA
    println!("\nPreparazione dei dati per il training...");
    
    let mut train_text = String::new();
    let mut test_text = String::new();
    
    // Formato per QA: "Contesto: {context} Domanda: {question} Risposta: {answer}"
    for i in 0..train_contexts.len() {
        train_text.push_str(&format!(
            "Contesto: {} Domanda: {} Risposta: {}\n",
            train_contexts[i], train_questions[i], train_answers[i]
        ));
    }
    
    for i in 0..test_contexts.len() {
        test_text.push_str(&format!(
            "Contesto: {} Domanda: {} Risposta: {}\n",
            test_contexts[i], test_questions[i], test_answers[i]
        ));
    }
    
    // 5. Tokenizzazione
    let mut tokenizer = BasicTokenizer::new();
    tokenizer.build_vocab(&train_text, 1);
    tokenizer.build_vocab(&test_text, 1);
    
    println!("Vocabolario costruito con {} token", tokenizer.get_vocab().len());
    
    // 6. Preparazione dei dati di training
    let train_tokens = tokenizer.encode(&train_text);
    let test_tokens = tokenizer.encode(&test_text);
    
    // 7. Creazione del trainer
    let mut trainer = Trainer::new(
        Box::new(tokenizer.clone()) as Box<dyn Tokenizer>,
        d_model,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate,
    );
    
    // 8. Training loop
    println!("\nInizio del training...");
    let num_epochs = 1; // Ridotto a 1 per debug
    
    println!("Debug: Preparazione degli input e target");
    
    // Prepara gli input e i target
    // Input: tutti i token tranne l'ultimo
    // Target: tutti i token tranne il primo (predizione del token successivo)
    let input_tokens = train_tokens[0..train_tokens.len()-1].to_vec();
    let target_tokens = train_tokens[1..train_tokens.len()].to_vec();
    
    println!("Debug: Creazione dell'array targets");
    
    // Converti target_tokens in Array2
    let mut targets = ndarray::Array2::zeros((1, target_tokens.len()));
    for (j, &token_id) in target_tokens.iter().enumerate() {
        targets[[0, j]] = token_id;
    }
    
    println!("Debug: Forward pass per calcolare accuracy");
    
    for epoch in 0..num_epochs {
        // Forward pass per calcolare l'accuracy prima dell'aggiornamento
        println!("Debug: Esecuzione forward pass - Inizio");
        let output = trainer.forward(&[input_tokens.clone()], Some(&targets));
        println!("Debug: Esecuzione forward pass - Fine");
        
        // Calcola l'accuracy
        println!("Debug: Calcolo accuracy");
        let mut correct = 0;
        let total = target_tokens.len();
        
        // Estrai i logits dell'output
        println!("Debug: Estrazione logits");
        let logits_data = output.logits.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        
        // Per ogni posizione, trova il token con la probabilità più alta
        println!("Debug: Elaborazione token per token");
        for j in 0..total {
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            
            for v in 0..logits_data.shape()[2] {
                let val = logits_data[[0, j, v]];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }
            
            // Confronta con il target
            if max_idx == targets[[0, j]] as usize {
                correct += 1;
            }
        }
        
        let accuracy = (correct as f32) / (total as f32) * 100.0;
        println!("Debug: Accuracy calcolata: {}%", accuracy);
        
        // Backward pass e aggiornamento dei parametri
        println!("Debug: Esecuzione train_step - Inizio");
        let loss = trainer.train_step(
            &[input_tokens.clone()], 
            &targets
        );
        println!("Debug: Esecuzione train_step - Fine");
        
        println!("Epoca {}/{}: loss = {:.6}, accuracy = {:.2}%", epoch + 1, num_epochs, loss, accuracy);
    }
    
    println!("Debug: Training completato");
    
    // 9. Valutazione sul test set
    let forward_fn = |model: &Trainer, token_ids: &Vec<Vec<usize>>, _: Option<ndarray::Array2<f32>>| {
        let mut targets = ndarray::Array2::zeros((token_ids.len(), token_ids[0].len()));
        for (i, seq) in token_ids.iter().enumerate() {
            for (j, &token_id) in seq.iter().enumerate() {
                targets[[i, j]] = token_id;
            }
        }
        model.forward(token_ids, Some(&targets))
    };
    
    // Prepara i dati di test
    let test_input = test_tokens[0..test_tokens.len()-1].to_vec();
    let test_dataset = vec![vec![test_input]];
    
    // Padding token ID (assumiamo 0 per [PAD])
    let padding_token_id = 0;
    
    // Valutazione
    let (avg_loss, accuracy) = training::evaluate::evaluate(
        &trainer,
        forward_fn,
        &test_dataset,
        &(Box::new(tokenizer.clone()) as Box<dyn Tokenizer>),
        Some(padding_token_id)
    );
    
    let perplexity = (avg_loss as f64).exp();
    
    println!("\nValutazione sul test set:");
    println!("- Loss: {:.4}", avg_loss);
    println!("- Perplexity: {:.4}", perplexity);
    println!("- Accuratezza: {:.2}%", accuracy * 100.0);
    
    // 10. Test di inferenza con domande specifiche
    println!("\nTest di domande e risposte:");
    
    // Estrai prompt dal JSON o usa quelli predefiniti
    let prompts = if let Some(prompts_json) = json_data.get("prompts").and_then(|v| v.as_array()) {
        prompts_json
            .iter()
            .filter_map(|p| p.as_str().map(String::from))
            .collect::<Vec<String>>()
    } else {
        vec![
            "Come gestisce Rust la memoria?".to_string(),
            "Qual è la capitale dell'Italia?".to_string(),
            "Per cosa è stato progettato Rust?".to_string(),
        ]
    };
    
    // Funzione per generare risposte
    let answer_question = |context: &str, question: &str| {
        let input = format!("Contesto: {} Domanda: {}", context, question);
        let temperature = 0.5;
        let max_tokens = 30;
        
        let response = trainer.generate(
            &input,
            max_tokens,
            temperature,
            None, // top_k
        );
        
        // Estrai solo la parte di risposta
        if let Some(resp_idx) = response.find("Risposta: ") {
            response[resp_idx + 10..].trim().to_string()
        } else {
            response
        }
    };
    
    // Test con i contesti del set di test
    if !test_contexts.is_empty() {
        let context = &test_contexts[0];
        
        for prompt in &prompts {
            println!("\nDomanda: {}", prompt);
            let answer = answer_question(context, prompt);
            println!("Risposta: {}", answer);
        }
    }
    
    println!("\nExample di Question Answering completato!");
}

