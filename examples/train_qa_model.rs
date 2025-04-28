use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use wall_e1::tokenizer::basic_tokenizer::BasicTokenizer;
use wall_e1::tokenizer::Tokenizer;
use wall_e1::training::Trainer;
use serde_json::Value;
use ndarray::Array2;
use std::io::{self, Write};
use std::time::Instant;
use ndarray::{Array, Axis, s};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Addestramento modello QA Wall-E1");
    println!("--------------------------------");
    
    // Carica il dataset QA
    println!("Caricamento del dataset da 'data/qa_dataset.json'...");
    let dataset_path = "data/qa_dataset.json";
    
    if !Path::new(dataset_path).exists() {
        eprintln!("Errore: Il dataset '{}' non esiste!", dataset_path);
        return Err("Dataset non trovato".into());
    }
    
    // Leggi il file JSON
    let mut file = File::open(dataset_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    
    // Deserializza il JSON
    let data: Value = serde_json::from_str(&contents)?;
    
    // Estrai i parametri del modello
    let model_params = &data["model_params"];
    let d_model = model_params["d_model"].as_u64().unwrap_or(96) as usize;
    let ff_dim = model_params["ff_dim"].as_u64().unwrap_or(192) as usize;
    let num_heads = model_params["num_heads"].as_u64().unwrap_or(6) as usize;
    let num_layers = model_params["num_layers"].as_u64().unwrap_or(3) as usize;
    let dropout_rate = model_params["dropout_rate"].as_f64().unwrap_or(0.15) as f32;
    let learning_rate = model_params["learning_rate"].as_f64().unwrap_or(0.0005) as f32;
    
    println!("Parametri del modello:");
    println!("  - Dimensione del modello: {}", d_model);
    println!("  - Dimensione feed-forward: {}", ff_dim);
    println!("  - Numero di teste di attenzione: {}", num_heads);
    println!("  - Numero di layer: {}", num_layers);
    println!("  - Dropout rate: {}", dropout_rate);
    println!("  - Learning rate: {}", learning_rate);
    
    // Estrai i dati di training
    let train_data = &data["train_data"];
    let mut training_texts = Vec::new();
    
    // Prepara un corpus per costruire il vocabolario
    let mut corpus = String::new();
    
    for item in train_data.as_array().unwrap() {
        let context = item["context"].as_str().unwrap();
        corpus.push_str(context);
        corpus.push_str(" ");
        
        training_texts.push(context.to_string());
        
        // Aggiungi anche domande e risposte al corpus
        for qa in item["questions"].as_array().unwrap() {
            let question = qa["question"].as_str().unwrap();
            let answer = qa["answer"].as_str().unwrap();
            corpus.push_str(question);
            corpus.push_str(" ");
            corpus.push_str(answer);
            corpus.push_str(" ");
            
            training_texts.push(format!("Domanda: {} Risposta: {}", question, answer));
        }
    }
    
    // Crea e inizializza il tokenizer
    let mut tokenizer = BasicTokenizer::new();
    println!("\nCostruzione del vocabolario dal corpus...");
    tokenizer.build_vocab(&corpus, 2); // Minima frequenza: 2
    
    let vocab_size = tokenizer.get_vocab().len();
    println!("Vocabolario costruito con {} token", vocab_size);
    
    // Crea il trainer con i parametri estratti
    println!("\nInizializzazione del modello...");
    let mut trainer = Trainer::new(
        Box::new(tokenizer.clone()),
        d_model,
        ff_dim,
        num_heads,
        num_layers,
        dropout_rate,
        learning_rate
    );
    
    // Prepara gli esempi di addestramento
    println!("Preparazione degli esempi di addestramento...");
    let mut training_examples = Vec::new();
    let mut training_targets = Vec::new();
    
    for text in &training_texts {
        let tokens = tokenizer.encode(text);
        if tokens.len() > 3 {  // Assicuriamoci che ci siano abbastanza token
            for i in 0..(tokens.len() - 1) {
                let mut input = Vec::new();
                let mut target = Vec::new();
                
                // Prendiamo massimo 10 token come contesto
                let start = if i > 9 { i - 9 } else { 0 };
                for j in start..=i {
                    input.push(tokens[j]);
                }
                
                // Il target è il token successivo
                target.push(tokens[i + 1]);
                
                training_examples.push(input);
                training_targets.push(target);
            }
        }
    }
    
    // Batch delle sequenze
    let mut batched_examples = Vec::new();
    let mut batched_targets = Vec::new();
    let batch_size = 1; // Ridotto a 1 per evitare problemi di compatibilità di forma
    
    for i in (0..training_examples.len()).step_by(batch_size) {
        let mut batch = Vec::new();
        let mut batch_targets = Vec::new();
        
        let end = std::cmp::min(i + batch_size, training_examples.len());
        
        // Trova la sequenza più lunga in questo batch per il padding
        let max_len = training_examples[i..end]
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(1);
            
        // Trova la lunghezza massima dei target in questo batch
        let max_target_len = training_targets[i..end]
            .iter()
            .map(|seq| seq.len())
            .max()
            .unwrap_or(1);
        
        for j in i..end {
            // Crea una versione padded della sequenza
            let mut padded_seq = training_examples[j].clone();
            // Padding con token 0 fino alla lunghezza massima
            while padded_seq.len() < max_len {
                padded_seq.push(0);
            }
            
            // Crea una versione padded del target
            let mut padded_target = training_targets[j].clone();
            // Padding con token 0 fino alla lunghezza massima dei target
            while padded_target.len() < max_target_len {
                padded_target.push(0);
            }
            
            batch.push(padded_seq);
            batch_targets.push(padded_target);
        }
        
        if !batch.is_empty() {
            batched_examples.push(batch);
            batched_targets.push(batch_targets);
        }
    }
    
    // Addestramento del modello
    println!("\nInizio addestramento con {} batch...", batched_examples.len());
    println!("\nNOTA: Usando batch size di 1 per compatibilità con l'implementazione attuale di attention.");
    let epochs = 1; // Ridotto a 1 epoca poiché abbiamo più batch con batch size 1
    
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        for (batch_idx, (batch, targets)) in batched_examples.iter().zip(batched_targets.iter()).enumerate() {
            // Converti i target in Array2
            let max_seq_len = targets.iter().map(|t| t.len()).max().unwrap_or(1);
            let mut targets_array = Array2::zeros((targets.len(), max_seq_len));
            
            for (i, target_seq) in targets.iter().enumerate() {
                for (j, &token) in target_seq.iter().enumerate() {
                    targets_array[[i, j]] = token;
                }
            }
            
            // Esegui un passo di training
            let loss = trainer.train_step(batch, &targets_array);
            total_loss += loss;
            num_batches += 1;
            
            if batch_idx % 10 == 0 {
                println!("  Epoch {}, Batch {}/{}: Loss = {:.4}", 
                         epoch, batch_idx + 1, batched_examples.len(), loss);
            }
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {} completata. Loss media: {:.4}", epoch, avg_loss);
    }
    
    // Salva il modello addestrato
    let model_path = "models/qa_model.bin";
    println!("\nSalvataggio del modello in '{}'...", model_path);
    trainer.save_model(model_path)?;
    
    // Test con le domande dal dataset
    println!("\nTest del modello con le domande dal dataset:");
    
    let test_prompts = data["prompts"].as_array().unwrap();
    for prompt in test_prompts {
        let question = prompt.as_str().unwrap();
        println!("\nDomanda: {}", question);
        
        // Codifichiamo il prompt
        let tokens = tokenizer.encode(question);
        println!("Token IDs: {:?}", tokens);
        
        // Prepariamo l'input (un batch con un singolo esempio)
        let input = vec![tokens.clone()];
        
        // Forward pass
        let output = trainer.forward(&input, None);
        
        // Prendiamo i token più probabili dall'output
        let logits = output.logits.data.slice(s![0, tokens.len() - 1, ..]);
        let mut token_probs: Vec<(usize, f32)> = Vec::new();
        
        for token_id in 0..vocab_size {
            let prob = logits[token_id];
            token_probs.push((token_id, prob));
        }
        
        // Ordiniamo per probabilità (decrescente)
        token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Mostriamo i top 3 token
        println!("Risposte più probabili:");
        for (i, (token_id, prob)) in token_probs.iter().take(3).enumerate() {
            let token_text = tokenizer.decode(&[*token_id]);
            println!("  {}. \"{}\" (probabilità: {:.3})", i+1, token_text, prob);
        }
    }
    
    println!("\nAddestramento e test completati!");
    println!("Modello salvato in: {}", model_path);
    Ok(())
} 