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
use ndarray_parallel::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Addestramento modello QA Wall-E1");
    println!("--------------------------------");
    println!("Utilizzo {} thread per il calcolo parallelo", rayon::current_num_threads());
    
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
    
    // Usiamo Rayon per elaborare i dati di training in parallelo
    let training_texts: Vec<String> = {
        let items = train_data.as_array().unwrap();
        let contexts_and_qa: Vec<(String, Vec<(String, String)>)> = items.par_iter().map(|item| {
            let context = item["context"].as_str().unwrap().to_string();
            
            let qa_pairs = item["questions"].as_array().unwrap().iter()
                .map(|qa| {
                    let question = qa["question"].as_str().unwrap().to_string();
                    let answer = qa["answer"].as_str().unwrap().to_string();
                    (question, answer)
                })
                .collect();
                
            (context, qa_pairs)
        }).collect();
        
        // Appiattire i risultati per ottenere un unico vettore di testi
        contexts_and_qa.into_par_iter().flat_map(|(context, qa_pairs)| {
            let mut texts = vec![context.clone()];
            
            qa_pairs.into_iter().for_each(|(question, answer)| {
                texts.push(format!("Domanda: {} Risposta: {}", question, answer));
            });
            
            texts
        }).collect()
    };
    
    // Prepara un corpus per costruire il vocabolario
    let corpus = {
        // Uniamo tutti i testi in parallelo
        let corpus_chunks: Vec<String> = training_texts.par_iter()
            .map(|text| text.clone() + " ")
            .collect();
            
        // Unisci i chunk in una stringa unica
        corpus_chunks.join("")
    };
    
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
    
    // Utilizziamo un vettore di esempi e target condiviso tra thread
    let training_examples_mutex = Arc::new(Mutex::new(Vec::new()));
    let training_targets_mutex = Arc::new(Mutex::new(Vec::new()));
    
    // Parallelizza la generazione degli esempi di addestramento
    training_texts.par_iter().for_each(|text| {
        let tokens = tokenizer.encode(text);
        if tokens.len() > 3 {  // Assicuriamoci che ci siano abbastanza token
            let mut local_examples = Vec::new();
            let mut local_targets = Vec::new();
            
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
                
                local_examples.push(input);
                local_targets.push(target);
            }
            
            // Acquisici il lock e aggiungi esempi e target
            if let Ok(mut examples) = training_examples_mutex.lock() {
                examples.extend(local_examples);
            }
            
            if let Ok(mut targets) = training_targets_mutex.lock() {
                targets.extend(local_targets);
            }
        }
    });
    
    // Estrai i dati dai mutex
    let training_examples = Arc::try_unwrap(training_examples_mutex)
        .map_err(|_| "Impossibile ottenere i dati di addestramento").unwrap()
        .into_inner().map_err(|_| "Errore di lock").unwrap();
    
    let training_targets = Arc::try_unwrap(training_targets_mutex)
        .map_err(|_| "Impossibile ottenere i target di addestramento").unwrap()
        .into_inner().map_err(|_| "Errore di lock").unwrap();
    
    println!("Generati {} esempi di addestramento", training_examples.len());
    
    // Batch delle sequenze - Utilizziamo Rayon per processare i batch in parallelo
    println!("Preparazione dei batch in parallelo...");
    let batch_size = 16; // Ridotto a 1 per evitare problemi di compatibilità di forma
    
    // Crea range di indici
    let indices: Vec<usize> = (0..training_examples.len()).step_by(batch_size).collect();
    
    // Struttura dati per i batch
    let batch_results = Arc::new(Mutex::new(Vec::new()));
    let target_results = Arc::new(Mutex::new(Vec::new()));
    
    // Processamento parallelo dei batch
    indices.par_iter().for_each(|&i| {
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
            // Aggiungi il batch ai risultati
            if let Ok(mut batches) = batch_results.lock() {
                batches.push(batch);
            }
            
            if let Ok(mut targets) = target_results.lock() {
                targets.push(batch_targets);
            }
        }
    });
    
    // Estrai i dati dai mutex
    let batched_examples = Arc::try_unwrap(batch_results)
        .map_err(|_| "Impossibile ottenere i batch").unwrap()
        .into_inner().map_err(|_| "Errore di lock").unwrap();
    
    let batched_targets = Arc::try_unwrap(target_results)
        .map_err(|_| "Impossibile ottenere i target batch").unwrap()
        .into_inner().map_err(|_| "Errore di lock").unwrap();
    
    // Addestramento del modello
    println!("\nInizio addestramento con {} batch...", batched_examples.len());
    println!("\nNOTA: Usando batch size di {}.", batch_size);
    let epochs = 20; // Aumentiamo a 3 epoche
    
    for epoch in 1..=epochs {
        // Utilizziamo Mutex per aggiornare in modo sicuro i contatori di loss
        let total_loss_mutex = Arc::new(Mutex::new(0.0));
        let num_batches_mutex = Arc::new(Mutex::new(0));
        
        // Creiamo un contatore per monitorare i batch
        let batch_counter = Arc::new(Mutex::new(0));
        
        // Trainer deve essere mutato sequenzialmente, quindi non possiamo usare par_iter qui
        for (batch_idx, (batch, targets)) in batched_examples.iter().zip(batched_targets.iter()).enumerate() {
            // Converti i target in Array2
            let max_seq_len = targets.iter().map(|t| t.len()).max().unwrap_or(1);
            let mut targets_array = Array2::zeros((targets.len(), max_seq_len));
            
            // Utilizziamo par_azip per impostare i valori di targets_array in parallelo
            for (i, target_seq) in targets.iter().enumerate() {
                for (j, &token) in target_seq.iter().enumerate() {
                    targets_array[[i, j]] = token;
                }
            }
            
            // Esegui un passo di training
            let loss = trainer.train_step(batch, &targets_array);
            
            // Aggiorna il conteggio in modo thread-safe
            if let Ok(mut total) = total_loss_mutex.lock() {
                *total += loss;
            }
            
            if let Ok(mut count) = num_batches_mutex.lock() {
                *count += 1;
            }
            
            // Incrementa e controlla se mostrare l'avanzamento
            let should_print = {
                let mut counter = batch_counter.lock().unwrap();
                *counter += 1;
                *counter % 10 == 0
            };
            
            if should_print {
                println!("  Epoch {}, Batch {}/{}: Loss = {:.4}", 
                         epoch, batch_idx + 1, batched_examples.len(), loss);
            }
        }
        
        // Estrai i valori finali dai mutex
        let total_loss = *total_loss_mutex.lock().unwrap();
        let num_batches = *num_batches_mutex.lock().unwrap();
        
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
    
    // Prepara una struttura per i risultati
    struct TestResult {
        question: String,
        top_responses: Vec<(String, f32)>
    }
    
    // Creiamo un mutex per i risultati
    let results_mutex = Arc::new(Mutex::new(Vec::new()));
    
    // Test sincronizzato per i prompts
    for prompt in test_prompts {
        let question = prompt.as_str().unwrap().to_string();
        
        // Codifichiamo il prompt
        let tokens = tokenizer.encode(&question);
        
        // Prepariamo l'input (un batch con un singolo esempio)
        let input = vec![tokens.clone()];
        
        // Forward pass
        let output = trainer.forward(&input, None);
        
        // Prendiamo i token più probabili dall'output
        let logits = output.logits.data.slice(s![0, tokens.len() - 1, ..]);
        
        // Parallelizziamo la creazione dell'array di probabilità
        let token_probs: Vec<(usize, f32)> = (0..vocab_size)
            .into_par_iter()
            .map(|token_id| (token_id, logits[token_id]))
            .collect();
        
        // Ordiniamo per probabilità (decrescente)
        let mut sorted_probs = token_probs;
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Prepara le risposte migliori
        let top_responses: Vec<(String, f32)> = sorted_probs.iter().take(3)
            .map(|(token_id, prob)| (tokenizer.decode(&[*token_id]), *prob))
            .collect();
            
        // Aggiungi il risultato
        if let Ok(mut results) = results_mutex.lock() {
            results.push(TestResult {
                question,
                top_responses
            });
        }
    }
    
    // Estrai i risultati e stampali
    let test_results = Arc::try_unwrap(results_mutex)
        .map_err(|_| "Impossibile ottenere i risultati del test").unwrap()
        .into_inner().map_err(|_| "Errore di lock").unwrap();
        
    // Stampa i risultati in parallelo
    for result in test_results {
        println!("\nDomanda: {}", result.question);
        println!("Risposte più probabili:");
        for (i, (token_text, prob)) in result.top_responses.iter().enumerate() {
            println!("  {}. \"{}\" (probabilità: {:.3})", i+1, token_text, prob);
        }
    }
    
    println!("\nAddestramento e test completati!");
    println!("Modello salvato in: {}", model_path);
    Ok(())
} 