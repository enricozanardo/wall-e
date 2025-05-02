use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use wall_e1::tokenizer::basic_tokenizer::BasicTokenizer;
use wall_e1::tokenizer::Tokenizer;
use wall_e1::training::Trainer;
use serde_json::Value;
use ndarray::Array2;
use std::time::Instant;
use ndarray::s;
use ndarray_parallel::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle};
use std::fmt;

// Struttura per rappresentare un parametro del modello
struct ModelParam {
    name: String,
    shape: Vec<usize>,
    param_count: usize,
}

impl fmt::Display for ModelParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<30} {:<25} {:>12}", 
               self.name, 
               format!("{:?}", self.shape), 
               format!("{}", self.param_count))
    }
}

// Funzione per calcolare i parametri del modello
#[allow(unused_variables)]
fn calculate_model_parameters(
    vocab_size: usize,
    d_model: usize,
    ff_dim: usize,
    num_heads: usize,
    num_layers: usize,
    max_seq_len: usize
) -> Vec<ModelParam> {
    let mut params = Vec::new();
    
    // Token embeddings
    params.push(ModelParam {
        name: "Token Embeddings".to_string(),
        shape: vec![vocab_size, d_model],
        param_count: vocab_size * d_model,
    });
    
    // Positional embeddings
    params.push(ModelParam {
        name: "Positional Embeddings".to_string(),
        shape: vec![max_seq_len, d_model],
        param_count: max_seq_len * d_model,
    });
    
    // Embedding Layer Norm
    params.push(ModelParam {
        name: "Embedding LayerNorm".to_string(),
        shape: vec![d_model],
        param_count: d_model * 2, // gamma e beta
    });
    
    // Encoder layers
    for i in 0..num_layers {
        // Self-attention
        // Query, Key, Value projections
        params.push(ModelParam {
            name: format!("Encoder {}/Self-Attention/Query", i),
            shape: vec![d_model, d_model],
            param_count: d_model * d_model,
        });
        
        params.push(ModelParam {
            name: format!("Encoder {}/Self-Attention/Key", i),
            shape: vec![d_model, d_model],
            param_count: d_model * d_model,
        });
        
        params.push(ModelParam {
            name: format!("Encoder {}/Self-Attention/Value", i),
            shape: vec![d_model, d_model],
            param_count: d_model * d_model,
        });
        
        // Output projection
        params.push(ModelParam {
            name: format!("Encoder {}/Self-Attention/Output", i),
            shape: vec![d_model, d_model],
            param_count: d_model * d_model,
        });
        
        // Layer normalization 1
        params.push(ModelParam {
            name: format!("Encoder {}/LayerNorm 1", i),
            shape: vec![d_model],
            param_count: d_model * 2, // gamma e beta
        });
        
        // Feed-forward network
        params.push(ModelParam {
            name: format!("Encoder {}/FFN/Linear 1", i),
            shape: vec![d_model, ff_dim],
            param_count: d_model * ff_dim,
        });
        
        params.push(ModelParam {
            name: format!("Encoder {}/FFN/Linear 2", i),
            shape: vec![ff_dim, d_model],
            param_count: ff_dim * d_model,
        });
        
        // Layer normalization 2
        params.push(ModelParam {
            name: format!("Encoder {}/LayerNorm 2", i),
            shape: vec![d_model],
            param_count: d_model * 2, // gamma e beta
        });
    }
    
    // Output projection
    params.push(ModelParam {
        name: "Output Projection".to_string(),
        shape: vec![d_model, vocab_size],
        param_count: d_model * vocab_size,
    });
    
    params
}

// Funzione per visualizzare il sommario del modello
fn print_model_summary(
    params: &Vec<ModelParam>,
    d_model: usize,
    ff_dim: usize,
    num_heads: usize,
    num_layers: usize,
    vocab_size: usize,
    max_seq_len: usize
) {
    println!("\n{}", "=".repeat(70));
    println!("                     MODELLO TRANSFORMER QA WALL-E1");
    println!("{}", "=".repeat(70));
    println!("Architettura del modello:");
    println!("  - Dimensione del vocabolario: {}", vocab_size);
    println!("  - Dimensione del modello (d_model): {}", d_model);
    println!("  - Dimensione feed-forward (ff_dim): {}", ff_dim);
    println!("  - Numero di teste di attenzione: {}", num_heads);
    println!("  - Numero di layer transformer: {}", num_layers);
    println!("  - Lunghezza massima sequenza: {}", max_seq_len);
    println!("{}", "-".repeat(70));
    println!("{:<30} {:<25} {:<12}", "Layer", "Shape", "Params");
    println!("{}", "-".repeat(70));
    
    let mut total_params = 0;
    for param in params {
        println!("{}", param);
        total_params += param.param_count;
    }
    
    println!("{}", "=".repeat(70));
    println!("Parametri totali: {}", total_params);
    
    // Stampa dimensione approssimativa del modello
    let model_size_mb = (total_params * 4) as f64 / (1024.0 * 1024.0); // 4 bytes per float32
    println!("Dimensione approssimativa del modello: {:.2} MB", model_size_mb);
    println!("{}", "=".repeat(70));
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Addestramento modello QA Wall-E1");
    println!("--------------------------------");
    println!("Utilizzo {} thread per il calcolo parallelo", rayon::current_num_threads());
    
    // Carica il dataset QA
    println!("Caricamento del dataset da 'data/qa_en_dataset.json'...");
    let dataset_path = "data/qa_en_dataset.json";
    
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
    let max_seq_len = 256; // Valore di default usato nel costruttore del Trainer
    let gradient_clip = model_params["gradient_clip"].as_f64().map(|c| c as f32); // Nuovo parametro
    
    println!("Parametri del modello:");
    println!("  - Dimensione del modello (d_model): {} - Dimensione degli embedding e stati nascosti", d_model);
    println!("  - Dimensione feed-forward: {}", ff_dim);
    println!("  - Numero di teste di attenzione: {}", num_heads);
    println!("  - Numero di layer: {}", num_layers);
    println!("  - Dropout rate: {}", dropout_rate);
    println!("  - Learning rate: {}", learning_rate);
    if let Some(clip) = gradient_clip {
        println!("  - Gradient clipping: {}", clip);
    } else {
        println!("  - Gradient clipping: disabilitato");
    }
    
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
    
    // Stampa il riassunto del modello
    let params = calculate_model_parameters(
        vocab_size,
        d_model,
        ff_dim,
        num_heads,
        num_layers,
        max_seq_len
    );
    
    print_model_summary(
        &params,
        d_model,
        ff_dim,
        num_heads,
        num_layers,
        vocab_size,
        max_seq_len
    );
    
    // Crea il trainer con i parametri estratti
    println!("\nInizializzazione del modello...");
    let mut trainer = {
        // Creiamo prima un trainer base
        let mut t = Trainer::new(
            Box::new(tokenizer.clone()),
            d_model,
            ff_dim,
            num_heads,
            num_layers,
            dropout_rate,
            learning_rate
        );
        
        // Modifichiamo l'ottimizzatore per supportare il gradient clipping se necessario
        if let Some(clip_threshold) = gradient_clip {
            println!("Configurazione del gradient clipping con soglia {}", clip_threshold);
            t.with_gradient_clipping(Some(clip_threshold));
        }
        
        t
    };
    
    // Prepara gli esempi di addestramento
    println!("Preparazione degli esempi di addestramento...");
    
    // Progress bar per la preparazione degli esempi
    let progress_bar = ProgressBar::new(training_texts.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    progress_bar.set_message("Preparazione esempi");
    
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
        progress_bar.inc(1);
    });
    
    progress_bar.finish_with_message("Esempi preparati");
    
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
    let batch_size = 128; // Ridotto a 1 per evitare problemi di compatibilità di forma
    
    // Crea range di indici
    let indices: Vec<usize> = (0..training_examples.len()).step_by(batch_size).collect();
    
    // Progress bar per la preparazione dei batch
    let batch_progress = ProgressBar::new(indices.len() as u64);
    batch_progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    batch_progress.set_message("Preparazione batch");
    
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
        batch_progress.inc(1);
    });
    
    batch_progress.finish_with_message("Batch preparati");
    
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
    let epochs = 100; // Numero di epoche
    
    // Tempo di inizio dell'addestramento
    let training_start_time = Instant::now();
    
    for epoch in 1..=epochs {
        // Tempo di inizio dell'epoca
        let epoch_start_time = Instant::now();
        
        // Progress bar per l'epoca corrente
        let epoch_progress = ProgressBar::new(batched_examples.len() as u64);
        epoch_progress.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Batch: {msg}")
                .unwrap()
                .progress_chars("##-")
        );
        epoch_progress.set_message(format!("Epoca {}/{}", epoch, epochs));
        
        // Utilizziamo Mutex per aggiornare in modo sicuro i contatori
        let total_loss_mutex = Arc::new(Mutex::new(0.0));
        let num_batches_mutex = Arc::new(Mutex::new(0));
        
        // Per il calcolo dell'accuratezza
        let correct_predictions_mutex = Arc::new(Mutex::new(0));
        let total_predictions_mutex = Arc::new(Mutex::new(0));
        
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
            
            // Calcola l'accuratezza per questo batch
            let batch_output = trainer.forward(batch, Some(&targets_array));
            let batch_size = batch.len();
            let seq_len = batch[0].len();
            
            let mut batch_correct = 0;
            let mut batch_total = 0;
            
            for i in 0..batch_size {
                for j in 0..seq_len {
                    if j < targets_array.shape()[1] {
                        let target_id = targets_array[[i, j]];
                        if target_id != 0 { // Ignora i token di padding
                            batch_total += 1;
                            
                            // Trova il token con maggiore probabilità dall'output
                            let mut max_token = 0;
                            let mut max_prob = f32::NEG_INFINITY;
                            
                            for k in 0..trainer.get_vocab_size() {
                                let prob = batch_output.logits.data[[i, j, k]];
                                if prob > max_prob {
                                    max_prob = prob;
                                    max_token = k;
                                }
                            }
                            
                            // Controlla se la predizione è corretta
                            if max_token == target_id {
                                batch_correct += 1;
                            }
                        }
                    }
                }
            }
            
            // Aggiorna le metriche in modo thread-safe
            if let Ok(mut total) = total_loss_mutex.lock() {
                *total += loss;
            }
            
            if let Ok(mut count) = num_batches_mutex.lock() {
                *count += 1;
            }
            
            if let Ok(mut correct) = correct_predictions_mutex.lock() {
                *correct += batch_correct;
            }
            
            if let Ok(mut total) = total_predictions_mutex.lock() {
                *total += batch_total;
            }
            
            // Aggiorna la progress bar
            epoch_progress.set_position(batch_idx as u64 + 1);
            epoch_progress.set_message(format!("Epoca {}/{} - Loss: {:.4}", epoch, epochs, loss));
        }
        
        // Tempo di fine dell'epoca
        let epoch_duration = epoch_start_time.elapsed();
        
        // Estrai i valori finali dai mutex
        let total_loss = *total_loss_mutex.lock().unwrap();
        let num_batches = *num_batches_mutex.lock().unwrap();
        let correct_predictions = *correct_predictions_mutex.lock().unwrap();
        let total_predictions = *total_predictions_mutex.lock().unwrap();
        
        let avg_loss = total_loss / num_batches as f32;
        let accuracy = if total_predictions > 0 {
            (correct_predictions as f32 / total_predictions as f32) * 100.0
        } else {
            0.0
        };
        
        epoch_progress.finish();
        println!("Epoca {} completata in {:?}.", epoch, epoch_duration);
        println!("  Loss media: {:.4}", avg_loss);
        println!("  Accuratezza: {:.2}% ({}/{})", accuracy, correct_predictions, total_predictions);
    }
    
    // Tempo totale di addestramento
    let total_training_time = training_start_time.elapsed();
    println!("\nAddestramento completato in {:?}.", total_training_time);
    
    // Salva il modello addestrato
    let model_path = "models/qa_en_model.bin";
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
    
    // Progress bar per il test
    let test_progress = ProgressBar::new(test_prompts.len() as u64);
    test_progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    test_progress.set_message("Testando domande");
    
    // Test sincronizzato per i prompts
    for (i, prompt) in test_prompts.iter().enumerate() {
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
        
        test_progress.set_position(i as u64 + 1);
    }
    
    test_progress.finish_with_message("Test completato");
    
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
    println!("Tempo totale: {:?}", total_training_time);
    Ok(())
} 