use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
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
use wall_e1::tokenizer::BPETokenizer;
use num_cpus;
use wall_e1::nabla;
use std::env;
use rand::prelude::*;
use rand::seq::SliceRandom;

// Struct to hold command line arguments for faster testing
struct TrainingArgs {
    fast_mode: bool,         // Run a quick test with reduced data
    max_stories: usize,      // Maximum number of stories to use
    epochs: usize,           // Number of epochs to train for
    sample_every: usize,     // Generate sample text every N epochs
    checkpoint_every: usize, // Save checkpoint every N epochs
    vocab_size: usize,       // Size of the vocabulary
    dropout: f32,            // Dropout rate
    temperature: f32,        // Temperature for text generation
}

impl Default for TrainingArgs {
    fn default() -> Self {
        Self {
            fast_mode: false,
            max_stories: 10000,
            epochs: 5,
            sample_every: 1,
            checkpoint_every: 1,
            vocab_size: 5000,
            dropout: 0.15,
            temperature: 0.8,
        }
    }
}

// Function to parse command line arguments
fn parse_args() -> TrainingArgs {
    let args: Vec<String> = env::args().collect();
    let mut training_args = TrainingArgs::default();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fast" => {
                training_args.fast_mode = true;
                training_args.max_stories = 1000;
                training_args.epochs = 2;
                println!("Fast mode enabled: using 1000 stories and 2 epochs");
            },
            "--stories" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        training_args.max_stories = val;
                        i += 1; // Skip the next argument since we used it
                    }
                }
            },
            "--epochs" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        training_args.epochs = val;
                        i += 1;
                    }
                }
            },
            "--sample-every" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        training_args.sample_every = val;
                        i += 1;
                    }
                }
            },
            "--checkpoint-every" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        training_args.checkpoint_every = val;
                        i += 1;
                    }
                }
            },
            "--vocab-size" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse() {
                        training_args.vocab_size = val;
                        i += 1;
                    }
                }
            },
            "--dropout" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        training_args.dropout = val;
                        i += 1;
                    }
                }
            },
            "--temperature" => {
                if i + 1 < args.len() {
                    if let Ok(val) = args[i + 1].parse::<f32>() {
                        training_args.temperature = val;
                        i += 1;
                    }
                }
            },
            _ => {
                // Ignore unknown arguments
            }
        }
        i += 1;
    }
    
    training_args
}

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
    println!("                     MODELLO TRANSFORMER STORY WALL-E1");
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

// Function to generate text with temperature sampling and improved word separation
fn generate_text(trainer: &Trainer, tokenizer: &BPETokenizer, prompt: &str, max_tokens: usize, temperature: f32) -> String {
    let mut tokens = tokenizer.encode(prompt);
    let mut generated_text = prompt.to_string();
    let mut rng = rand::thread_rng();
    
    for _ in 0..max_tokens {
        // Prepare input
        let input = vec![tokens.clone()];
        
        // Forward pass
        let output = trainer.forward(&input, None);
        
        // Take the logits for the last token
        let logits = output.logits.data.slice(s![0, tokens.len() - 1, ..]);
        
        // Apply temperature to soften the distribution
        let vocab_size = trainer.get_vocab_size();
        let mut softmax_probs = vec![0.0; vocab_size];
        let mut max_logit = f32::NEG_INFINITY;
        
        // Find the max logit for numerical stability
        for i in 0..vocab_size {
            if logits[i] > max_logit {
                max_logit = logits[i];
            }
        }
        
        // Compute softmax with temperature
        let mut sum = 0.0;
        for i in 0..vocab_size {
            softmax_probs[i] = ((logits[i] - max_logit) / temperature).exp();
            sum += softmax_probs[i];
        }
        
        // Normalize
        for i in 0..vocab_size {
            softmax_probs[i] /= sum;
        }
        
        // Sample from the distribution
        let mut cumulative = 0.0;
        let sample = rng.gen_range(0.0..1.0);
        let mut next_token = 0;
        
        for i in 0..vocab_size {
            cumulative += softmax_probs[i];
            if sample < cumulative {
                next_token = i;
                break;
            }
        }
        
        // Add the sampled token
        tokens.push(next_token);
        let token_text = tokenizer.decode(&[next_token]);
        generated_text.push_str(&token_text);
        
        // Stop on period, newline, or question mark after reaching at least 15 tokens
        if tokens.len() > 15 && (token_text.contains(".") || token_text.contains("\n") || token_text.contains("?")) {
            break;
        }
    }
    
    // Post-process the generated text to improve spacing
    post_process_text(&generated_text)
}

// Function to improve spacing and text formatting
fn post_process_text(text: &str) -> String {
    // Step 1: Normalize spaces by removing extra spaces
    let mut normalized = String::new();
    let mut last_was_space = false;
    
    for c in text.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                normalized.push(' ');
                last_was_space = true;
            }
        } else {
            normalized.push(c);
            last_was_space = false;
        }
    }
    
    // Step 2: Add spaces between words if they're missing
    let mut with_spaces = String::new();
    let mut last_char_type = CharType::Other;
    
    for c in normalized.chars() {
        let current_type = if c.is_alphabetic() {
            CharType::Letter
        } else if c.is_numeric() {
            CharType::Number
        } else if c == ' ' {
            CharType::Space
        } else if ".,:;!?\"'()[]{}".contains(c) {
            CharType::Punctuation
        } else {
            CharType::Other
        };
        
        // Add space when transitioning from letter to letter case boundaries (camelCase -> camel Case)
        let should_add_space = match (last_char_type, current_type) {
            (CharType::Letter, CharType::Letter) => {
                let last_char = with_spaces.chars().last().unwrap_or(' ');
                // If transitioning from lowercase to uppercase, add a space
                last_char.is_lowercase() && c.is_uppercase()
            },
            (CharType::Letter, CharType::Number) => true,
            (CharType::Number, CharType::Letter) => true,
            (CharType::Letter, CharType::Punctuation) => false, // No space before punctuation
            (CharType::Punctuation, CharType::Letter) => true,  // Space after punctuation
            (CharType::Punctuation, CharType::Number) => true,  // Space after punctuation
            _ => false,
        };
        
        if should_add_space && !with_spaces.ends_with(' ') {
            with_spaces.push(' ');
        }
        
        with_spaces.push(c);
        last_char_type = current_type;
    }
    
    // Step 3: Fix common spacing issues around punctuation
    let fixed = with_spaces
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" :", ":")
        .replace(" ;", ";")
        .replace("( ", "(")
        .replace(" )", ")")
        .replace("\" ", "\"")
        .replace(" \"", "\"")
        .replace(" ' s ", "'s ")
        .replace(" n't ", "n't ")
        .replace(" 's ", "'s ");
    
    // Step 4: Ensure first character is capitalized
    if let Some(first_char) = fixed.chars().next() {
        if first_char.is_alphabetic() && first_char.is_lowercase() {
            return first_char.to_uppercase().to_string() + &fixed[1..];
        }
    }
    
    fixed
}

// Helper enum for character type classification
#[derive(Debug, PartialEq, Copy, Clone)]
enum CharType {
    Letter,
    Number,
    Space,
    Punctuation,
    Other,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args = parse_args();
    
    // Set number of threads based on CPU
    let num_cores = num_cpus::get();
    let threads_to_use = std::cmp::min(num_cores - 1, 12); // Leave 1 core free, max 12 threads
    nabla::tensor::set_num_threads(threads_to_use);
    
    println!("Addestramento modello Story Wall-E1");
    println!("----------------------------------");
    println!("Utilizzo {} thread per il calcolo parallelo", rayon::current_num_threads());
    
    // Carica il dataset TinyStories
    println!("Caricamento del dataset da 'data/tiny_stories_sample_updated.json'...");
    let dataset_path = "data/tiny_stories_sample_updated.json";
    
    if !Path::new(dataset_path).exists() {
        eprintln!("Errore: Il dataset '{}' non esiste!", dataset_path);
        println!("Esegui prima update_tiny_stories.py per creare il dataset aggiornato.");
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
    let d_model = model_params["d_model"].as_u64().unwrap_or(128) as usize;
    let ff_dim = model_params["ff_dim"].as_u64().unwrap_or(256) as usize;
    let num_heads = model_params["num_heads"].as_u64().unwrap_or(8) as usize;
    let num_layers = model_params["num_layers"].as_u64().unwrap_or(6) as usize;
    // Use dropout from command line arguments
    let dropout_rate = args.dropout;
    let learning_rate = model_params["learning_rate"].as_f64().unwrap_or(0.0015) as f32;
    let max_seq_len = model_params["max_seq_len"].as_u64().unwrap_or(128) as usize;
    let gradient_clip = model_params["gradient_clip"].as_f64().map(|c| c as f32);
    
    println!("Parametri del modello:");
    println!("  - Dimensione del modello (d_model): {} - Dimensione degli embedding e stati nascosti", d_model);
    println!("  - Dimensione feed-forward: {}", ff_dim);
    println!("  - Numero di teste di attenzione: {}", num_heads);
    println!("  - Numero di layer: {}", num_layers);
    println!("  - Dropout rate: {}", dropout_rate);
    println!("  - Learning rate: {}", learning_rate);
    println!("  - Lunghezza massima sequenza: {}", max_seq_len);
    if let Some(clip) = gradient_clip {
        println!("  - Gradient clipping: {}", clip);
    } else {
        println!("  - Gradient clipping: disabilitato");
    }
    
    // Estrai le storie dal dataset
    let stories = &data["stories"];
    
    // Usiamo Rayon per elaborare le storie in parallelo
    println!("Preparazione dei testi di addestramento...");
    let training_texts: Vec<String> = {
        let stories_array = stories.as_array().unwrap();
        let progress_bar = ProgressBar::new(stories_array.len() as u64);
        progress_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-")
        );
        progress_bar.set_message("Caricamento storie");
        
        // Take specified number of stories, shuffled for better training diversity
        let mut indices: Vec<usize> = (0..stories_array.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        let max_stories = if args.fast_mode {
            println!("Modalità fast mode: limitando a {} storie", args.max_stories);
            std::cmp::min(stories_array.len(), args.max_stories)
        } else {
            std::cmp::min(stories_array.len(), args.max_stories)
        };
        
        let selected_stories: Vec<String> = indices.into_iter()
            .take(max_stories)
            .map(|idx| {
                let story_text = stories_array[idx].as_str().unwrap().to_string();
                progress_bar.inc(1);
                story_text
            })
            .collect();
            
        progress_bar.finish_with_message("Storie caricate");
        selected_stories
    };
    
    println!("Caricate {} storie per l'addestramento", training_texts.len());
    
    // Prepara un corpus per costruire il vocabolario
    let corpus = {
        // Improve spacing in the corpus by adding extra spaces around punctuation
        let preprocessed_texts: Vec<String> = training_texts.par_iter()
            .map(|text| {
                // Replace common punctuation with space + punctuation + space
                let text = text.replace(".", " . ")
                               .replace(",", " , ")
                               .replace("!", " ! ")
                               .replace("?", " ? ")
                               .replace("\"", " \" ")
                               .replace("'", " ' ")
                               .replace("(", " ( ")
                               .replace(")", " ) ");
                               
                // Normalize spaces (replace multiple spaces with a single space)
                let mut normalized = String::new();
                let mut last_was_space = false;
                
                for c in text.chars() {
                    if c.is_whitespace() {
                        if !last_was_space {
                            normalized.push(' ');
                            last_was_space = true;
                        }
                    } else {
                        normalized.push(c);
                        last_was_space = false;
                    }
                }
                
                normalized
            })
            .collect();
            
        // Uniamo tutti i testi preprocessati in parallelo
        let corpus_chunks: Vec<String> = preprocessed_texts.par_iter()
            .map(|text| text.clone() + " ")
            .collect();
            
        // Unisci i chunk in una stringa unica
        corpus_chunks.join("")
    };
    
    println!("Corpus preparato con spacing migliorato per tokenization");
    
    // Crea e inizializza il tokenizer
    let mut tokenizer = BPETokenizer::new();
    
    // Assicurati che lo spazio sia un token speciale per migliorare la separazione delle parole
    tokenizer.learn_bpe(&corpus, args.vocab_size, 2); // Use vocab size from arguments
    
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
    training_texts.par_iter().for_each(|story| {
        let tokens = tokenizer.encode(story);
        if tokens.len() > 3 {  // Assicuriamoci che ci siano abbastanza token
            let mut local_examples = Vec::new();
            let mut local_targets = Vec::new();
            
            // Per le storie, creiamo finestre scorrevoli per l'addestramento
            let window_size = std::cmp::min(max_seq_len, 64); // Usiamo finestre di 64 token o meno
            
            for i in 0..(tokens.len() - 1) {
                if i % 16 == 0 {  // Prendiamo un esempio ogni 16 token per ridurre la ridondanza
                    let mut input = Vec::new();
                    let mut target = Vec::new();
                    
                    // Prendiamo una finestra di token come contesto
                    let end = std::cmp::min(i + window_size, tokens.len());
                    for j in i..end {
                        if j < end - 1 {
                            input.push(tokens[j]);
                            target.push(tokens[j + 1]);
                        }
                    }
                    
                    if !input.is_empty() && !target.is_empty() {
                        local_examples.push(input);
                        local_targets.push(target);
                    }
                }
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
    let batch_size = 32; // Batch size ridotto per dataset più grande
    
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
    
    // Parametri di training
    let epochs = args.epochs;
    let warmup_epochs = model_params["warmup_epochs"].as_u64().unwrap_or(1) as usize;
    let plateau_epochs = model_params["plateau_epochs"].as_u64().unwrap_or(3) as usize;
    
    println!("Parametri di addestramento:");
    println!("  - Epoche totali: {}", epochs);
    println!("  - Epoche di warmup: {}", warmup_epochs);
    println!("  - Epoche di plateau: {}", plateau_epochs);
    
    // Tempo di inizio dell'addestramento
    let training_start_time = Instant::now();
    
    // Rimuoviamo lo scheduler del learning rate poiché Trainer non supporta set_learning_rate
    // e useremo il learning rate fisso impostato durante la creazione del trainer
    println!("Usando learning rate fisso: {}", learning_rate);
    println!("Temperatura per la generazione: {}", args.temperature);
    
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
        
        // In fast mode, process only a fraction of the batches
        let batches_to_process = if args.fast_mode {
            std::cmp::min(batched_examples.len(), 100) // Process only 100 batches in fast mode
        } else {
            batched_examples.len()
        };
        
        // Trainer must be mutated sequentially
        for (batch_idx, (batch, targets)) in batched_examples.iter()
                                                        .zip(batched_targets.iter())
                                                        .enumerate()
                                                        .take(batches_to_process) {
            // Converti i target in Array2
            let max_seq_len = targets.iter().map(|t| t.len()).max().unwrap_or(1);
            let mut targets_array = Array2::zeros((targets.len(), max_seq_len));
            
            // Impostiamo i valori di targets_array
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
            let seq_len = targets_array.shape()[1];
            
            let mut batch_correct = 0;
            let mut batch_total = 0;
            
            for i in 0..batch_size {
                for j in 0..seq_len {
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
            
            // In fast mode, show text samples every 20 batches
            if args.fast_mode && batch_idx > 0 && batch_idx % 20 == 0 {
                let test_prompt = "Once upon a time,";
                let generated = generate_text(&trainer, &tokenizer, test_prompt, 20, args.temperature);
                println!("\nRapido esempio (batch {}): \"{}{}\"", 
                        batch_idx, test_prompt, generated.strip_prefix(test_prompt).unwrap_or(&generated));
            }
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
        
        // Generate text samples at the end of each epoch or more frequently in fast mode
        if epoch % args.sample_every == 0 || args.fast_mode || epoch == epochs {
            println!("\nGenerazione di esempi di testo (Epoca {}):", epoch);
            
            let test_prompts = [
                "Once upon a time,",
                "The little dog",
                "In the garden,",
                "Today I will",
            ];
            
            for prompt in test_prompts.iter() {
                let generated = generate_text(&trainer, &tokenizer, prompt, 30, args.temperature);
                println!("\nPrompt: \"{}\"", prompt);
                println!("Generato: \"{}\"", generated);
            }
        }
        
        // Save checkpoint if needed
        if epoch % args.checkpoint_every == 0 || epoch == epochs {
            let checkpoint_path = format!("models/story_model_epoch_{}.bin", epoch);
            println!("\nSalvataggio checkpoint in '{}'...", checkpoint_path);
            trainer.save_model(&checkpoint_path)?;
        }
    }
    
    // Tempo totale di addestramento
    let total_training_time = training_start_time.elapsed();
    println!("\nAddestramento completato in {:?}.", total_training_time);
    
    // Salva il modello addestrato finale
    let model_path = "models/story_model.bin";
    println!("\nSalvataggio del modello in '{}'...", model_path);
    trainer.save_model(model_path)?;
    
    println!("\nAddestramento e test completati!");
    println!("Modello salvato in: {}", model_path);
    println!("Tempo totale: {:?}", total_training_time);
    Ok(())
} 