use std::error::Error;
use std::path::Path;
use wall_e1::tokenizer::basic_tokenizer::BasicTokenizer;
use wall_e1::tokenizer::Tokenizer;
use wall_e1::training::Trainer;
use ndarray::{Array2, s};
use rand::{Rng, thread_rng};

// Funzione per generare testo dato un prompt
fn generate_text(trainer: &Trainer, tokenizer: &dyn Tokenizer, prompt: &str, max_length: usize) -> String {
    println!("Generazione di testo a partire dal prompt: \"{}\"", prompt);
    
    // Tokenizza il prompt
    let token_ids = tokenizer.encode(prompt);
    let mut token_ids = token_ids;  // Per poter modificare la lista
    
    println!("Token IDs iniziali: {:?}", token_ids);
    
    // Genera token aggiuntivi fino a raggiungere max_length
    for _ in 0..max_length {
        // Prepara l'input (batch di un solo elemento)
        let input = vec![token_ids.clone()];
        
        // Esegui il forward pass
        let output = trainer.forward(&input, None);
        
        // Ottieni i logits per l'ultimo token
        if token_ids.is_empty() {
            continue;
        }
        
        let last_token_idx = token_ids.len() - 1;
        let vocab_size = trainer.get_vocab_size();
        
        // Estrai i logits per l'ultimo token usando ndarray slice
        // Le dimensioni sono [batch, seq_len, vocab_size]
        let logits_slice = output.logits.data.slice(s![0, last_token_idx, ..]);
        let mut logits = Vec::with_capacity(vocab_size);
        
        for i in 0..vocab_size {
            logits.push(logits_slice[i]);
        }
        
        // Converti logits in probabilità usando softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |max, &val| f32::max(max, val));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();
        
        // Selezione del token successivo (temperature sampling)
        let mut rng = thread_rng();
        let next_token = sample_from_probs(&probs, &mut rng);
        
        println!("Token selezionato: {} (\"{}\")", next_token, 
            tokenizer.decode(&[next_token]));
        
        // Aggiungi il token alla sequenza
        token_ids.push(next_token);
        
        // Interrompi se generiamo un token di fine frase o [PAD]
        if next_token == 0 {
            break;
        }
    }
    
    // Decodifica la sequenza completa
    tokenizer.decode(&token_ids)
}

// Funzione per campionare un token dalle probabilità
fn sample_from_probs(probs: &[f32], rng: &mut impl Rng) -> usize {
    // Implementazione base: temperatura = 1.0
    let random_value = rng.gen_range(0.0..1.0);
    let mut cum_prob = 0.0;
    
    for (i, &p) in probs.iter().enumerate() {
        cum_prob += p;
        if random_value < cum_prob {
            return i;
        }
    }
    
    // Fallback al token più probabile
    probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Test di utilizzo del modello Wall-E1");
    println!("------------------------------------");
    
    // Percorso del modello
    let model_path = "models/qa_en_model.bin";
    
    // Verifica che il modello esista
    if !Path::new(model_path).exists() {
        eprintln!("Errore: Il modello '{}' non esiste!", model_path);
        return Err("Modello non trovato".into());
    }
    
    println!("Inizializzazione del tokenizer...");
    
    // Crea un tokenizer vuoto - il vocabolario verrà caricato dal modello
    let mut tokenizer = BasicTokenizer::new();
    
    println!("Caricamento del modello '{}'...", model_path);
    
    // Carica il modello - questo caricherà anche il vocabolario nel tokenizer
    let trainer = match Trainer::load_model(model_path, &mut tokenizer, Some(0.0005)) {
        Ok(trainer) => {
            println!("Modello caricato con successo!");
            println!("Dimensione del vocabolario: {}", trainer.get_vocab_size());
            println!("Lunghezza massima sequenza: {}", trainer.get_max_seq_len());
            trainer
        },
        Err(e) => {
            eprintln!("Errore durante il caricamento del modello: {}", e);
            return Err(Box::new(e));
        }
    };
    
    println!("\nTest di generazione di testo:");
    println!("-----------------------------");
    
    // Prompt di esempio
    let prompt = "what is a Software Defined Network?";
    
    // Genera testo (max 30 token aggiuntivi)
    let generated_text = generate_text(&trainer, &tokenizer, prompt, 30);
    
    println!("\nTesto generato:");
    println!("{}", generated_text);
    
    println!("\nTest completato!");
    Ok(())
} 