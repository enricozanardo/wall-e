use ndarray::{Array, Array2, Axis, Dimension};
use rand::{thread_rng, Rng};
use crate::nabla::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use crate::training::ModelOutput;

/// Valuta il modello su un set di dati di test
pub fn evaluate<M, F>(
    model: &M,
    forward_fn: F,
    test_data: &[Vec<Vec<usize>>],
    _tokenizer: &Box<dyn Tokenizer>,
    padding_idx: Option<usize>,
) -> (f32, f32)
where
    M: Sized,
    F: Fn(&M, &Vec<Vec<usize>>, Option<Array2<f32>>) -> ModelOutput,
{
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_tokens = 0;
    
    for batch in test_data {
        // Prepara i target: ogni token predice il successivo
        let targets = prepare_targets(batch);
        
        // Forward pass
        let output = forward_fn(model, batch, None);
        
        // Accumulazione della loss
        total_loss += output.loss.unwrap_or(0.0);
        
        // Calcolo dell'accuratezza
        let (correct, tokens) = count_correct_predictions(
            &output.logits,
            &targets,
            padding_idx,
        );
        
        total_correct += correct;
        total_tokens += tokens;
    }
    
    let avg_loss = total_loss / test_data.len() as f32;
    let accuracy = total_correct as f32 / total_tokens as f32;
    
    (avg_loss, accuracy)
}

/// Prepara i target per la valutazione
fn prepare_targets(batch: &Vec<Vec<usize>>) -> Array2<usize> {
    let batch_size = batch.len();
    let max_seq_len = batch.iter().map(|seq| seq.len()).max().unwrap_or(0);
    
    let mut targets = Array2::<usize>::zeros((batch_size, max_seq_len));
    
    for (i, seq) in batch.iter().enumerate() {
        for (j, &token_id) in seq.iter().enumerate() {
            if j < max_seq_len {
                targets[[i, j]] = token_id;
            }
        }
    }
    
    targets
}

/// Conta le predizioni corrette nel batch
fn count_correct_predictions(
    logits: &Tensor,
    targets: &Array2<usize>,
    padding_idx: Option<usize>,
) -> (usize, usize) {
    // Ottieni le dimensioni
    let shape = logits.data.shape();
    if shape.len() != 3 {
        return (0, 0); // Forma non valida
    }
    
    let batch_size = shape[0];
    let seq_len = shape[1];
    
    let mut correct = 0;
    let mut total = 0;
    
    for i in 0..batch_size {
        for j in 0..seq_len {
            let target_idx = targets[[i, j]];
            
            // Ignora i token di padding
            if let Some(pad_idx) = padding_idx {
                if target_idx == pad_idx {
                    continue;
                }
            }
            
            // Trova l'indice del valore massimo nelle logits per questa posizione
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            
            for v in 0..shape[2] {
                if let Some(&val) = logits.data.get([i, j, v]) {
                    if val > max_val {
                        max_val = val;
                        max_idx = v;
                    }
                }
            }
            
            // Controlla se la predizione è corretta
            if max_idx == target_idx {
                correct += 1;
            }
            
            total += 1;
        }
    }
    
    (correct, total)
}

/// Stampa esempi di testo generato
pub fn print_generated_samples<M, F>(
    model: &M,
    forward_fn: F,
    prompts: &[&str],
    tokenizer: &Box<dyn Tokenizer>,
    max_tokens: usize,
    temperature: f32,
) where
    M: Sized,
    F: Fn(&M, &Vec<Vec<usize>>, Option<Array2<f32>>) -> ModelOutput + Copy,
{
    for prompt in prompts {
        println!("Prompt: {}", prompt);
        
        // Tokenizza il prompt
        let tokens = tokenizer.encode(prompt);
        let token_batch = vec![tokens.clone()];
        
        // Genera testo
        let generated = generate_text(
            model,
            forward_fn,
            token_batch,
            tokenizer,
            max_tokens,
            temperature,
        );
        
        // Decodifica e stampa il risultato
        let generated_text = tokenizer.decode(&generated[0]);
        println!("Generato: {}\n", generated_text);
    }
}

/// Genera testo a partire da un input
pub fn generate_text<M, F>(
    model: &M,
    forward_fn: F,
    mut token_batch: Vec<Vec<usize>>,
    _tokenizer: &Box<dyn Tokenizer>,
    max_tokens: usize,
    temperature: f32,
) -> Vec<Vec<usize>>
where
    M: Sized,
    F: Fn(&M, &Vec<Vec<usize>>, Option<Array2<f32>>) -> ModelOutput,
{
    let mut result = token_batch.clone();
    
    for _ in 0..max_tokens {
        // Forward pass per ottenere le probabilità del prossimo token
        let output = forward_fn(model, &token_batch, None);
        
        // Per ogni sequenza nel batch
        for (i, seq) in token_batch.iter_mut().enumerate() {
            // Ottieni le probabilità dell'ultimo token
            let last_pos = seq.len() - 1;
            
            // Estrai logits dell'ultimo token
            let shape = output.logits.data.shape();
            let vocab_size = shape[2];
            
            let mut last_token_logits = Vec::with_capacity(vocab_size);
            for v in 0..vocab_size {
                if let Some(&val) = output.logits.data.get([i, last_pos, v]) {
                    last_token_logits.push(val);
                } else {
                    last_token_logits.push(0.0);
                }
            }
            
            // Applica la temperatura
            let scaled_logits = if temperature > 0.0 {
                last_token_logits.iter().map(|&x| x / temperature).collect::<Vec<f32>>()
            } else {
                last_token_logits
            };
            
            // Converti in probabilità usando softmax
            let probs = softmax(&scaled_logits);
            
            // Campiona il prossimo token (implementazione semplice)
            let next_token = if temperature > 0.0 {
                // Campionamento proporzionale alle probabilità
                sample_from_probs(&probs)
            } else {
                // Greedy sampling (prendi il token con la probabilità più alta)
                probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            };
            
            // Aggiungi il token alla sequenza
            seq.push(next_token);
            result[i].push(next_token);
        }
    }
    
    result
}

/// Implementazione semplice del softmax per array 1D
fn softmax(logits: &[f32]) -> Vec<f32> {
    // Per stabilità numerica, sottrai il massimo
    let max_logit = logits.iter().cloned().fold(f32::MIN, f32::max);
    
    // Calcola l'exp e la somma
    let exp_logits: Vec<f32> = logits.iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    
    let sum: f32 = exp_logits.iter().sum();
    
    // Normalizza
    exp_logits.iter().map(|&x| x / sum).collect()
}

/// Campionamento da un vettore di probabilità
fn sample_from_probs(probs: &[f32]) -> usize {
    // Genera un numero casuale tra 0 e 1
    let r: f32 = rand::random();
    
    // Seleziona l'indice basato sulla distribuzione cumulativa
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    
    // In caso di errori di arrotondamento, restituisce l'ultimo indice
    probs.len() - 1
} 