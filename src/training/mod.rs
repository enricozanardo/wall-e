use std::time::{Instant, Duration};
use std::collections::HashMap;
use ndarray::{Array, Array1, Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use rand_distr::Distribution;
use rayon::prelude::*;
use crate::nabla::tensor::Tensor;
use crate::tokenizer::{Tokenizer, Vocab};
use crate::embedding::TransformerEmbedding;
use crate::attention::{EncoderStack, create_causal_mask};

// Includi i moduli esterni
#[cfg(test)]
pub mod tests;
pub mod evaluate;

/// Struttura per i risultati del modello
#[derive(Clone)]
pub struct ModelOutput {
    /// Logits finali (probabilità non normalizzate)
    pub logits: Tensor,
    /// Loss calcolata (se disponibile)
    pub loss: Option<f32>,
}

/// Loss function di cross-entropy per task di linguaggio
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Crea una nuova istanza di CrossEntropyLoss
    pub fn new() -> Self {
        CrossEntropyLoss
    }
    
    /// Calcola la loss di cross-entropy tra logits e target
    /// 
    /// # Arguments
    /// * `logits` - Tensor di forma [batch_size, seq_len, vocab_size] con le previsioni
    /// * `targets` - Array di indici dei token target di forma [batch_size, seq_len]
    /// * `ignore_index` - Indice da ignorare (es. token di padding)
    /// 
    /// # Returns
    /// * Loss media e gradiente rispetto ai logits
    pub fn forward(&self, logits: &Tensor, targets: &Array2<usize>, ignore_index: Option<usize>) -> (f32, Tensor) {
        let batch_size = logits.data.shape()[0];
        let seq_len = logits.data.shape()[1];
        let vocab_size = logits.data.shape()[2];
        
        // Converti i logits in probabilità con softmax
        let logits_data = logits.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let mut loss_sum = 0.0;
        let mut valid_tokens = 0;
        
        // Inizializza il gradiente con zeri
        let mut grad_data = Array3::<f32>::zeros((batch_size, seq_len, vocab_size));
        
        // Calcola loss e gradiente
        for b in 0..batch_size {
            for s in 0..seq_len {
                let target_idx = targets[[b, s]];
                
                // Salta i token da ignorare (es. padding)
                if let Some(idx) = ignore_index {
                    if target_idx == idx {
                        continue;
                    }
                }
                
                // Estrai logits per questo token
                let token_logits = logits_data.slice(s![b, s, ..]).to_owned();
                
                // Trova il massimo per stabilità numerica
                let max_logit = token_logits.fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
                
                // Calcola softmax stabile
                let exp_logits: Array1<f32> = token_logits.mapv(|x| (x - max_logit).exp());
                let sum_exp = exp_logits.sum();
                let probs = exp_logits / sum_exp;
                
                // Calcola loss per questo token: -log(p_target)
                if target_idx < vocab_size {
                    loss_sum -= (probs[target_idx] + 1e-10).ln();
                    
                    // Calcola il gradiente: p_i - 1(i=target)
                    for v in 0..vocab_size {
                        let target_indicator = if v == target_idx { 1.0 } else { 0.0 };
                        grad_data[[b, s, v]] = probs[v] - target_indicator;
                    }
                    
                    valid_tokens += 1;
                }
            }
        }
        
        // Calcola la loss media per token
        let avg_loss = if valid_tokens > 0 {
            loss_sum / valid_tokens as f32
        } else {
            0.0
        };
        
        // Normalizza il gradiente per il numero di token validi
        if valid_tokens > 0 {
            grad_data.mapv_inplace(|x| x / valid_tokens as f32);
        }
        
        (avg_loss, Tensor::new_3d(grad_data))
    }
}

/// Ottimizzatore Adam per l'aggiornamento dei parametri
pub struct AdamOptimizer {
    /// Learning rate
    lr: f32,
    /// Parametro beta1 per il momento
    beta1: f32,
    /// Parametro beta2 per il momento secondo
    beta2: f32,
    /// Epsilon per stabilità numerica
    epsilon: f32,
    /// Momento primo
    m: HashMap<String, Tensor>,
    /// Momento secondo
    v: HashMap<String, Tensor>,
    /// Passo di training
    t: usize,
}

impl AdamOptimizer {
    /// Crea un nuovo ottimizzatore Adam
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        AdamOptimizer {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
    
    /// Aggiorna i parametri in base ai gradienti
    pub fn step(&mut self, params: &mut HashMap<String, Tensor>, grads: &HashMap<String, Tensor>) {
        self.t += 1;
        
        // Fattori di correzione
        let correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let correction2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr_t = self.lr * (correction2.sqrt() / correction1);
        
        // Aggiorna ogni parametro
        for (name, param) in params.iter_mut() {
            if let Some(grad) = grads.get(name) {
                // Ottieni la dimensionalità corretta
                let grad_data = grad.data.clone();
                
                // Aggiorna momento primo
                let m_t = if let Some(m) = self.m.get(name) {
                    &(&m.data * self.beta1) + &(&grad_data * (1.0 - self.beta1))
                } else {
                    &grad_data * (1.0 - self.beta1)
                };
                
                // Aggiorna momento secondo
                let v_t = if let Some(v) = self.v.get(name) {
                    &(&v.data * self.beta2) + &(&grad_data.mapv(|x| x * x) * (1.0 - self.beta2))
                } else {
                    &grad_data.mapv(|x| x * x) * (1.0 - self.beta2)
                };
                
                // Calcola l'aggiornamento
                let update = &m_t / &(v_t.mapv(|x| x.sqrt()) + self.epsilon) * lr_t;
                
                // Aggiorna il parametro
                param.data = &param.data - &update;
                
                // Salva i momenti
                let m_t_2d = m_t.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let v_t_2d = v_t.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                self.m.insert(name.clone(), Tensor::new(m_t_2d));
                self.v.insert(name.clone(), Tensor::new(v_t_2d));
            }
        }
    }
    
    /// Imposta il learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    /// Ottieni il learning rate corrente
    pub fn get_learning_rate(&self) -> f32 {
        self.lr
    }
}

/// Trainer per modelli transformer
pub struct Trainer {
    /// Tokenizer per il testo
    tokenizer: Box<dyn Tokenizer>,
    /// Embedding layer
    embedding: TransformerEmbedding,
    /// Encoder stack
    encoder: EncoderStack,
    /// Matrice di proiezione finale
    output_projection: Tensor,
    /// Dimensione del modello
    model_dim: usize,
    /// Dimensione del vocabolario
    vocab_size: usize,
    /// Funzione di loss
    loss_fn: CrossEntropyLoss,
    /// Ottimizzatore
    optimizer: AdamOptimizer,
    /// Parametri del modello
    params: HashMap<String, Tensor>,
}

impl Trainer {
    /// Crea un nuovo trainer
    pub fn new(
        tokenizer: Box<dyn Tokenizer>,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_rate: f32,
        learning_rate: f32,
    ) -> Self {
        // Accedi al vocabolario attraverso il trait
        let vocab_size = tokenizer.vocab_size();
        
        // Crea embedding layer
        let embedding = TransformerEmbedding::new(
            vocab_size,
            model_dim,
            1024, // max_seq_len
            dropout_rate,
        );
        
        // Crea encoder stack
        let encoder = EncoderStack::new(
            model_dim,
            ff_dim,
            num_heads,
            num_layers,
            dropout_rate,
        );
        
        // Crea la matrice di proiezione finale (per la predizione dei token)
        // Inizializza con una distribuzione normale con deviazione standard 0.02
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 0.02).unwrap();
        let random_matrix = Array::from_shape_fn((model_dim, vocab_size), |_| normal.sample(&mut rng));
        let output_projection = Tensor::new(random_matrix);
        
        // Funzione di loss
        let loss_fn = CrossEntropyLoss::new();
        
        // Ottimizzatore Adam
        let optimizer = AdamOptimizer::new(
            learning_rate,
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
        );
        
        // Parametri del modello (da implementare il recupero completo dei parametri)
        let mut params = HashMap::new();
        params.insert("output_projection".to_string(), output_projection.clone());
        
        Trainer {
            tokenizer: tokenizer,
            embedding,
            encoder,
            output_projection,
            model_dim,
            vocab_size,
            loss_fn,
            optimizer,
            params,
        }
    }
    
    /// Forward pass del modello
    pub fn forward(&self, token_ids: &[Vec<usize>], targets: Option<&Array2<usize>>) -> ModelOutput {
        let batch_size = token_ids.len();
        let seq_len = if !token_ids.is_empty() { token_ids[0].len() } else { 0 };
        
        // Ottieni gli embedding per il batch
        let embedded = self.embedding.forward_batch(token_ids);
        
        // Reshape degli embedding per l'encoder
        let embedded_data = embedded.data.clone()
            .into_shape((batch_size, seq_len, self.model_dim))
            .unwrap();
        let embedded_tensor = Tensor::new_3d(embedded_data.into_dimensionality::<ndarray::Ix3>().unwrap());
        
        // Crea una maschera causale se necessario
        let mask = create_causal_mask(seq_len);
        
        // Forward pass attraverso l'encoder
        let encoder_output = self.encoder.forward(&embedded_tensor, Some(&mask));
        
        // Proiezione finale per ottenere i logits
        let encoder_data = encoder_output.data.clone()
            .into_dimensionality::<ndarray::Ix3>().unwrap();
        
        // Calcola i logits [batch_size, seq_len, vocab_size]
        let mut logits_data = Array3::<f32>::zeros((batch_size, seq_len, self.vocab_size));
        
        // Calcola il prodotto matriciale per ottenere i logits
        for b in 0..batch_size {
            for s in 0..seq_len {
                for v in 0..self.vocab_size {
                    let mut sum = 0.0;
                    for d in 0..self.model_dim {
                        sum += encoder_data[[b, s, d]] * self.output_projection.data[[d, v]];
                    }
                    logits_data[[b, s, v]] = sum;
                }
            }
        }
        
        let logits = Tensor::new_3d(logits_data);
        
        // Calcola la loss se sono forniti i target
        let loss = if let Some(targets) = targets {
            let (loss_val, _) = self.loss_fn.forward(&logits, targets, None);
            Some(loss_val)
        } else {
            None
        };
        
        ModelOutput { logits, loss }
    }
    
    /// Addestra il modello su un batch di dati
    pub fn train_step(&mut self, token_ids: &[Vec<usize>], targets: &Array2<usize>) -> f32 {
        // Forward pass
        let output = self.forward(token_ids, Some(targets));
        
        // Calcola la loss e il gradiente
        let (loss, grad_logits) = self.loss_fn.forward(&output.logits, targets, None);
        
        // Converti il gradiente dei logits in gradiente della matrice di proiezione
        let batch_size = token_ids.len();
        let seq_len = if !token_ids.is_empty() { token_ids[0].len() } else { 0 };
        
        // Ottieni gli embedding per il batch
        let embedded = self.embedding.forward_batch(token_ids);
        
        // Reshape degli embedding per l'encoder
        let embedded_data = embedded.data.clone()
            .into_shape((batch_size, seq_len, self.model_dim))
            .unwrap();
        let embedded_tensor = Tensor::new_3d(embedded_data.into_dimensionality::<ndarray::Ix3>().unwrap());
        
        // Crea una maschera causale
        let mask = create_causal_mask(seq_len);
        
        // Forward pass attraverso l'encoder
        let encoder_output = self.encoder.forward(&embedded_tensor, Some(&mask));
        
        // Prepara il gradiente per la matrice di proiezione finale
        let grad_logits_data = grad_logits.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let encoder_data = encoder_output.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        
        // Inizializza gradiente per output_projection
        let mut grad_output_proj = Array::zeros((self.model_dim, self.vocab_size));
        
        // Calcola il gradiente rispetto alla matrice di proiezione
        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.model_dim {
                    for v in 0..self.vocab_size {
                        grad_output_proj[[d, v]] += encoder_data[[b, s, d]] * grad_logits_data[[b, s, v]];
                    }
                }
            }
        }
        
        // Crea un tensore dal gradiente
        let grad_output_proj_tensor = Tensor::new(grad_output_proj);
        
        // Aggiungi il gradiente all'hashtable
        let mut grads = HashMap::new();
        grads.insert("output_projection".to_string(), grad_output_proj_tensor);
        
        // Aggiorna i parametri
        self.optimizer.step(&mut self.params, &grads);
        
        // Aggiorna il parametro output_projection
        if let Some(param) = self.params.get("output_projection") {
            self.output_projection = param.clone();
        }
        
        loss
    }
    
    /// Addestra il modello per un numero di epoche
    pub fn train(&mut self, train_data: &[Vec<String>], num_epochs: usize, batch_size: usize) -> Vec<f32> {
        let mut losses = Vec::new();
        let num_samples = train_data.len();
        let num_batches = (num_samples + batch_size - 1) / batch_size;
        
        for epoch in 0..num_epochs {
            let start_time = Instant::now();
            let mut epoch_loss = 0.0;
            
            println!("Epoch {}/{}", epoch + 1, num_epochs);
            
            // Prepara i batch in parallelo
            let batches: Vec<_> = (0..num_batches)
                .map(|b| {
                    let start_idx = b * batch_size;
                    let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
                    let batch_data = &train_data[start_idx..end_idx];
                    
                    // Tokenizza le frasi
                    let batch_tokens: Vec<Vec<usize>> = batch_data
                        .iter()
                        .map(|sentence| {
                            // Converti il vettore di Stringhe in una singola stringa
                            let joined_sentence = sentence.join(" ");
                            self.tokenizer.encode(&joined_sentence)
                        })
                        .collect();
                    
                    // Determina la lunghezza massima della sequenza nel batch
                    let max_len = batch_tokens.iter().map(|seq| seq.len()).max().unwrap_or(0);
                    
                    // Prepara input e target
                    let mut input_ids = Vec::new();
                    let mut target_ids = Array2::<usize>::zeros((batch_tokens.len(), max_len));
                    
                    for (i, seq) in batch_tokens.iter().enumerate() {
                        // Input: tutti i token tranne l'ultimo
                        let input = if seq.len() > 1 {
                            seq[..seq.len() - 1].to_vec()
                        } else {
                            Vec::new()
                        };
                        
                        // Padding per raggiungere max_len - 1
                        let mut padded_input = input.clone();
                        padded_input.resize(max_len - 1, 0); // 0 come token di padding
                        input_ids.push(padded_input);
                        
                        // Target: tutti i token tranne il primo
                        for (j, &token) in seq[1..].iter().enumerate() {
                            target_ids[[i, j]] = token;
                        }
                    }
                    
                    (input_ids, target_ids)
                })
                .collect();
            
            // Addestra su ogni batch
            for (batch_idx, (batch_inputs, batch_targets)) in batches.into_iter().enumerate() {
                // Salta batch vuoti
                if batch_inputs.is_empty() {
                    continue;
                }
                
                // Addestra sul batch
                let batch_loss = self.train_step(&batch_inputs, &batch_targets);
                epoch_loss += batch_loss;
                
                // Stampa progresso
                if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == num_batches {
                    println!(
                        "Batch {}/{}, Loss: {:.4}, Time: {:?}",
                        batch_idx + 1,
                        num_batches,
                        batch_loss,
                        start_time.elapsed()
                    );
                }
            }
            
            // Loss media dell'epoca
            let avg_epoch_loss = epoch_loss / num_batches as f32;
            losses.push(avg_epoch_loss);
            
            println!(
                "Epoch {}/{} completed. Avg Loss: {:.4}, Time: {:?}",
                epoch + 1,
                num_epochs,
                avg_epoch_loss,
                start_time.elapsed()
            );
        }
        
        losses
    }
    
    /// Genera testo a partire da un prompt
    pub fn generate(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> String {
        // Tokenizza il prompt
        let mut tokens = self.tokenizer.encode(prompt);
        
        // Genera fino a max_length token
        for _ in 0..max_length {
            // Crea un batch con un singolo input
            let batch_input = vec![tokens.clone()];
            
            // Forward pass
            let output = self.forward(&batch_input, None);
            
            // Prendi i logits dell'ultimo token
            let logits_shape = output.logits.data.shape().to_vec();
            if logits_shape.len() != 3 {
                break; // Uscita di sicurezza se la forma non è corretta
            }
            
            // Accedi manualmente all'ultimo token per evitare problemi con i tipi
            let last_pos = tokens.len() - 1;
            let mut last_token_logits = Array1::<f32>::zeros(self.vocab_size);
            
            // Estrai manualmente i logits dell'ultimo token
            for v in 0..self.vocab_size {
                if let Some(val) = output.logits.data.get([0, last_pos, v]) {
                    last_token_logits[v] = *val;
                }
            }
            
            // Applica la temperatura per controllare la casualità
            let logits_temp = last_token_logits.mapv(|x| x / temperature);
            
            // Opzionalmente, filtra solo i top-k token
            let token_probs = if let Some(k) = top_k {
                // Trova i top-k indici e valori
                let mut pairs: Vec<_> = logits_temp
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (i, val))
                    .collect();
                
                // Ordina per valore decrescente
                pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                // Prendi solo i top-k
                let top_pairs = pairs.into_iter().take(k).collect::<Vec<_>>();
                
                // Calcola softmax solo sui top-k
                let max_logit = top_pairs.iter().map(|&(_, val)| val).fold(
                    std::f32::NEG_INFINITY,
                    |a, b| a.max(b)
                );
                
                let mut probs = Array1::<f32>::zeros(self.vocab_size);
                let sum_exp: f32 = top_pairs
                    .iter()
                    .map(|&(_, val)| (val - max_logit).exp())
                    .sum();
                
                for (idx, val) in top_pairs {
                    probs[idx] = (val - max_logit).exp() / sum_exp;
                }
                
                probs
            } else {
                // Softmax su tutti i logits
                let max_logit = logits_temp.fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits = logits_temp.mapv(|x| (x - max_logit).exp());
                let sum_exp = exp_logits.sum();
                exp_logits / sum_exp
            };
            
            // Campiona il prossimo token dalle probabilità
            let mut next_token = 0;
            let r: f32 = rand::random();
            let mut cumsum = 0.0;
                
            for (idx, &prob) in token_probs.iter().enumerate() {
                cumsum += prob;
                if r < cumsum {
                    next_token = idx;
                    break;
                }
            }
            
            // Fallback all'ultimo token se non ne abbiamo campionato uno
            if next_token == 0 && cumsum < r {
                next_token = self.vocab_size - 1;
            }
            
            // Aggiungi il token generato alla sequenza
            tokens.push(next_token);
            
            // Verifica se abbiamo generato un token speciale di fine sequenza
            // (da implementare in base al tokenizer specifico)
        }
        
        // Decodifica i token in testo
        self.tokenizer.decode(&tokens)
    }
}

/// Funzione per valutare il modello su un set di dati
pub fn evaluate(
    trainer: &Trainer,
    eval_data: &[Vec<String>],
    batch_size: usize,
) -> f32 {
    let num_samples = eval_data.len();
    let num_batches = (num_samples + batch_size - 1) / batch_size;
    
    let mut total_loss = 0.0;
    let mut total_tokens = 0;
    
    // Preparazione dei batch
    for b in 0..num_batches {
        let start_idx = b * batch_size;
        let end_idx = std::cmp::min(start_idx + batch_size, num_samples);
        let batch_data = &eval_data[start_idx..end_idx];
        
        // Tokenizza le frasi
        let tokenizer = &trainer.tokenizer;
        let batch_tokens: Vec<Vec<usize>> = batch_data
            .iter()
            .map(|sentence| {
                // Converti il vettore di Stringhe in una singola stringa
                let joined_sentence = sentence.join(" ");
                tokenizer.encode(&joined_sentence)
            })
            .collect();
        
        // Determina la lunghezza massima della sequenza nel batch
        let max_len = batch_tokens.iter().map(|seq| seq.len()).max().unwrap_or(0);
        
        // Prepara input e target
        let mut input_ids = Vec::new();
        let mut target_ids = Array2::<usize>::zeros((batch_tokens.len(), max_len));
        
        for (i, seq) in batch_tokens.iter().enumerate() {
            // Input: tutti i token tranne l'ultimo
            let input = if seq.len() > 1 {
                seq[..seq.len() - 1].to_vec()
            } else {
                Vec::new()
            };
            
            // Padding per raggiungere max_len - 1
            let mut padded_input = input.clone();
            padded_input.resize(max_len - 1, 0); // 0 come token di padding
            input_ids.push(padded_input);
            
            // Target: tutti i token tranne il primo
            for (j, &token) in seq[1..].iter().enumerate() {
                target_ids[[i, j]] = token;
                total_tokens += 1;
            }
        }
        
        // Forward pass
        let output = trainer.forward(&input_ids, Some(&target_ids));
        
        // Accumula la loss
        if let Some(loss) = output.loss {
            total_loss += loss * total_tokens as f32;
        }
    }
    
    // Calcola perplexity: exp(loss media)
    let avg_loss = if total_tokens > 0 {
        total_loss / total_tokens as f32
    } else {
        0.0
    };
    
    (avg_loss as f32).exp()
} 