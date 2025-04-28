use std::time::{Instant, Duration};
use std::collections::HashMap;
use std::path::Path;
use std::io::{self, BufWriter, BufReader, Write};
use std::fs::File;
use ndarray::{Array, Array1, Array2, Array3, Axis, s, Ix3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use indicatif::{ProgressBar, ProgressStyle};
use thiserror::Error;
use crate::tokenizer::{Tokenizer, Vocab};
use crate::embedding::TransformerEmbedding;
use crate::attention::{EncoderStack, Attention};
use crate::nabla::tensor::Tensor;
use crate::export;

/// Errori possibili durante l'uso del modello
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Errore IO: {0}")]
    Io(#[from] io::Error),
    #[error("Errore di serializzazione: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("File di modello non valido o corrotto")]
    InvalidModel,
    #[error("Versione del modello non compatibile")]
    IncompatibleVersion,
    #[error("Dimensione del vocabolario non compatibile")]
    IncompatibleVocabSize,
    #[error("Formato non valido: {0}")]
    InvalidFormat(String),
    #[error("Altro errore: {0}")]
    Other(String),
}

/// Tipo di risultato per le operazioni sul modello
pub type ModelResult<T> = Result<T, ModelError>;

/// Rappresenta l'output di un modello, inclusi i logits e la loss
pub struct ModelOutput {
    /// Logits finali (probabilità non normalizzate)
    pub logits: Tensor,
    /// Loss calcolata (se disponibile)
    pub loss: Option<f32>,
}

/// Implementazione della funzione di loss Cross Entropy
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Crea una nuova istanza della funzione di loss
    pub fn new() -> Self {
        Self {}
    }
    
    /// Forward pass della Cross Entropy loss
    /// 
    /// # Arguments
    /// * `logits` - Tensore di shape [batch_size, seq_len, vocab_size] contenente i logits
    /// * `targets` - Array di shape [batch_size, seq_len] contenente gli indici target
    /// * `ignore_index` - Opzionale, indice da ignorare nel calcolo della loss (es. padding)
    /// 
    /// # Returns
    /// * Loss media e gradiente rispetto ai logits
    pub fn forward(&self, logits: &Tensor, targets: &Array2<usize>, ignore_index: Option<usize>) -> (f32, Tensor) {
        // ... Implementazione invariata
        (0.0, Tensor::new_from_array(Array1::zeros(1).into_dyn()))  // Placeholder
    }
}

/// Implementa l'ottimizzatore Adam per l'aggiornamento dei parametri
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
    /// 
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `beta1` - Parametro beta1 per il momento primo (default: 0.9)
    /// * `beta2` - Parametro beta2 per il momento secondo (default: 0.999)
    /// * `epsilon` - Epsilon per stabilità numerica (default: 1e-8)
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
    
    /// Esegue un passo di ottimizzazione
    /// 
    /// # Arguments
    /// * `params` - Parametri da aggiornare
    /// * `grads` - Gradienti corrispondenti
    pub fn step(&mut self, params: &mut HashMap<String, Tensor>, grads: &HashMap<String, Tensor>) {
        // ... Implementazione invariata
    }
    
    /// Imposta il learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    /// Ottiene il learning rate corrente
    pub fn get_learning_rate(&self) -> f32 {
        self.lr
    }
}

/// Rappresenta il trainer per addestrare e utilizzare il modello
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
    /// Lunghezza massima della sequenza
    max_seq_len: usize,
    /// Dimensione del feed-forward network
    ff_dim: usize,
    /// Numero di teste di attenzione
    num_heads: usize,
    /// Numero di layer encoder
    num_layers: usize,
    /// Funzione di loss
    loss_fn: CrossEntropyLoss,
    /// Ottimizzatore
    optimizer: AdamOptimizer,
    /// Parametri del modello
    params: HashMap<String, Tensor>,
    /// Metadati extra
    metadata: HashMap<String, String>,
}

impl Trainer {
    /// Crea un nuovo trainer con i parametri specificati
    pub fn new(
        tokenizer: Box<dyn Tokenizer>,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_rate: f32,
        learning_rate: f32,
    ) -> Self {
        // Valore minimo per evitare errori di divisione per zero
        let min_value = 1;
        
        // Valori sicuri: imposta almeno 1 per ogni dimensione
        let safe_model_dim = model_dim.max(min_value);
        let safe_ff_dim = ff_dim.max(min_value);
        let safe_num_heads = num_heads.max(min_value);
        let safe_num_layers = num_layers.max(min_value);
        
        // Calcola la dimensione del vocabolario dal tokenizer
        let vocab_size = tokenizer.get_vocab().len();
        
        // Lunghezza massima della sequenza
        let max_seq_len = 256;  // Valore predefinito
        
        // Inizializza il TransformerEmbedding
        let embedding = TransformerEmbedding::new(
            vocab_size,
            safe_model_dim,
            max_seq_len,
            dropout_rate
        );
        
        // Inizializza l'EncoderStack
        let encoder = EncoderStack::new(
            safe_model_dim,
            safe_ff_dim,
            safe_num_heads,
            safe_num_layers,
            dropout_rate
        );
        
        println!("Inizializzazione output_projection: model_dim={}, vocab_size={}", safe_model_dim, vocab_size);
        
        // Output projection inizializzato con valori casuali
        // Matrice di proiezione dal model_dim al vocab_size per generare logits
        let output_proj_data = Array2::<f32>::zeros((safe_model_dim, vocab_size));
        let output_projection = Tensor::new(output_proj_data);
        
        // Aggiungi i parametri iniziali
        let mut params = HashMap::new();
        params.insert("output_projection".to_string(), output_projection.clone());
        
        Self {
            tokenizer,
            embedding,
            encoder,
            output_projection,
            model_dim: safe_model_dim,
            vocab_size,
            max_seq_len,
            ff_dim: safe_ff_dim,
            num_heads: safe_num_heads,
            num_layers: safe_num_layers,
            loss_fn: CrossEntropyLoss::new(),
            optimizer: AdamOptimizer::new(learning_rate, 0.9, 0.999, 1e-8),
            params,
            metadata: HashMap::new(),
        }
    }
    
    /// Esegue un forward pass del modello
    pub fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput {
        let batch_size = input.len();
        
        if batch_size == 0 {
            // Caso limite: batch vuoto
            return ModelOutput {
                logits: Tensor::new_from_array(Array::zeros((0, 0, 0)).into_dyn()),
                loss: None
            };
        }
        
        let seq_len = input[0].len();
        
        // Stampiamo informazioni di debug sulla forma dell'input
        println!("Debug: forward - batch_size: {}, seq_len: {}", batch_size, seq_len);
        
        // L'embedding ora ritorna direttamente un tensore 3D
        let encoder_input = self.embedding.forward_batch(input);
        
        println!("Debug: forward - shape dopo embedding: {:?}", encoder_input.data.shape());
        
        // Forward pass attraverso l'encoder con il tensore 3D
        let encoder_output = self.encoder.forward(&encoder_input, None);
        
        println!("Debug: forward - shape dopo encoder: {:?}", encoder_output.data.shape());
        
        // Utilizziamo matmul_with invece di dot per la proiezione nell'output space
        let logits = encoder_output.matmul_with(&self.output_projection);
        
        println!("Debug: forward - shape finale logits: {:?}", logits.data.shape());
        
        // Calcolo della loss se sono forniti i target
        let loss = if let Some(target_tokens) = target {
            let mut total_loss = 0.0;
            let mut total_tokens = 0;
            
            // Implementazione semplificata di cross-entropy loss
            for i in 0..batch_size {
                for j in 0..seq_len {
                    if j < target_tokens.shape()[1] {  // Verifica che j sia nel range valido
                        let target_id = target_tokens[[i, j]];
                        if target_id != 0 { // Ignora padding tokens
                            // Verifica che target_id sia nel range valido
                            if target_id < logits.data.shape()[2] {
                                let logit = logits.data[[i, j, target_id]];
                                total_loss -= logit; // Semplificazione della cross-entropy
                                total_tokens += 1;
                            } else {
                                println!("Warning: target_id {} fuori range (max {})", 
                                         target_id, logits.data.shape()[2]-1);
                            }
                        }
                    }
                }
            }
            
            if total_tokens > 0 {
                total_loss / total_tokens as f32
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        ModelOutput {
            logits,
            loss: if loss != 0.0 { Some(loss) } else { None }
        }
    }
    
    /// Esegue un passo di training
    pub fn train_step(&mut self, batch: &Vec<Vec<usize>>, targets: &Array2<usize>) -> f32 {
        // 1. Eseguiamo il forward pass
        let output = self.forward(batch, Some(targets));
        
        // 2. Calcoliamo la loss e il gradiente
        let (loss, logits_grad) = self.loss_fn.forward(&output.logits, targets, None);
        
        // 3. Backpropagation: in un'implementazione reale, calcoleremo i gradienti
        // per tutti i parametri. Per questo test, creiamo un gradiente di esempio
        // per output_projection
        let mut grads = HashMap::new();
        
        // Creiamo un gradiente di esempio per output_projection
        // In una implementazione reale, questo verrebbe calcolato dalla backpropagation
        let output_proj_grad = Tensor::new_from_array(
            Array::from_elem(self.output_projection.data.dim(), 0.01)
        );
        
        grads.insert("output_projection".to_string(), output_proj_grad);
        
        // 4. Aggiorniamo i parametri con l'ottimizzatore
        self.optimizer.step(&mut self.params, &grads);
        
        // 5. Aggiorniamo il riferimento a output_projection con il valore aggiornato
        if let Some(updated_output_proj) = self.params.get("output_projection") {
            self.output_projection = updated_output_proj.clone();
        }
        
        loss
    }
    
    /// Restituisce la lunghezza massima della sequenza
    pub fn get_max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    /// Restituisce la dimensione del vocabolario
    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Restituisce il numero totale di parametri nel modello
    pub fn get_parameter_count(&self) -> usize {
        // ... Implementazione invariata
        0  // Placeholder
    }
    
    /// Restituisce i tensori del modello
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.params
    }
    
    /// Salva il modello in formato binario
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        crate::export::save_model(
            path,
            self.model_dim,
            self.ff_dim,
            self.num_heads,
            self.num_layers,
            self.max_seq_len,
            self.tokenizer.as_ref(),
            &self.params,
            self.metadata.clone()
        )
    }
    
    /// Carica un modello da un file binario
    pub fn load_model<P: AsRef<Path>, T: Tokenizer + Clone + 'static>(
        path: P, 
        tokenizer: &mut T,
        learning_rate: Option<f32>
    ) -> ModelResult<Self> {
        // Crea un nuovo trainer con valori iniziali
        let mut trainer = Self::new(
            Box::new(tokenizer.clone()),
            0, 0, 0, 0, 0.0, 
            learning_rate.unwrap_or(0.001)
        );
        
        // Carica il modello utilizzando il modulo export e passa il tokenizer direttamente
        let (model_dim, ff_dim, num_heads, num_layers, max_seq_len, params, metadata) = 
            crate::export::load_model(path, tokenizer)?;
        
        // Aggiorna i parametri del trainer
        trainer.model_dim = model_dim;
        trainer.ff_dim = ff_dim;
        trainer.num_heads = num_heads;
        trainer.num_layers = num_layers;
        trainer.max_seq_len = max_seq_len;
        trainer.vocab_size = tokenizer.get_vocab().len();
        trainer.params = params;
        trainer.metadata = metadata;
        
        // Ricrea gli altri componenti
        trainer.embedding = TransformerEmbedding::new(
            trainer.vocab_size,
            trainer.model_dim,
            trainer.max_seq_len,
            0.1  // dropout_rate
        );
        
        trainer.encoder = EncoderStack::new(
            trainer.model_dim,
            trainer.ff_dim,
            trainer.num_heads,
            trainer.num_layers,
            0.1  // dropout_rate
        );
        
        // Imposta l'output projection o lo prende dai parametri se disponibile
        if let Some(output_proj) = trainer.params.get("output_projection") {
            trainer.output_projection = output_proj.clone();
        }
        
        // Aggiorna il tokenizer del trainer con quello fornito
        trainer.tokenizer = Box::new(tokenizer.clone());
        
        Ok(trainer)
    }
}

// Include i moduli aggiuntivi
pub mod tests;
pub mod evaluate; 