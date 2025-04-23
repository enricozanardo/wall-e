use crate::nabla::tensor::Tensor;
use super::create_weight_matrix;
use ndarray::Array2;
use rayon::prelude::*;

/// Implementazione della rete feed-forward come descritto nel paper "Attention is All You Need"
///
/// La rete feed-forward consiste di due trasformazioni lineari con una funzione ReLU in mezzo:
/// FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
pub struct FeedForward {
    /// Dimensione del modello (d_model)
    d_model: usize,
    /// Dimensione dello strato interno (d_ff), tipicamente 4 * d_model
    d_ff: usize,
    /// Matrice dei pesi per la prima trasformazione lineare
    w1: Tensor,
    /// Matrice dei pesi per la seconda trasformazione lineare
    w2: Tensor,
}

impl FeedForward {
    /// Crea una nuova istanza della rete feed-forward
    ///
    /// # Arguments
    /// * `d_model` - Dimensione del modello/embedding
    /// * `d_ff` - Dimensione dello strato interno (default = 4 * d_model)
    ///
    /// # Returns
    /// Una nuova istanza di FeedForward
    pub fn new(d_model: usize, d_ff: Option<usize>) -> Self {
        let d_ff = d_ff.unwrap_or(4 * d_model);
        
        // Inizializza i pesi per le due trasformazioni lineari
        let w1 = Tensor::new(create_weight_matrix(d_model, d_ff, 0.02));
        let w2 = Tensor::new(create_weight_matrix(d_ff, d_model, 0.02));
        
        Self {
            d_model,
            d_ff,
            w1,
            w2,
        }
    }
    
    /// Esegue il forward pass attraverso la rete feed-forward
    ///
    /// # Arguments
    /// * `input` - Tensore di input [batch_size * seq_len, d_model]
    ///
    /// # Returns
    /// Tensore di output [batch_size * seq_len, d_model]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Prima trasformazione lineare: xW₁
        let hidden = input.matmul_with(&self.w1);
        
        // Applicazione di ReLU: max(0, xW₁)
        let activated = hidden.relu();
        
        // Seconda trasformazione lineare: max(0, xW₁)W₂
        activated.matmul_with(&self.w2)
    }
    
    /// Restituisce la dimensione del modello
    pub fn model_dim(&self) -> usize {
        self.d_model
    }
}

/// Implementa una versione ottimizzata del feed-forward network
/// che divide il lavoro in batch sequenziali
pub struct BatchedFeedForward {
    /// Feed-forward network sottostante
    ff: FeedForward,
    /// Dimensione del batch per l'elaborazione
    batch_size: usize,
    /// Soglia oltre la quale applicare il parallelismo
    parallelism_threshold: usize,
}

impl BatchedFeedForward {
    /// Crea un nuovo feed-forward network con batch
    ///
    /// # Arguments
    /// * `d_model` - Dimensione del modello (input e output)
    /// * `d_ff` - Dimensione interna del feed-forward
    /// * `batch_size` - Dimensione del batch per l'elaborazione
    ///
    /// # Returns
    /// Una nuova istanza di BatchedFeedForward
    pub fn new(d_model: usize, d_ff: usize, batch_size: usize) -> Self {
        Self {
            ff: FeedForward::new(d_model, Some(d_ff)),
            batch_size,
            parallelism_threshold: 64, // Soglia predefinita
        }
    }
    
    /// Forward pass con batch
    ///
    /// # Arguments
    /// * `x` - Tensore di input [seq_len, d_model]
    ///
    /// # Returns
    /// Tensore di output [seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.data.shape()[0];
        
        // Per sequenze brevi, usa l'implementazione normale
        if seq_len < self.parallelism_threshold {
            return self.ff.forward(x);
        }
        
        // Per sequenze lunghe, dividi il lavoro in batch paralleli
        let batch_size = self.batch_size;
        let d_model = self.ff.d_model;
        
        // Crea array di risultati parziali per ogni batch
        let results: Vec<_> = (0..seq_len)
            .into_par_iter()
            .map(|i| {
                let batch_idx = i / batch_size;
                let batch_start = batch_idx * batch_size;
                let batch_end = (batch_start + batch_size).min(seq_len);
                
                // Ignora i batch fuori dal range
                if batch_start >= seq_len {
                    return vec![0.0; d_model];
                }
                
                // Estrai il batch
                let batch_slice = x.data.slice(ndarray::s![batch_start..batch_end, ..]).to_owned();
                let batch_input = Tensor::new(batch_slice);
                
                // Calcola l'output per questo batch
                let batch_output = self.ff.forward(&batch_input);
                
                // Restituisci la riga corretta
                let local_i = i - batch_start;
                if local_i < batch_output.data.shape()[0] {
                    batch_output.data.slice(ndarray::s![local_i, ..]).to_vec()
                } else {
                    vec![0.0; d_model] // questo non dovrebbe mai accadere
                }
            })
            .collect();
        
        // Ricostruisci il tensore di output
        let mut output_data = Array2::zeros((seq_len, d_model));
        for (i, row) in results.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                output_data[[i, j]] = val;
            }
        }
        
        Tensor::new(output_data)
    }
    
    /// Restituisce la dimensione del modello
    pub fn model_dim(&self) -> usize {
        self.ff.model_dim()
    }
    
    /// Imposta una nuova soglia per il parallelismo
    ///
    /// # Arguments
    /// * `threshold` - La nuova soglia
    pub fn set_parallelism_threshold(&mut self, threshold: usize) {
        self.parallelism_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_feed_forward_creation() {
        let d_model = 64;
        
        // Test con d_ff predefinito (4 * d_model)
        let ff_default = FeedForward::new(d_model, None);
        assert_eq!(ff_default.d_model, d_model);
        assert_eq!(ff_default.d_ff, 4 * d_model);
        assert_eq!(ff_default.w1.data.shape(), &[d_model, 4 * d_model]);
        assert_eq!(ff_default.w2.data.shape(), &[4 * d_model, d_model]);
        
        // Test con d_ff personalizzato
        let custom_d_ff = 128;
        let ff_custom = FeedForward::new(d_model, Some(custom_d_ff));
        assert_eq!(ff_custom.d_model, d_model);
        assert_eq!(ff_custom.d_ff, custom_d_ff);
        assert_eq!(ff_custom.w1.data.shape(), &[d_model, custom_d_ff]);
        assert_eq!(ff_custom.w2.data.shape(), &[custom_d_ff, d_model]);
        
        // Verifica che model_dim ritorni il valore corretto
        assert_eq!(ff_default.model_dim(), d_model);
    }
    
    #[test]
    fn test_feed_forward_forward() {
        let d_model = 64;
        let seq_len = 10;
        let batch_size = 2;
        
        let ff = FeedForward::new(d_model, None);
        
        // Crea un input di test
        let input_data = Array2::ones((batch_size * seq_len, d_model));
        let input = Tensor::new(input_data);
        
        // Forward pass
        let output = ff.forward(&input);
        
        // Verifica che l'output abbia la forma corretta
        assert_eq!(output.data.shape(), &[batch_size * seq_len, d_model]);
        
        // Verifica che l'output non sia tutto zero
        assert!(output.data.sum() != 0.0);
    }
    
    #[test]
    fn test_feed_forward_relu_activation() {
        let d_model = 4;
        let d_ff = 8;
        
        let ff = FeedForward::new(d_model, Some(d_ff));
        
        // Crea un input con alcuni valori negativi
        let mut input_data = Array2::ones((1, d_model));
        input_data[[0, 0]] = -1.0; // Imposta un valore negativo
        
        let input = Tensor::new(input_data);
        
        // Forward pass
        let hidden = input.matmul_with(&ff.w1);
        
        // Verifica che ci siano valori negativi nell'output della prima trasformazione
        let has_negative = hidden.data.iter().any(|&x| x < 0.0);
        
        // Attiva con ReLU
        let activated = hidden.relu();
        
        // Verifica che non ci siano valori negativi dopo ReLU
        let all_non_negative = activated.data.iter().all(|&x| x >= 0.0);
        
        assert!(has_negative, "L'output della prima trasformazione dovrebbe avere valori negativi");
        assert!(all_non_negative, "Dopo ReLU, tutti i valori dovrebbero essere non negativi");
    }
    
    #[test]
    fn test_batched_feed_forward() {
        let d_model = 64;
        let d_ff = 128;
        let batch_size = 8;
        let seq_len = 100;
        
        let mut batched_ff = BatchedFeedForward::new(d_model, d_ff, batch_size);
        batched_ff.set_parallelism_threshold(32); // Imposta una soglia bassa per il test
        
        // Crea un input di test
        let input_data = Array2::ones((seq_len, d_model));
        let input = Tensor::new(input_data);
        
        // Forward pass
        let output = batched_ff.forward(&input);
        
        // Verifica che l'output abbia la forma corretta
        assert_eq!(output.data.shape(), &[seq_len, d_model]);
    }
} 