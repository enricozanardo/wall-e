use crate::nabla::tensor::Tensor;
use crate::tokenizer::Vocab;
use ndarray::{Array, Array2, Array3, Axis};
use rand::thread_rng;
use rand_distr::{Normal, Distribution};

use super::Embedding;

/// Embedding di token che converte gli ID dei token in vettori densi
#[derive(Clone)]
pub struct TokenEmbedding {
    /// Matrice di embedding: [vocab_size, embedding_dim]
    embedding_matrix: Tensor,
    /// Dimensione del vocabolario
    vocab_size: usize,
    /// Dimensione dell'embedding
    embedding_dim: usize,
}

impl TokenEmbedding {
    /// Crea un nuovo embedding di token con inizializzazione normale
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Crea la distribuzione normale con media 0 e deviazione standard 0.02
        let normal = Normal::new(0.0, 0.02).unwrap();
        let mut rng = rand::rng();
        
        // Inizializza la matrice di embedding da una distribuzione normale
        let mut embedding_data = Array::zeros((vocab_size, embedding_dim));
        for i in 0..vocab_size {
            for j in 0..embedding_dim {
                embedding_data[[i, j]] = normal.sample(&mut rng);
            }
        }
        
        TokenEmbedding {
            embedding_matrix: Tensor::new(embedding_data),
            vocab_size,
            embedding_dim,
        }
    }
    
    /// Forward pass: restituisce gli embedding per una sequenza di token IDs
    /// Input: token_ids - array di token IDs di forma [seq_len]
    /// Output: Tensor [seq_len, embedding_dim]
    pub fn forward_tensor(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        
        // Crea una nuova matrice per memorizzare i risultati
        let mut result_data = Array2::<f32>::zeros((seq_len, self.embedding_dim));
        
        // Copia i vettori di embedding per ciascun token ID
        for (i, &token_id) in token_ids.iter().enumerate() {
            let effective_id = token_id.min(self.vocab_size - 1);
            for j in 0..self.embedding_dim {
                result_data[[i, j]] = self.embedding_matrix.data[[effective_id, j]];
            }
        }
        
        Tensor::new(result_data)
    }
    
    /// Forward pass con supporto per batch di token IDs
    /// Input: batch_token_ids - array di batch di token IDs
    /// Output: Tensor [batch_size, seq_len, embedding_dim]
    pub fn forward_batch(&self, batch_token_ids: &[Vec<usize>]) -> Tensor {
        let batch_size = batch_token_ids.len();
        if batch_size == 0 {
            return Tensor::new(Array2::<f32>::zeros((0, 0)));
        }
        
        let seq_len = batch_token_ids[0].len();
        
        // Crea una matrice 3D per memorizzare i risultati [batch_size, seq_len, embedding_dim]
        let mut result_data = ndarray::Array3::<f32>::zeros((batch_size, seq_len, self.embedding_dim));
        
        // Processa ogni sequenza nel batch
        for (b, token_ids) in batch_token_ids.iter().enumerate() {
            for (i, &token_id) in token_ids.iter().enumerate().take(seq_len) {
                let effective_id = token_id.min(self.vocab_size - 1);
                for j in 0..self.embedding_dim {
                    result_data[[b, i, j]] = self.embedding_matrix.data[[effective_id, j]];
                }
            }
        }
        
        // Converti il tensore 3D in un tensore 2D con forma [batch_size, seq_len * embedding_dim]
        let flattened = result_data.into_shape((batch_size, seq_len * self.embedding_dim)).unwrap();
        Tensor::new(flattened)
    }
    
    /// Accesso alla matrice di embedding
    pub fn embedding_matrix(&self) -> &Tensor {
        &self.embedding_matrix
    }
}

impl Embedding for TokenEmbedding {
    /// Forward pass: restituisce gli embedding per una sequenza di token IDs
    fn forward(&self, token_ids: &[usize]) -> Tensor {
        self.forward_tensor(token_ids)
    }
    
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayView;
    
    #[test]
    fn test_token_embedding_creation() {
        let embedding = TokenEmbedding::new(1000, 64);
        
        assert_eq!(embedding.vocab_size, 1000);
        assert_eq!(embedding.embedding_dim(), 64);
        assert_eq!(embedding.embedding_matrix().data.shape(), &[1000, 64]);
    }
    
    #[test]
    fn test_token_embedding_normal_distribution() {
        let embedding = TokenEmbedding::new(1000, 64);
        
        // Calcola la media e la varianza per verificare l'inizializzazione normale
        let matrix = &embedding.embedding_matrix().data;
        let flat_view = matrix.view().into_shape(1000 * 64).unwrap();
        
        let sum: f32 = flat_view.sum();
        let mean = sum / (1000.0 * 64.0);
        
        let sum_sq: f32 = flat_view.iter().map(|&x| (x - mean).powi(2)).sum();
        let variance = sum_sq / (1000.0 * 64.0);
        let std_dev = variance.sqrt();
        
        // Verifica che la media sia vicina a 0 e la deviazione standard vicina a 0.02
        assert!(mean.abs() < 0.01, "Mean should be close to 0, got {}", mean);
        assert!((std_dev - 0.02).abs() < 0.01, "Std dev should be close to 0.02, got {}", std_dev);
    }
    
    #[test]
    fn test_token_embedding_forward() {
        let embedding = TokenEmbedding::new(1000, 64);
        
        // Token IDs di test
        let token_ids = vec![5, 10, 15, 20, 25];
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Verifica la forma: [5, 64]
        assert_eq!(output.data.shape(), &[5, 64]);
        
        // Verifica che i valori corrispondano alla matrice di embedding
        for (i, &token_id) in token_ids.iter().enumerate() {
            for j in 0..64 {
                assert_eq!(
                    output.data[[i, j]],
                    embedding.embedding_matrix().data[[token_id, j]]
                );
            }
        }
    }
    
    #[test]
    fn test_token_embedding_out_of_bounds() {
        // Crea un embedding con vocabolario piccolo
        let embedding = TokenEmbedding::new(10, 16);
        
        // Prova con token ID fuori dai limiti
        let token_ids = vec![5, 20, 3]; // 20 è fuori limite
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Verifica la forma: [3, 16]
        assert_eq!(output.data.shape(), &[3, 16]);
        
        // Token ID 20 dovrebbe essere mappato all'ultimo token (9)
        for j in 0..16 {
            assert_eq!(
                output.data[[1, j]],
                embedding.embedding_matrix().data[[9, j]]
            );
        }
    }
    
    #[test]
    fn test_token_embedding_forward_batch() {
        let embedding = TokenEmbedding::new(100, 32);
        
        // Crea un batch di token IDs
        let batch_token_ids = vec![
            vec![1, 2, 3],  // prima sequenza
            vec![4, 5, 6],  // seconda sequenza
        ];
        
        // Forward pass con batch
        let output = embedding.forward_batch(&batch_token_ids);
        
        // Verifica la forma [batch_size, seq_len * embedding_dim]
        assert_eq!(output.data.shape(), &[2, 3 * 32]);
        
        // Verifica che i valori del primo batch siano corretti rispetto ai token IDs
        for j in 0..32 {
            assert_eq!(
                embedding.embedding_matrix().data[[1, j]],  // token ID 1
                output.data[[0, j]]  // primo batch, primo token (prima parte del vettore flattened)
            );
        }
        
        for j in 0..32 {
            assert_eq!(
                embedding.embedding_matrix().data[[4, j]],  // token ID 4
                output.data[[1, j]]  // secondo batch, primo token
            );
        }
    }
} 