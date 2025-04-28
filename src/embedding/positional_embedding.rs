use crate::nabla::tensor::Tensor;
use ndarray::{Array, Array2, Axis};
use std::f32::consts::PI;
use rayon::prelude::*;

use super::Embedding;

/// Embedding posizionale che aggiunge informazione sulla posizione ai token
/// Implementa il positional encoding del paper "Attention Is All You Need"
#[derive(Clone)]
pub struct PositionalEmbedding {
    /// Matrice di embedding posizionale: [max_len, d_model]
    embedding_matrix: Tensor,
    max_len: usize,
    embedding_dim: usize,
}

impl PositionalEmbedding {
    /// Crea un nuovo embedding posizionale usando la funzione sinusoidale
    pub fn new(max_len: usize, embedding_dim: usize) -> Self {
        // Usiamo la formula del paper:
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        let mut embedding_data = Array::zeros((max_len, embedding_dim));
        
        // Crea un vettore di tutte le posizioni e dimensioni
        let indices: Vec<(usize, usize)> = (0..max_len)
            .flat_map(|pos| (0..embedding_dim/2).map(move |i| (pos, i)))
            .collect();
        
        // Calcola i valori in parallelo
        let results: Vec<(usize, usize, f32, f32)> = indices.par_iter()
            .map(|&(pos, i)| {
                let denominator = 10000_f32.powf(2.0 * i as f32 / embedding_dim as f32);
                let angle = pos as f32 / denominator;
                let sin_val = angle.sin();
                let cos_val = angle.cos();
                (pos, i, sin_val, cos_val)
            })
            .collect();
        
        // Assegna i valori calcolati alla matrice
        for (pos, i, sin_val, cos_val) in results {
            embedding_data[[pos, 2 * i]] = sin_val;
            if 2 * i + 1 < embedding_dim {
                embedding_data[[pos, 2 * i + 1]] = cos_val;
            }
        }
        
        PositionalEmbedding {
            embedding_matrix: Tensor::new(embedding_data),
            max_len,
            embedding_dim,
        }
    }
    
    /// Restituisce l'embedding matrix
    pub fn embedding_matrix(&self) -> &Tensor {
        &self.embedding_matrix
    }

    /// Forward pass con supporto per batch
    /// Output: Tensor con forma [batch_size, seq_len, embedding_dim]
    pub fn forward_batch(&self, batch_size: usize, seq_len: usize) -> Tensor {
        let effective_len = std::cmp::min(seq_len, self.max_len);
        
        // Ottieni gli embedding posizionali standard
        let pos_embeddings = self.embedding_matrix.data.slice(
            ndarray::s![0..effective_len, ..],
        ).to_owned();
        
        // Crea una matrice 3D per memorizzare i risultati [batch_size, seq_len, embedding_dim]
        let mut result_data = ndarray::Array3::<f32>::zeros((batch_size, effective_len, self.embedding_dim));
        
        // Replica gli stessi embedding posizionali per ogni elemento del batch
        // Questo metodo è sequenziale, ma funziona con tutte le implementazioni di Tensor
        for b in 0..batch_size {
            for i in 0..effective_len {
                for j in 0..self.embedding_dim {
                    result_data[[b, i, j]] = pos_embeddings[[i, j]];
                }
            }
        }
        
        // Converti il tensore 3D in un tensore 2D con forma [batch_size, seq_len * embedding_dim]
        let flattened = result_data.into_shape((batch_size, effective_len * self.embedding_dim)).unwrap();
        Tensor::new(flattened)
    }

    /// Forward pass con supporto per batch
    /// Output: Tensor con forma [batch_size, seq_len, embedding_dim]
    pub fn forward_batch_3d(&self, batch_size: usize, seq_len: usize) -> Tensor {
        let effective_len = std::cmp::min(seq_len, self.max_len);
        
        // Ottieni gli embedding posizionali standard
        let pos_embeddings = self.embedding_matrix.data.slice(
            ndarray::s![0..effective_len, ..],
        ).to_owned();
        
        // Crea una matrice 3D per memorizzare i risultati [batch_size, seq_len, embedding_dim]
        let mut result_data = ndarray::Array3::<f32>::zeros((batch_size, effective_len, self.embedding_dim));
        
        // Replica gli stessi embedding posizionali per ogni elemento del batch
        for b in 0..batch_size {
            for i in 0..effective_len {
                for j in 0..self.embedding_dim {
                    result_data[[b, i, j]] = pos_embeddings[[i, j]];
                }
            }
        }
        
        // Restituisci direttamente il tensore 3D
        Tensor::new_3d(result_data)
    }
}

impl Embedding for PositionalEmbedding {
    /// Forward pass: restituisce gli embedding posizionali per una sequenza
    /// Input: token_ids - array di token IDs (non usato, serve solo la lunghezza)
    /// Output: Tensor [seq_len, embedding_dim]
    fn forward(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        let effective_len = std::cmp::min(seq_len, self.max_len);
        
        // Prende le prime `effective_len` righe dalla matrice di embedding
        let slice = self.embedding_matrix.data.slice(
            ndarray::s![0..effective_len, ..],
        );
        
        // Se la sequenza è più corta di max_len, prendi solo i primi elementi
        let result_data = slice.to_owned();
        
        Tensor::new(result_data)
    }
    
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_positional_embedding_creation() {
        let embedding = PositionalEmbedding::new(100, 64);
        
        assert_eq!(embedding.max_len, 100);
        assert_eq!(embedding.embedding_dim(), 64);
        assert_eq!(embedding.embedding_matrix().data.shape(), &[100, 64]);
    }
    
    #[test]
    fn test_positional_embedding_formula() {
        let embedding = PositionalEmbedding::new(10, 8);
        let matrix = &embedding.embedding_matrix().data;
        
        // Test di alcuni valori specifici per verificare la formula
        // Per position=4, dim=0 (sin)
        let pos = 4;
        let dim = 0;
        let i = dim / 2; // i = 0 per dim = 0
        let denominator = 10000_f32.powf(2.0 * i as f32 / 8.0);
        let expected = (pos as f32 / denominator).sin();
        assert!((matrix[[pos, dim]] - expected).abs() < 1e-5);
        
        // Per position=4, dim=1 (cos)
        let dim = 1;
        let i = dim / 2; // i = 0 per dim = 1 (primo coseno)
        let denominator = 10000_f32.powf(2.0 * i as f32 / 8.0);
        let expected = (pos as f32 / denominator).cos();
        assert!((matrix[[pos, dim]] - expected).abs() < 1e-5);
    }
    
    #[test]
    fn test_positional_embedding_forward() {
        let embedding = PositionalEmbedding::new(10, 64);
        
        // Token IDs (non usati, ma necessari per l'interfaccia)
        let token_ids = vec![0, 0, 0, 0, 0]; // 5 tokens
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Verifica la forma: [5, 64]
        assert_eq!(output.data.shape(), &[5, 64]);
        
        // Verifica che i valori corrispondano alla matrice di embedding
        for i in 0..5 {
            for j in 0..64 {
                assert_eq!(
                    output.data[[i, j]],
                    embedding.embedding_matrix().data[[i, j]]
                );
            }
        }
    }
    
    #[test]
    fn test_positional_embedding_too_long() {
        // Crea un embedding con max_len piccolo
        let embedding = PositionalEmbedding::new(5, 16);
        
        // Prova con una sequenza più lunga di max_len
        let token_ids = vec![0; 10]; // 10 tokens
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Dovrebbe troncare a max_len (5)
        assert_eq!(output.data.shape(), &[5, 16]);
        
        // Verifica che i valori corrispondano alla matrice di embedding
        for i in 0..5 {
            for j in 0..16 {
                assert_eq!(
                    output.data[[i, j]],
                    embedding.embedding_matrix().data[[i, j]]
                );
            }
        }
    }
    
    #[test]
    fn test_positional_embedding_forward_batch() {
        let embedding = PositionalEmbedding::new(10, 32);
        
        // Parametri di batch
        let batch_size = 2;
        let seq_len = 4;
        
        // Forward pass con batch
        let output = embedding.forward_batch(batch_size, seq_len);
        
        // Verifica la forma [batch_size, seq_len * embedding_dim]
        assert_eq!(output.data.shape(), &[2, 4 * 32]);
        
        // Verifica che gli embedding posizionali siano stati replicati per ogni batch
        // Gli embedding della prima e seconda sequenza nel batch dovrebbero essere identici
        // perché gli embedding posizionali dipendono solo dalla posizione, non dal batch
        for j in 0..seq_len * 32 {
            assert_eq!(
                output.data[[0, j]],  // primo batch
                output.data[[1, j]]   // secondo batch
            );
        }
        
        // Verifica che i valori corrispondano alla matrice di embedding originale
        for i in 0..seq_len {
            for j in 0..32 {
                assert_eq!(
                    embedding.embedding_matrix().data[[i, j]],
                    output.data[[0, i * 32 + j]]  // nel formato flattened
                );
            }
        }
    }
} 