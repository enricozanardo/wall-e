use ndarray::{Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::nabla::tensor::Tensor;
use crate::embedding::{Embedding, BatchEmbedding};

/// A structure that represents a token embedding
///
/// TokenEmbedding maps token IDs to embedding vectors.
/// Each token in the vocabulary is associated with a fixed-dimension vector.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    /// The embedding matrix
    pub weights: Array2<f32>,
    /// The embedding dimensionality
    pub embedding_dim: usize,
    /// The vocabulary size
    pub vocab_size: usize,
}

impl TokenEmbedding {
    /// Creates a new token embedding with randomly initialized weights
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of the vocabulary
    /// * `embedding_dim` - Dimension of the embedding vectors
    ///
    /// # Returns
    ///
    /// A new TokenEmbedding instance
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::token_embedding::TokenEmbedding;
    ///
    /// let embedding = TokenEmbedding::new(
    ///     1000,   // vocab_size
    ///     128,    // embedding_dim
    /// );
    /// ```
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Initialize weights with a uniform distribution between -1.0 and 1.0
        let weights = Array2::random_using(
            (vocab_size, embedding_dim),
            Uniform::new(-1.0, 1.0),
            &mut rand::thread_rng(),
        );
        
        TokenEmbedding {
            weights,
            embedding_dim,
            vocab_size,
        }
    }

    /// Updates the embedding for a specific token ID
    ///
    /// # Arguments
    ///
    /// * `token_id` - The ID of the token to update
    /// * `embedding` - The new embedding vector for the token
    ///
    /// # Panics
    ///
    /// Panics if the token ID is out of bounds
    pub fn update_embedding(&mut self, token_id: usize, embedding: Array1<f32>) {
        assert!(token_id < self.vocab_size, "Token ID out of bounds");
        assert_eq!(embedding.len(), self.embedding_dim, "Embedding dimension mismatch");
        
        for i in 0..self.embedding_dim {
            self.weights[[token_id, i]] = embedding[i];
        }
    }

    /// Retrieves the embedding for a specific token ID
    /// 
    /// # Arguments
    /// 
    /// * `token_id` - The ID of the token to retrieve
    /// 
    /// # Returns
    /// 
    /// The embedding vector for the specified token
    /// 
    /// # Panics
    /// 
    /// Panics if the token ID is out of bounds
    pub fn get_embedding(&self, token_id: usize) -> Array1<f32> {
        assert!(token_id < self.vocab_size, "Token ID out of bounds");
        
        let mut result = Array1::zeros(self.embedding_dim);
        for i in 0..self.embedding_dim {
            result[i] = self.weights[[token_id, i]];
        }
        result
    }
}

impl Embedding for TokenEmbedding {
    /// Performs the forward pass of token embedding on a group of token IDs
    /// 
    /// # Arguments
    /// 
    /// * `token_ids` - A slice of token IDs to embed
    /// 
    /// # Returns
    /// 
    /// A tensor containing the embeddings for the specified tokens
    fn forward(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        let mut embeddings = Array2::zeros((seq_len, self.embedding_dim));
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id < self.vocab_size {
                for j in 0..self.embedding_dim {
                    embeddings[[i, j]] = self.weights[[token_id, j]];
                }
            }
        }
        
        Tensor::new(embeddings)
    }
    
    /// Returns the embedding dimension
    ///
    /// # Returns
    ///
    /// The dimension of the embedding vectors
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

impl BatchEmbedding for TokenEmbedding {
    /// Performs the forward pass on a batch of token IDs
    /// 
    /// # Arguments
    /// 
    /// * `token_ids` - A slice of token ID sequences
    /// 
    /// # Returns
    /// 
    /// A tensor containing the embeddings for all tokens in the batch
    /// 
    /// # Panics
    /// 
    /// Panics if the sequences in the batch have different lengths
    fn forward_batch(&self, token_ids: &[Vec<usize>]) -> Tensor {
        if token_ids.is_empty() {
            return Tensor::new_3d(Array3::<f32>::zeros((0, 0, 0)));
        }

        // Check if all sequences have the same length
        let seq_length = token_ids[0].len();
        for seq in token_ids {
            assert_eq!(seq.len(), seq_length, "All sequences in a batch must have the same length");
        }

        let batch_size = token_ids.len();
        let mut batch_embeddings = Array3::zeros((batch_size, seq_length, self.embedding_dim));

        for (batch_idx, sequence) in token_ids.iter().enumerate() {
            for (seq_idx, &token_id) in sequence.iter().enumerate() {
                if token_id < self.vocab_size {
                    for dim_idx in 0..self.embedding_dim {
                        batch_embeddings[[batch_idx, seq_idx, dim_idx]] = self.weights[[token_id, dim_idx]];
                    }
                }
            }
        }

        Tensor::new_3d(batch_embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_token_embedding_creation() {
        let vocab_size = 1000;
        let embedding_dim = 64;
        let embedding = TokenEmbedding::new(vocab_size, embedding_dim);
        
        // Check dimensions
        assert_eq!(embedding.weights.shape(), &[vocab_size, embedding_dim]);
        assert_eq!(embedding.embedding_dim, embedding_dim);
    }
    
    #[test]
    fn test_get_embedding() {
        let vocab_size = 100;
        let embedding_dim = 32;
        let embedding = TokenEmbedding::new(vocab_size, embedding_dim);
        
        // Get embedding for token with ID 5
        let token_id = 5;
        let embed_vector = embedding.get_embedding(token_id);
        
        assert_eq!(embed_vector.shape(), &[embedding_dim]);
        
        // Check if the vector from get_embedding matches the weights matrix
        let weights_row = embedding.weights.row(token_id).to_owned();
        assert_eq!(embed_vector, weights_row);
    }
    
    #[test]
    fn test_update_embedding() {
        let vocab_size = 100;
        let embedding_dim = 32;
        let mut embedding = TokenEmbedding::new(vocab_size, embedding_dim);
        
        let token_id = 10;
        let original_embedding = embedding.get_embedding(token_id);
        
        // Create a gradient of all 1s
        let gradient = Array1::ones(embedding_dim);
        let learning_rate = 0.01;
        
        // Update the embedding
        embedding.update_embedding(token_id, gradient);
        
        // Get the updated embedding
        let updated_embedding = embedding.get_embedding(token_id);
        
        // Verify each element decreased by learning_rate * gradient
        for i in 0..embedding_dim {
            assert!((updated_embedding[i] - (original_embedding[i] - learning_rate)).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_forward() {
        let vocab_size = 100;
        let embedding_dim = 32;
        let embedding = TokenEmbedding::new(vocab_size, embedding_dim);
        
        // Input sequence: [1, 5, 10]
        let token_ids = vec![1, 5, 10];
        let seq_len = token_ids.len();
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Check output shape: [seq_len, embedding_dim]
        assert_eq!(output.data.shape(), &[seq_len, embedding_dim]);
        
        // Verify that the output matches the corresponding embedding
        for (i, &token_id) in token_ids.iter().enumerate() {
            let expected = embedding.get_embedding(token_id);
            for j in 0..embedding_dim {
                assert!((output.data[[i, j]] - expected[j]).abs() < 1e-6);
            }
        }
    }
    
    #[test]
    fn test_forward_batch() {
        let vocab_size = 100;
        let embedding_dim = 32;
        let embedding = TokenEmbedding::new(vocab_size, embedding_dim);
        
        // Batch of sequences: batch_size=2, seq_len=3
        let batch_token_ids = vec![
            vec![1, 5, 10],  // First sequence
            vec![2, 6, 11],  // Second sequence
        ];
        
        let batch_size = batch_token_ids.len();
        let seq_len = batch_token_ids[0].len();
        
        // Forward pass with batch
        let output = embedding.forward_batch(&batch_token_ids);
        
        // Check output shape: [batch_size, seq_len, embedding_dim]
        assert_eq!(output.data.shape(), &[batch_size, seq_len, embedding_dim]);
        
        // Verify that each element in the batch matches the corresponding embedding
        for (i, sequence) in batch_token_ids.iter().enumerate() {
            for (j, &token_id) in sequence.iter().enumerate() {
                let expected = embedding.get_embedding(token_id);
                for k in 0..embedding_dim {
                    assert!((output.data[[i, j, k]] - expected[k]).abs() < 1e-6);
                }
            }
        }
    }
} 