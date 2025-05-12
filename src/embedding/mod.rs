pub mod token_embedding;
pub mod positional_embedding;

use crate::nabla::tensor::Tensor;
use crate::tokenizer::Vocab;

/// Trait that defines common operations for embeddings
///
/// This trait provides the interface for converting token IDs into vector
/// representations (embeddings) and retrieving the embedding dimension.
pub trait Embedding {
    /// Converts token IDs into embeddings (vector representations)
    ///
    /// # Arguments
    ///
    /// * `token_ids` - A slice of token IDs to convert into embeddings
    ///
    /// # Returns
    ///
    /// A Tensor containing the embeddings with shape [seq_len, embedding_dim]
    fn forward(&self, token_ids: &[usize]) -> Tensor;
    
    /// Returns the embedding dimension (d_model)
    ///
    /// # Returns
    ///
    /// The size of the embedding vectors
    fn embedding_dim(&self) -> usize;
}

/// Trait that defines operations with batch support
///
/// This trait extends the basic embedding functionality to handle
/// batches of token sequences efficiently.
pub trait BatchEmbedding {
    /// Converts batches of token IDs into embeddings
    ///
    /// # Arguments
    ///
    /// * `batch_token_ids` - A slice of vectors, where each vector contains the token IDs for one sequence
    ///
    /// # Returns
    ///
    /// A Tensor containing the batch embeddings
    fn forward_batch(&self, batch_token_ids: &[Vec<usize>]) -> Tensor;
}

/// A struct that combines word embedding and positional embedding
///
/// TransformerEmbedding combines token embeddings and positional embeddings
/// to create the input representation for transformer models, as described in
/// the "Attention Is All You Need" paper.
#[allow(dead_code)]
pub struct TransformerEmbedding {
    /// Token embeddings
    token_emb: token_embedding::TokenEmbedding,
    /// Positional embeddings
    pos_emb: positional_embedding::PositionalEmbedding,
    /// Embedding dimension
    embedding_dim: usize,
    /// Dropout rate
    dropout_rate: f32,
}

impl TransformerEmbedding {
    /// Creates a new transformer embedding
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of the vocabulary
    /// * `embedding_dim` - Dimension of the embedding vectors
    /// * `max_seq_len` - Maximum sequence length
    /// * `dropout_rate` - Rate for dropout regularization
    ///
    /// # Returns
    ///
    /// A new TransformerEmbedding instance
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::TransformerEmbedding;
    ///
    /// let embedding = TransformerEmbedding::new(
    ///     1000,   // vocab_size
    ///     128,    // embedding_dim
    ///     512,    // max_seq_len
    ///     0.1,    // dropout_rate
    /// );
    /// ```
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        max_seq_len: usize,
        dropout_rate: f32,
    ) -> Self {
        TransformerEmbedding {
            token_emb: token_embedding::TokenEmbedding::new(vocab_size, embedding_dim),
            pos_emb: positional_embedding::PositionalEmbedding::new(max_seq_len, embedding_dim),
            embedding_dim,
            dropout_rate,
        }
    }
    
    /// Builds an embedding from an existing vocabulary
    ///
    /// # Arguments
    ///
    /// * `vocab` - Reference to a vocabulary
    /// * `embedding_dim` - Dimension of the embedding vectors
    /// * `max_seq_len` - Maximum sequence length
    /// * `dropout_rate` - Rate for dropout regularization
    ///
    /// # Returns
    ///
    /// A new TransformerEmbedding instance initialized with the provided vocabulary
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::TransformerEmbedding;
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let vocab = Vocab::new(); // Create a vocabulary
    /// let embedding = TransformerEmbedding::from_vocab(
    ///     &vocab,
    ///     128,    // embedding_dim
    ///     512,    // max_seq_len
    ///     0.1,    // dropout_rate
    /// );
    /// ```
    pub fn from_vocab(
        vocab: &Vocab,
        embedding_dim: usize,
        max_seq_len: usize,
        dropout_rate: f32,
    ) -> Self {
        TransformerEmbedding {
            token_emb: token_embedding::TokenEmbedding::new(vocab.len(), embedding_dim),
            pos_emb: positional_embedding::PositionalEmbedding::new(max_seq_len, embedding_dim),
            embedding_dim,
            dropout_rate,
        }
    }
    
    /// Applies the forward pass to convert token IDs into embeddings with positions
    ///
    /// # Arguments
    ///
    /// * `token_ids` - A slice of token IDs
    ///
    /// # Returns
    ///
    /// A Tensor containing the combined token and positional embeddings
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::TransformerEmbedding;
    ///
    /// let embedding = TransformerEmbedding::new(1000, 64, 512, 0.1);
    /// let token_ids = vec![1, 2, 3];
    /// let output = embedding.forward(&token_ids);
    /// assert_eq!(output.data.shape(), &[3, 64]);
    /// ```
    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        // Get token embeddings
        let token_embeddings = self.token_emb.forward(token_ids);
        
        // Get positional embeddings for sequence length
        let positional_embeddings = self.pos_emb.forward(token_ids);
        
        // Sum the embeddings
        let embeddings = Tensor::add(&token_embeddings, &positional_embeddings);
        
        // Apply dropout (in a real implementation)
        // For now just return the embeddings
        // TODO: Implement dropout
        
        embeddings
    }
    
    /// Forward pass with support for batches of token IDs
    ///
    /// # Arguments
    ///
    /// * `batch_token_ids` - A slice of vectors containing token IDs, where each vector represents a sequence
    ///
    /// # Returns
    ///
    /// A Tensor with shape [batch_size, seq_len, embedding_dim]
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::TransformerEmbedding;
    ///
    /// let embedding = TransformerEmbedding::new(1000, 64, 512, 0.1);
    /// 
    /// // Batch of sequences: [batch_size=2, seq_len=3]
    /// let batch_token_ids = vec![
    ///     vec![1, 2, 3],   // first sequence
    ///     vec![4, 5, 6],   // second sequence
    /// ];
    /// 
    /// let output = embedding.forward_batch(&batch_token_ids);
    /// // Check the shape is correct
    /// assert_eq!(output.data.shape()[0], 2); // batch_size
    /// ```
    pub fn forward_batch(&self, batch_token_ids: &[Vec<usize>]) -> Tensor {
        if batch_token_ids.is_empty() {
            return Tensor::new_3d(ndarray::Array3::<f32>::zeros((0, 0, 0)));
        }
        
        let batch_size = batch_token_ids.len();
        let seq_len = batch_token_ids[0].len();
        
        // Get token embeddings with batch dimension
        let token_embeddings = self.token_emb.forward_batch(batch_token_ids);
        
        // Get positional embeddings with batch dimension
        // Ensure the positional embeddings are also 3D
        let positional_embeddings = self.pos_emb.forward_batch_3d(batch_size, seq_len);
        
        // println!("Debug: TransformerEmbedding - token_embeddings shape: {:?}", token_embeddings.data.shape());
        // println!("Debug: TransformerEmbedding - pos_embeddings shape: {:?}", positional_embeddings.data.shape());
        
        // Sum the embeddings
        let embeddings = Tensor::add(&token_embeddings, &positional_embeddings);
        
        // Apply dropout (in a real implementation)
        // For now just return the embeddings
        // TODO: Implement dropout
        
        embeddings
    }
    
    /// Sets the token embedding weights
    ///
    /// # Arguments
    /// * `embedding_matrix` - Matrix containing token embeddings
    ///
    /// # Returns
    /// * `()` - Unit return
    pub fn set_token_embedding(&mut self, embedding_matrix: Tensor) {
        let shape = embedding_matrix.data.shape();
        
        // Validate dimensions
        if shape.len() != 2 {
            println!("Warning: Expected 2D matrix for token embeddings, got {}-D", shape.len());
            return;
        }
        
        // Match expected dimensions with token embedding
        if shape[0] == self.embedding_dim && shape[1] == self.token_emb.vocab_size {
            // The matrix has dimensions [d_model x vocab_size], but token embedding expects [vocab_size x d_model]
            // We need to transpose it
            println!("Transposing embedding matrix from {}x{} to {}x{}", 
                     shape[0], shape[1], shape[1], shape[0]);
            
            let embedding_data = embedding_matrix.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            let mut transposed = ndarray::Array2::<f32>::zeros((shape[1], shape[0]));
            
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    transposed[[j, i]] = embedding_data[[i, j]];
                }
            }
            
            // The transposed matrix should match the token_emb.weights dimensions
            self.token_emb.weights = transposed;
            println!("Successfully set token embeddings");
        } else if shape[0] == self.token_emb.vocab_size && shape[1] == self.embedding_dim {
            // The dimensions match the token embedding directly
            let embedding_data = embedding_matrix.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            self.token_emb.weights = embedding_data;
            println!("Successfully set token embeddings");
        } else {
            println!("Warning: Unexpected embedding matrix dimensions: {:?}, expected: {}x{} or {}x{}",
                     shape, self.embedding_dim, self.token_emb.vocab_size, 
                     self.token_emb.vocab_size, self.embedding_dim);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transformer_embedding_creation() {
        let embedding = TransformerEmbedding::new(
            1000,   // vocab_size
            128,    // embedding_dim
            512,    // max_seq_len
            0.1,    // dropout_rate
        );
        
        assert_eq!(embedding.embedding_dim, 128);
        assert_eq!(embedding.token_emb.embedding_dim(), 128);
        assert_eq!(embedding.pos_emb.embedding_dim(), 128);
    }
    
    #[test]
    fn test_transformer_embedding_forward() {
        let embedding = TransformerEmbedding::new(
            1000,   // vocab_size
            64,     // embedding_dim
            512,    // max_seq_len
            0.1,    // dropout_rate
        );
        
        // Input sequence [seq_len=3]
        let token_ids = vec![1, 2, 3];
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Check output shape: should be [seq_len, embedding_dim]
        assert_eq!(output.data.shape(), &[3, 64]);
    }
    
    #[test]
    fn test_transformer_embedding_forward_batch() {
        let embedding = TransformerEmbedding::new(
            1000,   // vocab_size
            64,     // embedding_dim
            512,    // max_seq_len
            0.1,    // dropout_rate
        );
        
        // Batch of sequences: [batch_size=2, seq_len=3]
        let batch_token_ids = vec![
            vec![1, 2, 3],   // first sequence
            vec![4, 5, 6],   // second sequence
        ];
        
        // Forward pass with batch
        let output = embedding.forward_batch(&batch_token_ids);
        
        // Check output shape: should be [batch_size, seq_len, embedding_dim]
        assert_eq!(output.data.shape(), &[2, 3, 64]);
    }
} 