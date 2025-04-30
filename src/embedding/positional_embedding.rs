use crate::nabla::tensor::Tensor;
use ndarray::{Array, Array2, Axis};
use std::f32::consts::PI;
use rayon::prelude::*;

use super::Embedding;

/// Positional embedding that adds position information to tokens.
///
/// This struct implements the positional encoding described in the paper 
/// "Attention Is All You Need" by Vaswani et al. Positional embeddings
/// enable transformer models to incorporate sequence order information
/// despite their inherently parallelized architecture.
///
/// The positional encoding uses sine and cosine functions of different frequencies:
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// # Examples
///
/// ```
/// use wall_e1::embedding::positional_embedding::PositionalEmbedding;
///
/// // Create a positional embedding for sequences up to length 512 with dimension 64
/// let embedding = PositionalEmbedding::new(512, 64);
///
/// // Get positional embeddings for a sequence of 10 tokens
/// let token_ids = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let positional_embeddings = embedding.forward(&token_ids);
/// ```
#[derive(Clone)]
pub struct PositionalEmbedding {
    /// Positional embedding matrix: [max_len, d_model]
    embedding_matrix: Tensor,
    /// Maximum sequence length supported
    max_len: usize,
    /// Dimensionality of the embeddings
    embedding_dim: usize,
}

impl PositionalEmbedding {
    /// Creates a new positional embedding using the sinusoidal function.
    ///
    /// Initializes the positional embedding matrix according to the formula from
    /// "Attention Is All You Need" paper:
    /// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    ///
    /// This implementation uses parallel processing to efficiently compute the values.
    ///
    /// # Arguments
    ///
    /// * `max_len` - Maximum sequence length that will be supported
    /// * `embedding_dim` - Dimensionality of the embedding vectors
    ///
    /// # Returns
    ///
    /// A new `PositionalEmbedding` instance with a pre-computed embedding matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::positional_embedding::PositionalEmbedding;
    ///
    /// // Create a positional embedding for sequences up to length 512 with dimension 64
    /// let embedding = PositionalEmbedding::new(512, 64);
    /// ```
    pub fn new(max_len: usize, embedding_dim: usize) -> Self {
        // We use the formula from the paper:
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        let mut embedding_data = Array::zeros((max_len, embedding_dim));
        
        // Create a vector of all positions and dimensions
        let indices: Vec<(usize, usize)> = (0..max_len)
            .flat_map(|pos| (0..embedding_dim/2).map(move |i| (pos, i)))
            .collect();
        
        // Calculate values in parallel
        let results: Vec<(usize, usize, f32, f32)> = indices.par_iter()
            .map(|&(pos, i)| {
                let denominator = 10000_f32.powf(2.0 * i as f32 / embedding_dim as f32);
                let angle = pos as f32 / denominator;
                let sin_val = angle.sin();
                let cos_val = angle.cos();
                (pos, i, sin_val, cos_val)
            })
            .collect();
        
        // Assign calculated values to the matrix
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
    
    /// Returns the underlying embedding matrix.
    ///
    /// # Returns
    ///
    /// A reference to the pre-computed positional embedding matrix tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::positional_embedding::PositionalEmbedding;
    ///
    /// let embedding = PositionalEmbedding::new(512, 64);
    /// let matrix = embedding.embedding_matrix();
    /// assert_eq!(matrix.data.shape(), &[512, 64]);
    /// ```
    pub fn embedding_matrix(&self) -> &Tensor {
        &self.embedding_matrix
    }

    /// Performs forward pass with batch support, returning a 2D tensor.
    ///
    /// This method takes a batch size and sequence length as input and returns
    /// positional embeddings for the entire batch in a flattened 2D format.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of sequences in the batch
    /// * `seq_len` - Length of each sequence
    ///
    /// # Returns
    ///
    /// A tensor with shape [batch_size, seq_len * embedding_dim] containing
    /// positional embeddings for each position in each sequence of the batch.
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::positional_embedding::PositionalEmbedding;
    ///
    /// let embedding = PositionalEmbedding::new(512, 64);
    /// let batch_size = 2;
    /// let seq_len = 10;
    /// let batch_embeddings = embedding.forward_batch(batch_size, seq_len);
    /// assert_eq!(batch_embeddings.data.shape(), &[batch_size, seq_len * 64]);
    /// ```
    pub fn forward_batch(&self, batch_size: usize, seq_len: usize) -> Tensor {
        let effective_len = std::cmp::min(seq_len, self.max_len);
        
        // Get the standard positional embeddings
        let pos_embeddings = self.embedding_matrix.data.slice(
            ndarray::s![0..effective_len, ..],
        ).to_owned();
        
        // Create a 3D matrix to store results [batch_size, seq_len, embedding_dim]
        let mut result_data = ndarray::Array3::<f32>::zeros((batch_size, effective_len, self.embedding_dim));
        
        // Replicate the same positional embeddings for each batch element
        // This method is sequential, but works with all Tensor implementations
        for b in 0..batch_size {
            for i in 0..effective_len {
                for j in 0..self.embedding_dim {
                    result_data[[b, i, j]] = pos_embeddings[[i, j]];
                }
            }
        }
        
        // Convert the 3D tensor to a 2D tensor with shape [batch_size, seq_len * embedding_dim]
        let flattened = result_data.into_shape((batch_size, effective_len * self.embedding_dim)).unwrap();
        Tensor::new(flattened)
    }

    /// Performs forward pass with batch support, returning a 3D tensor.
    ///
    /// This method takes a batch size and sequence length as input and returns
    /// positional embeddings for the entire batch in a 3D format, which is more
    /// suitable for operations like attention that need to preserve the sequence
    /// dimension.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of sequences in the batch
    /// * `seq_len` - Length of each sequence
    ///
    /// # Returns
    ///
    /// A tensor with shape [batch_size, seq_len, embedding_dim] containing
    /// positional embeddings for each position in each sequence of the batch.
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::positional_embedding::PositionalEmbedding;
    ///
    /// let embedding = PositionalEmbedding::new(512, 64);
    /// let batch_size = 2;
    /// let seq_len = 10;
    /// let batch_embeddings = embedding.forward_batch_3d(batch_size, seq_len);
    /// assert_eq!(batch_embeddings.data.shape(), &[batch_size, seq_len, 64]);
    /// ```
    pub fn forward_batch_3d(&self, batch_size: usize, seq_len: usize) -> Tensor {
        let effective_len = std::cmp::min(seq_len, self.max_len);
        
        // Get the standard positional embeddings
        let pos_embeddings = self.embedding_matrix.data.slice(
            ndarray::s![0..effective_len, ..],
        ).to_owned();
        
        // Create a 3D matrix to store results [batch_size, seq_len, embedding_dim]
        let mut result_data = ndarray::Array3::<f32>::zeros((batch_size, effective_len, self.embedding_dim));
        
        // Replicate the same positional embeddings for each batch element
        for b in 0..batch_size {
            for i in 0..effective_len {
                for j in 0..self.embedding_dim {
                    result_data[[b, i, j]] = pos_embeddings[[i, j]];
                }
            }
        }
        
        // Return the 3D tensor directly
        Tensor::new_3d(result_data)
    }
}

impl Embedding for PositionalEmbedding {
    /// Performs the forward pass to generate positional embeddings for a sequence.
    ///
    /// This method implements the `Embedding` trait's forward method. It extracts
    /// positional embeddings from the pre-computed matrix based on the length of
    /// the input sequence.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - A slice of token IDs. The actual values are not used, only the
    ///   length matters as positional embeddings depend solely on position.
    ///
    /// # Returns
    ///
    /// A tensor with shape [seq_len, embedding_dim] containing positional embeddings
    /// for each position in the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::embedding::{Embedding, positional_embedding::PositionalEmbedding};
    ///
    /// let embedding = PositionalEmbedding::new(512, 64);
    /// let token_ids = vec![1, 2, 3, 4, 5]; // 5 tokens
    /// let embeddings = embedding.forward(&token_ids);
    /// assert_eq!(embeddings.data.shape(), &[5, 64]);
    /// ```
    fn forward(&self, token_ids: &[usize]) -> Tensor {
        let seq_len = token_ids.len();
        let effective_len = std::cmp::min(seq_len, self.max_len);
        
        // Takes the first `effective_len` rows from the embedding matrix
        let slice = self.embedding_matrix.data.slice(
            ndarray::s![0..effective_len, ..],
        );
        
        // If the sequence is shorter than max_len, only take the first elements
        let result_data = slice.to_owned();
        
        Tensor::new(result_data)
    }
    
    /// Returns the embedding dimension.
    ///
    /// # Returns
    ///
    /// The dimension of the embedding vectors.
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
        
        // Test some specific values to verify the formula
        // For position=4, dim=0 (sin)
        let pos = 4;
        let dim = 0;
        let i = dim / 2; // i = 0 for dim = 0
        let denominator = 10000_f32.powf(2.0 * i as f32 / 8.0);
        let expected = (pos as f32 / denominator).sin();
        assert!((matrix[[pos, dim]] - expected).abs() < 1e-5);
        
        // For position=4, dim=1 (cos)
        let dim = 1;
        let i = dim / 2; // i = 0 for dim = 1 (first cosine)
        let denominator = 10000_f32.powf(2.0 * i as f32 / 8.0);
        let expected = (pos as f32 / denominator).cos();
        assert!((matrix[[pos, dim]] - expected).abs() < 1e-5);
    }
    
    #[test]
    fn test_positional_embedding_forward() {
        let embedding = PositionalEmbedding::new(10, 64);
        
        // Token IDs (not used, but required for the interface)
        let token_ids = vec![0, 0, 0, 0, 0]; // 5 tokens
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Verify shape: [5, 64]
        assert_eq!(output.data.shape(), &[5, 64]);
        
        // Verify that values match the embedding matrix
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
        // Create an embedding with small max_len
        let embedding = PositionalEmbedding::new(5, 16);
        
        // Try with a sequence longer than max_len
        let token_ids = vec![0; 10]; // 10 tokens
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Should truncate to max_len (5)
        assert_eq!(output.data.shape(), &[5, 16]);
        
        // Verify that values match the embedding matrix
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
        
        // Batch parameters
        let batch_size = 2;
        let seq_len = 4;
        
        // Forward pass with batch
        let output = embedding.forward_batch(batch_size, seq_len);
        
        // Verify shape [batch_size, seq_len * embedding_dim]
        assert_eq!(output.data.shape(), &[2, 4 * 32]);
        
        // Verify that positional embeddings were replicated for each batch
        // The embeddings of the first and second sequence in the batch should be identical
        // because positional embeddings depend only on position, not on the batch
        for j in 0..seq_len * 32 {
            assert_eq!(
                output.data[[0, j]],  // first batch
                output.data[[1, j]]   // second batch
            );
        }
        
        // Verify that values match the original embedding matrix
        for i in 0..seq_len {
            for j in 0..32 {
                assert_eq!(
                    embedding.embedding_matrix().data[[i, j]],
                    output.data[[0, i * 32 + j]]  // in flattened format
                );
            }
        }
    }
} 