use crate::nabla::tensor::Tensor;
use super::create_weight_matrix;
use ndarray::Array2;
use rayon::prelude::*;

/// Implementation of the feed-forward network as described in the paper "Attention is All You Need"
///
/// The feed-forward network consists of two linear transformations with a ReLU function in between:
/// FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
#[allow(dead_code)]
pub struct FeedForward {
    /// Model dimension (d_model)
    d_model: usize,
    /// Inner layer dimension (d_ff), typically 4 * d_model
    d_ff: usize,
    /// Weight matrix for the first linear transformation
    w1: Tensor,
    /// Weight matrix for the second linear transformation
    w2: Tensor,
}

impl FeedForward {
    /// Creates a new feed-forward network instance
    ///
    /// # Arguments
    /// * `d_model` - Model/embedding dimension
    /// * `d_ff` - Inner layer dimension (default = 4 * d_model)
    ///
    /// # Returns
    /// A new FeedForward instance
    pub fn new(d_model: usize, d_ff: Option<usize>) -> Self {
        let d_ff = d_ff.unwrap_or(4 * d_model);
        
        // Initialize weights for the two linear transformations
        let w1 = Tensor::new(create_weight_matrix(d_model, d_ff, 0.02));
        let w2 = Tensor::new(create_weight_matrix(d_ff, d_model, 0.02));
        
        Self {
            d_model,
            d_ff,
            w1,
            w2,
        }
    }
    
    /// Performs the forward pass through the feed-forward network
    ///
    /// # Arguments
    /// * `input` - Input tensor [batch_size * seq_len, d_model]
    ///
    /// # Returns
    /// Output tensor [batch_size * seq_len, d_model]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // First linear transformation: xW₁
        let hidden = input.matmul_with(&self.w1);
        
        // Apply ReLU: max(0, xW₁)
        let activated = hidden.relu();
        
        // Second linear transformation: max(0, xW₁)W₂
        activated.matmul_with(&self.w2)
    }
    
    /// Returns the model dimension
    pub fn model_dim(&self) -> usize {
        self.d_model
    }
}

/// Implements an optimized version of the feed-forward network
/// that divides the work into sequential batches
pub struct BatchedFeedForward {
    /// Underlying feed-forward network
    ff: FeedForward,
    /// Batch size for processing
    batch_size: usize,
    /// Threshold above which to apply parallelism
    parallelism_threshold: usize,
}

impl BatchedFeedForward {
    /// Creates a new batched feed-forward network
    ///
    /// # Arguments
    /// * `d_model` - Model dimension (input and output)
    /// * `d_ff` - Inner dimension of the feed-forward network
    /// * `batch_size` - Batch size for processing
    ///
    /// # Returns
    /// A new BatchedFeedForward instance
    pub fn new(d_model: usize, d_ff: usize, batch_size: usize) -> Self {
        Self {
            ff: FeedForward::new(d_model, Some(d_ff)),
            batch_size,
            parallelism_threshold: 64, // Default threshold
        }
    }
    
    /// Forward pass with batching
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, d_model]
    ///
    /// # Returns
    /// Output tensor [seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.data.shape()[0];
        
        // For short sequences, use the normal implementation
        if seq_len < self.parallelism_threshold {
            return self.ff.forward(x);
        }
        
        // For long sequences, divide the work into parallel batches
        let batch_size = self.batch_size;
        let d_model = self.ff.d_model;
        
        // Create array of partial results for each batch
        let results: Vec<_> = (0..seq_len)
            .into_par_iter()
            .map(|i| {
                let batch_idx = i / batch_size;
                let batch_start = batch_idx * batch_size;
                let batch_end = (batch_start + batch_size).min(seq_len);
                
                // Skip batches out of range
                if batch_start >= seq_len {
                    return vec![0.0; d_model];
                }
                
                // Extract the batch
                let batch_slice = x.data.slice(ndarray::s![batch_start..batch_end, ..]).to_owned();
                let batch_input = Tensor::new(batch_slice);
                
                // Calculate output for this batch
                let batch_output = self.ff.forward(&batch_input);
                
                // Return the correct row
                let local_i = i - batch_start;
                if local_i < batch_output.data.shape()[0] {
                    batch_output.data.slice(ndarray::s![local_i, ..]).to_vec()
                } else {
                    vec![0.0; d_model] // this should never happen
                }
            })
            .collect();
        
        // Reconstruct the output tensor
        let mut output_data = Array2::zeros((seq_len, d_model));
        for (i, row) in results.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                output_data[[i, j]] = val;
            }
        }
        
        Tensor::new(output_data)
    }
    
    /// Returns the model dimension
    pub fn model_dim(&self) -> usize {
        self.ff.model_dim()
    }
    
    /// Sets a new threshold for parallelism
    ///
    /// # Arguments
    /// * `threshold` - The new threshold
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
        
        // Test with default d_ff (4 * d_model)
        let ff_default = FeedForward::new(d_model, None);
        assert_eq!(ff_default.d_model, d_model);
        assert_eq!(ff_default.d_ff, 4 * d_model);
        assert_eq!(ff_default.w1.data.shape(), &[d_model, 4 * d_model]);
        assert_eq!(ff_default.w2.data.shape(), &[4 * d_model, d_model]);
        
        // Test with custom d_ff
        let custom_d_ff = 128;
        let ff_custom = FeedForward::new(d_model, Some(custom_d_ff));
        assert_eq!(ff_custom.d_model, d_model);
        assert_eq!(ff_custom.d_ff, custom_d_ff);
        assert_eq!(ff_custom.w1.data.shape(), &[d_model, custom_d_ff]);
        assert_eq!(ff_custom.w2.data.shape(), &[custom_d_ff, d_model]);
        
        // Verify that model_dim returns the correct value
        assert_eq!(ff_default.model_dim(), d_model);
    }
    
    #[test]
    fn test_feed_forward_forward() {
        let d_model = 64;
        let seq_len = 10;
        let batch_size = 2;
        
        let ff = FeedForward::new(d_model, None);
        
        // Create a test input
        let input_data = Array2::ones((batch_size * seq_len, d_model));
        let input = Tensor::new(input_data);
        
        // Forward pass
        let output = ff.forward(&input);
        
        // Verify that the output has the correct shape
        assert_eq!(output.data.shape(), &[batch_size * seq_len, d_model]);
        
        // Verify that the output is not all zero
        assert!(output.data.sum() != 0.0);
    }
    
    #[test]
    fn test_feed_forward_relu_activation() {
        let d_model = 4;
        let d_ff = 8;
        
        let ff = FeedForward::new(d_model, Some(d_ff));
        
        // Create an input with some negative values
        let mut input_data = Array2::ones((1, d_model));
        input_data[[0, 0]] = -1.0; // Set a negative value
        
        let input = Tensor::new(input_data);
        
        // Forward pass
        let hidden = input.matmul_with(&ff.w1);
        
        // Verify that there are negative values in the output of the first transformation
        let has_negative = hidden.data.iter().any(|&x| x < 0.0);
        
        // Activate with ReLU
        let activated = hidden.relu();
        
        // Verify that there are no negative values after ReLU
        let all_non_negative = activated.data.iter().all(|&x| x >= 0.0);
        
        assert!(has_negative, "The output of the first transformation should have negative values");
        assert!(all_non_negative, "After ReLU, all values should be non-negative");
    }
    
    #[test]
    fn test_batched_feed_forward() {
        let d_model = 64;
        let d_ff = 128;
        let batch_size = 8;
        let seq_len = 100;
        
        let mut batched_ff = BatchedFeedForward::new(d_model, d_ff, batch_size);
        batched_ff.set_parallelism_threshold(32); // Set a low threshold for testing
        
        // Create a test input
        let input_data = Array2::ones((seq_len, d_model));
        let input = Tensor::new(input_data);
        
        // Forward pass
        let output = batched_ff.forward(&input);
        
        // Verify that the output has the correct shape
        assert_eq!(output.data.shape(), &[seq_len, d_model]);
    }
} 