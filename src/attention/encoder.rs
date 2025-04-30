use ndarray::{Array, Array2, Array3, Axis};
use crate::nabla::tensor::Tensor;
use super::{Attention, MultiHeadAttention, FeedForward};

/// Implements Layer Normalization as described in the paper "Attention is All You Need"
/// 
/// # Arguments
/// 
/// * `x` - Input tensor
/// * `eps` - Epsilon for numerical stability
/// 
/// # Returns
/// 
/// The normalized tensor
pub fn layer_norm(x: &Tensor, eps: f32) -> Tensor {
    // Get tensor dimensions
    let shape = x.data.shape();
    
    // Check if tensor is 2D or 3D
    let dimensionality = shape.len();
    
    if dimensionality == 2 {
        // Implement Layer Normalization along the last dimension for 2D tensors
        let last_dim = shape[1];
        
        // Prepare result tensor
        let mut norm_data = x.data.clone();
        
        // For each row in the 2D tensor
        for i in 0..shape[0] {
            // Calculate mean for this row
            let mut sum = 0.0;
            for j in 0..last_dim {
                sum += x.data[[i, j]];
            }
            let mean = sum / (last_dim as f32);
            
            // Calculate variance for this row
            let mut variance = 0.0;
            for j in 0..last_dim {
                variance += (x.data[[i, j]] - mean).powi(2);
            }
            variance /= last_dim as f32;
            
            // Normalize this row
            for j in 0..last_dim {
                norm_data[[i, j]] = (x.data[[i, j]] - mean) / (variance + eps).sqrt();
            }
        }
        
        return Tensor::new(norm_data.into_dimensionality::<ndarray::Ix2>().unwrap());
    } else if dimensionality == 3 {
        // Implement Layer Normalization along the last dimension for 3D tensors
        let last_dim = shape[2];
        
        // Prepare result tensor
        let mut norm_data = x.data.clone();
        
        // For each batch and for each row in the 3D tensor
        for b in 0..shape[0] {
            for i in 0..shape[1] {
                // Calculate mean for this slice
                let mut sum = 0.0;
                for j in 0..last_dim {
                    sum += x.data[[b, i, j]];
                }
                let mean = sum / (last_dim as f32);
                
                // Calculate variance for this slice
                let mut variance = 0.0;
                for j in 0..last_dim {
                    variance += (x.data[[b, i, j]] - mean).powi(2);
                }
                variance /= last_dim as f32;
                
                // Normalize this slice
                for j in 0..last_dim {
                    norm_data[[b, i, j]] = (x.data[[b, i, j]] - mean) / (variance + eps).sqrt();
                }
            }
        }
        
        return Tensor::new_3d(norm_data.into_dimensionality::<ndarray::Ix3>().unwrap());
    } else {
        panic!("layer_norm only supports 2D or 3D tensors, received: {}-D", dimensionality);
    }
}

/// Implementation of an Encoder Layer as described in the paper "Attention is All You Need"
pub struct EncoderLayer {
    /// Multi-head attention
    attention: MultiHeadAttention,
    /// Feed-forward network
    feed_forward: FeedForward,
    /// Model dimension
    d_model: usize,
    /// Epsilon for layer normalization
    eps: f32,
    /// Dropout rate
    dropout_rate: f32,
}

impl EncoderLayer {
    /// Creates a new encoder layer
    /// 
    /// # Arguments
    /// 
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `d_ff` - Feed-forward layer dimension (default: 4 * d_model)
    /// * `dropout_rate` - Dropout rate
    /// * `eps` - Epsilon for layer normalization
    /// 
    /// # Returns
    /// 
    /// A new encoder layer
    pub fn new(d_model: usize, num_heads: usize, d_ff: Option<usize>, dropout_rate: f32, eps: f32) -> Self {
        let d_ff = d_ff.unwrap_or(4 * d_model);
        
        EncoderLayer {
            attention: MultiHeadAttention::new(d_model, num_heads, 0.1),
            feed_forward: FeedForward::new(d_model, Some(d_ff)),
            d_model,
            eps,
            dropout_rate,
        }
    }
    
    /// Forward pass through the encoder layer
    /// 
    /// # Arguments
    /// 
    /// * `x` - Input tensor [batch_size, seq_len, d_model]
    /// * `mask` - Attention mask (optional)
    /// 
    /// # Returns
    /// 
    /// Output tensor [batch_size, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // 1. Layer Normalization before attention
        let norm1 = layer_norm(x, self.eps);
        
        // 2. Multi-head attention
        let attn_output = self.attention.forward(&norm1, &norm1, &norm1, mask);
        
        // 3. Residual connection with input
        let residual1 = Tensor::add(x, &attn_output);
        
        // 4. Layer Normalization before feed-forward
        let norm2 = layer_norm(&residual1, self.eps);
        
        // 5. Feed-forward network
        // We need to reshape the 3D tensor to 2D for the feed-forward
        let shape = norm2.data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        // Reshape from [batch, seq_len, d_model] to [batch*seq_len, d_model]
        let reshaped_data = norm2.data.clone()
            .into_dimensionality::<ndarray::Ix3>().unwrap()
            .into_shape((batch_size * seq_len, self.d_model)).unwrap();
        
        let reshaped_tensor = Tensor::new(reshaped_data.into_dimensionality::<ndarray::Ix2>().unwrap());
        
        // Forward pass feed-forward
        let ff_output = self.feed_forward.forward(&reshaped_tensor);
        
        // Reshape back from [batch*seq_len, d_model] to [batch, seq_len, d_model]
        let ff_output_reshaped = ff_output.data.clone()
            .into_shape((batch_size, seq_len, self.d_model)).unwrap();
        
        let ff_output_tensor = Tensor::new_3d(ff_output_reshaped);
        
        // 6. Residual connection with attention output
        Tensor::add(&residual1, &ff_output_tensor)
    }
    
    /// Returns the model dimension
    pub fn model_dim(&self) -> usize {
        self.d_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_encoder_layer_creation() {
        let d_model = 64;
        let num_heads = 4;
        let dropout_rate = 0.1;
        let eps = 1e-6;
        
        let encoder = EncoderLayer::new(d_model, num_heads, None, dropout_rate, eps);
        
        assert_eq!(encoder.d_model, d_model);
        assert_eq!(encoder.dropout_rate, dropout_rate);
        assert_eq!(encoder.eps, eps);
    }
    
    #[test]
    fn test_encoder_layer_forward() {
        let d_model = 64;
        let num_heads = 4;
        let seq_len = 5;
        let batch_size = 2;
        
        let encoder = EncoderLayer::new(d_model, num_heads, None, 0.1, 1e-6);
        
        // Create test input
        let x_data = Array3::<f32>::zeros((batch_size, seq_len, d_model));
        let x = Tensor::new_3d(x_data);
        
        // Forward pass without mask
        let output = encoder.forward(&x, None);
        
        // Verify output dimensions
        let output_shape = output.data.shape();
        assert_eq!(output_shape[0], batch_size);
        assert_eq!(output_shape[1], seq_len);
        assert_eq!(output_shape[2], d_model);
    }
    
    #[test]
    fn test_layer_norm() {
        // Create a test tensor with known values
        let data = Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Tensor::new(data);
        
        // Normalize
        let normalized = layer_norm(&x, 1e-6);
        
        // Calculate mean and variance of the output
        let mean: f32 = normalized.data.mean().unwrap();
        
        // Mean should be approximately zero
        assert!(mean.abs() < 1e-5, "Mean should be zero, but is {}", mean);
        
        // Variance should be approximately 1
        let mut variance = 0.0;
        for &val in normalized.data.iter() {
            variance += (val - mean).powi(2);
        }
        variance /= normalized.data.len() as f32;
        
        assert!((variance - 1.0).abs() < 1e-5, "Variance should be 1, but is {}", variance);
    }
    
    #[test]
    fn test_encoder_layer_with_mask() {
        let d_model = 64;
        let num_heads = 4;
        let seq_len = 5;
        let batch_size = 2;
        
        let encoder = EncoderLayer::new(d_model, num_heads, None, 0.1, 1e-6);
        
        // Create input data with a pattern that makes the causal mask effect evident
        let mut x_data = Array3::<f32>::zeros((batch_size, seq_len, d_model));
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..d_model {
                    // Use a pattern that creates dependencies between future and previous positions
                    // Early positions (1, 2) have small values
                    // Later positions (3, 4, 5) have large values
                    // This will make the effect of the causal mask more evident
                    if i < 2 {
                        x_data[[b, i, j]] = 0.01 * (i + 1) as f32;
                    } else {
                        x_data[[b, i, j]] = 10.0 * (i + 1) as f32; // Much larger values in future positions
                    }
                }
            }
        }
        
        let x = Tensor::new_3d(x_data);
        
        // Create a 3D mask [batch_size, seq_len, seq_len]
        let mut mask_data = Array3::<f32>::zeros((batch_size, seq_len, seq_len));
        
        // Lower triangular mask (causal) for each batch
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..=i {  // j <= i (lower triangular)
                    mask_data[[b, i, j]] = 1.0;
                }
            }
        }
        
        let mask = Tensor::new_3d(mask_data);
        
        // Forward pass with mask
        let output_with_mask = encoder.forward(&x, Some(&mask));
        
        // Forward pass without mask
        let output_no_mask = encoder.forward(&x, None);
        
        // The two outputs should be different
        let mut all_equal = true;
        let output_with_mask_data = output_with_mask.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        let output_no_mask_data = output_no_mask.data.clone().into_dimensionality::<ndarray::Ix3>().unwrap();
        
        let mut diff_count = 0;
        let mut max_diff = 0.0;
        
        for ((b, i, j), &v1) in output_with_mask_data.indexed_iter() {
            let v2 = output_no_mask_data[[b, i, j]];
            let diff = (v1 - v2).abs();
            if diff > 1e-5 {
                all_equal = false;
                diff_count += 1;
                max_diff = f32::max(max_diff, diff);
            }
        }
        
        // Print debug information
        println!("Differences found in encoder test: {}, max diff: {}", diff_count, max_diff);
        
        // We expect that the output with mask is different from the output without mask
        // If the test fails, it means the mask is not having any effect
        assert!(!all_equal, "Output with mask should be different from output without mask. 
                             This could happen if the mask is not being applied correctly
                             or if the input values are such that the mask makes no difference.");
    }
} 