use ndarray::{Array, Array2, Array3, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use crate::nabla::tensor::Tensor;

// Export modules
pub mod self_attention;
pub mod multi_head_attention;
pub mod feed_forward;
pub mod encoder;
pub mod encoder_stack;

// Export structures
pub use self_attention::SelfAttention;
pub use multi_head_attention::MultiHeadAttention;
pub use feed_forward::{FeedForward, BatchedFeedForward};
pub use encoder::EncoderLayer;
pub use encoder_stack::EncoderStack;
pub use encoder::layer_norm;

/// Trait that defines a common interface for attention mechanisms
pub trait Attention {
    /// Performs the forward pass of the attention mechanism
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor;
    
    /// Returns the model dimension
    fn model_dim(&self) -> usize;
}

/// Creates a causal mask to prevent attention to future positions
/// The mask has 1 on the diagonal and below, 0 above the diagonal
/// 
/// # Arguments
/// * `seq_len` - Sequence length
/// 
/// # Returns
/// Tensor [1, seq_len, seq_len] containing the causal mask
pub fn create_causal_mask(seq_len: usize) -> Tensor {
    let mut mask_data = Array3::zeros((1, seq_len, seq_len));
    
    // Set 1 on the diagonal and below (lower triangular)
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[[0, i, j]] = 1.0;
        }
    }
    
    Tensor::new_3d(mask_data)
}

/// Creates a padding mask to ignore padding tokens
/// 
/// # Arguments
/// * `seq_len` - Maximum sequence length
/// * `valid_lens` - Vector with valid lengths for each sequence in the batch
/// 
/// # Returns
/// Tensor [batch_size, seq_len, seq_len] containing masks for each sequence
pub fn create_padding_mask(seq_len: usize, valid_lens: &[usize]) -> Tensor {
    let batch_size = valid_lens.len();
    let mut mask_data = Array3::zeros((batch_size, seq_len, seq_len));
    
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        for i in 0..seq_len {
            // If i is a valid position, allow attending up to valid_len
            if i < valid_len {
                for j in 0..valid_len {
                    mask_data[[b, i, j]] = 1.0;
                }
            }
            // Otherwise, don't attend to any position (row of all 0s)
        }
    }
    
    Tensor::new_3d(mask_data)
}

/// Combines a causal mask with a padding mask
/// Useful for decoders with padding
/// 
/// # Arguments
/// * `seq_len` - Maximum sequence length
/// * `valid_lens` - Vector with valid lengths for each sequence in the batch
/// 
/// # Returns
/// Tensor [batch_size, seq_len, seq_len] with the combined mask
pub fn create_combined_mask(seq_len: usize, valid_lens: &[usize]) -> Tensor {
    let batch_size = valid_lens.len();
    let mut mask_data = Array3::zeros((batch_size, seq_len, seq_len));
    
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        for i in 0..seq_len {
            // If i is a valid position
            if i < valid_len {
                // Apply both causal and padding masks
                for j in 0..=i {
                    if j < valid_len {
                        mask_data[[b, i, j]] = 1.0;
                    }
                }
            }
            // Otherwise, don't attend to any position
        }
    }
    
    Tensor::new_3d(mask_data)
}

/// Creates a weight matrix using a normal distribution
/// 
/// # Arguments
/// 
/// * `in_features` - Number of input features
/// * `out_features` - Number of output features
/// * `std` - Standard deviation for initialization
/// 
/// # Returns
/// 
/// * A weight matrix initialized with normal distribution
pub fn create_weight_matrix(in_features: usize, out_features: usize, std: f32) -> Array2<f32> {
    Array::random((in_features, out_features), Normal::new(0.0, std as f64).unwrap())
        .mapv(|x| x as f32)
}

/// Computes 3D softmax along a specific axis
/// 
/// # Arguments
/// 
/// * `x` - 3D input tensor
/// * `axis` - Axis along which to compute softmax (0, 1, or 2)
/// 
/// # Returns
/// 
/// A new 3D tensor with softmax applied
pub fn softmax_3d(x: &Array3<f32>, axis: usize) -> Array3<f32> {
    assert!(axis <= 2, "Axis must be 0, 1, or 2");
    
    let shape = x.shape();
    let mut result = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    
    // Very negative but finite value to replace -infinity
    let very_negative_value = -1e30f32;
    
    // Apply softmax based on the specified axis
    if axis == 0 {
        // Softmax along axis 0 (batch)
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Extract the slice
                let mut slice = Vec::with_capacity(shape[0]);
                for i in 0..shape[0] {
                    // Replace -infinity with a very negative but finite value
                    if x[[i, j, k]].is_infinite() && x[[i, j, k]] < 0.0 {
                        slice.push(very_negative_value);
                    } else {
                        slice.push(x[[i, j, k]]);
                    }
                }
                
                // Find the maximum value for numerical stability
                let max_val = slice.iter().fold(f32::MIN, |a, &b| a.max(b));
                
                // Calculate exp(x - max) for each element
                let mut exp_vals = Vec::with_capacity(shape[0]);
                for val in slice.iter() {
                    exp_vals.push((*val - max_val).exp());
                }
                
                // Calculate sum for normalization
                let sum: f32 = exp_vals.iter().sum();
                
                // Handle the case where sum is close to zero
                let safe_sum = if sum < 1e-10 {
                    // If sum is almost zero, distribute uniformly
                    for i in 0..shape[0] {
                        result[[i, j, k]] = 1.0 / (shape[0] as f32);
                    }
                    continue;
                } else {
                    sum
                };
                
                // Apply normalization
                for i in 0..shape[0] {
                    result[[i, j, k]] = exp_vals[i] / safe_sum;
                }
                
                // Verify and correct any numerical issues to ensure sum = 1
                let actual_sum: f32 = (0..shape[0]).map(|i| result[[i, j, k]]).sum();
                if (actual_sum - 1.0).abs() > 1e-5 {
                    // Adjust the first value to ensure sum = 1
                    result[[0, j, k]] += 1.0 - actual_sum;
                }
            }
        }
    } else if axis == 1 {
        // Softmax along axis 1 (sequence)
        for i in 0..shape[0] {
            for k in 0..shape[2] {
                // Extract the slice
                let mut slice = Vec::with_capacity(shape[1]);
                for j in 0..shape[1] {
                    // Replace -infinity with a very negative but finite value
                    if x[[i, j, k]].is_infinite() && x[[i, j, k]] < 0.0 {
                        slice.push(very_negative_value);
                    } else {
                        slice.push(x[[i, j, k]]);
                    }
                }
                
                // Find the maximum value for numerical stability
                let max_val = slice.iter().fold(f32::MIN, |a, &b| a.max(b));
                
                // Calculate exp(x - max) for each element
                let mut exp_vals = Vec::with_capacity(shape[1]);
                for val in slice.iter() {
                    exp_vals.push((*val - max_val).exp());
                }
                
                // Calculate sum for normalization
                let sum: f32 = exp_vals.iter().sum();
                
                // Handle the case where sum is close to zero
                let safe_sum = if sum < 1e-10 {
                    // If sum is almost zero, distribute uniformly
                    for j in 0..shape[1] {
                        result[[i, j, k]] = 1.0 / (shape[1] as f32);
                    }
                    continue;
                } else {
                    sum
                };
                
                // Apply normalization
                for j in 0..shape[1] {
                    result[[i, j, k]] = exp_vals[j] / safe_sum;
                }
                
                // Verify and correct any numerical issues to ensure sum = 1
                let actual_sum: f32 = (0..shape[1]).map(|j| result[[i, j, k]]).sum();
                if (actual_sum - 1.0).abs() > 1e-5 {
                    // Adjust the first value to ensure sum = 1
                    result[[i, 0, k]] += 1.0 - actual_sum;
                }
            }
        }
    } else if axis == 2 {
        // Softmax along axis 2 (feature)
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                // Extract the slice
                let mut slice = Vec::with_capacity(shape[2]);
                for k in 0..shape[2] {
                    // Replace -infinity with a very negative but finite value
                    if x[[i, j, k]].is_infinite() && x[[i, j, k]] < 0.0 {
                        slice.push(very_negative_value);
                    } else {
                        slice.push(x[[i, j, k]]);
                    }
                }
                
                // Find the maximum value for numerical stability
                let max_val = slice.iter().fold(f32::MIN, |a, &b| a.max(b));
                
                // Calculate exp(x - max) for each element
                let mut exp_vals = Vec::with_capacity(shape[2]);
                for val in slice.iter() {
                    exp_vals.push((*val - max_val).exp());
                }
                
                // Calculate sum for normalization
                let sum: f32 = exp_vals.iter().sum();
                
                // Handle the case where sum is close to zero
                let safe_sum = if sum < 1e-10 {
                    // If sum is almost zero, distribute uniformly
                    for k in 0..shape[2] {
                        result[[i, j, k]] = 1.0 / (shape[2] as f32);
                    }
                    continue;
                } else {
                    sum
                };
                
                // Apply normalization
                for k in 0..shape[2] {
                    result[[i, j, k]] = exp_vals[k] / safe_sum;
                }
                
                // Verify and correct any numerical issues to ensure sum = 1
                let actual_sum: f32 = (0..shape[2]).map(|k| result[[i, j, k]]).sum();
                if (actual_sum - 1.0).abs() > 1e-5 {
                    // Adjust the first value to ensure sum = 1
                    result[[i, j, 0]] += 1.0 - actual_sum;
                }
            }
        }
    }
    
    // Final check to ensure there are no NaN values
    for val in result.iter_mut() {
        if val.is_nan() {
            *val = 0.0; // Replace NaN with 0
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_weight_matrix() {
        let w = create_weight_matrix(10, 20, 0.1);
        
        // Verify dimensions
        assert_eq!(w.shape(), &[10, 20]);
        
        // Verify that the mean is approximately zero
        let mean = w.mean().unwrap();
        assert!(mean.abs() < 0.1);
        
        // Verify that the standard deviation is approximately 0.1
        let std_dev = (w.mapv(|x| x.powi(2)).sum() / (w.len() as f32)).sqrt();
        assert!((std_dev - 0.1).abs() < 0.05);
        
        // Verify that values are distributed uniformly around zero
        let positive_count = w.iter().filter(|&&x| x > 0.0).count();
        let total_count = w.len();
        let positive_ratio = positive_count as f32 / total_count as f32;
        assert!((positive_ratio - 0.5).abs() < 0.1, "Expected uniform distribution, found {:.2}% positive values", positive_ratio * 100.0);
        
        // Verify that there are no NaN or infinite values
        for &val in w.iter() {
            assert!(!val.is_nan() && !val.is_infinite(), "Found invalid value in weight matrix");
        }
    }
    
    #[test]
    fn test_softmax_3d() {
        // Create a sample 3D tensor
        let x = Array3::from_shape_vec((2, 3, 4), 
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0,
                21.0, 22.0, 23.0, 24.0,
            ]
        ).unwrap();
        
        // Calculate softmax along axis 2
        let result = softmax_3d(&x, 2);
        
        // Verify that sums are 1.0 for each slice
        for i in 0..2 {
            for j in 0..3 {
                let sum: f32 = result.slice(s![i, j, ..]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "Sum = {} for slice [{}][{}], should be 1.0", sum, i, j);
            }
        }
        
        // Verify that all values are positive
        for v in result.iter() {
            assert!(*v > 0.0, "Softmax value must be positive, found: {}", v);
        }
        
        // Test with softmax along axis 0
        let result_axis0 = softmax_3d(&x, 0);
        
        // Verify sums along axis 0
        for j in 0..3 {
            for k in 0..4 {
                let sum: f32 = result_axis0.slice(s![.., j, k]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "Sum = {} for slice [*][{}][{}], should be 1.0", sum, j, k);
            }
        }
        
        // Test with softmax along axis 1
        let result_axis1 = softmax_3d(&x, 1);
        
        // Verify sums along axis 1
        for i in 0..2 {
            for k in 0..4 {
                let sum: f32 = result_axis1.slice(s![i, .., k]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "Sum = {} for slice [{}][*][{}], should be 1.0", sum, i, k);
            }
        }
        
        // Test with extreme values (very large)
        let mut large_vals = Array3::<f32>::zeros((2, 2, 2));
        large_vals[[0, 0, 0]] = 1000.0;
        large_vals[[0, 0, 1]] = 0.0;
        large_vals[[0, 1, 0]] = 0.0;
        large_vals[[0, 1, 1]] = 1000.0;
        large_vals[[1, 0, 0]] = 0.0;
        large_vals[[1, 0, 1]] = 1000.0;
        large_vals[[1, 1, 0]] = 1000.0;
        large_vals[[1, 1, 1]] = 0.0;
        
        let result_large = softmax_3d(&large_vals, 2);
        
        // For extreme values, the result should be almost 0-1
        for i in 0..2 {
            for j in 0..2 {
                let max_idx = if large_vals[[i, j, 0]] > large_vals[[i, j, 1]] { 0 } else { 1 };
                let min_idx = 1 - max_idx;
                assert!(result_large[[i, j, max_idx]] > 0.99, 
                        "For extreme values, softmax should be close to 1.0 for the maximum value");
                assert!(result_large[[i, j, min_idx]] < 0.01, 
                        "For extreme values, softmax should be close to 0.0 for the minimum value");
            }
        }
        
        // Test with -infinity (mask simulation)
        let mut mask_test = Array3::<f32>::zeros((2, 2, 2));
        mask_test[[0, 0, 0]] = 1.0;
        mask_test[[0, 0, 1]] = std::f32::NEG_INFINITY;
        mask_test[[0, 1, 0]] = std::f32::NEG_INFINITY;
        mask_test[[0, 1, 1]] = 1.0;
        mask_test[[1, 0, 0]] = std::f32::NEG_INFINITY; 
        mask_test[[1, 0, 1]] = 1.0;
        mask_test[[1, 1, 0]] = 1.0;
        mask_test[[1, 1, 1]] = std::f32::NEG_INFINITY;
        
        let result_mask = softmax_3d(&mask_test, 2);
        
        // Positions with -infinity should have probability 0
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    if mask_test[[i, j, k]].is_infinite() && mask_test[[i, j, k]] < 0.0 {
                        assert!(result_mask[[i, j, k]] < 1e-6, 
                                "Masked positions (-inf) should have probability ≈ 0");
                    } else {
                        assert!((result_mask[[i, j, k]] - 1.0).abs() < 1e-5, 
                                "Unmasked positions should have probability ≈ 1");
                    }
                }
                
                // The sum of probabilities should be 1 even with masked values
                let sum: f32 = result_mask.slice(s![i, j, ..]).sum();
                assert!((sum - 1.0).abs() < 1e-5, 
                        "Sum of probabilities should be 1 even with masks");
            }
        }
    }
    
    #[test]
    fn test_causal_mask() {
        let seq_len = 5;
        
        // Create a causal mask
        let mask = create_causal_mask(seq_len);
        
        // Verify mask dimensions
        assert_eq!(mask.data.shape(), &[1, seq_len, seq_len]);
        
        // Verify that the mask has 1 on the diagonal and below, 0 above
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[0, i, j]], expected, 
                           "Causal mask at position [{}, {}] should be {}", i, j, expected);
            }
        }
        
        // Test for sequence length = 1
        let mask_1 = create_causal_mask(1);
        assert_eq!(mask_1.data.shape(), &[1, 1, 1]);
        assert_eq!(mask_1.data[[0, 0, 0]], 1.0);
        
        // Test for large sequence length
        let large_seq_len = 100;
        let large_mask = create_causal_mask(large_seq_len);
        assert_eq!(large_mask.data.shape(), &[1, large_seq_len, large_seq_len]);
        
        // Check some example points
        assert_eq!(large_mask.data[[0, 0, 0]], 1.0);  // Diagonal
        assert_eq!(large_mask.data[[0, 99, 99]], 1.0); // Diagonal
        assert_eq!(large_mask.data[[0, 99, 0]], 1.0);  // Below diagonal
        assert_eq!(large_mask.data[[0, 0, 99]], 0.0);  // Above diagonal
    }
    
    #[test]
    fn test_padding_mask() {
        // Basic test
        let seq_len = 5;
        let valid_lens = vec![3, 4];
        
        let mask = create_padding_mask(seq_len, &valid_lens);
        
        // Verify mask dimensions
        assert_eq!(mask.data.shape(), &[valid_lens.len(), seq_len, seq_len]);
        
        // Verify mask values for first batch (valid_len = 3)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if i < valid_lens[0] && j < valid_lens[0] { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[0, i, j]], expected, 
                           "Padding mask [0, {}, {}] should be {}", i, j, expected);
            }
        }
        
        // Verify mask values for second batch (valid_len = 4)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if i < valid_lens[1] && j < valid_lens[1] { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[1, i, j]], expected, 
                           "Padding mask [1, {}, {}] should be {}", i, j, expected);
            }
        }
    }
    
    #[test]
    fn test_combined_mask() {
        let seq_len = 5;
        let valid_lens = vec![3, 4];
        
        let mask = create_combined_mask(seq_len, &valid_lens);
        
        // Verify mask dimensions
        assert_eq!(mask.data.shape(), &[valid_lens.len(), seq_len, seq_len]);
        
        // Verify mask values for first batch (valid_len = 3)
        for i in 0..seq_len {
            for j in 0..seq_len {
                // In the combined mask, an element is visible if:
                // 1. It's in the valid part (not padding)
                // 2. It's in a causal position (j <= i)
                let expected = if i < valid_lens[0] && j < valid_lens[0] && j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[0, i, j]], expected, 
                           "Combined mask [0, {}, {}] should be {}", i, j, expected);
            }
        }
        
        // Verify mask values for second batch (valid_len = 4)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if i < valid_lens[1] && j < valid_lens[1] && j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[1, i, j]], expected, 
                           "Combined mask [1, {}, {}] should be {}", i, j, expected);
            }
        }
    }
} 