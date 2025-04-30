use ndarray::{Array, Array2, Array3, Axis, Ix3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use crate::nabla::tensor::Tensor;
use crate::attention::{Attention, create_weight_matrix, softmax_3d};

/// Implementation of Self-Attention as described in the paper "Attention is All You Need"
pub struct SelfAttention {
    /// Model dimension (d_model)
    model_dimension: usize,
    
    /// Scale factor for attention (1/sqrt(d_k))
    scale_factor: f32,
    
    /// Projection matrix for queries
    w_query: Array2<f32>,
    
    /// Projection matrix for keys
    w_key: Array2<f32>,
    
    /// Projection matrix for values
    w_value: Array2<f32>,
    
    /// Projection matrix for output
    w_output: Array2<f32>,
}

impl SelfAttention {
    /// Creates a new SelfAttention instance
    /// 
    /// # Arguments
    /// 
    /// * `model_dimension` - Model dimension (d_model)
    /// * `std` - Standard deviation for weight initialization
    /// 
    /// # Returns
    /// 
    /// * A new SelfAttention instance
    pub fn new(model_dimension: usize, std: f32) -> Self {
        let scale_factor = 1.0 / (model_dimension as f32).sqrt();
        
        // Initialize projection matrices for query, key, value and output
        let w_query = create_weight_matrix(model_dimension, model_dimension, std);
        let w_key = create_weight_matrix(model_dimension, model_dimension, std);
        let w_value = create_weight_matrix(model_dimension, model_dimension, std);
        let w_output = create_weight_matrix(model_dimension, model_dimension, std);
        
        SelfAttention {
            model_dimension,
            scale_factor,
            w_query,
            w_key,
            w_value,
            w_output,
        }
    }
    
    /// Computes attention scores
    /// 
    /// # Arguments
    /// 
    /// * `q` - Query tensor
    /// * `k` - Key tensor
    /// * `mask` - Optional attention mask
    /// 
    /// # Returns
    /// 
    /// * Tensor of attention scores
    fn compute_attention_scores(&self, q: &Array3<f32>, k: &Array3<f32>, mask: Option<&Array3<f32>>) -> Array3<f32> {
        // Transpose keys for matrix-matrix product
        // k_transposed will be of shape [batch_size, d_k, seq_len]
        let mut k_transposed = Array3::<f32>::zeros((k.shape()[0], k.shape()[2], k.shape()[1]));
        
        for b in 0..k.shape()[0] {
            for i in 0..k.shape()[1] {
                for j in 0..k.shape()[2] {
                    k_transposed[[b, j, i]] = k[[b, i, j]];
                }
            }
        }
        
        // Calculate matrix-matrix product q * k_t
        // Result will be of shape [batch_size, seq_len_q, seq_len_k]
        let mut scores = Array3::<f32>::zeros((q.shape()[0], q.shape()[1], k_transposed.shape()[2]));
        
        for b in 0..q.shape()[0] {
            for i in 0..q.shape()[1] {
                for j in 0..k_transposed.shape()[2] {
                    let mut sum = 0.0;
                    for k in 0..q.shape()[2] {
                        sum += q[[b, i, k]] * k_transposed[[b, k, j]];
                    }
                    scores[[b, i, j]] = sum * self.scale_factor;
                }
            }
        }
        
        // Apply mask if present
        if let Some(mask) = mask {
            for b in 0..scores.shape()[0] {
                for i in 0..scores.shape()[1] {
                    for j in 0..scores.shape()[2] {
                        if mask[[b, i, j]] == 0.0 {
                            scores[[b, i, j]] = std::f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
        
        // Apply softmax to get attention weights
        softmax_3d(&scores, 2)
    }
    
    /// Applies attention weights to values
    /// 
    /// # Arguments
    /// 
    /// * `attention_weights` - Attention weights
    /// * `v` - Value tensor
    /// 
    /// # Returns
    /// 
    /// * Weighted output tensor
    fn apply_attention(&self, attention_weights: &Array3<f32>, v: &Array3<f32>) -> Array3<f32> {
        // attention_weights: [batch_size, seq_len_q, seq_len_k]
        // v: [batch_size, seq_len_k, d_v]
        // output: [batch_size, seq_len_q, d_v]
        
        let mut output = Array3::<f32>::zeros((
            attention_weights.shape()[0],  // batch_size
            attention_weights.shape()[1],  // seq_len_q
            v.shape()[2],                  // d_v
        ));
        
        for b in 0..attention_weights.shape()[0] {
            for i in 0..attention_weights.shape()[1] {
                for j in 0..v.shape()[2] {
                    let mut sum = 0.0;
                    for k in 0..attention_weights.shape()[2] {
                        sum += attention_weights[[b, i, k]] * v[[b, k, j]];
                    }
                    output[[b, i, j]] = sum;
                }
            }
        }
        
        output
    }
}

impl Attention for SelfAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Get data as Array3
        let q_data = q.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let k_data = k.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let v_data = v.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        // Project query, key, and value
        let mut q_proj = Array3::<f32>::zeros((
            q_data.shape()[0],         // batch_size
            q_data.shape()[1],         // seq_len
            self.model_dimension,      // d_model
        ));
        
        let mut k_proj = Array3::<f32>::zeros((
            k_data.shape()[0],         // batch_size
            k_data.shape()[1],         // seq_len
            self.model_dimension,      // d_model
        ));
        
        let mut v_proj = Array3::<f32>::zeros((
            v_data.shape()[0],         // batch_size
            v_data.shape()[1],         // seq_len
            self.model_dimension,      // d_model
        ));
        
        // Apply projections
        for b in 0..q_data.shape()[0] {
            for i in 0..q_data.shape()[1] {
                for j in 0..self.model_dimension {
                    let mut q_sum = 0.0;
                    let mut k_sum = 0.0;
                    let mut v_sum = 0.0;
                    
                    for k in 0..q_data.shape()[2] {
                        q_sum += q_data[[b, i, k]] * self.w_query[[k, j]];
                        k_sum += k_data[[b, i, k]] * self.w_key[[k, j]];
                        v_sum += v_data[[b, i, k]] * self.w_value[[k, j]];
                    }
                    
                    q_proj[[b, i, j]] = q_sum;
                    k_proj[[b, i, j]] = k_sum;
                    v_proj[[b, i, j]] = v_sum;
                }
            }
        }
        
        // Convert mask if present
        let mask_data = mask.map(|m| {
            m.data.clone().into_dimensionality::<Ix3>().unwrap()
        });
        
        // Calculate attention scores and apply attention
        let attention_weights = self.compute_attention_scores(&q_proj, &k_proj, mask_data.as_ref());
        let context = self.apply_attention(&attention_weights, &v_proj);
        
        // Project output
        let mut output = Array3::<f32>::zeros((
            context.shape()[0],        // batch_size
            context.shape()[1],        // seq_len
            self.model_dimension,      // d_model
        ));
        
        for b in 0..context.shape()[0] {
            for i in 0..context.shape()[1] {
                for j in 0..self.model_dimension {
                    let mut sum = 0.0;
                    for k in 0..context.shape()[2] {
                        sum += context[[b, i, k]] * self.w_output[[k, j]];
                    }
                    output[[b, i, j]] = sum;
                }
            }
        }
        
        // Convert result to a Tensor
        Tensor::new_3d(output)
    }
    
    fn model_dim(&self) -> usize {
        self.model_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_self_attention_forward() {
        // Create a SelfAttention instance
        let model_dim = 4;
        let attention = SelfAttention::new(model_dim, 0.1);
        
        // Create example input tensors
        let batch_size = 2;
        let seq_len = 3;
        
        let q_data = Array3::from_shape_vec((batch_size, seq_len, model_dim),
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                
                0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
            ]
        ).unwrap();
        
        let k_data = q_data.clone();
        let v_data = q_data.clone();
        
        let q = Tensor::new_3d(q_data);
        let k = Tensor::new_3d(k_data);
        let v = Tensor::new_3d(v_data);
        
        // Calculate attention output
        let output = attention.forward(&q, &k, &v, None);
        
        // Verify output dimensions
        let output_data = output.data.clone().into_dimensionality::<Ix3>().unwrap();
        assert_eq!(output_data.shape()[0], batch_size);
        assert_eq!(output_data.shape()[1], seq_len);
        assert_eq!(output_data.shape()[2], model_dim);
        
        // Verify that all values are valid numbers (not NaN or infinite)
        for v in output_data.iter() {
            assert!(!v.is_nan() && !v.is_infinite());
        }
    }
    
    #[test]
    fn test_self_attention_with_mask() {
        // Create a SelfAttention instance
        let model_dim = 4;
        let attention = SelfAttention::new(model_dim, 0.1);
        
        // Create example input tensors
        let batch_size = 2;
        let seq_len = 3;
        
        let q_data = Array3::from_shape_vec((batch_size, seq_len, model_dim),
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                
                0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
            ]
        ).unwrap();
        
        let k_data = q_data.clone();
        let v_data = q_data.clone();
        
        // Create a mask that allows only causal attention
        // (each position can only see previous positions)
        let mut mask_data = Array3::<f32>::ones((batch_size, seq_len, seq_len));
        
        // Causal mask: position i can only see positions j <= i
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in (i+1)..seq_len {
                    mask_data[[b, i, j]] = 0.0;
                }
            }
        }
        
        let q = Tensor::new_3d(q_data);
        let k = Tensor::new_3d(k_data);
        let v = Tensor::new_3d(v_data);
        let mask = Tensor::new_3d(mask_data);
        
        // Calculate attention output with mask
        let output = attention.forward(&q, &k, &v, Some(&mask));
        
        // Verify output dimensions
        let output_data = output.data.clone().into_dimensionality::<Ix3>().unwrap();
        assert_eq!(output_data.shape()[0], batch_size);
        assert_eq!(output_data.shape()[1], seq_len);
        assert_eq!(output_data.shape()[2], model_dim);
        
        // Verify that all values are valid numbers
        for v in output_data.iter() {
            assert!(!v.is_nan() && !v.is_infinite());
        }
        
        // Verify that output with mask is different from output without mask
        let output_no_mask = attention.forward(&q, &k, &v, None);
        let output_no_mask_data = output_no_mask.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let mut all_equal = true;
        
        for ((b, i, j), &v1) in output_data.indexed_iter() {
            let v2 = output_no_mask_data[[b, i, j]];
            if (v1 - v2).abs() > 1e-5 {
                all_equal = false;
                break;
            }
        }
        
        // Output with mask should be different from output without mask
        assert!(!all_equal, "Output with mask should be different from output without mask");
    }
} 