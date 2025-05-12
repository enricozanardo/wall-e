use ndarray::{Array2, Array3, Ix3};

use crate::nabla::tensor::Tensor;
use crate::attention::{Attention, create_weight_matrix, softmax_3d};

/// Implementation of Multi-Head Attention as described in the paper "Attention is All You Need"
pub struct MultiHeadAttention {
    /// Model dimension (d_model)
    model_dimension: usize,
    
    /// Number of attention heads
    num_heads: usize,
    
    /// Dimension of each head (d_k)
    head_dimension: usize,
    
    /// Scale factor for attention (1/sqrt(d_k))
    scale_factor: f32,
    
    /// Projection matrices for queries (one per head)
    w_queries: Vec<Array2<f32>>,
    
    /// Projection matrices for keys (one per head)
    w_keys: Vec<Array2<f32>>,
    
    /// Projection matrices for values (one per head)
    w_values: Vec<Array2<f32>>,
    
    /// Projection matrix for the combined output
    w_output: Array2<f32>,
}

impl MultiHeadAttention {
    /// Creates a new MultiHeadAttention instance
    /// 
    /// # Arguments
    /// 
    /// * `model_dimension` - Model dimension (d_model)
    /// * `num_heads` - Number of attention heads
    /// * `std` - Standard deviation for weight initialization
    /// 
    /// # Returns
    /// 
    /// * A new MultiHeadAttention instance
    pub fn new(model_dimension: usize, num_heads: usize, std: f32) -> Self {
        assert!(model_dimension % num_heads == 0, 
                "Model dimension ({}) must be divisible by the number of heads ({})", 
                model_dimension, num_heads);
        
        let head_dimension = model_dimension / num_heads;
        let scale_factor = 1.0 / (head_dimension as f32).sqrt();
        
        // Initialize projection matrices for each head
        let mut w_queries = Vec::with_capacity(num_heads);
        let mut w_keys = Vec::with_capacity(num_heads);
        let mut w_values = Vec::with_capacity(num_heads);
        
        for _ in 0..num_heads {
            w_queries.push(create_weight_matrix(model_dimension, head_dimension, std));
            w_keys.push(create_weight_matrix(model_dimension, head_dimension, std));
            w_values.push(create_weight_matrix(model_dimension, head_dimension, std));
        }
        
        // Projection matrix for the combined output
        let w_output = create_weight_matrix(model_dimension, model_dimension, std);
        
        MultiHeadAttention {
            model_dimension,
            num_heads,
            head_dimension,
            scale_factor,
            w_queries,
            w_keys,
            w_values,
            w_output,
        }
    }
    
    /// Computes attention scores for a single head
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
            // Verify mask shape and adapt if necessary
            if mask.shape().len() == 3 {
                let mask_shape = mask.shape();
                
                // If the mask has shape [batch_size, seq_len, seq_len], 
                // apply it directly to the attention scores
                if mask_shape.len() == 3 && mask_shape[0] == scores.shape()[0] &&
                   mask_shape[1] == scores.shape()[1] && mask_shape[2] == scores.shape()[2] {
                    
                    // Print debug info for mask application
                    // println!("Applying mask in compute_attention_scores");
                    // println!("Mask shape: {:?}, Scores shape: {:?}", mask_shape, scores.shape());
                    
                    let mut neg_inf_count = 0;
                    for b in 0..scores.shape()[0] {
                        for i in 0..scores.shape()[1] {
                            for j in 0..scores.shape()[2] {
                                if mask[[b, i, j]] == 0.0 {
                                    scores[[b, i, j]] = std::f32::NEG_INFINITY;
                                    neg_inf_count += 1;
                                }
                            }
                        }
                    }
                    
                    // println!("Number of values set to NEG_INFINITY: {}", neg_inf_count);
                    if neg_inf_count == 0 {
                        println!("WARNING: No mask value is 0.0, so no value was masked!");
                    }
                    
                } else {
                    panic!("Mask has incompatible shape: {:?}, expected: {:?}", 
                           mask_shape, scores.shape());
                }
            } else {
                panic!("Mask must be a 3D tensor, received: {}-D", mask.shape().len());
            }
        }
        
        // Apply softmax to get attention weights
        softmax_3d(&scores, 2)
    }
    
    /// Applies attention weights to values for a single head
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
    
    /// Loads weights into the attention component
    ///
    /// # Arguments
    /// * `qkv_matrix` - Combined matrix for query, key, value projections
    /// * `output_matrix` - Matrix for output projection
    ///
    /// # Returns
    /// * `()` - Unit return
    pub fn load_weights(&mut self, qkv_matrix: &Tensor, output_matrix: &Tensor) {
        // Validate the matrix dimensions
        let qkv_shape = qkv_matrix.data.shape();
        let output_shape = output_matrix.data.shape();
        
        if qkv_shape.len() != 2 || output_shape.len() != 2 {
            println!("Warning: Expected 2D matrices for attention weights");
            return;
        }
        
        // Extract the query, key, value projections from the combined matrix
        if qkv_shape[0] == self.model_dimension && qkv_shape[1] == self.model_dimension * 3 {
            // This is a combined QKV matrix (d_model x 3*d_model)
            // Split it into individual matrices for each head
            
            let qkv_data = qkv_matrix.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            
            let head_size = qkv_shape[1] / 3 / self.num_heads;
            
            for h in 0..self.num_heads {
                // Calculate offsets for the q, k, v projections
                let q_offset = 0;
                let k_offset = self.model_dimension;
                let v_offset = 2 * self.model_dimension;
                
                let q_slice = qkv_data.slice(ndarray::s![.., q_offset + h * head_size..q_offset + (h+1) * head_size]);
                let k_slice = qkv_data.slice(ndarray::s![.., k_offset + h * head_size..k_offset + (h+1) * head_size]);
                let v_slice = qkv_data.slice(ndarray::s![.., v_offset + h * head_size..v_offset + (h+1) * head_size]);
                
                // Copy the weights into our projection matrices
                for i in 0..self.model_dimension {
                    for j in 0..head_size {
                        if j < self.head_dimension {
                            self.w_queries[h][[i, j]] = q_slice[[i, j]];
                            self.w_keys[h][[i, j]] = k_slice[[i, j]];
                            self.w_values[h][[i, j]] = v_slice[[i, j]];
                        }
                    }
                }
            }
            
            println!("Successfully loaded QKV projections for {} attention heads", self.num_heads);
        } else {
            println!("Warning: Unexpected QKV matrix dimensions: {:?}, expected: {}x{}", 
                     qkv_shape, self.model_dimension, self.model_dimension * 3);
        }
        
        // Load the output projection matrix
        if output_shape[0] == self.model_dimension && output_shape[1] == self.model_dimension {
            let output_data = output_matrix.data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
            
            // Copy the weights into our output projection matrix
            for i in 0..self.model_dimension {
                for j in 0..self.model_dimension {
                    self.w_output[[i, j]] = output_data[[i, j]];
                }
            }
            
            println!("Successfully loaded output projection matrix");
        } else {
            println!("Warning: Unexpected output matrix dimensions: {:?}, expected: {}x{}", 
                     output_shape, self.model_dimension, self.model_dimension);
        }
    }
}

impl Attention for MultiHeadAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Get data as Array3
        let q_data = q.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let k_data = k.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let v_data = v.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        // Convert mask if present
        let mask_data = mask.map(|m| {
            m.data.clone().into_dimensionality::<Ix3>().unwrap()
        });
        
        // Create a tensor for the concatenated output of all heads
        let mut concatenated_heads = Array3::<f32>::zeros((
            q_data.shape()[0],         // batch_size
            q_data.shape()[1],         // seq_len
            self.model_dimension,      // d_model (= num_heads * head_dimension)
        ));
        
        // Process each head separately
        for h in 0..self.num_heads {
            // Project query, key, and value for this head
            let mut q_proj = Array3::<f32>::zeros((
                q_data.shape()[0],         // batch_size
                q_data.shape()[1],         // seq_len
                self.head_dimension,       // d_k
            ));
            
            let mut k_proj = Array3::<f32>::zeros((
                k_data.shape()[0],         // batch_size
                k_data.shape()[1],         // seq_len
                self.head_dimension,       // d_k
            ));
            
            let mut v_proj = Array3::<f32>::zeros((
                v_data.shape()[0],         // batch_size
                v_data.shape()[1],         // seq_len
                self.head_dimension,       // d_v
            ));
            
            // Apply projections
            for b in 0..q_data.shape()[0] {
                for i in 0..q_data.shape()[1] {
                    for j in 0..self.head_dimension {
                        let mut q_sum = 0.0;
                        let mut k_sum = 0.0;
                        let mut v_sum = 0.0;
                        
                        for k in 0..q_data.shape()[2] {
                            q_sum += q_data[[b, i, k]] * self.w_queries[h][[k, j]];
                            k_sum += k_data[[b, i, k]] * self.w_keys[h][[k, j]];
                            v_sum += v_data[[b, i, k]] * self.w_values[h][[k, j]];
                        }
                        
                        q_proj[[b, i, j]] = q_sum;
                        k_proj[[b, i, j]] = k_sum;
                        v_proj[[b, i, j]] = v_sum;
                    }
                }
            }
            
            // Calculate attention scores and apply attention for this head
            let attention_weights = self.compute_attention_scores(&q_proj, &k_proj, mask_data.as_ref());
            let head_output = self.apply_attention(&attention_weights, &v_proj);
            
            // Concatenate the output of this head
            let head_offset = h * self.head_dimension;
            for b in 0..head_output.shape()[0] {
                for i in 0..head_output.shape()[1] {
                    for j in 0..head_output.shape()[2] {
                        concatenated_heads[[b, i, head_offset + j]] = head_output[[b, i, j]];
                    }
                }
            }
        }
        
        // Project the concatenated output
        let mut output = Array3::<f32>::zeros((
            concatenated_heads.shape()[0],  // batch_size
            concatenated_heads.shape()[1],  // seq_len
            self.model_dimension,           // d_model
        ));
        
        for b in 0..concatenated_heads.shape()[0] {
            for i in 0..concatenated_heads.shape()[1] {
                for j in 0..self.model_dimension {
                    let mut sum = 0.0;
                    for k in 0..concatenated_heads.shape()[2] {
                        sum += concatenated_heads[[b, i, k]] * self.w_output[[k, j]];
                    }
                    output[[b, i, j]] = sum;
                }
            }
        }
        
        // Convert the result to a Tensor
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
    fn test_multi_head_attention_creation() {
        // Verify that creation fails if model_dimension is not divisible by num_heads
        let result = std::panic::catch_unwind(|| {
            MultiHeadAttention::new(10, 3, 0.1);
        });
        assert!(result.is_err());
        
        // Verify that creation succeeds if model_dimension is divisible by num_heads
        let mha = MultiHeadAttention::new(12, 3, 0.1);
        assert_eq!(mha.model_dimension, 12);
        assert_eq!(mha.num_heads, 3);
        assert_eq!(mha.head_dimension, 4);
    }
    
    #[test]
    fn test_multi_head_attention_forward() {
        // Create a MultiHeadAttention instance
        let model_dim = 12;
        let num_heads = 3;
        let attention = MultiHeadAttention::new(model_dim, num_heads, 0.1);
        
        // Create example input tensors
        let batch_size = 2;
        let seq_len = 4;
        
        let q_data = Array3::<f32>::ones((batch_size, seq_len, model_dim));
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
    fn test_multi_head_attention_with_mask() {
        // Create a MultiHeadAttention instance
        let model_dim = 12;
        let num_heads = 3;
        let attention = MultiHeadAttention::new(model_dim, num_heads, 0.1);
        
        // Create example input tensors
        let batch_size = 2;
        let seq_len = 4;
        
        // Create non-uniform inputs to make the effect of the mask more evident
        let mut q_data = Array3::<f32>::zeros((batch_size, seq_len, model_dim));
        let mut k_data = Array3::<f32>::zeros((batch_size, seq_len, model_dim));
        let mut v_data = Array3::<f32>::zeros((batch_size, seq_len, model_dim));
        
        // Initialize data with values that create a strong dependency on future positions
        // For q_data, increasing values in the sequence dimension
        // For k_data, decreasing values in the sequence dimension
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..model_dim {
                    q_data[[i, j, k]] = (j + 1) as f32;                    // Increasing values in seq
                    k_data[[i, j, k]] = (seq_len - j) as f32;              // Decreasing values in seq
                    v_data[[i, j, k]] = (j + 1) as f32 * (seq_len - j) as f32; // Products of both
                }
            }
        }
        
        // Create a causal mask (each position can only see previous positions)
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
        let output_with_mask = attention.forward(&q, &k, &v, Some(&mask));
        
        // Calculate attention output without mask
        let output_no_mask = attention.forward(&q, &k, &v, None);
        
        // Verify that output with mask is different from output without mask
        let output_with_mask_data = output_with_mask.data.clone().into_dimensionality::<Ix3>().unwrap();
        let output_no_mask_data = output_no_mask.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let mut all_equal = true;
        let mut diff_count = 0;
        let mut max_diff: f32 = 0.0;
        
        for ((b, i, j), &v1) in output_with_mask_data.indexed_iter() {
            let v2 = output_no_mask_data[[b, i, j]];
            let diff = (v1 - v2).abs();
            if diff > 1e-5 {
                all_equal = false;
                diff_count += 1;
                max_diff = max_diff.max(diff);
            }
        }
        
        // Print information about the difference
        println!("Differences found: {}, max diff: {}", diff_count, max_diff);
        
        // Output with mask should be different from output without mask
        assert!(!all_equal, "Output with mask should be different from output without mask");
    }
} 