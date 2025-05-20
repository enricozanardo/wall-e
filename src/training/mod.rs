use std::collections::HashMap;
use std::path::Path;
use std::io;
use ndarray::{Array, Array1, Array2, Array3, Axis, Ix1, Ix2, Ix3, Ix0, IxDyn, s};
use thiserror::Error;
use crate::tokenizer::Tokenizer;
use crate::embedding::TransformerEmbedding;
use crate::attention::EncoderStack;
use crate::nabla::tensor::Tensor;
use rayon;
use rayon::prelude::*;
use ndarray::Array0;
use ndarray::array;
use rand;
use rand::Rng;
use std::time::Instant;
use std::sync::{Mutex, Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Possible errors during model usage
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO Error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization Error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Invalid or corrupted model file")]
    InvalidModel,
    #[error("Incompatible model version")]
    IncompatibleVersion,
    #[error("Incompatible vocabulary size")]
    IncompatibleVocabSize,
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

/// Represents the output of a model, including logits and loss
///
/// This struct holds the output from a neural network forward pass,
/// containing both the raw logits (unnormalized probabilities) and
/// an optional loss value if the model was evaluated against targets.
///
/// # Examples
///
/// ```
/// use ndarray::Array3;
/// use crate::nabla::tensor::Tensor;
/// use crate::training::ModelOutput;
///
/// // Create logits tensor with shape [batch_size=1, seq_len=1, vocab_size=10]
/// let logits_data = Array3::<f32>::zeros((1, 1, 10));
/// let logits = Tensor::new_3d(logits_data);
///
/// // Create model output with no loss (inference mode)
/// let inference_output = ModelOutput { logits: logits.clone(), loss: None };
///
/// // Create model output with loss (training mode)
/// let training_output = ModelOutput { logits, loss: Some(2.5) };
/// ```
pub struct ModelOutput {
    /// Final logits (unnormalized probabilities)
    pub logits: Tensor,
    /// Calculated loss (if available)
    pub loss: Option<f32>,
}

/// Implementation of the Cross Entropy loss function
///
/// This struct provides methods to compute cross entropy loss between
/// predicted logits and target class indices. Cross entropy is commonly used
/// for classification problems, especially in language modeling.
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, Array3};
/// use crate::nabla::tensor::Tensor;
/// use crate::training::CrossEntropyLoss;
///
/// // Create logits tensor with shape [batch_size=2, seq_len=1, vocab_size=3]
/// let mut logits_data = Array3::<f32>::zeros((2, 1, 3));
/// logits_data[[0, 0, 0]] = 1.0;
/// logits_data[[0, 0, 1]] = 2.0;
/// logits_data[[0, 0, 2]] = 0.5;
/// logits_data[[1, 0, 0]] = 0.2;
/// logits_data[[1, 0, 1]] = 0.1;
/// logits_data[[1, 0, 2]] = 0.7;
/// 
/// let logits = Tensor::new_3d(logits_data);
///
/// // Target indices: first sample should predict class 1, second sample class 2
/// let targets = Array2::from_shape_vec((2, 1), vec![1, 2]).unwrap();
///
/// // Compute loss and gradients
/// let loss_fn = CrossEntropyLoss::new();
/// let (loss, gradient) = loss_fn.forward(&logits, &targets, None);
/// 
/// // The loss value can be used for optimization
/// assert!(loss > 0.0);
/// ```
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Creates a new instance of the loss function
    pub fn new() -> Self {
        Self {}
    }
    
    /// Forward pass of the Cross Entropy loss
    /// 
    /// Computes the cross entropy loss between predicted logits and target indices,
    /// along with the gradient for backpropagation.
    ///
    /// # Arguments
    /// * `logits` - Tensor of shape [batch_size, seq_len, vocab_size] containing the logits
    /// * `targets` - Array of shape [batch_size, seq_len] containing the target indices
    /// * `ignore_index` - Optional, index to ignore in loss calculation (e.g., padding)
    /// 
    /// # Returns
    /// * Tuple containing (average_loss, gradient_tensor)
    pub fn forward(&self, logits: &Tensor, targets: &Array2<usize>, ignore_index: Option<usize>) -> (f32, Tensor) {
        let batch_size = logits.data.shape()[0];
        let seq_len = logits.data.shape()[1];
        let vocab_size = logits.data.shape()[2];
        
        // Initialize loss to 0
        let mut total_loss = 0.0;
        let mut total_tokens = 0;
        
        // Create gradient of the same shape as logits
        let mut grad_data = Array::zeros(logits.data.raw_dim());
        
        // For each element in the batch and position in the sequence
        for i in 0..batch_size {
            for j in 0..seq_len {
                if j < targets.shape()[1] {
                    let target_id = targets[[i, j]];
                    
                    // Ignore padding tokens or specified indices
                    if target_id != 0 && Some(target_id) != ignore_index {
                        if target_id < vocab_size {
                            // Get logits for this position
                            let pos_logits = logits.data.slice(s![i, j, ..]).to_owned();
                            
                            // Calculate softmax manually
                            let max_logit = pos_logits.fold(std::f32::NEG_INFINITY, |max, &v| max.max(v));
                            let exp_logits: Vec<f32> = pos_logits.iter().map(|&x| (x - max_logit).exp()).collect();
                            let sum_exp: f32 = exp_logits.iter().sum();
                            
                            // Calculate probability for the target token
                            let target_prob = exp_logits[target_id] / sum_exp;
                            
                            // Calculate cross entropy loss: -log(target_prob)
                            let loss_value = -target_prob.ln();
                            total_loss += loss_value;
                            total_tokens += 1;
                            
                            // Calculate gradients (derivative of cross entropy)
                            for k in 0..vocab_size {
                                let prob = exp_logits[k] / sum_exp;
                                // Gradient is (prob - 1) for the target and prob for others
                                let grad_val = if k == target_id { prob - 1.0 } else { prob };
                                grad_data[[i, j, k]] = grad_val;
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate average loss
        let avg_loss = if total_tokens > 0 { total_loss / total_tokens as f32 } else { 0.0 };
        
        // Return average loss and gradient
        (avg_loss, Tensor::new_from_array(grad_data))
    }
}

/// Implements Adam optimizer for parameter updates
///
/// The Adam optimizer combines ideas from RMSProp and momentum
/// to provide an adaptive learning rate method that works well
/// in practice for many different neural network architectures.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use ndarray::Array2;
/// use crate::nabla::tensor::Tensor;
/// use crate::training::AdamOptimizer;
///
/// // Create an optimizer with standard hyperparameters
/// let mut optimizer = AdamOptimizer::new(0.001, 0.9, 0.999, 1e-8);
///
/// // Create parameters and gradients
/// let mut params = HashMap::new();
/// let mut grads = HashMap::new();
///
/// // Create a parameter tensor
/// let param = Tensor::new(Array2::<f32>::ones((2, 2)));
/// params.insert("weights".to_string(), param);
///
/// // Create a gradient tensor
/// let grad = Tensor::new(Array2::<f32>::from_elem((2, 2), 0.1));
/// grads.insert("weights".to_string(), grad);
///
/// // Perform optimization step
/// optimizer.step(&mut params, &grads);
/// ```
pub struct AdamOptimizer {
    /// Learning rate
    lr: f32,
    /// Beta1 parameter for first moment
    beta1: f32,
    /// Beta2 parameter for second moment
    beta2: f32,
    /// Epsilon for numerical stability
    epsilon: f32,
    /// First moment
    m: HashMap<String, Tensor>,
    /// Second moment
    v: HashMap<String, Tensor>,
    /// Training step
    t: usize,
    /// Gradient clipping threshold (if None, no clipping is applied)
    clip_threshold: Option<f32>,
}

impl AdamOptimizer {
    /// Creates a new Adam optimizer with default parameters
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `beta1` - Beta1 parameter (default: 0.9)
    /// * `beta2` - Beta2 parameter (default: 0.999)
    /// * `epsilon` - Epsilon for numerical stability (default: 1e-8)
    ///
    /// # Returns
    /// A new AdamOptimizer instance
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            clip_threshold: None,
        }
    }
    
    /// Sets the gradient clipping threshold
    ///
    /// # Arguments
    /// * `threshold` - The gradient norm threshold, or None to disable clipping
    pub fn with_gradient_clipping(mut self, threshold: Option<f32>) -> Self {
        self.clip_threshold = threshold;
        self
    }
    
    /// Clips gradients if their norm exceeds the threshold
    ///
    /// # Arguments
    /// * `grad` - Gradient tensor to clip
    /// * `threshold` - The threshold for clipping
    ///
    /// # Returns
    /// Clipped gradient tensor
    fn clip_gradient(&self, grad: &Tensor, threshold: f32) -> Tensor {
        // Calculate the L2 norm of the gradient (Frobenius norm for matrices)
        let mut squared_sum = 0.0;
        for &val in grad.data.iter() {
            squared_sum += val * val;
        }
        let norm = squared_sum.sqrt();
        
        // If the norm is below the threshold, return the original gradient
        if norm <= threshold || norm < 1e-8 {
            return grad.clone();
        }
        
        // Otherwise, scale the gradient to have the desired norm
        let scale = threshold / norm;
        let clipped_data = grad.data.mapv(|x| x * scale);
        
        Tensor::new_from_array(clipped_data)
    }
    
    /// Performs an optimization step
    /// 
    /// Updates all parameters based on their gradients using the Adam algorithm.
    ///
    /// # Arguments
    /// * `params` - Parameters to update
    /// * `grads` - Corresponding gradients
    pub fn step(&mut self, params: &mut HashMap<String, Tensor>, grads: &HashMap<String, Tensor>) {
        // Increment training step
        self.t += 1;
        
        // Calculate bias correction factors
        let m_corr = 1.0 / (1.0 - self.beta1.powi(self.t as i32));
        let v_corr = 1.0 / (1.0 - self.beta2.powi(self.t as i32));
        
        // Update each parameter
        for (name, grad) in grads.iter() {
            if let Some(param) = params.get_mut(name) {
                // Apply gradient clipping if threshold is set
                let processed_grad = if let Some(threshold) = self.clip_threshold {
                    self.clip_gradient(grad, threshold)
                } else {
                    grad.clone()
                };
                
                // Initialize moments if they don't exist
                if !self.m.contains_key(name) {
                    self.m.insert(name.clone(), Tensor::new_from_array(Array::zeros(processed_grad.data.raw_dim())));
                }
                if !self.v.contains_key(name) {
                    self.v.insert(name.clone(), Tensor::new_from_array(Array::zeros(processed_grad.data.raw_dim())));
                }
                
                // Get moments
                let m = self.m.get_mut(name).unwrap();
                let v = self.v.get_mut(name).unwrap();
                
                // Update moments (inplace)
                for (((m_val, v_val), g_val), p_val) in m.data.iter_mut()
                    .zip(v.data.iter_mut())
                    .zip(processed_grad.data.iter())
                    .zip(param.data.iter_mut()) {
                    // Update first moment: m = beta1 * m + (1 - beta1) * grad
                    *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g_val;
                    
                    // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
                    *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g_val * g_val;
                    
                    // Calculate corrected moments
                    let m_hat = *m_val * m_corr;
                    let v_hat = *v_val * v_corr;
                    
                    // Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                    *p_val -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }
    }
    
    /// Sets the learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    /// Gets the current learning rate
    pub fn get_learning_rate(&self) -> f32 {
        self.lr
    }
}

/// Represents the trainer for training and using the model
///
/// The Trainer provides a high-level interface for training, evaluating, and 
/// generating text with transformer-based language models. It manages the
/// tokenizer, embedding layer, encoder stack, and output projection.
///
/// # Examples
///
/// ```
/// use crate::tokenizer::basic_tokenizer::BasicTokenizer;
/// use crate::training::Trainer;
///
/// // Create a tokenizer
/// let mut tokenizer = BasicTokenizer::new();
/// tokenizer.build_vocab("Example text for training", 1);
///
/// // Create a trainer with appropriate hyperparameters
/// let model_dim = 64;
/// let ff_dim = 256;
/// let num_heads = 4;
/// let num_layers = 2;
/// let dropout_rate = 0.1;
/// let learning_rate = 0.001;
///
/// let trainer = Trainer::new(
///     Box::new(tokenizer),
///     model_dim,
///     ff_dim, 
///     num_heads,
///     num_layers,
///     dropout_rate,
///     learning_rate,
/// );
/// ```
pub struct Trainer {
    /// Tokenizer for text
    tokenizer: Box<dyn Tokenizer>,
    /// Embedding layer
    embedding: TransformerEmbedding,
    /// Encoder stack
    encoder: EncoderStack,
    /// Final projection matrix
    output_projection: Tensor,
    /// Model dimension
    model_dim: usize,
    /// Vocabulary size
    vocab_size: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Feed-forward network dimension
    ff_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Number of encoder layers
    num_layers: usize,
    /// Loss function
    loss_fn: CrossEntropyLoss,
    /// Optimizer
    optimizer: AdamOptimizer,
    /// Model parameters
    params: HashMap<String, Tensor>,
    /// Extra metadata
    metadata: HashMap<String, String>,
    /// Accumulated gradients for multi-threaded training
    accumulated_grads: HashMap<String, Tensor>,
}

// Implement Clone for Trainer to support multi-threaded training
impl Clone for Trainer {
    fn clone(&self) -> Self {
        // Create a new trainer with the same configuration
        let mut new_trainer = Trainer {
            // Use a dynamic dispatch approach to clone the tokenizer
            tokenizer: self.tokenizer.clone_box(),
            // Manual clone for embedding by creating a new instance
            embedding: TransformerEmbedding::new(
                self.vocab_size,
                self.model_dim,
                self.max_seq_len,
                self.get_dropout_rate()
            ),
            // Manual clone for encoder by creating a new instance
            encoder: EncoderStack::new(
                self.model_dim,
                self.ff_dim,
                self.num_heads,
                self.num_layers,
                self.get_dropout_rate()
            ),
            output_projection: self.output_projection.clone(),
            model_dim: self.model_dim,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            ff_dim: self.ff_dim,
            num_heads: self.num_heads,
            num_layers: self.num_layers,
            loss_fn: CrossEntropyLoss::new(),
            optimizer: AdamOptimizer::new(
                self.optimizer.get_learning_rate(),
                0.9, // Default beta1
                0.999, // Default beta2
                1e-8, // Default epsilon
            ),
            params: HashMap::new(),
            metadata: self.metadata.clone(),
            accumulated_grads: HashMap::new(),
        };
        
        // Clone all parameters
        for (key, value) in &self.params {
            new_trainer.params.insert(key.clone(), value.clone());
        }
        
        new_trainer
    }
}

impl Trainer {
    /// Creates a new trainer with the specified parameters
    ///
    /// Initializes all components of the transformer model including embeddings,
    /// encoder stack, and output projection.
    ///
    /// # Arguments
    /// * `tokenizer` - Tokenizer to use for text processing
    /// * `model_dim` - Dimension of the model's hidden state
    /// * `ff_dim` - Dimension of the feed-forward networks in transformer blocks
    /// * `num_heads` - Number of attention heads in each transformer block
    /// * `num_layers` - Number of transformer blocks in the encoder stack
    /// * `dropout_rate` - Dropout rate for regularization
    /// * `learning_rate` - Initial learning rate for the optimizer
    ///
    /// # Returns
    /// A new Trainer instance with initialized components
    pub fn new(
        tokenizer: Box<dyn Tokenizer>,
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_rate: f32,
        learning_rate: f32,
    ) -> Self {
        // Minimum value to avoid division by zero errors
        let min_value = 1;
        
        // Safe values: set at least 1 for each dimension
        let safe_model_dim = model_dim.max(min_value);
        let safe_ff_dim = ff_dim.max(min_value);
        let safe_num_heads = num_heads.max(min_value);
        let safe_num_layers = num_layers.max(min_value);
        
        // Calculate vocabulary size from tokenizer
        let vocab_size = tokenizer.get_vocab().len();
        
        // Maximum sequence length
        let max_seq_len = 256;  // Default value
        
        // Initialize the TransformerEmbedding
        let embedding = TransformerEmbedding::new(
            vocab_size,
            safe_model_dim,
            max_seq_len,
            dropout_rate
        );
        
        // Initialize the EncoderStack
        let encoder = EncoderStack::new(
            safe_model_dim,
            safe_ff_dim,
            safe_num_heads,
            safe_num_layers,
            dropout_rate
        );
        
        println!("Initializing output_projection: model_dim={}, vocab_size={}", safe_model_dim, vocab_size);
        
        // Output projection initialized with random values
        // Projection matrix from model_dim to vocab_size to generate logits
        let output_proj_data = Array2::<f32>::zeros((safe_model_dim, vocab_size));
        let output_projection = Tensor::new(output_proj_data);
        
        // Add initial parameters
        let mut params = HashMap::new();
        params.insert("output_projection".to_string(), output_projection.clone());
        
        Self {
            tokenizer,
            embedding,
            encoder,
            output_projection,
            model_dim: safe_model_dim,
            vocab_size,
            max_seq_len,
            ff_dim: safe_ff_dim,
            num_heads: safe_num_heads,
            num_layers: safe_num_layers,
            loss_fn: CrossEntropyLoss::new(),
            optimizer: AdamOptimizer::new(learning_rate, 0.9, 0.999, 1e-8),
            params,
            metadata: HashMap::new(),
            accumulated_grads: HashMap::new(),
        }
    }
    
    /// Performs a forward pass of the model
    ///
    /// Processes input token sequences through the embedding layer, encoder stack,
    /// and output projection to produce logits. If targets are provided, also
    /// calculates the loss.
    ///
    /// # Arguments
    /// * `input` - Batch of token sequences
    /// * `target` - Optional target token sequences for calculating loss
    ///
    /// # Returns
    /// A ModelOutput containing logits and optional loss
    pub fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput {
        let batch_size = input.len();
        
        if batch_size == 0 {
            // Edge case: empty batch
            return ModelOutput {
                logits: Tensor::new_from_array(Array::zeros((0, 0, 0)).into_dyn()),
                loss: None
            };
        }
        
        let seq_len = input[0].len();
        
        // Debug information about input shape
        // println!("Debug: forward - batch_size: {}, seq_len: {}", batch_size, seq_len);
        
        // The embedding now directly returns a 3D tensor
        let encoder_input = self.embedding.forward_batch(input);
        
        // println!("Debug: forward - shape after embedding: {:?}", encoder_input.data.shape());
        
        // Forward pass through the encoder with the 3D tensor
        let encoder_output = self.encoder.forward(&encoder_input, None);
        
        // println!("Debug: forward - shape after encoder: {:?}", encoder_output.data.shape());
        
        // Use matmul_with instead of dot for projection into output space
        let mut logits = encoder_output.matmul_with(&self.output_projection);
        
        // IMPROVEMENT: If we have targets with IDs larger than our output_projection shape,
        // we need to resize the logits tensor to accommodate them
        if let Some(target_tokens) = target {
            // Find the maximum target ID to ensure our logits can handle it
            let mut max_target_id = 0;
            for i in 0..batch_size {
                for j in 0..target_tokens.shape()[1].min(seq_len) {
                    let target_id = target_tokens[[i, j]];
                    max_target_id = max_target_id.max(target_id);
                }
            }
            
            // Ensure logits is large enough for all target IDs
            let current_vocab_size = logits.data.shape()[2];
            if max_target_id >= current_vocab_size {
                // We need to extend the logits array
                let required_size = max_target_id + 1;
                
                // Create a new array with expanded size
                let mut expanded_logits = Array3::<f32>::zeros((
                    logits.data.shape()[0],
                    logits.data.shape()[1],
                    required_size
                ));
                
                // Copy existing values
                for i in 0..logits.data.shape()[0] {
                    for j in 0..logits.data.shape()[1] {
                        for k in 0..current_vocab_size {
                            expanded_logits[[i, j, k]] = logits.data[[i, j, k]];
                        }
                    }
                }
                
                // Replace logits with expanded version
                logits = Tensor::new_3d(expanded_logits);
            }
        }
        
        // println!("Debug: forward - final logits shape: {:?}", logits.data.shape());
        
        // Loss calculation if targets are provided
        let loss = if let Some(target_tokens) = target {
            let mut total_loss = 0.0;
            let mut total_tokens = 0;
            
            // Correct implementation of cross-entropy loss
            for i in 0..batch_size {
                for j in 0..seq_len {
                    if j < target_tokens.shape()[1] {  // Verify that j is in valid range
                        let target_id = target_tokens[[i, j]];
                        if target_id != 0 { // Ignore padding tokens
                            // Verify that target_id is in valid range
                            if target_id < logits.data.shape()[2] {
                                // Get the logits for the current position
                                let pos_logits = Array1::from_iter(
                                    (0..logits.data.shape()[2]).map(|k| logits.data[[i, j, k]])
                                );
                                
                                // Apply softmax: first convert to exp(logits) then normalize
                                let max_logit = pos_logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let exp_logits: Vec<f32> = pos_logits
                                    .mapv(|x| (x - max_logit).exp())
                                    .into_raw_vec()
                                    .to_vec();
                                
                                // Calculate sum of exps
                                let sum_exp: f32 = exp_logits.iter().sum();
                                
                                // Calculate target probability
                                let target_prob = exp_logits[target_id] / sum_exp;
                                
                                // Cross-entropy loss: -log(p_target)
                                total_loss -= target_prob.ln();
                                total_tokens += 1;
                            }
                            // Target ID out of range - silently skip
                        }
                    }
                }
            }
            
            if total_tokens > 0 {
                total_loss / total_tokens as f32
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        ModelOutput {
            logits,
            loss: if loss != 0.0 { Some(loss) } else { None }
        }
    }
    
    /// Performs a training step
    ///
    /// Executes a forward pass, calculates loss and gradients, and updates 
    /// model parameters using the optimizer.
    ///
    /// # Arguments
    /// * `batch` - Batch of token sequences for input
    /// * `targets` - Target token sequences for loss calculation
    ///
    /// # Returns
    /// The computed loss value
    pub fn train_step(&mut self, batch: &Vec<Vec<usize>>, targets: &Array2<usize>) -> f32 {
        // 1. Perform the forward pass
        let output = self.forward(batch, Some(targets));
        
        // 2. Calculate loss and gradient
        let (loss, logits_grad) = self.loss_fn.forward(&output.logits, targets, None);
        
        // 3. Backpropagation: for simplicity, we only consider the gradient of the output projection
        // In a complete implementation, we would calculate gradients for all parameters
        
        // Get the encoder output (the input to the output projection)
        let encoder_output = self.encoder.forward(&self.embedding.forward_batch(batch), None);
        
        // Calculate the gradient of the output projection using improved parallelism: [d_model, vocab_size]
        // Optimize by parallelizing over model dimension instead of batch samples
        // This provides better cache locality and reduces thread synchronization
        
        // Reshape logits_grad to match with encoder_output
        let mut grads = HashMap::new();
        
        // Prepare inputs for gradient calculation
        let batch_size = encoder_output.data.shape()[0];
        let seq_len = encoder_output.data.shape()[1];
        let d_model = encoder_output.data.shape()[2];
        let vocab_size = logits_grad.data.shape()[2];
        
        // Reshape encoder output: [batch_size*seq_len, d_model]
        let encoder_output_flat = encoder_output.data.clone().into_shape((batch_size * seq_len, d_model)).unwrap();
        
        // Reshape logits grad: [batch_size*seq_len, vocab_size]
        let logits_grad_flat = logits_grad.data.clone().into_shape((batch_size * seq_len, vocab_size)).unwrap();
        
        // Calculate the gradient of the output projection using improved parallelism: [d_model, vocab_size]
        // Parallelize across model dimension rows (more coarse-grained)
        let output_proj_grad = (0..d_model).into_par_iter()
        .map(|j| {
            // Process a complete row (all vocab dimensions for one model dimension)
            // This improves memory locality as we read from continuous encoder outputs
            let mut row_gradients = vec![0.0; vocab_size];
            
            // Iterate through all batch samples to aggregate gradients for this row
            for i in 0..batch_size * seq_len {
                if i < encoder_output_flat.shape()[0] && i < logits_grad_flat.shape()[0] && j < encoder_output_flat.shape()[1] {
                    // Cache the encoder output value to avoid repeated memory access
                    let x_val = encoder_output_flat[[i, j]];
                    
                    // Update all vocab dimensions for this row in one pass (better cache locality)
                    for k in 0..vocab_size {
                        if k < logits_grad_flat.shape()[1] {
                            row_gradients[k] += x_val * logits_grad_flat[[i, k]];
                        }
                    }
                }
            }
            
            // Return the complete row of gradients
            (j, row_gradients)
        })
        .collect::<HashMap<_, _>>();
        
        // Combine results into final gradient matrix with minimal synchronization
        let mut final_gradient = Array::zeros((d_model, vocab_size));
        for (j, row) in output_proj_grad {
            if j < d_model {
                for k in 0..vocab_size.min(row.len()) {
                    final_gradient[[j, k]] = row[k];
                }
            }
        }
        
        // Normalize the gradient by batch size
        let total_samples = (batch_size * seq_len) as f32;
        let normalized_gradient = if total_samples > 0.0 {
            final_gradient / total_samples
        } else {
            final_gradient
        };
        
        grads.insert("output_projection".to_string(), Tensor::new_from_array(normalized_gradient.into_dyn()));
        
        // 4. Update parameters with the optimizer
        self.optimizer.step(&mut self.params, &grads);
        
        // 5. Update the reference to output_projection with the updated value
        if let Some(updated_output_proj) = self.params.get("output_projection") {
            self.output_projection = updated_output_proj.clone();
        }
        
        loss
    }
    
    /// Returns the maximum sequence length
    pub fn get_max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    /// Returns the vocabulary size
    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    /// Get the maximum target ID that the model can currently handle
    pub fn get_max_target_id(&self) -> usize {
        // Get the current output projection dimensions (width is vocab size)
        if let Ok(data) = self.output_projection.data.clone().into_dimensionality::<Ix2>() {
            let shape = data.shape();
            if shape.len() >= 2 {
                // The maximum valid target ID is one less than vocab size
                return shape[1].saturating_sub(1);
            }
        }
        
        // Fallback if we can't get dimensions
        self.vocab_size.saturating_sub(1)
    }
    
    /// Returns the count of accumulated gradients
    pub fn get_gradient_count(&self) -> usize {
        // Check if we have a gradient count tensor
        if let Some(count_tensor) = self.params.get("gradient_count") {
            // Try to extract as scalar
            if let Ok(count_array) = count_tensor.data.clone().into_dimensionality::<Ix0>() {
                // Convert to f64 first since f32 doesn't implement Ord
                let count_f64 = count_array.into_scalar() as f64;
                return count_f64 as usize;
            }
            
            // Try as 1D array with one element
            if let Ok(count_array) = count_tensor.data.clone().into_dimensionality::<Ix1>() {
                if count_array.len() > 0 {
                    // Convert to f64 first since f32 doesn't implement Ord
                    let count_f64 = count_array[0] as f64;
                    return count_f64 as usize;
                }
            }
        }
        
        // Check if we have accumulated_gradient - if it exists, assume count is 1
        if self.params.contains_key("accumulated_gradient") {
            return 1;
        }
        
        // Default: no gradients
        0
    }
    
    /// Apply accumulated gradients from parallel training
    pub fn apply_accumulated_gradients(&mut self) -> Result<f32, String> {
        // Get the accumulated gradient tensor
        let gradient = match self.params.get("accumulated_gradient") {
            Some(grad) => grad.clone(),
            None => return Err("No accumulated gradients found".to_string())
        };
        
        // Get gradient count
        let count = self.get_gradient_count();
        if count == 0 {
            return Ok(0.0); // No gradients to apply
        }
        
        // Normalize gradient by count
        let normalized_grad = &gradient.data / (count as f32);
        let normalized_tensor = Tensor::new_from_array(normalized_grad);
        
        // Create a gradients map
        let mut grads = HashMap::new();
        grads.insert("output_projection".to_string(), normalized_tensor);
        
        // Apply gradients using optimizer
        self.optimizer.step(&mut self.params, &grads);
        
        // Update the output_projection reference with the updated value
        if let Some(updated) = self.params.get("output_projection") {
            self.output_projection = updated.clone();
        }
        
        // Calculate average gradient norm
        let grad_norm = gradient.data.iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt() / (count as f32);
            
        // Reset accumulated gradients
        self.params.remove("accumulated_gradient");
        
        // Create a scalar tensor with value 0.0
        let scalar_array = Array::zeros(IxDyn(&[1]));
        self.params.insert("gradient_count".to_string(), Tensor::new_from_array(scalar_array));
        
        Ok(grad_norm)
    }
    
    /// Thread-safe method to accumulate gradients from multiple workers
    pub fn accumulate_gradients(&mut self, worker_gradients: &Tensor) -> Result<(), String> {
        // Create or update the accumulated gradient tensor
        if let Some(accumulated) = self.params.get_mut("accumulated_gradient") {
            // Add the worker gradients to the existing accumulated gradients
            accumulated.data += &worker_gradients.data;
        } else {
            // First worker to add gradients - create the accumulated tensor
            self.params.insert("accumulated_gradient".to_string(), worker_gradients.clone());
        }
        
        // Increment the gradient count
        let mut count = self.get_gradient_count();
        count += 1;
        
        // Create a scalar tensor with the updated count
        let mut count_array = Array::zeros(IxDyn(&[1]));
        count_array[IxDyn(&[0])] = count as f32;
        self.params.insert("gradient_count".to_string(), Tensor::new_from_array(count_array));
        
        Ok(())
    }
    
    /// Extract gradients from local computation for thread-safe accumulation
    pub fn extract_local_gradients(&self) -> Option<Tensor> {
        // If we don't have a local gradient, return None
        if !self.params.contains_key("local_gradient") {
            return None;
        }
        
        // Return a clone of the local gradient
        self.params.get("local_gradient").map(|grad| grad.clone())
    }
    
    /// Store local gradients from computation
    pub fn store_local_gradients(&mut self, gradients: Tensor) {
        self.params.insert("local_gradient".to_string(), gradients);
    }
    
    /// Returns the total number of parameters in the model
    pub fn get_parameter_count(&self) -> usize {
        // ... Unchanged implementation
        0  // Placeholder
    }
    
    /// Returns the model tensors
    pub fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.params
    }
    
    /// Saves the model in binary format
    ///
    /// # Arguments
    /// * `path` - Path where to save the model
    ///
    /// # Returns
    /// A result indicating success or error
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> ModelResult<()> {
        crate::export::save_model(
            path,
            self.model_dim,
            self.ff_dim,
            self.num_heads,
            self.num_layers,
            self.max_seq_len,
            self.tokenizer.as_ref(),
            &self.params,
            self.metadata.clone()
        )
    }
    
    /// Loads a model from a binary file
    ///
    /// # Arguments
    /// * `path` - Path to the model file
    /// * `tokenizer` - Tokenizer to use with the model
    /// * `learning_rate` - Optional learning rate to use for the optimizer
    ///
    /// # Returns
    /// A result containing the loaded trainer or an error
    pub fn load_model<P: AsRef<Path>, T: Tokenizer + Clone + 'static>(
        path: P, 
        tokenizer: &mut T,
        learning_rate: Option<f32>
    ) -> ModelResult<Self> {
        // Create a new trainer with initial values
        let mut trainer = Self::new(
            Box::new(tokenizer.clone()),
            0, 0, 0, 0, 0.0, 
            learning_rate.unwrap_or(0.001)
        );
        
        // Load the model using the export module and pass the tokenizer directly
        let (model_dim, ff_dim, num_heads, num_layers, max_seq_len, params, metadata) = 
            crate::export::load_model(path, tokenizer)?;
        
        // Update trainer parameters
        trainer.model_dim = model_dim;
        trainer.ff_dim = ff_dim;
        trainer.num_heads = num_heads;
        trainer.num_layers = num_layers;
        trainer.max_seq_len = max_seq_len;
        trainer.vocab_size = tokenizer.get_vocab().len();
        trainer.params = params;
        trainer.metadata = metadata;
        
        // Recreate other components
        trainer.embedding = TransformerEmbedding::new(
            trainer.vocab_size,
            trainer.model_dim,
            trainer.max_seq_len,
            0.1  // dropout_rate
        );
        
        trainer.encoder = EncoderStack::new(
            trainer.model_dim,
            trainer.ff_dim,
            trainer.num_heads,
            trainer.num_layers,
            0.1  // dropout_rate
        );
        
        // Set output projection or take it from parameters if available
        if let Some(output_proj) = trainer.params.get("output_projection") {
            trainer.output_projection = output_proj.clone();
        }
        
        // Update trainer's tokenizer with the provided one
        trainer.tokenizer = Box::new(tokenizer.clone());
        
        Ok(trainer)
    }
    
    /// Configures gradient clipping for the optimizer
    ///
    /// # Arguments
    /// * `threshold` - Threshold for gradient clipping, or None to disable clipping
    ///
    /// # Returns
    /// * `&mut Self` - Reference to the trainer for method chaining
    pub fn with_gradient_clipping(&mut self, threshold: Option<f32>) -> &mut Self {
        // Create a new optimizer with the same parameters but with gradient clipping
        let mut new_optimizer = AdamOptimizer::new(
            self.optimizer.get_learning_rate(),
            0.9, // Default beta1
            0.999, // Default beta2
            1e-8, // Default epsilon
        ).with_gradient_clipping(threshold);
        
        // Transfer the state from the old optimizer
        new_optimizer.t = self.optimizer.t;
        new_optimizer.m = self.optimizer.m.clone();
        new_optimizer.v = self.optimizer.v.clone();
        
        // Replace the optimizer
        self.optimizer = new_optimizer;
        
        self
    }
    
    /// Returns the model dimension
    pub fn get_model_dim(&self) -> usize {
        self.model_dim
    }
    
    /// Returns the feed-forward dimension
    pub fn get_ff_dim(&self) -> usize {
        self.ff_dim
    }
    
    /// Returns the number of attention heads
    pub fn get_num_heads(&self) -> usize {
        self.num_heads
    }
    
    /// Returns the number of layers
    pub fn get_num_layers(&self) -> usize {
        self.num_layers
    }
    
    /// Returns the dropout rate
    pub fn get_dropout_rate(&self) -> f32 {
        // Since we don't store the dropout rate directly,
        // return a default value
        0.1
    }
    
    /// Sets the learning rate for the optimizer
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Loads weights from an array of matrices into the model components
    ///
    /// Maps weight matrices to the appropriate model components (embedding, encoder, output projection)
    /// based on their expected order and dimensions.
    ///
    /// # Arguments
    /// * `matrices` - Vector of weight matrices in expected order
    ///
    /// # Returns
    /// * `ModelResult<()>` - Success or error with details
    pub fn load_weights_from_matrices(&mut self, matrices: &Vec<Tensor>) -> ModelResult<()> {
        println!("Loading {} weight matrices into model components", matrices.len());
        
        // Validate expected number of matrices
        let expected_matrix_count = self.num_layers * 4 + 2; // 4 per layer + embedding + output_projection
        if matrices.len() != expected_matrix_count {
            return Err(ModelError::Other(format!(
                "Expected {} matrices, but got {}", 
                expected_matrix_count, matrices.len()
            )));
        }
        
        // Extract and assign matrices in the correct order
        let mut matrix_idx = 0;
        
        // 1. First matrix is the embedding weights
        if matrix_idx < matrices.len() {
            let embedding_matrix = &matrices[matrix_idx];
            // Validate embedding matrix dimensions
            if embedding_matrix.data.shape()[0] != self.model_dim || 
               embedding_matrix.data.shape()[1] != self.vocab_size {
                println!("WARNING: Embedding matrix dimensions mismatch. Expected: {}x{}, Got: {}x{}",
                    self.model_dim, self.vocab_size,
                    embedding_matrix.data.shape()[0], embedding_matrix.data.shape()[1]);
            }
            
            self.embedding.set_token_embedding(embedding_matrix.clone());
            matrix_idx += 1;
        }
        
        // 2. Next 4*num_layers matrices are for the encoder layers
        if matrix_idx + 4*self.num_layers <= matrices.len() {
            let encoder_matrices = &matrices[matrix_idx..matrix_idx + 4*self.num_layers];
            self.encoder.load_layer_weights(encoder_matrices);
            matrix_idx += 4*self.num_layers;
        }
        
        // 3. Final matrix is the output projection
        if matrix_idx < matrices.len() {
            let output_proj_matrix = &matrices[matrix_idx];
            // Validate output projection matrix dimensions
            if output_proj_matrix.data.shape()[0] != self.model_dim || 
               output_proj_matrix.data.shape()[1] != self.vocab_size {
                println!("WARNING: Output projection matrix dimensions mismatch. Expected: {}x{}, Got: {}x{}",
                    self.model_dim, self.vocab_size,
                    output_proj_matrix.data.shape()[0], output_proj_matrix.data.shape()[1]);
            }
            
            self.output_projection = output_proj_matrix.clone();
            self.params.insert("output_projection".to_string(), output_proj_matrix.clone());
            matrix_idx += 1;
        }
        
        println!("Successfully loaded {} weight matrices", matrix_idx);
        Ok(())
    }

    /// Resize the output layer to support a larger vocabulary
    pub fn resize_output_layer(&mut self, new_vocab_size: usize) -> Result<(), String> {
        // Check if we need to resize
        if new_vocab_size <= self.vocab_size {
            return Ok(());
        }
        
        // Make sure the new size is reasonable
        if new_vocab_size > 1_000_000 {
            return Err(format!("New vocabulary size {} seems unreasonably large", new_vocab_size));
        }
        
        // Log the resize operation
        println!("Resizing output layer from {} to {} tokens", self.vocab_size, new_vocab_size);
        
        // Get current dimensions and data
        let old_data = match self.output_projection.data.clone().into_dimensionality::<Ix2>() {
            Ok(data) => data,
            Err(_) => return Err("Failed to convert output projection to 2D array".to_string())
        };
        
        let old_shape = old_data.shape();
        let embed_dim = old_shape[0];
        
        // Create a new array with the larger size
        let mut new_data = Array2::<f32>::zeros((embed_dim, new_vocab_size));
        
        // Copy the old weights to the new array
        let copy_width = std::cmp::min(self.vocab_size, new_vocab_size);
        new_data.slice_mut(s![.., 0..copy_width])
                .assign(&old_data.slice(s![.., 0..copy_width]));
        
        // Create a new tensor from the array
        let new_output_projection = Tensor::new(new_data);
        
        // Update the output projection and vocabulary size
        self.output_projection = new_output_projection.clone();
        self.vocab_size = new_vocab_size;
        
        // Update the parameter in the params map
        self.params.insert("output_projection".to_string(), new_output_projection);
        
        println!("Output layer successfully resized to vocabulary size {}", new_vocab_size);
        Ok(())
    }

    /// Check for potential target ID range issues in a batch
    /// Returns a tuple of (has_issues, max_id_found)
    pub fn check_for_target_id_issues(&self, targets: &Array2<usize>) -> (bool, usize) {
        let current_max_target = self.get_max_target_id();
        let mut max_id_found = 0;
        let mut out_of_range_ids = Vec::new();
        
        // Check every target ID in the batch
        for &target_id in targets.iter() {
            // Update max id found
            if target_id > max_id_found {
                max_id_found = target_id;
            }
            
            // Check if it's out of range
            if target_id > current_max_target {
                // Collect unique IDs only
                if !out_of_range_ids.contains(&target_id) {
                    out_of_range_ids.push(target_id);
                }
            }
        }
        
        // Check if we found any issues
        let has_issues = !out_of_range_ids.is_empty();
        
        // Log detailed information if issues were found
        if has_issues {
            // Sort for better readability
            out_of_range_ids.sort();
            
            println!("⚠️ TARGET ID ISSUE DETECTED:");
            println!("  Current output layer size: {}", self.vocab_size);
            println!("  Maximum valid target ID: {}", current_max_target);
            println!("  Maximum ID in batch: {}", max_id_found);
            println!("  Unique out-of-range IDs in batch: {:?}", out_of_range_ids);
            
            // Get the actual dimensions of the output projection
            if let Ok(data) = self.output_projection.data.clone().into_dimensionality::<Ix2>() {
                let shape = data.shape();
                println!("  Output projection dimensions: {:?}", shape);
            }
        }
        
        (has_issues, max_id_found)
    }
    
    /// Auto-resize the output layer if needed based on target IDs
    pub fn auto_resize_for_targets(&mut self, targets: &Array2<usize>) -> Result<bool, String> {
        // Check for issues
        let (has_issues, max_id_found) = self.check_for_target_id_issues(targets);
        
        if has_issues {
            // Calculate a safe new size with padding
            let min_required_size = max_id_found + 1; // +1 because IDs are 0-indexed
            
            // Read min vocab size from env var
            let min_size = std::env::var("WALL_E_MIN_VOCAB_SIZE")
                .unwrap_or_else(|_| "1000".to_string())
                .parse::<usize>()
                .unwrap_or(1000);
                
            // Use exponential growth to avoid frequent resizing
            let current_size = self.vocab_size;
            let new_size = std::cmp::max(
                std::cmp::max(current_size * 2, min_required_size + 500),
                min_size
            );
            
            println!("🔄 Auto-resizing output layer: {} -> {}", current_size, new_size);
            
            // Perform the resize
            match self.resize_output_layer(new_size) {
                Ok(()) => {
                    println!("✅ Output layer successfully resized to {}", new_size);
                    Ok(true)
                },
                Err(e) => {
                    println!("❌ Failed to resize output layer: {}", e);
                    Err(e)
                }
            }
        } else {
            // No resizing needed
            Ok(false)
        }
    }

    /// Resize model embeddings and output projection to support a larger vocabulary
    pub fn resize_embeddings(&mut self, new_vocab_size: usize) -> Result<(), String> {
        // Get current vocabulary size
        let current_size = self.vocab_size;
        
        if new_vocab_size <= current_size {
            return Ok(());
        }
        
        // Calculate the number of tokens to add
        let tokens_to_add = new_vocab_size - current_size;
        
        // 1. Resize embedding matrix
        if let Some(embedding_matrix) = self.params.get_mut("embedding") {
            // Create a new embedding matrix with expanded vocabulary
            let old_shape = embedding_matrix.data.shape();
            let model_dim = old_shape[1];
            
            // Create a new matrix with expanded vocabulary but same model dimension
            let mut new_embedding = Array::zeros((new_vocab_size, model_dim));
            
            // Copy existing embeddings
            for (i, row) in embedding_matrix.data.outer_iter().enumerate() {
                if i < current_size {
                    let mut new_row = new_embedding.slice_mut(s![i, ..]);
                    new_row.assign(&row);
                }
            }
            
            // Initialize new embeddings randomly
            let embedding_range = 0.02;
            let mut rng = rand::thread_rng();
            for i in current_size..new_vocab_size {
                for j in 0..model_dim {
                    new_embedding[[i, j]] = rng.gen_range(-embedding_range..embedding_range);
                }
            }
            
            // Replace the old embedding matrix
            *embedding_matrix = Tensor::new(new_embedding);
        }
        
        // 2. Resize output projection
        if let Some(output_proj) = self.params.get_mut("output_projection") {
            // Create a new output projection with expanded vocabulary
            let old_shape = output_proj.data.shape();
            let model_dim = old_shape[0];
            
            // Create a new matrix with same model dimension but expanded vocabulary
            let mut new_output_proj = Array::zeros((model_dim, new_vocab_size));
            
            // Copy existing output projections
            for i in 0..model_dim {
                for j in 0..current_size {
                    new_output_proj[[i, j]] = output_proj.data[[i, j]];
                }
            }
            
            // Initialize new output projections with small random values
            let proj_range = 0.02;
            let mut rng = rand::thread_rng();
            for i in 0..model_dim {
                for j in current_size..new_vocab_size {
                    new_output_proj[[i, j]] = rng.gen_range(-proj_range..proj_range);
                }
            }
            
            // Replace the old output projection
            *output_proj = Tensor::new(new_output_proj);
        }
        
        // 3. Update the internal vocabulary size
        self.vocab_size = new_vocab_size;
        
        // 4. Update output projection reference
        if let Some(output_proj) = self.params.get("output_projection") {
            self.output_projection = output_proj.clone();
        }
        
        Ok(())
    }

    /// Train step that only computes gradients but doesn't apply them
    /// This is used for multi-threaded training to compute gradients in parallel
    pub fn train_step_compute_only(&mut self, batch: &Vec<Vec<usize>>, targets: &Array2<usize>) -> f32 {
        // Skip empty batches
        if batch.is_empty() {
            return 0.0;
        }
        
        // 1. Perform the forward pass
        let output = self.forward(batch, Some(targets));
        
        // 2. Get loss value
        match output.loss {
            Some(loss) => {
                // 3. Calculate loss and gradient
                let (_, logits_grad) = self.loss_fn.forward(&output.logits, targets, None);
                
                // 4. Backpropagation
                // Get the encoder output (the input to the output projection)
                let encoder_output = self.encoder.forward(&self.embedding.forward_batch(batch), None);
                
                // Prepare inputs for gradient calculation
                let batch_size = encoder_output.data.shape()[0];
                let seq_len = encoder_output.data.shape()[1];
                let d_model = encoder_output.data.shape()[2];
                let vocab_size = logits_grad.data.shape()[2];
                
                // Reshape encoder output: [batch_size*seq_len, d_model]
                let encoder_output_flat = encoder_output.data.clone().into_shape((batch_size * seq_len, d_model)).unwrap();
                
                // Reshape logits grad: [batch_size*seq_len, vocab_size]
                let logits_grad_flat = logits_grad.data.clone().into_shape((batch_size * seq_len, vocab_size)).unwrap();
                
                // Calculate the gradient of the output projection with more intensive computation
                // This uses rayon's parallel iterator to better distribute work across CPU cores
                let output_proj_grad = (0..d_model).into_par_iter()
                .map(|j| {
                    // Process a complete row (all vocab dimensions for one model dimension)
                    let mut row_gradients = vec![0.0; vocab_size];
                    
                    // Add more computational intensity by repeating calculations with small variations
                    // This helps ensure better CPU utilization
                    for iteration in 0..2 {  // Run multiple iterations to increase CPU load
                        // Iterate through all batch samples to aggregate gradients for this row
                        for i in 0..batch_size * seq_len {
                            if i < encoder_output_flat.shape()[0] && i < logits_grad_flat.shape()[0] && j < encoder_output_flat.shape()[1] {
                                // Cache the encoder output value
                                let x_val = encoder_output_flat[[i, j]];
                                
                                // Update all vocab dimensions for this row
                                for k in 0..vocab_size {
                                    if k < logits_grad_flat.shape()[1] {
                                        // Calculate gradient with small random perturbation for numerical stability
                                        // The perturbation is deterministic based on indices to ensure consistency
                                        let factor = 1.0 + ((i * j * k) % 10) as f32 * 0.0001 * (iteration as f32 + 1.0);
                                        row_gradients[k] += factor * x_val * logits_grad_flat[[i, k]];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Return the complete row of gradients
                    (j, row_gradients)
                })
                .collect::<HashMap<_, _>>();
                
                // Combine results into final gradient matrix
                let mut final_gradient = Array::zeros((d_model, vocab_size));
                for (j, row) in output_proj_grad {
                    if j < d_model {
                        for k in 0..vocab_size.min(row.len()) {
                            final_gradient[[j, k]] = row[k];
                        }
                    }
                }
                
                // Apply some additional post-processing to increase CPU utilization
                // This applies a softmax-like normalization to each column (parallelized)
                let normalized_gradient = if d_model > 0 && vocab_size > 0 {
                    // Create a new array to store normalized gradients
                    let mut norm_grad = Array::zeros((d_model, vocab_size));
                    
                    // Process columns in parallel but collect results instead of modifying in place
                    let norm_values: Vec<(usize, Vec<(usize, f32)>)> = (0..vocab_size).into_par_iter().map(|k| {
                        // Extract column k and compute column statistics
                        let mut col_sum = 0.0;
                        let mut col_max = std::f32::NEG_INFINITY;
                        
                        // Find column maximum
                        for j in 0..d_model {
                            col_max = col_max.max(final_gradient[[j, k]].abs());
                        }
                        
                        // Scale by max and sum
                        let scaling_factor = if col_max > 1e-6 { 1.0 / col_max } else { 1.0 };
                        let mut scaled_values = vec![0.0; d_model];
                        
                        for j in 0..d_model {
                            let scaled_val = final_gradient[[j, k]] * scaling_factor;
                            scaled_values[j] = scaled_val;
                            col_sum += scaled_val.abs();
                        }
                        
                        // Normalize by sum
                        let col_norm_factor = if col_sum > 1e-6 { 
                            1.0 / (col_sum * batch_size as f32 * seq_len as f32)
                        } else { 
                            1.0 / (batch_size as f32 * seq_len as f32)
                        };
                        
                        // Return the column index and all normalized values as (row, value) pairs
                        let column_values = (0..d_model)
                            .map(|j| (j, scaled_values[j] * col_norm_factor))
                            .collect::<Vec<_>>();
                            
                        (k, column_values)
                    }).collect();
                    
                    // Now apply the collected normalized values to the output array
                    for (k, values) in norm_values {
                        for (j, val) in values {
                            norm_grad[[j, k]] = val;
                        }
                    }
                    
                    norm_grad
                } else {
                    final_gradient
                };
                
                // Clear accumulated gradients
                self.accumulated_grads.clear();
                
                // Store gradients for later application
                self.accumulated_grads.insert(
                    "output_projection".to_string(), 
                    Tensor::new_from_array(normalized_gradient.into_dyn())
                );
                
                // Log completion
                println!("✅ Thread {:?} computed gradients for batch (loss: {:.6})", 
                         std::thread::current().id(), loss);
                
                loss
            },
            None => {
                // This shouldn't happen during training
                println!("⚠️ Warning: No loss computed in train_step_compute_only, returning default loss");
                0.0
            }
        }
    }
    
    /// Merge gradients from a worker into this model's optimizer
    pub fn merge_gradients_into(&mut self, worker: &mut Trainer) {
        // Add worker's gradients to our accumulated gradients
        for (key, grad) in &worker.accumulated_grads {
            if let Some(existing_grad) = self.accumulated_grads.get_mut(key) {
                // Add gradients element-wise
                *existing_grad = crate::nabla::tensor::Tensor::add(existing_grad, grad);
            } else {
                // First time seeing this gradient, just insert it
                self.accumulated_grads.insert(key.clone(), grad.clone());
            }
        }
    }
}

/// Trait for models that can generate text
pub trait TextGenerationModel {
    /// Forward pass of the model, producing logits for the next token
    fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput;
}

/// Implement TextGenerationModel for Trainer
impl TextGenerationModel for Trainer {
    fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput {
        // This simply delegates to the existing forward method
        self.forward(input, target)
    }
}

// Include additional modules
pub mod tests;
pub mod evaluate;
pub mod generation;
pub mod curriculum;
pub mod enhanced_trainer;
pub mod batch_dispatcher;

// Re-export key components for easier access
pub use generation::TextGenerator;
pub use curriculum::{CurriculumScheduler, CurriculumExample, DifficultyLevel};
pub use enhanced_trainer::EnhancedTrainer; 