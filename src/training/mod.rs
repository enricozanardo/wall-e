use std::collections::HashMap;
use std::path::Path;
use std::io;
use ndarray::{Array, Array1, Array2,s};
use thiserror::Error;
use crate::tokenizer::Tokenizer;
use crate::embedding::TransformerEmbedding;
use crate::attention::EncoderStack;
use crate::nabla::tensor::Tensor;

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
        let logits = encoder_output.matmul_with(&self.output_projection);
        
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
                                // Calculate softmax for this position
                                let logits_row = Array1::from_iter(
                                    (0..logits.data.shape()[2])
                                        .map(|k| logits.data[[i, j, k]])
                                );
                                
                                // Find the maximum value for numerical stability
                                let max_logit = logits_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                
                                // Calculate exp of (logits - max_logit)
                                let exp_logits: Vec<f32> = logits_row
                                    .iter()
                                    .map(|&l| (l - max_logit).exp())
                                    .collect();
                                
                                // Calculate sum of exps
                                let sum_exp: f32 = exp_logits.iter().sum();
                                
                                // Calculate target probability
                                let target_prob = exp_logits[target_id] / sum_exp;
                                
                                // Cross-entropy loss: -log(p_target)
                                total_loss -= target_prob.ln();
                                total_tokens += 1;
                            } else {
                                println!("Warning: target_id {} out of range (max {})", 
                                         target_id, logits.data.shape()[2]-1);
                            }
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
        
        // Calculate the gradient of the output projection using the chain rule
        // dL/dW = dL/dO * dO/dW = dL/dO * X^T where O = XW
        // The gradient with respect to weights is the product between the gradient of logits
        // and the transpose of the encoder output
        
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
        
        // Calculate the gradient of the output projection: [d_model, vocab_size]
        let mut output_proj_grad = Array::zeros((d_model, vocab_size));
        
        // Add bounds checking to prevent index out of bounds errors
        for i in 0..(batch_size * seq_len) {
            if i >= encoder_output_flat.shape()[0] || i >= logits_grad_flat.shape()[0] {
                // Skip invalid indices
                continue;
            }
            
            for j in 0..d_model {
                if j >= encoder_output_flat.shape()[1] {
                    // Skip invalid indices
                    continue;
                }
                
                for k in 0..vocab_size {
                    if k >= logits_grad_flat.shape()[1] {
                        // Skip invalid indices
                        continue;
                    }
                    
                    output_proj_grad[[j, k]] += encoder_output_flat[[i, j]] * logits_grad_flat[[i, k]];
                }
            }
        }
        
        // Normalize the gradient by batch size
        output_proj_grad /= (batch_size * seq_len) as f32;
        
        grads.insert("output_projection".to_string(), Tensor::new_from_array(output_proj_grad.into_dyn()));
        
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

// Re-export key components for easier access
pub use generation::TextGenerator;
pub use curriculum::{CurriculumScheduler, CurriculumExample, DifficultyLevel};
pub use enhanced_trainer::EnhancedTrainer; 