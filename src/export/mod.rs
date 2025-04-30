use std::fs::{self, File};
use std::io::{self, BufWriter, BufReader, Write, Read};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use ndarray::{Array, ArrayD};
use bincode;
use serde::{Serialize, Deserialize};

use crate::tokenizer::Tokenizer;
use crate::nabla::tensor::Tensor;
use crate::training::ModelError;
use crate::training::ModelResult;

/// Structure for serializing model parameters
///
/// This structure contains all the information needed to save and restore
/// a model, including its architecture parameters, weights, vocabulary,
/// and additional metadata.
///
/// # Fields
///
/// * `version` - Format version number for compatibility checking
/// * `model_dim` - Hidden dimension size of the model
/// * `ff_dim` - Feed-forward network dimension
/// * `num_heads` - Number of attention heads
/// * `num_layers` - Number of transformer layers
/// * `vocab_size` - Size of the vocabulary
/// * `max_seq_len` - Maximum sequence length supported
/// * `vocab` - Serialized vocabulary data
/// * `params` - Serialized model parameters (weights)
/// * `param_shapes` - Original shapes of each parameter tensor
/// * `metadata` - Additional metadata key-value pairs
#[derive(Serialize, Deserialize)]
pub struct ModelParameters {
    /// Format version
    pub version: u32,
    /// Model dimension
    pub model_dim: usize,
    /// Feed-forward network dimension
    pub ff_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Serialized vocabulary
    pub vocab: Option<Vec<u8>>, 
    /// Serialized parameters
    pub params: HashMap<String, Vec<f32>>,
    /// Parameter shapes
    pub param_shapes: HashMap<String, Vec<usize>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Saves a model in binary format
///
/// This function serializes a model to a binary file, including all its parameters,
/// architecture configuration, and vocabulary.
///
/// # Arguments
///
/// * `path` - Path where the model will be saved
/// * `model_dim` - Model's hidden dimension size
/// * `ff_dim` - Feed-forward network dimension
/// * `num_heads` - Number of attention heads
/// * `num_layers` - Number of transformer layers
/// * `max_seq_len` - Maximum supported sequence length
/// * `tokenizer` - Tokenizer with vocabulary to be saved
/// * `params` - Model parameters (tensors)
/// * `metadata` - Additional metadata to include
///
/// # Returns
///
/// A result indicating success or an error with details
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use std::path::Path;
/// use crate::tokenizer::basic_tokenizer::BasicTokenizer;
/// use crate::nabla::tensor::Tensor;
/// use crate::export::save_model;
///
/// // Create a tokenizer
/// let tokenizer = BasicTokenizer::new();
///
/// // Create model parameters
/// let mut params = HashMap::new();
/// params.insert("output_projection".to_string(), Tensor::new_default());
///
/// // Create metadata
/// let mut metadata = HashMap::new();
/// metadata.insert("created_date".to_string(), "2023-06-01".to_string());
///
/// // Save the model
/// let result = save_model(
///     "models/my_model.bin",
///     64,   // model_dim
///     256,  // ff_dim
///     4,    // num_heads
///     2,    // num_layers
///     128,  // max_seq_len
///     &tokenizer,
///     &params,
///     metadata
/// );
/// ```
pub fn save_model<P: AsRef<Path>>(
    path: P,
    model_dim: usize,
    ff_dim: usize,
    num_heads: usize,
    num_layers: usize,
    max_seq_len: usize,
    tokenizer: &dyn Tokenizer,
    params: &HashMap<String, Tensor>,
    metadata: HashMap<String, String>
) -> ModelResult<()> {
    println!("Saving model in binary format...");
    
    // Create directory if it doesn't exist
    if let Some(dir) = path.as_ref().parent() {
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
    }
    
    // Serialize the vocabulary
    let vocab_bytes = bincode::serialize(tokenizer.get_vocab())
        .map_err(|e| ModelError::Serialization(e))?;
        
    // Prepare parameters for serialization
    let mut params_data = HashMap::new();
    let mut param_shapes = HashMap::new();
    
    for (name, tensor) in params {
        // Convert tensors to flat vectors for easier serialization
        let flat_data: Vec<f32> = tensor.data.clone().into_raw_vec();
        params_data.insert(name.clone(), flat_data);
        
        // Save the original shape
        let shape = tensor.data.shape().to_vec();
        param_shapes.insert(name.clone(), shape);
    }
    
    // Create model parameters structure
    let model_params = ModelParameters {
        version: 1,  // Current format version
        model_dim,
        ff_dim,
        num_heads,
        num_layers,
        vocab_size: tokenizer.get_vocab().len(),
        max_seq_len,
        vocab: Some(vocab_bytes),
        params: params_data,
        param_shapes,
        metadata,
    };
    
    // Serialize in binary format
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    
    bincode::serialize_into(&mut writer, &model_params)
        .map_err(|e| ModelError::Serialization(e))?;
    
    writer.flush()?;
    
    println!("Model successfully saved to: {}", path.as_ref().display());
    Ok(())
}

/// Loads a model from a binary file
///
/// This function deserializes a model from a binary file, reconstructing its
/// parameters, architecture configuration, and vocabulary.
///
/// # Arguments
///
/// * `path` - Path to the model file
/// * `tokenizer` - Tokenizer that will be updated with the model's vocabulary
///
/// # Returns
///
/// On success, returns a tuple containing:
/// * `model_dim` - Model's hidden dimension size
/// * `ff_dim` - Feed-forward network dimension
/// * `num_heads` - Number of attention heads
/// * `num_layers` - Number of transformer layers
/// * `max_seq_len` - Maximum supported sequence length
/// * `params` - HashMap of model parameters (tensors)
/// * `metadata` - Additional metadata
///
/// # Errors
///
/// Returns an error if:
/// * The file doesn't exist
/// * The file format is invalid or corrupted
/// * The model version is incompatible
///
/// # Examples
///
/// ```
/// use crate::tokenizer::basic_tokenizer::BasicTokenizer;
/// use crate::export::load_model;
///
/// // Create a tokenizer
/// let mut tokenizer = BasicTokenizer::new();
///
/// // Load the model
/// let result = load_model("models/my_model.bin", &mut tokenizer);
/// match result {
///     Ok((model_dim, ff_dim, num_heads, num_layers, max_seq_len, params, metadata)) => {
///         println!("Model loaded successfully with {} parameters", params.len());
///     },
///     Err(e) => {
///         println!("Failed to load model: {:?}", e);
///     }
/// }
/// ```
pub fn load_model<P: AsRef<Path>>(
    path: P,
    tokenizer: &mut dyn Tokenizer
) -> ModelResult<(usize, usize, usize, usize, usize, HashMap<String, Tensor>, HashMap<String, String>)> {
    println!("Loading model from {}", path.as_ref().display());
    
    // Verify that the file exists
    if !path.as_ref().exists() {
        return Err(ModelError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("File not found: {}", path.as_ref().display())
        )));
    }
    
    // Open and read the binary file
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    
    // Deserialize model parameters
    let model_params: ModelParameters = bincode::deserialize_from(reader)
        .map_err(|e| {
            eprintln!("Deserialization error: {}", e);
            ModelError::Serialization(e)
        })?;
    
    // Verify version
    if model_params.version != 1 {
        return Err(ModelError::IncompatibleVersion);
    }
    
    // Restore vocabulary if present
    if let Some(vocab_bytes) = model_params.vocab {
        let vocab = bincode::deserialize(&vocab_bytes)
            .map_err(|e| ModelError::Serialization(e))?;
        
        // Warn if there are vocabulary size differences
        if tokenizer.get_vocab().len() != model_params.vocab_size {
            println!("WARNING: Different vocabulary size! Model: {}, Tokenizer: {}", 
                     model_params.vocab_size, tokenizer.get_vocab().len());
        }
        
        // Use model vocabulary if the tokenizer supports it
        if let Some(tokenizer_mut) = tokenizer.as_vocab_mut() {
            *tokenizer_mut = vocab;
        }
    }
    
    // Reconstruct tensors from serialized data
    let mut params = HashMap::new();
    
    for (name, flat_data) in &model_params.params {
        // Retrieve original shape
        if let Some(shape) = model_params.param_shapes.get(name) {
            // Reconstruct ArrayD with correct shape
            let arr = Array::from_shape_vec(shape.clone(), flat_data.clone())
                .map_err(|e| ModelError::Other(format!("Error reconstructing tensor: {}", e)))?;
            
            // Create tensor
            let tensor = Tensor::new_from_array(arr.into_dyn());
            params.insert(name.clone(), tensor);
        } else {
            return Err(ModelError::InvalidModel);
        }
    }
    
    println!("Model loaded successfully!");
    
    // Return model parameters and tensors
    Ok((
        model_params.model_dim,
        model_params.ff_dim,
        model_params.num_heads,
        model_params.num_layers,
        model_params.max_seq_len,
        params,
        model_params.metadata
    ))
}

/// Verifies that a model file is valid
///
/// This function checks if a model file exists and has a valid format,
/// without loading the entire model into memory.
///
/// # Arguments
///
/// * `path` - Path to the model file to verify
///
/// # Returns
///
/// A result indicating if the model is valid or an error with details
///
/// # Examples
///
/// ```
/// use crate::export::verify_model;
///
/// // Verify a model file
/// match verify_model("models/my_model.bin") {
///     Ok(_) => {
///         println!("Model is valid");
///     },
///     Err(e) => {
///         println!("Model is invalid: {:?}", e);
///     }
/// }
/// ```
pub fn verify_model<P: AsRef<Path>>(path: P) -> ModelResult<()> {
    println!("Verifying model: {}", path.as_ref().display());
    
    // Verify that the file exists
    if !path.as_ref().exists() {
        return Err(ModelError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("File not found: {}", path.as_ref().display())
        )));
    }
    
    // Open and read the binary file
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    
    // Attempt to deserialize model parameters
    let model_params: ModelParameters = bincode::deserialize_from(reader)
        .map_err(|e| ModelError::Serialization(e))?;
    
    // Verify version
    if model_params.version != 1 {
        println!("WARNING: Unsupported model version: {}", model_params.version);
        return Err(ModelError::IncompatibleVersion);
    }
    
    // Verify essential parameters are present
    if model_params.params.is_empty() {
        println!("ERROR: Model contains no parameters");
        return Err(ModelError::InvalidModel);
    }
    
    // Verify all shapes are defined
    for name in model_params.params.keys() {
        if !model_params.param_shapes.contains_key(name) {
            println!("ERROR: Missing shape for parameter: {}", name);
            return Err(ModelError::InvalidModel);
        }
    }
    
    // Print model information
    println!("Valid model:");
    println!("- Version: {}", model_params.version);
    println!("- Model dimension: {}", model_params.model_dim);
    println!("- Number of layers: {}", model_params.num_layers);
    println!("- Vocabulary size: {}", model_params.vocab_size);
    println!("- Number of parameters: {}", model_params.params.len());
    println!("- File size: {} bytes", fs::metadata(path)?.len());
    
    Ok(())
} 