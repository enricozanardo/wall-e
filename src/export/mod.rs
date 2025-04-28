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

/// Struttura per serializzare i parametri del modello
#[derive(Serialize, Deserialize)]
pub struct ModelParameters {
    /// Versione del formato
    pub version: u32,
    /// Dimensione del modello
    pub model_dim: usize,
    /// Dimensione del feed-forward network
    pub ff_dim: usize,
    /// Numero di teste di attenzione
    pub num_heads: usize,
    /// Numero di layer
    pub num_layers: usize,
    /// Dimensione del vocabolario
    pub vocab_size: usize,
    /// Lunghezza massima della sequenza
    pub max_seq_len: usize,
    /// Vocabolario serializzato
    pub vocab: Option<Vec<u8>>, // Vocabolario serializzato
    /// Parametri serializzati
    pub params: HashMap<String, Vec<f32>>,
    /// Forme dei parametri
    pub param_shapes: HashMap<String, Vec<usize>>,
    /// Metadati aggiuntivi
    pub metadata: HashMap<String, String>,
}

/// Salva un modello in formato binario
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
    println!("Salvataggio del modello in formato binario...");
    
    // Crea directory se non esiste
    if let Some(dir) = path.as_ref().parent() {
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
    }
    
    // Serializza il vocabolario
    let vocab_bytes = bincode::serialize(tokenizer.get_vocab())
        .map_err(|e| ModelError::Serialization(e))?;
        
    // Prepara i parametri per la serializzazione
    let mut params_data = HashMap::new();
    let mut param_shapes = HashMap::new();
    
    for (name, tensor) in params {
        // Converte i tensori in vettori flat per facilità di serializzazione
        let flat_data: Vec<f32> = tensor.data.clone().into_raw_vec();
        params_data.insert(name.clone(), flat_data);
        
        // Salva la forma originale
        let shape = tensor.data.shape().to_vec();
        param_shapes.insert(name.clone(), shape);
    }
    
    // Crea la struttura dei parametri del modello
    let model_params = ModelParameters {
        version: 1,  // Versione corrente del formato
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
    
    // Serializza in formato binario
    let file = File::create(path.as_ref())?;
    let mut writer = BufWriter::new(file);
    
    bincode::serialize_into(&mut writer, &model_params)
        .map_err(|e| ModelError::Serialization(e))?;
    
    writer.flush()?;
    
    println!("Modello salvato con successo in: {}", path.as_ref().display());
    Ok(())
}

/// Carica un modello da un file binario
pub fn load_model<P: AsRef<Path>>(
    path: P,
    tokenizer: &mut dyn Tokenizer
) -> ModelResult<(usize, usize, usize, usize, usize, HashMap<String, Tensor>, HashMap<String, String>)> {
    println!("Caricamento del modello da {}", path.as_ref().display());
    
    // Verifica che il file esista
    if !path.as_ref().exists() {
        return Err(ModelError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("File non trovato: {}", path.as_ref().display())
        )));
    }
    
    // Apri e leggi il file binario
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    
    // Deserializza i parametri del modello
    let model_params: ModelParameters = bincode::deserialize_from(reader)
        .map_err(|e| {
            eprintln!("Errore nella deserializzazione: {}", e);
            ModelError::Serialization(e)
        })?;
    
    // Verifica la versione
    if model_params.version != 1 {
        return Err(ModelError::IncompatibleVersion);
    }
    
    // Ripristina il vocabolario se presente
    if let Some(vocab_bytes) = model_params.vocab {
        let vocab = bincode::deserialize(&vocab_bytes)
            .map_err(|e| ModelError::Serialization(e))?;
        
        // Avvisa se ci sono differenze nel vocabolario
        if tokenizer.get_vocab().len() != model_params.vocab_size {
            println!("ATTENZIONE: Dimensione del vocabolario diversa! Modello: {}, Tokenizer: {}", 
                     model_params.vocab_size, tokenizer.get_vocab().len());
        }
        
        // Usa il vocabolario del modello se il tokenizer lo supporta
        if let Some(tokenizer_mut) = tokenizer.as_vocab_mut() {
            *tokenizer_mut = vocab;
        }
    }
    
    // Ricostruisci i tensori dai dati serializzati
    let mut params = HashMap::new();
    
    for (name, flat_data) in &model_params.params {
        // Recupera la forma originale
        if let Some(shape) = model_params.param_shapes.get(name) {
            // Ricostruisci l'ArrayD con la forma corretta
            let arr = Array::from_shape_vec(shape.clone(), flat_data.clone())
                .map_err(|e| ModelError::Other(format!("Errore nella ricostruzione del tensore: {}", e)))?;
            
            // Crea il tensore
            let tensor = Tensor::new_from_array(arr.into_dyn());
            params.insert(name.clone(), tensor);
        } else {
            return Err(ModelError::InvalidModel);
        }
    }
    
    println!("Modello caricato con successo!");
    
    // Restituisci i parametri del modello e i tensori
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

/// Verifica che un modello sia valido
pub fn verify_model<P: AsRef<Path>>(path: P) -> ModelResult<()> {
    println!("Verifica del modello: {}", path.as_ref().display());
    
    // Verifica che il file esista
    if !path.as_ref().exists() {
        return Err(ModelError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            format!("File non trovato: {}", path.as_ref().display())
        )));
    }
    
    // Apri e leggi il file binario
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    
    // Tenta di deserializzare i parametri del modello
    let model_params: ModelParameters = bincode::deserialize_from(reader)
        .map_err(|e| ModelError::Serialization(e))?;
    
    // Verifica la versione
    if model_params.version != 1 {
        println!("ATTENZIONE: Versione del modello non supportata: {}", model_params.version);
        return Err(ModelError::IncompatibleVersion);
    }
    
    // Verifica la presenza dei parametri essenziali
    if model_params.params.is_empty() {
        println!("ERRORE: Il modello non contiene parametri");
        return Err(ModelError::InvalidModel);
    }
    
    // Verifica che tutte le forme siano definite
    for name in model_params.params.keys() {
        if !model_params.param_shapes.contains_key(name) {
            println!("ERRORE: Forma mancante per il parametro: {}", name);
            return Err(ModelError::InvalidModel);
        }
    }
    
    // Stampa informazioni sul modello
    println!("Modello valido:");
    println!("- Versione: {}", model_params.version);
    println!("- Dimensione del modello: {}", model_params.model_dim);
    println!("- Numero di layer: {}", model_params.num_layers);
    println!("- Dimensione del vocabolario: {}", model_params.vocab_size);
    println!("- Numero di parametri: {}", model_params.params.len());
    println!("- Dimensione del file: {} bytes", fs::metadata(path)?.len());
    
    Ok(())
} 