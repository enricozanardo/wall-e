use ndarray::{Array, Array2, Array3, Axis};
use crate::nabla::tensor::Tensor;
use super::{Attention, MultiHeadAttention, FeedForward};

/// Implementa il Layer Normalization come descritto nel paper "Attention is All You Need"
/// 
/// # Arguments
/// 
/// * `x` - Tensore di input
/// * `eps` - Epsilon per stabilità numerica
/// 
/// # Returns
/// 
/// Il tensore normalizzato
pub fn layer_norm(x: &Tensor, eps: f32) -> Tensor {
    // Ottiene le dimensioni del tensore
    let shape = x.data.shape();
    
    // Verifica se il tensore è 2D o 3D
    let dimensionality = shape.len();
    
    if dimensionality == 2 {
        // Implementa Layer Normalization lungo l'ultima dimensione per tensori 2D
        let last_dim = shape[1];
        
        // Prepara il tensore risultato
        let mut norm_data = x.data.clone();
        
        // Per ogni riga nel tensore 2D
        for i in 0..shape[0] {
            // Calcola la media per questa riga
            let mut sum = 0.0;
            for j in 0..last_dim {
                sum += x.data[[i, j]];
            }
            let mean = sum / (last_dim as f32);
            
            // Calcola la varianza per questa riga
            let mut variance = 0.0;
            for j in 0..last_dim {
                variance += (x.data[[i, j]] - mean).powi(2);
            }
            variance /= last_dim as f32;
            
            // Normalizza questa riga
            for j in 0..last_dim {
                norm_data[[i, j]] = (x.data[[i, j]] - mean) / (variance + eps).sqrt();
            }
        }
        
        return Tensor::new(norm_data.into_dimensionality::<ndarray::Ix2>().unwrap());
    } else if dimensionality == 3 {
        // Implementa Layer Normalization lungo l'ultima dimensione per tensori 3D
        let last_dim = shape[2];
        
        // Prepara il tensore risultato
        let mut norm_data = x.data.clone();
        
        // Per ogni batch e per ogni riga nel tensore 3D
        for b in 0..shape[0] {
            for i in 0..shape[1] {
                // Calcola la media per questa slice
                let mut sum = 0.0;
                for j in 0..last_dim {
                    sum += x.data[[b, i, j]];
                }
                let mean = sum / (last_dim as f32);
                
                // Calcola la varianza per questa slice
                let mut variance = 0.0;
                for j in 0..last_dim {
                    variance += (x.data[[b, i, j]] - mean).powi(2);
                }
                variance /= last_dim as f32;
                
                // Normalizza questa slice
                for j in 0..last_dim {
                    norm_data[[b, i, j]] = (x.data[[b, i, j]] - mean) / (variance + eps).sqrt();
                }
            }
        }
        
        return Tensor::new_3d(norm_data.into_dimensionality::<ndarray::Ix3>().unwrap());
    } else {
        panic!("layer_norm supporta solo tensori 2D o 3D, ricevuto: {}-D", dimensionality);
    }
}

/// Implementazione di un Encoder Layer come descritto nel paper "Attention is All You Need"
pub struct EncoderLayer {
    /// Multi-head attention
    attention: MultiHeadAttention,
    /// Feed-forward network
    feed_forward: FeedForward,
    /// Dimensione del modello
    d_model: usize,
    /// Epsilon per layer normalization
    eps: f32,
    /// Dropout rate
    dropout_rate: f32,
}

impl EncoderLayer {
    /// Crea un nuovo encoder layer
    /// 
    /// # Arguments
    /// 
    /// * `d_model` - Dimensione del modello
    /// * `num_heads` - Numero di teste di attenzione
    /// * `d_ff` - Dimensione del layer feed-forward (default: 4 * d_model)
    /// * `dropout_rate` - Tasso di dropout
    /// * `eps` - Epsilon per layer normalization
    /// 
    /// # Returns
    /// 
    /// Un nuovo encoder layer
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
    
    /// Forward pass attraverso l'encoder layer
    /// 
    /// # Arguments
    /// 
    /// * `x` - Tensore di input [batch_size, seq_len, d_model]
    /// * `mask` - Maschera per l'attenzione (opzionale)
    /// 
    /// # Returns
    /// 
    /// Tensore di output [batch_size, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // 1. Layer Normalization prima dell'attenzione
        let norm1 = layer_norm(x, self.eps);
        
        // 2. Multi-head attention
        let attn_output = self.attention.forward(&norm1, &norm1, &norm1, mask);
        
        // 3. Residual connection con l'input
        let residual1 = Tensor::add(x, &attn_output);
        
        // 4. Layer Normalization prima del feed-forward
        let norm2 = layer_norm(&residual1, self.eps);
        
        // 5. Feed-forward network
        // Dobbiamo rimappare il tensore 3D a 2D per il feed-forward
        let shape = norm2.data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        
        // Reshape da [batch, seq_len, d_model] a [batch*seq_len, d_model]
        let reshaped_data = norm2.data.clone()
            .into_dimensionality::<ndarray::Ix3>().unwrap()
            .into_shape((batch_size * seq_len, self.d_model)).unwrap();
        
        let reshaped_tensor = Tensor::new(reshaped_data.into_dimensionality::<ndarray::Ix2>().unwrap());
        
        // Forward pass feed-forward
        let ff_output = self.feed_forward.forward(&reshaped_tensor);
        
        // Reshape back da [batch*seq_len, d_model] a [batch, seq_len, d_model]
        let ff_output_reshaped = ff_output.data.clone()
            .into_shape((batch_size, seq_len, self.d_model)).unwrap();
        
        let ff_output_tensor = Tensor::new_3d(ff_output_reshaped);
        
        // 6. Residual connection con l'output dell'attenzione
        Tensor::add(&residual1, &ff_output_tensor)
    }
    
    /// Restituisce la dimensione del modello
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
        
        // Crea input di test
        let x_data = Array3::<f32>::zeros((batch_size, seq_len, d_model));
        let x = Tensor::new_3d(x_data);
        
        // Forward pass senza maschera
        let output = encoder.forward(&x, None);
        
        // Verifica le dimensioni dell'output
        let output_shape = output.data.shape();
        assert_eq!(output_shape[0], batch_size);
        assert_eq!(output_shape[1], seq_len);
        assert_eq!(output_shape[2], d_model);
    }
    
    #[test]
    fn test_layer_norm() {
        // Crea un tensore di test con valori conosciuti
        let data = Array2::<f32>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Tensor::new(data);
        
        // Normalizza
        let normalized = layer_norm(&x, 1e-6);
        
        // Calcola la media e la varianza dell'output
        let mean: f32 = normalized.data.mean().unwrap();
        
        // La media dovrebbe essere approssimativamente zero
        assert!(mean.abs() < 1e-5, "La media dovrebbe essere zero, ma è {}", mean);
        
        // La varianza dovrebbe essere approssimativamente 1
        let mut variance = 0.0;
        for &val in normalized.data.iter() {
            variance += (val - mean).powi(2);
        }
        variance /= normalized.data.len() as f32;
        
        assert!((variance - 1.0).abs() < 1e-5, "La varianza dovrebbe essere 1, ma è {}", variance);
    }
    
    #[test]
    fn test_encoder_layer_with_mask() {
        let d_model = 64;
        let num_heads = 4;
        let seq_len = 5;
        let batch_size = 2;
        
        let encoder = EncoderLayer::new(d_model, num_heads, None, 0.1, 1e-6);
        
        // Creiamo dati di input con un pattern che rende evidente la maschera causale
        let mut x_data = Array3::<f32>::zeros((batch_size, seq_len, d_model));
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..d_model {
                    // Utilizziamo un pattern che crea dipendenze tra posizioni future e precedenti
                    // Le prime posizioni (1, 2) hanno valori piccoli
                    // Le ultime posizioni (3, 4, 5) hanno valori grandi
                    // Questo renderà più evidente l'effetto della maschera causale
                    if i < 2 {
                        x_data[[b, i, j]] = 0.01 * (i + 1) as f32;
                    } else {
                        x_data[[b, i, j]] = 10.0 * (i + 1) as f32; // Valori molto più grandi nelle posizioni future
                    }
                }
            }
        }
        
        let x = Tensor::new_3d(x_data);
        
        // Crea una maschera 3D [batch_size, seq_len, seq_len]
        let mut mask_data = Array3::<f32>::zeros((batch_size, seq_len, seq_len));
        
        // Maschera triangolare inferiore (causale) per ogni batch
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..=i {  // j <= i (triangolare inferiore)
                    mask_data[[b, i, j]] = 1.0;
                }
            }
        }
        
        let mask = Tensor::new_3d(mask_data);
        
        // Forward pass con maschera
        let output_with_mask = encoder.forward(&x, Some(&mask));
        
        // Forward pass senza maschera
        let output_no_mask = encoder.forward(&x, None);
        
        // Le due output dovrebbero essere diverse
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
        
        // Stampa debug informazioni
        println!("Differenze trovate nel test encoder: {}, max diff: {}", diff_count, max_diff);
        
        // Ci aspettiamo che l'output con maschera sia diverso dall'output senza maschera
        // Se il test fallisce, significa che la maschera non sta avendo alcun effetto
        assert!(!all_equal, "L'output con maschera dovrebbe essere diverso dall'output senza maschera. 
                             Questo potrebbe accadere se la maschera non viene applicata correttamente
                             o se i valori di input sono tali che la maschera non fa differenza.");
    }
} 