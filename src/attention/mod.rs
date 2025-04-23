use crate::nabla::tensor::Tensor;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use rayon::prelude::*;

// Re-export dei moduli
pub mod self_attention;
// Il modulo multi_head sarà implementato in seguito

pub use self_attention::SelfAttention;
// pub use multi_head::MultiHeadAttention;

/// Trait per implementazioni di meccanismi di attenzione
pub trait Attention {
    /// Forward pass: converte input in output utilizzando un meccanismo di attenzione
    /// Input: input - Tensor di forma [batch_size * seq_len, d_model]
    /// Output: Tensor di forma [batch_size * seq_len, d_model]
    fn forward(&self, input: &Tensor) -> Tensor;
    
    /// Forward pass con maschera di attenzione
    /// Input: input - Tensor di forma [batch_size * seq_len, d_model]
    ///        mask - Tensor di forma [batch_size * seq_len, seq_len]
    /// Output: Tensor di forma [batch_size * seq_len, d_model]
    fn forward_with_mask(&self, input: &Tensor, mask: &Tensor) -> Tensor;
    
    /// Dimensione del modello (d_model)
    fn model_dim(&self) -> usize;
}

/// Funzione di utilità per creare una matrice di pesi per una proiezione lineare
/// Utilizza ndarray-rand per inizializzazione ottimizzata e thread-safe
pub fn create_weight_matrix(input_dim: usize, output_dim: usize) -> Tensor {
    // Inizializzazione Xavier/Glorot per stabilizzare la varianza
    let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
    let weights = Array::random((input_dim, output_dim), Normal::new(0.0, scale).unwrap());
    Tensor::new(weights)
}

/// Calcola il softmax su un tensore 2D (batch_size * seq_len, seq_len)
/// applicando l'operazione sulle righe (per ogni riga in modo indipendente)
pub fn softmax(x: &Tensor) -> Tensor {
    // Estrai i dati
    let shape = x.data.shape();
    let rows = shape[0];
    let cols = shape[1];
    
    // Creiamo un nuovo array per il risultato
    let mut result_data = Array2::zeros((rows, cols));
    
    // Calcola softmax per ogni riga
    for i in 0..rows {
        let mut max_val = f32::NEG_INFINITY;
        
        // Trova il massimo nella riga (per stabilità numerica)
        for j in 0..cols {
            max_val = max_val.max(x.data[[i, j]]);
        }
        
        // Calcola exp(x_i - max) per ogni elemento
        let mut exp_vals = vec![0.0; cols];
        let mut exp_sum = 0.0;
        
        for j in 0..cols {
            let exp_val = (x.data[[i, j]] - max_val).exp();
            exp_vals[j] = exp_val;
            exp_sum += exp_val;
        }
        
        // Normalizza
        for j in 0..cols {
            result_data[[i, j]] = exp_vals[j] / exp_sum;
        }
    }
    
    Tensor::new(result_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_create_weight_matrix() {
        let input_dim = 64;
        let output_dim = 32;
        
        let weights = create_weight_matrix(input_dim, output_dim);
        
        // Verifica dimensioni corrette
        assert_eq!(weights.data.shape(), &[input_dim, output_dim]);
        
        // Verifica che i valori siano distribuiti attorno a 0
        let flat_values: Vec<f32> = weights.data.iter().cloned().collect();
        let sum: f32 = flat_values.iter().sum();
        let mean = sum / (input_dim * output_dim) as f32;
        
        // La media dovrebbe essere vicina a 0
        assert!(mean.abs() < 0.1);
    }
    
    #[test]
    fn test_softmax() {
        // Creiamo un tensore di test [6, 3] (rappresenta batch_size=2, seq_len=3)
        let mut test_data = Array2::zeros((6, 3));
        
        // Prima sequenza del primo batch
        test_data[[0, 0]] = 1.0;
        test_data[[0, 1]] = 2.0;
        test_data[[0, 2]] = 0.5;
        
        // Seconda sequenza del primo batch
        test_data[[1, 0]] = 0.0;
        test_data[[1, 1]] = 1.0;
        test_data[[1, 2]] = 0.0;
        
        // Terza sequenza del primo batch
        test_data[[2, 0]] = 1.0;
        test_data[[2, 1]] = 1.0;
        test_data[[2, 2]] = 1.0;
        
        // Prima sequenza del secondo batch
        test_data[[3, 0]] = 10.0;
        test_data[[3, 1]] = 10.0;
        test_data[[3, 2]] = 10.0;
        
        // Seconda sequenza del secondo batch
        test_data[[4, 0]] = -1.0;
        test_data[[4, 1]] = 0.0; 
        test_data[[4, 2]] = 1.0;
        
        // Terza sequenza del secondo batch
        test_data[[5, 0]] = 5.0;
        test_data[[5, 1]] = -5.0;
        test_data[[5, 2]] = 0.0;
        
        let input = Tensor::new(test_data);
        let result = softmax(&input);
        
        // Verifica le proprietà del softmax:
        // 1. La somma di ogni riga deve essere ~1.0
        // 2. Tutti i valori devono essere positivi e <= 1.0
        for i in 0..6 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                let val = result.data[[i, j]];
                assert!(val >= 0.0 && val <= 1.0, "Valore fuori range [0,1]: {}", val);
                row_sum += val;
            }
            assert!((row_sum - 1.0).abs() < 1e-5, "La somma della riga non è 1.0: {}", row_sum);
        }
        
        // Verifica alcuni valori specifici
        // Per la riga con valori uguali, il softmax dovrebbe dare probabilità uniformi
        assert!((result.data[[2, 0]] - 1.0/3.0).abs() < 1e-5);
        assert!((result.data[[2, 1]] - 1.0/3.0).abs() < 1e-5);
        assert!((result.data[[2, 2]] - 1.0/3.0).abs() < 1e-5);
        
        // Per la riga con un valore dominante, quel valore dovrebbe avere probabilità più alta
        assert!(result.data[[0, 1]] > result.data[[0, 0]]);
        assert!(result.data[[0, 1]] > result.data[[0, 2]]);
    }
} 