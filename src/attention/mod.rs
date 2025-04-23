use ndarray::{Array, Array2, Array3, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use crate::nabla::tensor::Tensor;

// Esporto i moduli
pub mod self_attention;
pub mod multi_head_attention;
pub mod feed_forward;
pub mod encoder;
pub mod encoder_stack;

// Esporto le strutture
pub use self_attention::SelfAttention;
pub use multi_head_attention::MultiHeadAttention;
pub use feed_forward::{FeedForward, BatchedFeedForward};
pub use encoder::EncoderLayer;
pub use encoder_stack::EncoderStack;
pub use encoder::layer_norm;

/// Trait che definisce un'interfaccia comune per meccanismi di attenzione
pub trait Attention {
    /// Esegue il forward pass del meccanismo di attenzione
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor;
    
    /// Restituisce la dimensione del modello
    fn model_dim(&self) -> usize;
}

/// Crea una maschera causale per impedire l'attenzione a posizioni future
/// La maschera ha 1 sulla diagonale e sotto, 0 sopra la diagonale
/// 
/// # Arguments
/// * `seq_len` - Lunghezza della sequenza
/// 
/// # Returns
/// Tensor [1, seq_len, seq_len] contenente la maschera causale
pub fn create_causal_mask(seq_len: usize) -> Tensor {
    let mut mask_data = Array3::zeros((1, seq_len, seq_len));
    
    // Imposta 1 sulla diagonale e sotto (triangolare inferiore)
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[[0, i, j]] = 1.0;
        }
    }
    
    Tensor::new_3d(mask_data)
}

/// Crea una maschera di padding per ignorare i token di padding
/// 
/// # Arguments
/// * `seq_len` - Lunghezza massima della sequenza
/// * `valid_lens` - Vettore con le lunghezze valide per ogni sequenza nel batch
/// 
/// # Returns
/// Tensor [batch_size, seq_len, seq_len] contenente maschere per ogni sequenza
pub fn create_padding_mask(seq_len: usize, valid_lens: &[usize]) -> Tensor {
    let batch_size = valid_lens.len();
    let mut mask_data = Array3::zeros((batch_size, seq_len, seq_len));
    
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        for i in 0..seq_len {
            // Se i è una posizione valida, permetti di prestare attenzione fino a valid_len
            if i < valid_len {
                for j in 0..valid_len {
                    mask_data[[b, i, j]] = 1.0;
                }
            }
            // Altrimenti, non prestare attenzione a nessuna posizione (riga di tutti 0)
        }
    }
    
    Tensor::new_3d(mask_data)
}

/// Combina una maschera causale con una maschera di padding
/// Utile per decoder con padding
/// 
/// # Arguments
/// * `seq_len` - Lunghezza massima della sequenza
/// * `valid_lens` - Vettore con le lunghezze valide per ogni sequenza nel batch
/// 
/// # Returns
/// Tensor [batch_size, seq_len, seq_len] con la maschera combinata
pub fn create_combined_mask(seq_len: usize, valid_lens: &[usize]) -> Tensor {
    let batch_size = valid_lens.len();
    let mut mask_data = Array3::zeros((batch_size, seq_len, seq_len));
    
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        for i in 0..seq_len {
            // Se i è una posizione valida
            if i < valid_len {
                // Applica sia la maschera causale che quella di padding
                for j in 0..=i {
                    if j < valid_len {
                        mask_data[[b, i, j]] = 1.0;
                    }
                }
            }
            // Altrimenti, non prestare attenzione a nessuna posizione
        }
    }
    
    Tensor::new_3d(mask_data)
}

/// Crea una matrice di pesi utilizzando una distribuzione normale
/// 
/// # Arguments
/// 
/// * `in_features` - Numero di feature di input
/// * `out_features` - Numero di feature di output
/// * `std` - Deviazione standard per l'inizializzazione
/// 
/// # Returns
/// 
/// * Una matrice di pesi inizializzata con distribuzione normale
pub fn create_weight_matrix(in_features: usize, out_features: usize, std: f32) -> Array2<f32> {
    Array::random((in_features, out_features), Normal::new(0.0, std as f64).unwrap())
        .mapv(|x| x as f32)
}

/// Calcola il softmax 3D lungo un asse specifico
/// 
/// # Arguments
/// 
/// * `x` - Tensore di input 3D
/// * `axis` - Asse lungo cui calcolare il softmax (0, 1, o 2)
/// 
/// # Returns
/// 
/// Un nuovo tensore 3D con il softmax applicato
pub fn softmax_3d(x: &Array3<f32>, axis: usize) -> Array3<f32> {
    assert!(axis <= 2, "L'asse deve essere 0, 1 o 2");
    
    let shape = x.shape();
    let mut result = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    
    // Valore molto negativo ma finito per sostituire -infinity
    let very_negative_value = -1e30f32;
    
    // Applicazione softmax in base all'asse specificato
    if axis == 0 {
        // Softmax lungo l'asse 0 (batch)
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Estrai la slice
                let mut slice = Vec::with_capacity(shape[0]);
                for i in 0..shape[0] {
                    // Sostituisci -infinity con un valore molto negativo ma finito
                    if x[[i, j, k]].is_infinite() && x[[i, j, k]] < 0.0 {
                        slice.push(very_negative_value);
                    } else {
                        slice.push(x[[i, j, k]]);
                    }
                }
                
                // Trova il valore massimo per stabilità numerica
                let max_val = slice.iter().fold(f32::MIN, |a, &b| a.max(b));
                
                // Calcola exp(x - max) per ciascun elemento
                let mut exp_vals = Vec::with_capacity(shape[0]);
                for val in slice.iter() {
                    exp_vals.push((*val - max_val).exp());
                }
                
                // Calcola la somma per normalizzare
                let sum: f32 = exp_vals.iter().sum();
                
                // Gestisci il caso in cui sum è vicino a zero
                let safe_sum = if sum < 1e-10 {
                    // Se sum è quasi zero, distribuisci uniformemente
                    for i in 0..shape[0] {
                        result[[i, j, k]] = 1.0 / (shape[0] as f32);
                    }
                    continue;
                } else {
                    sum
                };
                
                // Applica la normalizzazione
                for i in 0..shape[0] {
                    result[[i, j, k]] = exp_vals[i] / safe_sum;
                }
                
                // Verifica e correggi eventuali problemi numerici per garantire somma = 1
                let actual_sum: f32 = (0..shape[0]).map(|i| result[[i, j, k]]).sum();
                if (actual_sum - 1.0).abs() > 1e-5 {
                    // Aggiusta il primo valore per garantire somma = 1
                    result[[0, j, k]] += 1.0 - actual_sum;
                }
            }
        }
    } else if axis == 1 {
        // Softmax lungo l'asse 1 (sequenza)
        for i in 0..shape[0] {
            for k in 0..shape[2] {
                // Estrai la slice
                let mut slice = Vec::with_capacity(shape[1]);
                for j in 0..shape[1] {
                    // Sostituisci -infinity con un valore molto negativo ma finito
                    if x[[i, j, k]].is_infinite() && x[[i, j, k]] < 0.0 {
                        slice.push(very_negative_value);
                    } else {
                        slice.push(x[[i, j, k]]);
                    }
                }
                
                // Trova il valore massimo per stabilità numerica
                let max_val = slice.iter().fold(f32::MIN, |a, &b| a.max(b));
                
                // Calcola exp(x - max) per ciascun elemento
                let mut exp_vals = Vec::with_capacity(shape[1]);
                for val in slice.iter() {
                    exp_vals.push((*val - max_val).exp());
                }
                
                // Calcola la somma per normalizzare
                let sum: f32 = exp_vals.iter().sum();
                
                // Gestisci il caso in cui sum è vicino a zero
                let safe_sum = if sum < 1e-10 {
                    // Se sum è quasi zero, distribuisci uniformemente
                    for j in 0..shape[1] {
                        result[[i, j, k]] = 1.0 / (shape[1] as f32);
                    }
                    continue;
                } else {
                    sum
                };
                
                // Applica la normalizzazione
                for j in 0..shape[1] {
                    result[[i, j, k]] = exp_vals[j] / safe_sum;
                }
                
                // Verifica e correggi eventuali problemi numerici per garantire somma = 1
                let actual_sum: f32 = (0..shape[1]).map(|j| result[[i, j, k]]).sum();
                if (actual_sum - 1.0).abs() > 1e-5 {
                    // Aggiusta il primo valore per garantire somma = 1
                    result[[i, 0, k]] += 1.0 - actual_sum;
                }
            }
        }
    } else if axis == 2 {
        // Softmax lungo l'asse 2 (feature)
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                // Estrai la slice
                let mut slice = Vec::with_capacity(shape[2]);
                for k in 0..shape[2] {
                    // Sostituisci -infinity con un valore molto negativo ma finito
                    if x[[i, j, k]].is_infinite() && x[[i, j, k]] < 0.0 {
                        slice.push(very_negative_value);
                    } else {
                        slice.push(x[[i, j, k]]);
                    }
                }
                
                // Trova il valore massimo per stabilità numerica
                let max_val = slice.iter().fold(f32::MIN, |a, &b| a.max(b));
                
                // Calcola exp(x - max) per ciascun elemento
                let mut exp_vals = Vec::with_capacity(shape[2]);
                for val in slice.iter() {
                    exp_vals.push((*val - max_val).exp());
                }
                
                // Calcola la somma per normalizzare
                let sum: f32 = exp_vals.iter().sum();
                
                // Gestisci il caso in cui sum è vicino a zero
                let safe_sum = if sum < 1e-10 {
                    // Se sum è quasi zero, distribuisci uniformemente
                    for k in 0..shape[2] {
                        result[[i, j, k]] = 1.0 / (shape[2] as f32);
                    }
                    continue;
                } else {
                    sum
                };
                
                // Applica la normalizzazione
                for k in 0..shape[2] {
                    result[[i, j, k]] = exp_vals[k] / safe_sum;
                }
                
                // Verifica e correggi eventuali problemi numerici per garantire somma = 1
                let actual_sum: f32 = (0..shape[2]).map(|k| result[[i, j, k]]).sum();
                if (actual_sum - 1.0).abs() > 1e-5 {
                    // Aggiusta il primo valore per garantire somma = 1
                    result[[i, j, 0]] += 1.0 - actual_sum;
                }
            }
        }
    }
    
    // Verifica finale per assicurarsi che non ci siano NaN
    for val in result.iter_mut() {
        if val.is_nan() {
            *val = 0.0; // Sostituisci NaN con 0
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_create_weight_matrix() {
        let w = create_weight_matrix(10, 20, 0.1);
        
        // Verifica le dimensioni
        assert_eq!(w.shape(), &[10, 20]);
        
        // Verifica che la media sia approssimativamente zero
        let mean = w.mean().unwrap();
        assert!(mean.abs() < 0.1);
        
        // Verifica che la deviazione standard sia approssimativamente 0.1
        let std_dev = (w.mapv(|x| x.powi(2)).sum() / (w.len() as f32)).sqrt();
        assert!((std_dev - 0.1).abs() < 0.05);
        
        // Verifica che i valori siano distribuiti uniformemente attorno allo zero
        let positive_count = w.iter().filter(|&&x| x > 0.0).count();
        let total_count = w.len();
        let positive_ratio = positive_count as f32 / total_count as f32;
        assert!((positive_ratio - 0.5).abs() < 0.1, "Ci si aspetta una distribuzione uniforme, trovato {:.2}% di valori positivi", positive_ratio * 100.0);
        
        // Verifica che non ci siano valori NaN o infiniti
        for &val in w.iter() {
            assert!(!val.is_nan() && !val.is_infinite(), "Trovato valore non valido nella matrice di pesi");
        }
    }
    
    #[test]
    fn test_softmax_3d() {
        // Crea un tensore 3D di esempio
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
        
        // Calcola softmax lungo l'asse 2
        let result = softmax_3d(&x, 2);
        
        // Verifica che le somme siano 1.0 per ogni slice
        for i in 0..2 {
            for j in 0..3 {
                let sum: f32 = result.slice(s![i, j, ..]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "Somma = {} per slice [{}][{}], dovrebbe essere 1.0", sum, i, j);
            }
        }
        
        // Verifica che tutti i valori siano positivi
        for v in result.iter() {
            assert!(*v > 0.0, "Valore softmax deve essere positivo, trovato: {}", v);
        }
        
        // Test con softmax lungo asse 0
        let result_axis0 = softmax_3d(&x, 0);
        
        // Verifica le somme lungo l'asse 0
        for j in 0..3 {
            for k in 0..4 {
                let sum: f32 = result_axis0.slice(s![.., j, k]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "Somma = {} per slice [*][{}][{}], dovrebbe essere 1.0", sum, j, k);
            }
        }
        
        // Test con softmax lungo asse 1
        let result_axis1 = softmax_3d(&x, 1);
        
        // Verifica le somme lungo l'asse 1
        for i in 0..2 {
            for k in 0..4 {
                let sum: f32 = result_axis1.slice(s![i, .., k]).sum();
                assert!((sum - 1.0).abs() < 1e-5, "Somma = {} per slice [{}][*][{}], dovrebbe essere 1.0", sum, i, k);
            }
        }
        
        // Test con valori estremi (molto grandi)
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
        
        // Per valori estremi, il risultato dovrebbe essere quasi 0-1
        for i in 0..2 {
            for j in 0..2 {
                let max_idx = if large_vals[[i, j, 0]] > large_vals[[i, j, 1]] { 0 } else { 1 };
                let min_idx = 1 - max_idx;
                assert!(result_large[[i, j, max_idx]] > 0.99, 
                        "Per valori estremi, il softmax dovrebbe essere vicino a 1.0 per il valore massimo");
                assert!(result_large[[i, j, min_idx]] < 0.01, 
                        "Per valori estremi, il softmax dovrebbe essere vicino a 0.0 per il valore minimo");
            }
        }
        
        // Test con -infinity (simulazione di maschere)
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
        
        // Le posizioni con -infinity dovrebbero avere probabilità 0
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    if mask_test[[i, j, k]].is_infinite() && mask_test[[i, j, k]] < 0.0 {
                        assert!(result_mask[[i, j, k]] < 1e-6, 
                                "Posizioni mascherate (-inf) dovrebbero avere probabilità ≈ 0");
                    } else {
                        assert!((result_mask[[i, j, k]] - 1.0).abs() < 1e-5, 
                                "Posizioni non mascherate dovrebbero avere probabilità ≈ 1");
                    }
                }
                
                // La somma delle probabilità dovrebbe essere 1 anche con valori mascherati
                let sum: f32 = result_mask.slice(s![i, j, ..]).sum();
                assert!((sum - 1.0).abs() < 1e-5, 
                        "La somma delle probabilità dovrebbe essere 1 anche con maschere");
            }
        }
    }
    
    #[test]
    fn test_causal_mask() {
        let seq_len = 5;
        
        // Crea una maschera causale
        let mask = create_causal_mask(seq_len);
        
        // Verifica le dimensioni della maschera
        assert_eq!(mask.data.shape(), &[1, seq_len, seq_len]);
        
        // Verifica che la maschera abbia 1 sulla diagonale e sotto, 0 sopra
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[0, i, j]], expected, 
                           "La maschera causale nella posizione [{}, {}] dovrebbe essere {}", i, j, expected);
            }
        }
        
        // Test per lunghezza di sequenza = 1
        let mask_1 = create_causal_mask(1);
        assert_eq!(mask_1.data.shape(), &[1, 1, 1]);
        assert_eq!(mask_1.data[[0, 0, 0]], 1.0);
        
        // Test per lunghezza di sequenza grande
        let large_seq_len = 100;
        let large_mask = create_causal_mask(large_seq_len);
        assert_eq!(large_mask.data.shape(), &[1, large_seq_len, large_seq_len]);
        
        // Verifica alcuni punti di esempio
        assert_eq!(large_mask.data[[0, 0, 0]], 1.0);  // Diagonale
        assert_eq!(large_mask.data[[0, 99, 99]], 1.0); // Diagonale
        assert_eq!(large_mask.data[[0, 99, 0]], 1.0);  // Sotto diagonale
        assert_eq!(large_mask.data[[0, 0, 99]], 0.0);  // Sopra diagonale
    }
    
    #[test]
    fn test_padding_mask() {
        // Test base
        let seq_len = 5;
        let valid_lens = vec![3, 4];
        
        let mask = create_padding_mask(seq_len, &valid_lens);
        
        // Verifica le dimensioni della maschera
        assert_eq!(mask.data.shape(), &[valid_lens.len(), seq_len, seq_len]);
        
        // Verifica i valori della maschera per il primo batch (valid_len = 3)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if i < valid_lens[0] && j < valid_lens[0] { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[0, i, j]], expected, 
                           "La maschera di padding [0, {}, {}] dovrebbe essere {}", i, j, expected);
            }
        }
        
        // Verifica i valori della maschera per il secondo batch (valid_len = 4)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if i < valid_lens[1] && j < valid_lens[1] { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[1, i, j]], expected, 
                           "La maschera di padding [1, {}, {}] dovrebbe essere {}", i, j, expected);
            }
        }
    }
    
    #[test]
    fn test_combined_mask() {
        let seq_len = 5;
        let valid_lens = vec![3, 4];
        
        let mask = create_combined_mask(seq_len, &valid_lens);
        
        // Verifica le dimensioni della maschera
        assert_eq!(mask.data.shape(), &[valid_lens.len(), seq_len, seq_len]);
        
        // Verifica i valori della maschera per il primo batch (valid_len = 3)
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Nella maschera combinata, un elemento è visibile se:
                // 1. È nella parte valida (non padding)
                // 2. È in una posizione causale (j <= i)
                let expected = if i < valid_lens[0] && j < valid_lens[0] && j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[0, i, j]], expected, 
                           "La maschera combinata [0, {}, {}] dovrebbe essere {}", i, j, expected);
            }
        }
        
        // Verifica i valori della maschera per il secondo batch (valid_len = 4)
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if i < valid_lens[1] && j < valid_lens[1] && j <= i { 1.0 } else { 0.0 };
                assert_eq!(mask.data[[1, i, j]], expected, 
                           "La maschera combinata [1, {}, {}] dovrebbe essere {}", i, j, expected);
            }
        }
    }
} 