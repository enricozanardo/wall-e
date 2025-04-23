use ndarray::{Array, Array2, Array3, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use crate::nabla::tensor::Tensor;

// Esporto i moduli
pub mod self_attention;
pub mod multi_head_attention;
pub mod feed_forward;

// Esporto le strutture
pub use self_attention::SelfAttention;
pub use multi_head_attention::MultiHeadAttention;
pub use feed_forward::{FeedForward, BatchedFeedForward};

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
/// Tensor [seq_len, seq_len] contenente la maschera causale
pub fn create_causal_mask(seq_len: usize) -> Tensor {
    let mut mask_data = Array2::zeros((seq_len, seq_len));
    
    // Imposta 1 sulla diagonale e sotto (triangolare inferiore)
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[[i, j]] = 1.0;
        }
    }
    
    Tensor::new(mask_data)
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
    let mut mask_data = Array2::zeros((batch_size * seq_len, seq_len));
    
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        for i in 0..seq_len {
            let row_idx = b * seq_len + i;
            
            // Se i è una posizione valida, permetti di prestare attenzione fino a valid_len
            if i < valid_len {
                for j in 0..valid_len {
                    mask_data[[row_idx, j]] = 1.0;
                }
            }
            // Altrimenti, non prestare attenzione a nessuna posizione (riga di tutti 0)
        }
    }
    
    Tensor::new(mask_data)
}

/// Combina una maschera causale con una maschera di padding
/// Utile per decoder con padding
/// 
/// # Arguments
/// * `seq_len` - Lunghezza massima della sequenza
/// * `valid_lens` - Vettore con le lunghezze valide per ogni sequenza nel batch
/// 
/// # Returns
/// Tensor [batch_size * seq_len, seq_len] con la maschera combinata
pub fn create_combined_mask(seq_len: usize, valid_lens: &[usize]) -> Tensor {
    let batch_size = valid_lens.len();
    let mut mask_data = Array2::zeros((batch_size * seq_len, seq_len));
    
    for (b, &valid_len) in valid_lens.iter().enumerate() {
        for i in 0..seq_len {
            let row_idx = b * seq_len + i;
            
            // Se i è una posizione valida
            if i < valid_len {
                // Applica sia la maschera causale che quella di padding
                for j in 0..=i {
                    if j < valid_len {
                        mask_data[[row_idx, j]] = 1.0;
                    }
                }
            }
            // Altrimenti, non prestare attenzione a nessuna posizione
        }
    }
    
    Tensor::new(mask_data)
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

/// Calcola la funzione softmax su un tensore 3D lungo l'asse specificato
/// 
/// # Arguments
/// 
/// * `x` - Il tensore di input
/// * `axis` - L'asse lungo cui calcolare softmax
/// 
/// # Returns
/// 
/// * Il tensore con softmax applicato
pub fn softmax_3d(x: &Array3<f32>, axis: usize) -> Array3<f32> {
    let shape = x.shape();
    let mut result = Array3::<f32>::zeros(x.raw_dim());
    
    if axis == 2 {
        // Caso più comune: softmax lungo l'ultima dimensione
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                // Estrai la slice
                let slice = x.slice(s![i, j, ..]);
                
                // Calcola il massimo per stabilità numerica
                let max_val = slice.fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
                
                // Calcola exp(x - max)
                let mut exp_values = vec![0.0; shape[2]];
                let mut sum = 0.0;
                
                for k in 0..shape[2] {
                    let exp_val = (slice[k] - max_val).exp();
                    exp_values[k] = exp_val;
                    sum += exp_val;
                }
                
                // Normalizza in modo che la somma sia esattamente 1.0
                if sum <= f32::EPSILON {
                    // Se la somma è quasi zero, usa una distribuzione uniforme
                    let uniform_val = 1.0 / shape[2] as f32;
                    for k in 0..shape[2] {
                        result[[i, j, k]] = uniform_val;
                    }
                } else {
                    for k in 0..shape[2] {
                        result[[i, j, k]] = exp_values[k] / sum;
                    }
                }
                
                // Controllo di sicurezza: verifica che la somma sia 1.0
                let final_sum: f32 = result.slice(s![i, j, ..]).sum();
                if (final_sum - 1.0).abs() > 1e-5 {
                    // Aggiusta il primo valore per ottenere una somma esatta di 1.0
                    result[[i, j, 0]] += 1.0 - final_sum;
                }
            }
        }
    } else if axis == 1 {
        // Softmax lungo la seconda dimensione
        for i in 0..shape[0] {
            for k in 0..shape[2] {
                // Estrai la slice
                let mut exp_values = vec![0.0; shape[1]];
                let mut max_val = std::f32::NEG_INFINITY;
                
                // Trova il massimo
                for j in 0..shape[1] {
                    max_val = max_val.max(x[[i, j, k]]);
                }
                
                // Calcola exp(x - max) e la somma
                let mut sum = 0.0;
                for j in 0..shape[1] {
                    let exp_val = (x[[i, j, k]] - max_val).exp();
                    exp_values[j] = exp_val;
                    sum += exp_val;
                }
                
                // Normalizza
                if sum <= f32::EPSILON {
                    let uniform_val = 1.0 / shape[1] as f32;
                    for j in 0..shape[1] {
                        result[[i, j, k]] = uniform_val;
                    }
                } else {
                    for j in 0..shape[1] {
                        result[[i, j, k]] = exp_values[j] / sum;
                    }
                }
                
                // Controllo di sicurezza
                let mut final_sum = 0.0;
                for j in 0..shape[1] {
                    final_sum += result[[i, j, k]];
                }
                
                if (final_sum - 1.0).abs() > 1e-5 {
                    result[[i, 0, k]] += 1.0 - final_sum;
                }
            }
        }
    } else if axis == 0 {
        // Softmax lungo la prima dimensione
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Estrai la slice
                let mut exp_values = vec![0.0; shape[0]];
                let mut max_val = std::f32::NEG_INFINITY;
                
                // Trova il massimo
                for i in 0..shape[0] {
                    max_val = max_val.max(x[[i, j, k]]);
                }
                
                // Calcola exp(x - max) e la somma
                let mut sum = 0.0;
                for i in 0..shape[0] {
                    let exp_val = (x[[i, j, k]] - max_val).exp();
                    exp_values[i] = exp_val;
                    sum += exp_val;
                }
                
                // Normalizza
                if sum <= f32::EPSILON {
                    let uniform_val = 1.0 / shape[0] as f32;
                    for i in 0..shape[0] {
                        result[[i, j, k]] = uniform_val;
                    }
                } else {
                    for i in 0..shape[0] {
                        result[[i, j, k]] = exp_values[i] / sum;
                    }
                }
                
                // Controllo di sicurezza
                let mut final_sum = 0.0;
                for i in 0..shape[0] {
                    final_sum += result[[i, j, k]];
                }
                
                if (final_sum - 1.0).abs() > 1e-5 {
                    result[[0, j, k]] += 1.0 - final_sum;
                }
            }
        }
    } else {
        panic!("Asse non valido per softmax_3d, deve essere 0, 1 o 2");
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
                assert!((sum - 1.0).abs() < 1e-5);
            }
        }
        
        // Verifica che tutti i valori siano positivi
        for v in result.iter() {
            assert!(*v > 0.0);
        }
    }
    
    #[test]
    fn test_causal_mask() {
        let seq_len = 5;
        
        // Crea una maschera causale
        let mask = create_causal_mask(seq_len);
        
        // Verifica le dimensioni della maschera
        assert_eq!(mask.data.shape(), &[seq_len, seq_len]);
        
        // Verifica che la maschera abbia 1 sulla diagonale e sotto, 0 sopra
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    assert_eq!(mask.data[[i, j]], 1.0, "Maschera dovrebbe essere 1 a [{}, {}]", i, j);
                } else {
                    assert_eq!(mask.data[[i, j]], 0.0, "Maschera dovrebbe essere 0 a [{}, {}]", i, j);
                }
            }
        }
    }
    
    #[test]
    fn test_padding_mask() {
        let seq_len = 4;
        let valid_lens = vec![2, 3]; // Batch di 2 sequenze con lunghezze valide 2 e 3
        
        // Crea una maschera di padding
        let mask = create_padding_mask(seq_len, &valid_lens);
        
        // Verifica le dimensioni della maschera
        assert_eq!(mask.data.shape(), &[valid_lens.len() * seq_len, seq_len]);
        
        // Verifica prima sequenza (lunghezza 2)
        // Prima riga (token 0)
        assert_eq!(mask.data[[0, 0]], 1.0);
        assert_eq!(mask.data[[0, 1]], 1.0);
        assert_eq!(mask.data[[0, 2]], 0.0);
        assert_eq!(mask.data[[0, 3]], 0.0);
        
        // Seconda riga (token 1)
        assert_eq!(mask.data[[1, 0]], 1.0);
        assert_eq!(mask.data[[1, 1]], 1.0);
        assert_eq!(mask.data[[1, 2]], 0.0);
        assert_eq!(mask.data[[1, 3]], 0.0);
        
        // Terza e quarta riga (padding)
        for j in 0..seq_len {
            assert_eq!(mask.data[[2, j]], 0.0);
            assert_eq!(mask.data[[3, j]], 0.0);
        }
        
        // Verifica seconda sequenza (lunghezza 3)
        // Prima riga (token 0)
        assert_eq!(mask.data[[4, 0]], 1.0);
        assert_eq!(mask.data[[4, 1]], 1.0);
        assert_eq!(mask.data[[4, 2]], 1.0);
        assert_eq!(mask.data[[4, 3]], 0.0);
        
        // Seconda riga (token 1)
        assert_eq!(mask.data[[5, 0]], 1.0);
        assert_eq!(mask.data[[5, 1]], 1.0);
        assert_eq!(mask.data[[5, 2]], 1.0);
        assert_eq!(mask.data[[5, 3]], 0.0);
        
        // Terza riga (token 2)
        assert_eq!(mask.data[[6, 0]], 1.0);
        assert_eq!(mask.data[[6, 1]], 1.0);
        assert_eq!(mask.data[[6, 2]], 1.0);
        assert_eq!(mask.data[[6, 3]], 0.0);
        
        // Quarta riga (padding)
        for j in 0..seq_len {
            assert_eq!(mask.data[[7, j]], 0.0);
        }
    }
    
    #[test]
    fn test_combined_mask() {
        let seq_len = 4;
        let valid_lens = vec![2, 3]; // Batch di 2 sequenze con lunghezze valide 2 e 3
        
        // Crea una maschera combinata
        let mask = create_combined_mask(seq_len, &valid_lens);
        
        // Verifica le dimensioni della maschera
        assert_eq!(mask.data.shape(), &[valid_lens.len() * seq_len, seq_len]);
        
        // Verifica prima sequenza (lunghezza 2)
        // Prima riga (token 0) - può vedere solo sé stesso
        assert_eq!(mask.data[[0, 0]], 1.0);
        assert_eq!(mask.data[[0, 1]], 0.0);
        assert_eq!(mask.data[[0, 2]], 0.0);
        assert_eq!(mask.data[[0, 3]], 0.0);
        
        // Seconda riga (token 1) - può vedere sé stesso e token 0
        assert_eq!(mask.data[[1, 0]], 1.0);
        assert_eq!(mask.data[[1, 1]], 1.0);
        assert_eq!(mask.data[[1, 2]], 0.0);
        assert_eq!(mask.data[[1, 3]], 0.0);
        
        // Righe di padding - tutti 0
        for i in 2..4 {
            for j in 0..seq_len {
                assert_eq!(mask.data[[i, j]], 0.0);
            }
        }
        
        // Verifica seconda sequenza (lunghezza 3)
        // Prima riga (token 0) - può vedere solo sé stesso
        assert_eq!(mask.data[[4, 0]], 1.0);
        assert_eq!(mask.data[[4, 1]], 0.0);
        assert_eq!(mask.data[[4, 2]], 0.0);
        assert_eq!(mask.data[[4, 3]], 0.0);
        
        // Seconda riga (token 1) - può vedere sé stesso e token 0
        assert_eq!(mask.data[[5, 0]], 1.0);
        assert_eq!(mask.data[[5, 1]], 1.0);
        assert_eq!(mask.data[[5, 2]], 0.0);
        assert_eq!(mask.data[[5, 3]], 0.0);
        
        // Terza riga (token 2) - può vedere sé stesso e token 0, 1
        assert_eq!(mask.data[[6, 0]], 1.0);
        assert_eq!(mask.data[[6, 1]], 1.0);
        assert_eq!(mask.data[[6, 2]], 1.0);
        assert_eq!(mask.data[[6, 3]], 0.0);
        
        // Riga di padding - tutti 0
        for j in 0..seq_len {
            assert_eq!(mask.data[[7, j]], 0.0);
        }
    }
} 