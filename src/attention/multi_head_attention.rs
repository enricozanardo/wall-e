use ndarray::{Array, Array2, Array3, Axis, Ix3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use crate::nabla::tensor::Tensor;
use crate::attention::{Attention, create_weight_matrix, softmax_3d};

/// Implementazione di Multi-Head Attention come descritto nel paper "Attention is All You Need"
pub struct MultiHeadAttention {
    /// Dimensione del modello (d_model)
    model_dimension: usize,
    
    /// Numero di teste di attenzione
    num_heads: usize,
    
    /// Dimensione di ciascuna testa (d_k)
    head_dimension: usize,
    
    /// Fattore di scala per l'attenzione (1/sqrt(d_k))
    scale_factor: f32,
    
    /// Matrici di proiezione per le query (una per testa)
    w_queries: Vec<Array2<f32>>,
    
    /// Matrici di proiezione per le chiavi (una per testa)
    w_keys: Vec<Array2<f32>>,
    
    /// Matrici di proiezione per i valori (una per testa)
    w_values: Vec<Array2<f32>>,
    
    /// Matrice di proiezione per l'output combinato
    w_output: Array2<f32>,
}

impl MultiHeadAttention {
    /// Crea una nuova istanza di MultiHeadAttention
    /// 
    /// # Arguments
    /// 
    /// * `model_dimension` - Dimensione del modello (d_model)
    /// * `num_heads` - Numero di teste di attenzione
    /// * `std` - Deviazione standard per l'inizializzazione dei pesi
    /// 
    /// # Returns
    /// 
    /// * Una nuova istanza di MultiHeadAttention
    pub fn new(model_dimension: usize, num_heads: usize, std: f32) -> Self {
        assert!(model_dimension % num_heads == 0, 
                "La dimensione del modello ({}) deve essere divisibile per il numero di teste ({})", 
                model_dimension, num_heads);
        
        let head_dimension = model_dimension / num_heads;
        let scale_factor = 1.0 / (head_dimension as f32).sqrt();
        
        // Inizializza le matrici di proiezione per ogni testa
        let mut w_queries = Vec::with_capacity(num_heads);
        let mut w_keys = Vec::with_capacity(num_heads);
        let mut w_values = Vec::with_capacity(num_heads);
        
        for _ in 0..num_heads {
            w_queries.push(create_weight_matrix(model_dimension, head_dimension, std));
            w_keys.push(create_weight_matrix(model_dimension, head_dimension, std));
            w_values.push(create_weight_matrix(model_dimension, head_dimension, std));
        }
        
        // Matrice di proiezione per l'output combinato
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
    
    /// Calcola i punteggi di attenzione per una singola testa
    /// 
    /// # Arguments
    /// 
    /// * `q` - Tensore delle query
    /// * `k` - Tensore delle chiavi
    /// * `mask` - Maschera di attenzione opzionale
    /// 
    /// # Returns
    /// 
    /// * Tensore dei punteggi di attenzione
    fn compute_attention_scores(&self, q: &Array3<f32>, k: &Array3<f32>, mask: Option<&Array3<f32>>) -> Array3<f32> {
        // Trasposizione delle chiavi per il prodotto matrice-matrice
        // k_transposed sarà di forma [batch_size, d_k, seq_len]
        let mut k_transposed = Array3::<f32>::zeros((k.shape()[0], k.shape()[2], k.shape()[1]));
        
        for b in 0..k.shape()[0] {
            for i in 0..k.shape()[1] {
                for j in 0..k.shape()[2] {
                    k_transposed[[b, j, i]] = k[[b, i, j]];
                }
            }
        }
        
        // Calcola il prodotto matrice-matrice q * k_t
        // Risultato sarà di forma [batch_size, seq_len_q, seq_len_k]
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
        
        // Applica la maschera se presente
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
        
        // Applica softmax per ottenere i pesi di attenzione
        softmax_3d(&scores, 2)
    }
    
    /// Applica i pesi di attenzione ai valori per una singola testa
    /// 
    /// # Arguments
    /// 
    /// * `attention_weights` - Pesi di attenzione
    /// * `v` - Tensore dei valori
    /// 
    /// # Returns
    /// 
    /// * Tensore dell'output ponderato
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

impl Attention for MultiHeadAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        // Ottieni i dati come Array3
        let q_data = q.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let k_data = k.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        let v_data = v.data.clone().into_dimensionality::<Ix3>().unwrap();
        
        // Converti la maschera se presente
        let mask_data = mask.map(|m| {
            m.data.clone().into_dimensionality::<Ix3>().unwrap()
        });
        
        // Crea un tensore per l'output concatenato di tutte le teste
        let mut concatenated_heads = Array3::<f32>::zeros((
            q_data.shape()[0],         // batch_size
            q_data.shape()[1],         // seq_len
            self.model_dimension,      // d_model (= num_heads * head_dimension)
        ));
        
        // Elabora ogni testa separatamente
        for h in 0..self.num_heads {
            // Proietta query, key e value per questa testa
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
            
            // Applica le proiezioni
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
            
            // Calcola i punteggi di attenzione e applica l'attenzione per questa testa
            let attention_weights = self.compute_attention_scores(&q_proj, &k_proj, mask_data.as_ref());
            let head_output = self.apply_attention(&attention_weights, &v_proj);
            
            // Concatena l'output di questa testa
            let head_offset = h * self.head_dimension;
            for b in 0..head_output.shape()[0] {
                for i in 0..head_output.shape()[1] {
                    for j in 0..head_output.shape()[2] {
                        concatenated_heads[[b, i, head_offset + j]] = head_output[[b, i, j]];
                    }
                }
            }
        }
        
        // Proietta l'output concatenato
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
        
        // Converti il risultato in un Tensor
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
        // Verifica che la creazione fallisca se model_dimension non è divisibile per num_heads
        let result = std::panic::catch_unwind(|| {
            MultiHeadAttention::new(10, 3, 0.1);
        });
        assert!(result.is_err());
        
        // Verifica che la creazione abbia successo se model_dimension è divisibile per num_heads
        let mha = MultiHeadAttention::new(12, 3, 0.1);
        assert_eq!(mha.model_dimension, 12);
        assert_eq!(mha.num_heads, 3);
        assert_eq!(mha.head_dimension, 4);
    }
    
    #[test]
    fn test_multi_head_attention_forward() {
        // Crea una istanza di MultiHeadAttention
        let model_dim = 12;
        let num_heads = 3;
        let attention = MultiHeadAttention::new(model_dim, num_heads, 0.1);
        
        // Crea tensori di input di esempio
        let batch_size = 2;
        let seq_len = 4;
        
        let q_data = Array3::<f32>::ones((batch_size, seq_len, model_dim));
        let k_data = q_data.clone();
        let v_data = q_data.clone();
        
        let q = Tensor::new_3d(q_data);
        let k = Tensor::new_3d(k_data);
        let v = Tensor::new_3d(v_data);
        
        // Calcola l'output dell'attenzione
        let output = attention.forward(&q, &k, &v, None);
        
        // Verifica le dimensioni dell'output
        let output_data = output.data.clone().into_dimensionality::<Ix3>().unwrap();
        assert_eq!(output_data.shape()[0], batch_size);
        assert_eq!(output_data.shape()[1], seq_len);
        assert_eq!(output_data.shape()[2], model_dim);
        
        // Verifica che tutti i valori siano numeri validi (non NaN o infiniti)
        for v in output_data.iter() {
            assert!(!v.is_nan() && !v.is_infinite());
        }
    }
    
    #[test]
    fn test_multi_head_attention_with_mask() {
        // Crea una istanza di MultiHeadAttention
        let model_dim = 12;
        let num_heads = 3;
        let attention = MultiHeadAttention::new(model_dim, num_heads, 0.1);
        
        // Crea tensori di input di esempio
        let batch_size = 2;
        let seq_len = 4;
        
        // Creiamo input non uniformi per rendere più evidente l'effetto della maschera
        let mut q_data = Array3::<f32>::zeros((batch_size, seq_len, model_dim));
        let mut k_data = Array3::<f32>::zeros((batch_size, seq_len, model_dim));
        let mut v_data = Array3::<f32>::zeros((batch_size, seq_len, model_dim));
        
        // Inizializziamo i dati con valori che creano una forte dipendenza sulle posizioni future
        // Nel caso di q_data, valori crescenti nella dimensione della sequenza
        // Nel caso di k_data, valori decrescenti nella dimensione della sequenza
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..model_dim {
                    q_data[[i, j, k]] = (j + 1) as f32;                    // Valori crescenti nella seq
                    k_data[[i, j, k]] = (seq_len - j) as f32;              // Valori decrescenti nella seq
                    v_data[[i, j, k]] = (j + 1) as f32 * (seq_len - j) as f32; // Prodotti dei due
                }
            }
        }
        
        // Crea una maschera causale (ogni posizione può vedere solo le posizioni precedenti)
        let mut mask_data = Array3::<f32>::ones((batch_size, seq_len, seq_len));
        
        // Maschera causale: posizione i può vedere solo posizioni j <= i
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
        
        // Calcola l'output dell'attenzione con maschera
        let output_with_mask = attention.forward(&q, &k, &v, Some(&mask));
        
        // Calcola l'output dell'attenzione senza maschera
        let output_no_mask = attention.forward(&q, &k, &v, None);
        
        // Verifica che l'output con maschera sia diverso dall'output senza maschera
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
        
        // Stampiamo informazioni sulla differenza
        println!("Differenze trovate: {}, max diff: {}", diff_count, max_diff);
        
        // L'output con maschera dovrebbe essere diverso dall'output senza maschera
        assert!(!all_equal, "L'output con maschera dovrebbe essere diverso dall'output senza maschera");
    }
} 