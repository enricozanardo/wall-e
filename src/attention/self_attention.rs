use crate::nabla::tensor::Tensor;
use crate::attention::{Attention, create_weight_matrix, softmax};
use ndarray::{Array2, Axis};
use std::f32;

/// Implementazione di Self-Attention come descritto nel paper "Attention is All You Need"
/// 
/// Self-Attention permette ad ogni posizione nella sequenza di prestare attenzione
/// a tutte le altre posizioni nella stessa sequenza.
pub struct SelfAttention {
    /// Dimensione del modello (d_model)
    d_model: usize,
    /// Dimensione delle chiavi (d_k)
    d_k: usize,
    /// Dimensione dei valori (d_v)
    d_v: usize,
    /// Matrice di proiezione di query
    W_q: Tensor,
    /// Matrice di proiezione di key
    W_k: Tensor,
    /// Matrice di proiezione di value
    W_v: Tensor,
    /// Matrice di proiezione per l'output
    W_o: Tensor,
}

impl SelfAttention {
    /// Crea una nuova istanza di Self-Attention
    ///
    /// # Arguments
    /// * `d_model` - La dimensione del modello
    /// * `d_k` - La dimensione delle chiavi (di solito d_model / num_heads)
    /// * `d_v` - La dimensione dei valori (di solito uguale a d_k)
    ///
    /// # Returns
    /// Una nuova istanza di SelfAttention
    pub fn new(d_model: usize) -> Self {
        let d_k = d_model;  // per semplicità, usiamo d_k = d_model

        // Inizializza le matrici di proiezione con pesi casuali invece di zeri
        let W_q = create_weight_matrix(d_model, d_k);
        let W_k = create_weight_matrix(d_model, d_k);
        let W_v = create_weight_matrix(d_model, d_k);
        let W_o = create_weight_matrix(d_k, d_model);

        Self {
            d_model,
            d_k,
            d_v: d_k,
            W_q,
            W_k,
            W_v,
            W_o,
        }
    }
    
    /// Calcola l'attenzione scalata tra query e key
    ///
    /// # Arguments
    /// * `q` - Tensore di query [batch_size * seq_len, d_k]
    /// * `k` - Tensore di key [batch_size * seq_len, d_k]
    /// * `v` - Tensore di value [batch_size * seq_len, d_v]
    /// * `seq_len` - Lunghezza della sequenza
    /// * `mask` - Opzionale, maschera per l'attenzione [batch_size * seq_len, seq_len]
    ///
    /// # Returns
    /// Tensore di output [batch_size * seq_len, d_v]
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>
    ) -> Tensor {
        let d_k = self.d_k as f32;
        
        // QK^T
        // Utilizziamo il metodo transpose() e matmul_with()
        let k_transposed = k.transpose();
        let scores = q.matmul_with(&k_transposed);
        
        // Scala i punteggi per la stabilità numerica
        let scaling_factor = (d_k).sqrt();
        let scores_scaled_data = &scores.data / scaling_factor;
        let scores_scaled = Tensor::new(scores_scaled_data);
        
        // Applica la maschera se presente
        let scores_masked = if let Some(mask_tensor) = mask {
            // Applica la maschera usando una moltiplicazione elemento per elemento
            // Nei punti dove la maschera è 0, impostiamo -inf
            let mut masked_data = scores_scaled.data.clone();
            
            for ((i, j), val) in masked_data.indexed_iter_mut() {
                if mask_tensor.data[[i, j]] == 0.0 {
                    *val = f32::NEG_INFINITY;
                }
            }
            
            Tensor::new(masked_data)
        } else {
            scores_scaled
        };
        
        // Applica softmax alle righe
        let mut attention_weights_data = scores_masked.data.clone();
        
        // Calcola softmax per ogni riga
        for mut row in attention_weights_data.axis_iter_mut(Axis(0)) {
            // Trova il massimo per stabilità numerica
            let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            // Calcola exp(x_i - max) per ogni elemento
            let mut exp_sum = 0.0;
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
                exp_sum += *val;
            }
            
            // Normalizza
            for val in row.iter_mut() {
                *val /= exp_sum;
            }
        }
        
        let attention_weights = Tensor::new(attention_weights_data);
        
        // Calcola il risultato finale: attention_weights * V
        attention_weights.matmul_with(v)
    }
    
    /// Proietta l'input in query, key e value
    ///
    /// # Arguments
    /// * `x` - Tensore di input [batch_size * seq_len, d_model]
    ///
    /// # Returns
    /// Tripla di tensori (q, k, v) di dimensioni rispettive:
    /// q: [batch_size * seq_len, d_k]
    /// k: [batch_size * seq_len, d_k]
    /// v: [batch_size * seq_len, d_v]
    fn project_qkv(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let q = x.matmul_with(&self.W_q);
        let k = x.matmul_with(&self.W_k);
        let v = x.matmul_with(&self.W_v);
        (q, k, v)
    }
}

impl Attention for SelfAttention {
    /// Forward pass dell'attenzione senza maschera
    ///
    /// # Arguments
    /// * `input` - Tensore di input [batch_size * seq_len, d_model]
    ///
    /// # Returns
    /// Tensore di output [batch_size * seq_len, d_model]
    fn forward(&self, input: &Tensor) -> Tensor {
        let (q, k, v) = self.project_qkv(input);
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v, None);
        attention_output.matmul_with(&self.W_o)
    }
    
    /// Forward pass dell'attenzione con maschera
    ///
    /// # Arguments
    /// * `input` - Tensore di input [batch_size * seq_len, d_model]
    /// * `mask` - Maschera [batch_size * seq_len, seq_len]
    ///
    /// # Returns
    /// Tensore di output [batch_size * seq_len, d_model]
    fn forward_with_mask(&self, input: &Tensor, mask: &Tensor) -> Tensor {
        let (q, k, v) = self.project_qkv(input);
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v, Some(mask));
        attention_output.matmul_with(&self.W_o)
    }
    
    /// Restituisce la dimensione del modello
    fn model_dim(&self) -> usize {
        self.d_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array};
    
    #[test]
    fn test_self_attention_dimensions() {
        let d_model = 64;
        let batch_size = 2;
        let seq_len = 4;
        let total_seq = batch_size * seq_len;
        
        let attention = SelfAttention::new(d_model);
        
        // Crea un input di test [batch_size * seq_len, d_model]
        let input_data = Array2::ones((total_seq, d_model));
        let input = Tensor::new(input_data);
        
        // Test forward pass
        let output = attention.forward(&input);
        
        // Verifica che le dimensioni di output siano corrette
        assert_eq!(output.data.shape(), &[total_seq, d_model]);
    }
    
    #[test]
    fn test_self_attention_with_mask() {
        let d_model = 8;
        let batch_size = 1;
        let seq_len = 3;
        let total_seq = batch_size * seq_len;
        
        let attention = SelfAttention::new(d_model);
        
        // Crea un input di test [batch_size * seq_len, d_model]
        let mut input_data = Array2::zeros((total_seq, d_model));
        
        // Assegna valori diversi a ciascuna posizione della sequenza
        for i in 0..total_seq {
            for j in 0..d_model {
                input_data[[i, j]] = (i * d_model + j) as f32 * 0.1;
            }
        }
        
        let input = Tensor::new(input_data);
        
        // Crea una maschera per impedire alla posizione 0 di vedere le posizioni 1 e 2
        // e alla posizione 1 di vedere la posizione 2 (maschera triangolare inferiore)
        let mut mask_data = Array2::zeros((total_seq, seq_len));
        
        // Attenzione: per la maschera, 1 permette l'attenzione, 0 la blocca
        // La maschera viene convertita internamente a -inf per i valori 0
        mask_data[[0, 0]] = 1.0;
        mask_data[[0, 1]] = 0.0;
        mask_data[[0, 2]] = 0.0;
        
        mask_data[[1, 0]] = 1.0;
        mask_data[[1, 1]] = 1.0;
        mask_data[[1, 2]] = 0.0;
        
        mask_data[[2, 0]] = 1.0;
        mask_data[[2, 1]] = 1.0;
        mask_data[[2, 2]] = 1.0;
        
        let mask = Tensor::new(mask_data);
        
        // Test forward pass con maschera
        let output_with_mask = attention.forward_with_mask(&input, &mask);
        let output_no_mask = attention.forward(&input);
        
        // Verifica che le dimensioni di output siano corrette
        assert_eq!(output_with_mask.data.shape(), &[total_seq, d_model]);
        
        // Verifica che l'output con maschera sia diverso dall'output senza maschera
        let mut is_different = false;
        for i in 0..total_seq {
            for j in 0..d_model {
                if (output_with_mask.data[[i, j]] - output_no_mask.data[[i, j]]).abs() > 1e-5 {
                    is_different = true;
                    break;
                }
            }
            if is_different {
                break;
            }
        }
        
        assert!(is_different, "L'output con maschera dovrebbe essere diverso dall'output senza maschera");
    }
} 