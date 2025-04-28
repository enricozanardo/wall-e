pub mod token_embedding;
pub mod positional_embedding;

use crate::nabla::tensor::Tensor;
use crate::tokenizer::Vocab;

/// Trait che definisce le operazioni comuni per gli embedding
pub trait Embedding {
    /// Converte gli ID dei token in embedding (rappresentazioni vettoriali)
    fn forward(&self, token_ids: &[usize]) -> Tensor;
    
    /// Dimensione dell'embedding (d_model)
    fn embedding_dim(&self) -> usize;
}

/// Trait che definisce operazioni con supporto per batch
pub trait BatchEmbedding {
    /// Converte i batch di token IDs in embedding
    fn forward_batch(&self, batch_token_ids: &[Vec<usize>]) -> Tensor;
}

/// Struct che combina word embedding e positional embedding
pub struct TransformerEmbedding {
    /// Embedding dei token
    token_emb: token_embedding::TokenEmbedding,
    /// Embedding posizionali
    pos_emb: positional_embedding::PositionalEmbedding,
    /// Dimensione dell'embedding
    embedding_dim: usize,
    /// Tasso di dropout
    dropout_rate: f32,
}

impl TransformerEmbedding {
    /// Crea un nuovo transformer embedding
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        max_seq_len: usize,
        dropout_rate: f32,
    ) -> Self {
        TransformerEmbedding {
            token_emb: token_embedding::TokenEmbedding::new(vocab_size, embedding_dim),
            pos_emb: positional_embedding::PositionalEmbedding::new(max_seq_len, embedding_dim),
            embedding_dim,
            dropout_rate,
        }
    }
    
    /// Costruisce un embedding da un vocabolario esistente
    pub fn from_vocab(
        vocab: &Vocab,
        embedding_dim: usize,
        max_seq_len: usize,
        dropout_rate: f32,
    ) -> Self {
        TransformerEmbedding {
            token_emb: token_embedding::TokenEmbedding::new(vocab.len(), embedding_dim),
            pos_emb: positional_embedding::PositionalEmbedding::new(max_seq_len, embedding_dim),
            embedding_dim,
            dropout_rate,
        }
    }
    
    /// Applica il forward pass per convertire token IDs in embedding con posizioni
    pub fn forward(&self, token_ids: &[usize]) -> Tensor {
        // Get token embeddings
        let token_embeddings = self.token_emb.forward(token_ids);
        
        // Get positional embeddings for sequence length
        let positional_embeddings = self.pos_emb.forward(token_ids);
        
        // Sum the embeddings
        let embeddings = Tensor::add(&token_embeddings, &positional_embeddings);
        
        // Apply dropout (in a real implementation)
        // For now just return the embeddings
        // TODO: Implement dropout
        
        embeddings
    }
    
    /// Forward pass con supporto per batch di token IDs
    /// Input: batch_token_ids - array di batch di token IDs
    /// Output: Tensor con forma [batch_size, seq_len, embedding_dim]
    pub fn forward_batch(&self, batch_token_ids: &[Vec<usize>]) -> Tensor {
        if batch_token_ids.is_empty() {
            return Tensor::new_3d(ndarray::Array3::<f32>::zeros((0, 0, 0)));
        }
        
        let batch_size = batch_token_ids.len();
        let seq_len = batch_token_ids[0].len();
        
        // Get token embeddings con dimensione batch
        let token_embeddings = self.token_emb.forward_batch(batch_token_ids);
        
        // Get positional embeddings con dimensione batch
        // Assicuriamoci che anche gli embedding posizionali siano 3D
        let positional_embeddings = self.pos_emb.forward_batch_3d(batch_size, seq_len);
        
        println!("Debug: TransformerEmbedding - token_embeddings shape: {:?}", token_embeddings.data.shape());
        println!("Debug: TransformerEmbedding - pos_embeddings shape: {:?}", positional_embeddings.data.shape());
        
        // Sum the embeddings
        let embeddings = Tensor::add(&token_embeddings, &positional_embeddings);
        
        // Apply dropout (in a real implementation)
        // For now just return the embeddings
        // TODO: Implement dropout
        
        embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_transformer_embedding_creation() {
        let embedding = TransformerEmbedding::new(
            1000,   // vocab_size
            128,    // embedding_dim
            512,    // max_seq_len
            0.1,    // dropout_rate
        );
        
        assert_eq!(embedding.embedding_dim, 128);
        assert_eq!(embedding.token_emb.embedding_dim(), 128);
        assert_eq!(embedding.pos_emb.embedding_dim(), 128);
    }
    
    #[test]
    fn test_transformer_embedding_forward() {
        let embedding = TransformerEmbedding::new(
            1000,   // vocab_size
            64,     // embedding_dim
            512,    // max_seq_len
            0.1,    // dropout_rate
        );
        
        // Input sequence [seq_len=3]
        let token_ids = vec![1, 2, 3];
        
        // Forward pass
        let output = embedding.forward(&token_ids);
        
        // Check output shape: should be [seq_len, embedding_dim]
        assert_eq!(output.data.shape(), &[3, 64]);
    }
    
    #[test]
    fn test_transformer_embedding_forward_batch() {
        let embedding = TransformerEmbedding::new(
            1000,   // vocab_size
            64,     // embedding_dim
            512,    // max_seq_len
            0.1,    // dropout_rate
        );
        
        // Batch di sequenze: [batch_size=2, seq_len=3]
        let batch_token_ids = vec![
            vec![1, 2, 3],   // prima sequenza
            vec![4, 5, 6],   // seconda sequenza
        ];
        
        // Forward pass con batch
        let output = embedding.forward_batch(&batch_token_ids);
        
        // Check output shape: should be [batch_size, seq_len * embedding_dim]
        assert_eq!(output.data.shape(), &[2, 3 * 64]);
    }
} 