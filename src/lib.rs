pub mod nabla;
pub mod tokenizer;
pub mod embedding;
pub mod attention;
pub mod training;
pub mod dataset;
pub mod export;

// Esporta le strutture principali per una facile importazione
pub use nabla::tensor::Tensor;
pub use tokenizer::{Tokenizer, Vocab, BasicTokenizer, BPETokenizer};
pub use embedding::{Embedding, TransformerEmbedding};
pub use attention::{Attention, SelfAttention, MultiHeadAttention, EncoderLayer, EncoderStack};
pub use training::{Trainer, ModelOutput, CrossEntropyLoss, AdamOptimizer};
pub use dataset::{DatasetStream, DatasetSplit, split_dataset, k_fold_split}; 