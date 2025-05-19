pub mod nabla;
pub mod tokenizer;
pub mod embedding;
pub mod attention;
pub mod training;
pub mod dataset;
pub mod export;
pub mod utils;

// Esporta le strutture principali per una facile importazione
pub use nabla::tensor::Tensor;
pub use tokenizer::{Tokenizer, Vocab, BasicTokenizer, BPETokenizer};
pub use embedding::{Embedding, TransformerEmbedding};
pub use attention::{Attention, SelfAttention, MultiHeadAttention, EncoderLayer, EncoderStack};
pub use training::{Trainer, ModelOutput, CrossEntropyLoss, AdamOptimizer};
pub use training::enhanced_trainer::EnhancedTrainer;
pub use dataset::{DatasetStream, DatasetSplit, split_dataset, k_fold_split}; 
pub use utils::thread_pool::get_global_thread_pool; 