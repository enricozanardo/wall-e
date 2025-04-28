pub mod basic_tokenizer;
pub mod vocab;
pub mod bpe;

pub use basic_tokenizer::BasicTokenizer;
pub use vocab::Vocab;
pub use bpe::BPETokenizer;

/// Trait che definisce le funzionalità di un tokenizer
pub trait Tokenizer: Send + Sync {
    /// Tokenizza una stringa in una lista di token
    fn tokenize(&self, text: &str) -> Vec<String>;
    
    /// Trasforma un testo in una lista di ID di token
    fn encode(&self, text: &str) -> Vec<usize>;
    
    /// Trasforma una lista di ID di token in un testo
    fn decode(&self, token_ids: &[usize]) -> String;
    
    /// Restituisce la dimensione del vocabolario
    fn vocab_size(&self) -> usize;
    
    /// Restituisce un riferimento al vocabolario
    fn get_vocab(&self) -> &Vocab;
    
    /// Restituisce un riferimento mutabile al vocabolario se supportato
    fn as_vocab_mut(&mut self) -> Option<&mut Vocab> {
        None  // Default implementation returns None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // I test per i tokenizer specifici saranno nei rispettivi moduli
} 