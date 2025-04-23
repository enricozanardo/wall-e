pub mod basic_tokenizer;
pub mod vocab;
pub mod bpe;

pub use basic_tokenizer::BasicTokenizer;
pub use vocab::Vocab;
pub use bpe::BPETokenizer;

/// Trait che definisce l'interfaccia comune per tutti i tokenizer
pub trait Tokenizer {
    /// Converte una stringa in una lista di token (stringhe)
    fn tokenize(&self, text: &str) -> Vec<String>;
    
    /// Converte una stringa in una lista di ID (interi)
    fn encode(&self, text: &str) -> Vec<usize>;
    
    /// Converte una lista di ID in una stringa
    fn decode(&self, ids: &[usize]) -> String;
    
    /// Restituisce la dimensione del vocabolario
    fn vocab_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // I test per i tokenizer specifici saranno nei rispettivi moduli
} 