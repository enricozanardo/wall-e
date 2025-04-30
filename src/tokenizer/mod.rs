pub mod basic_tokenizer;
pub mod vocab;
pub mod bpe;

pub use basic_tokenizer::BasicTokenizer;
pub use vocab::Vocab;
pub use bpe::BPETokenizer;

/// Trait that defines the functionalities of a tokenizer
///
/// This trait provides the core methods that all tokenizers in the library must implement.
/// It handles tokenization, encoding, and decoding of text, as well as vocabulary management.
///
/// # Examples
///
/// ```
/// use wall_e1::tokenizer::{Tokenizer, BasicTokenizer};
///
/// let mut tokenizer = BasicTokenizer::new();
/// 
/// // Add tokens to the vocabulary
/// tokenizer.get_vocab_mut().add_token("hello");
/// tokenizer.get_vocab_mut().add_token("world");
///
/// // Tokenize text
/// let tokens = tokenizer.tokenize("Hello world!");
/// assert_eq!(tokens, vec!["hello", "world", "[UNK]"]);
///
/// // Encode text to token IDs
/// let ids = tokenizer.encode("Hello world!");
/// 
/// // Decode token IDs back to text
/// let text = tokenizer.decode(&ids);
/// ```
pub trait Tokenizer: Send + Sync {
    /// Tokenizes a string into a list of tokens
    fn tokenize(&self, text: &str) -> Vec<String>;
    
    /// Transforms text into a list of token IDs
    fn encode(&self, text: &str) -> Vec<usize>;
    
    /// Transforms a list of token IDs into text
    fn decode(&self, token_ids: &[usize]) -> String;
    
    /// Returns the size of the vocabulary
    fn vocab_size(&self) -> usize;
    
    /// Returns a reference to the vocabulary
    fn get_vocab(&self) -> &Vocab;
    
    /// Returns a mutable reference to the vocabulary if supported
    fn as_vocab_mut(&mut self) -> Option<&mut Vocab> {
        None  // Default implementation returns None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Tests for specific tokenizers will be in their respective modules
} 