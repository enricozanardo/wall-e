use std::collections::HashMap;
use crate::tokenizer::{Tokenizer, Vocab};
use rayon::prelude::*;

/// Basic tokenizer that uses spaces and punctuation to split text
///
/// This tokenizer provides a simple approach to text tokenization by splitting
/// on whitespace and treating punctuation marks as separate tokens. It also converts
/// all text to lowercase.
///
/// # Examples
///
/// ```
/// use wall_e1::tokenizer::{BasicTokenizer, Tokenizer};
///
/// // Create a new tokenizer
/// let mut tokenizer = BasicTokenizer::new();
///
/// // Add tokens to the vocabulary
/// tokenizer.get_vocab_mut().add_token("hello");
/// tokenizer.get_vocab_mut().add_token("world");
///
/// // Tokenize a sentence
/// let tokens = tokenizer.tokenize("Hello, world!");
/// assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
/// ```
#[derive(Debug, Clone)]
pub struct BasicTokenizer {
    vocab: Vocab,
    unk_token: String,
}

impl BasicTokenizer {
    /// Creates a new basic tokenizer with an empty vocabulary
    ///
    /// The tokenizer is initialized with default special tokens:
    /// - "[PAD]" for padding sequences
    /// - "[UNK]" for unknown tokens
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::BasicTokenizer;
    ///
    /// let tokenizer = BasicTokenizer::new();
    /// assert_eq!(tokenizer.vocab_size(), 2); // PAD and UNK tokens
    /// ```
    pub fn new() -> Self {
        let mut vocab = Vocab::new();
        
        // Add default special tokens
        vocab.add_special_token("[PAD]"); // Padding token
        let _unk_id = vocab.add_special_token("[UNK]"); // Unknown token
        
        BasicTokenizer {
            vocab,
            unk_token: "[UNK]".to_string(),
        }
    }
    
    /// Creates a new basic tokenizer with a predefined vocabulary
    ///
    /// # Arguments
    ///
    /// * `vocab` - The vocabulary to use
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::{BasicTokenizer, Vocab};
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.add_token("hello");
    /// vocab.add_token("world");
    ///
    /// let tokenizer = BasicTokenizer::with_vocab(vocab);
    /// assert!(tokenizer.get_vocab().token_to_id("hello").is_some());
    /// ```
    pub fn with_vocab(vocab: Vocab) -> Self {
        // Make sure the unknown token exists
        let mut vocab = vocab;
        if vocab.token_to_id("[UNK]").is_none() {
            vocab.add_special_token("[UNK]");
        }
        
        BasicTokenizer {
            vocab,
            unk_token: "[UNK]".to_string(),
        }
    }
    
    /// Builds a vocabulary from text
    ///
    /// This method tokenizes the provided text and builds a vocabulary
    /// containing all tokens that appear at least `min_freq` times.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to build the vocabulary from
    /// * `min_freq` - The minimum frequency required for a token to be included
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::{BasicTokenizer, Tokenizer};
    ///
    /// let mut tokenizer = BasicTokenizer::new();
    /// tokenizer.build_vocab("hello world hello hello world test", 2);
    ///
    /// // "hello" and "world" are frequent enough
    /// assert!(tokenizer.get_vocab().token_to_id("hello").is_some());
    /// assert!(tokenizer.get_vocab().token_to_id("world").is_some());
    ///
    /// // "test" is not frequent enough
    /// assert!(tokenizer.get_vocab().token_to_id("test").is_none());
    /// ```
    pub fn build_vocab(&mut self, text: &str, min_freq: usize) {
        // Tokenize the text
        let tokens = self.tokenize_raw(text);
        
        // Count frequencies in parallel using Rayon
        let freqs = tokens.par_iter()
            .fold(
                || HashMap::new(), 
                |mut acc, token| {
                    *acc.entry(token.clone()).or_insert(0) += 1;
                    acc
                }
            )
            .reduce(
                || HashMap::new(),
                |mut acc, map| {
                    for (token, count) in map {
                        *acc.entry(token).or_insert(0) += count;
                    }
                    acc
                }
            );
        
        // Add tokens that exceed the minimum frequency
        for (token, freq) in freqs {
            if freq >= min_freq {
                self.vocab.add_token(&token);
            }
        }
    }
    
    /// Tokenizes text without using the vocabulary (for vocabulary building)
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    ///
    /// A vector of string tokens
    fn tokenize_raw(&self, text: &str) -> Vec<String> {
        // Simple tokenization based on spaces and punctuation
        
        // Replace punctuation with spaces + punctuation + spaces
        let text = text.replace('.', " . ")
                   .replace(',', " , ")
                   .replace('!', " ! ")
                   .replace('?', " ? ")
                   .replace(':', " : ")
                   .replace(';', " ; ")
                   .replace('(', " ( ")
                   .replace(')', " ) ")
                   .replace('[', " [ ")
                   .replace(']', " ] ")
                   .replace('{', " { ")
                   .replace('}', " } ");
        
        // Split by spaces and convert to lowercase in parallel
        let tokens: Vec<String> = text.split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
            .into_par_iter()
            .map(|s| s.to_lowercase())
            .collect();
        
        tokens
    }
    
    /// Gets the vocabulary
    ///
    /// # Returns
    ///
    /// A reference to the vocabulary
    pub fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    /// Gets a mutable reference to the vocabulary
    ///
    /// # Returns
    ///
    /// A mutable reference to the vocabulary
    pub fn get_vocab_mut(&mut self) -> &mut Vocab {
        &mut self.vocab
    }
}

impl Tokenizer for BasicTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let raw_tokens = self.tokenize_raw(text);
        
        // Map unknown tokens to the unknown token
        raw_tokens.into_par_iter()
            .map(|token| {
                if self.vocab.token_to_id(&token).is_some() {
                    token
                } else {
                    self.unk_token.clone()
                }
            })
            .collect()
    }
    
    fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let unk_id = self.vocab.token_to_id(&self.unk_token).unwrap();
        
        tokens.iter()
              .map(|token| self.vocab.token_to_id(token).unwrap_or(unk_id))
              .collect()
    }
    
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
           .filter_map(|&id| self.vocab.id_to_token(id))
           .collect::<Vec<_>>()
           .join(" ")
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }
    
    fn as_vocab_mut(&mut self) -> Option<&mut Vocab> {
        Some(&mut self.vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokenizer_tokenize() {
        let mut tokenizer = BasicTokenizer::new();
        
        // Add necessary tokens to the vocabulary 
        tokenizer.get_vocab_mut().add_token("hello");
        tokenizer.get_vocab_mut().add_token("world");
        tokenizer.get_vocab_mut().add_token(",");
        tokenizer.get_vocab_mut().add_token("!");
        
        let text = "Hello, world!";
        let tokens = tokenizer.tokenize(text);
        
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }
    
    #[test]
    fn test_basic_tokenizer_encode_decode() {
        let mut tokenizer = BasicTokenizer::new();
        
        // Build a small vocabulary
        tokenizer.get_vocab_mut().add_token("hello");
        tokenizer.get_vocab_mut().add_token("world");
        
        let text = "Hello, world!";
        let ids = tokenizer.encode(text);
        
        // "hello" and "world" must be in the vocabulary, "," and "!" are [UNK]
        let unk_id = tokenizer.get_vocab().token_to_id("[UNK]").unwrap();
        assert_eq!(ids[0], tokenizer.get_vocab().token_to_id("hello").unwrap());
        assert_eq!(ids[1], unk_id); // ","
        assert_eq!(ids[2], tokenizer.get_vocab().token_to_id("world").unwrap());
        assert_eq!(ids[3], unk_id); // "!"
        
        // Decode (note that we lose punctuation)
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hello [UNK] world [UNK]");
    }
    
    #[test]
    fn test_build_vocab() {
        let mut tokenizer = BasicTokenizer::new();
        
        let text = "hello world hello hello world test";
        tokenizer.build_vocab(text, 2); // Only tokens with frequency >= 2
        
        // "hello" and "world" are frequent enough
        assert!(tokenizer.get_vocab().token_to_id("hello").is_some());
        assert!(tokenizer.get_vocab().token_to_id("world").is_some());
        
        // "test" is not frequent enough
        assert!(tokenizer.get_vocab().token_to_id("test").is_none());
    }
} 