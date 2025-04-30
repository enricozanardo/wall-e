use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use serde::{Serialize, Deserialize};

/// Type that represents a vocabulary for a tokenizer
///
/// A vocabulary stores the mapping between tokens (strings) and their corresponding IDs,
/// as well as tracking which tokens are special tokens (like padding, unknown, etc.).
///
/// # Examples
///
/// ```
/// use wall_e1::tokenizer::Vocab;
///
/// let mut vocab = Vocab::new();
///
/// // Add regular tokens
/// let hello_id = vocab.add_token("hello");
/// let world_id = vocab.add_token("world");
///
/// // Add special tokens
/// let pad_id = vocab.add_special_token("[PAD]");
/// let unk_id = vocab.add_special_token("[UNK]");
///
/// // Look up token IDs
/// assert_eq!(vocab.token_to_id("hello"), Some(hello_id));
/// assert_eq!(vocab.token_to_id("[PAD]"), Some(pad_id));
///
/// // Look up tokens from IDs
/// assert_eq!(vocab.id_to_token(world_id), Some("world"));
///
/// // Check if a token is special
/// assert!(vocab.is_special_token("[UNK]"));
/// assert!(!vocab.is_special_token("hello"));
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocab {
    /// Mapping from token to ID
    token_to_id: HashMap<String, usize>,
    /// Mapping from ID to token
    id_to_token: HashMap<usize, String>,
    /// Set of special tokens
    special_tokens: HashSet<String>,
}

impl Vocab {
    /// Creates a new empty vocabulary
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let vocab = Vocab::new();
    /// assert_eq!(vocab.len(), 0);
    /// assert!(vocab.is_empty());
    /// ```
    pub fn new() -> Self {
        Vocab {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashSet::new(),
        }
    }
    
    /// Adds a token to the vocabulary
    ///
    /// If the token already exists in the vocabulary, returns its existing ID.
    /// Otherwise, adds the token with a new ID and returns that ID.
    ///
    /// # Arguments
    ///
    /// * `token` - The token to add
    ///
    /// # Returns
    ///
    /// The ID of the token
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// let id1 = vocab.add_token("hello");
    /// let id2 = vocab.add_token("world");
    ///
    /// // Adding the same token again returns the same ID
    /// let id3 = vocab.add_token("hello");
    /// assert_eq!(id1, id3);
    /// ```
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(id) = self.token_to_id.get(token) {
            return *id;
        }
        
        let id = self.token_to_id.len();
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        id
    }
    
    /// Adds a special token to the vocabulary
    ///
    /// Special tokens are treated differently during tokenization processes.
    /// Examples include padding tokens, unknown tokens, etc.
    ///
    /// # Arguments
    ///
    /// * `token` - The special token to add
    ///
    /// # Returns
    ///
    /// The ID of the special token
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// let pad_id = vocab.add_special_token("[PAD]");
    /// let unk_id = vocab.add_special_token("[UNK]");
    ///
    /// assert!(vocab.is_special_token("[PAD]"));
    /// assert!(vocab.is_special_token("[UNK]"));
    /// ```
    pub fn add_special_token(&mut self, token: &str) -> usize {
        let id = self.add_token(token);
        self.special_tokens.insert(token.to_string());
        id
    }
    
    /// Returns the ID of a token
    ///
    /// # Arguments
    ///
    /// * `token` - The token to look up
    ///
    /// # Returns
    ///
    /// The ID of the token if it exists in the vocabulary, or None if it doesn't
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// let id = vocab.add_token("hello");
    ///
    /// assert_eq!(vocab.token_to_id("hello"), Some(id));
    /// assert_eq!(vocab.token_to_id("unknown"), None);
    /// ```
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }
    
    /// Returns the token corresponding to an ID
    ///
    /// # Arguments
    ///
    /// * `id` - The ID to look up
    ///
    /// # Returns
    ///
    /// The token corresponding to the ID if it exists, or None if it doesn't
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// let id = vocab.add_token("hello");
    ///
    /// assert_eq!(vocab.id_to_token(id), Some("hello"));
    /// assert_eq!(vocab.id_to_token(999), None); // Non-existent ID
    /// ```
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
    
    /// Checks if a token is a special token
    ///
    /// # Arguments
    ///
    /// * `token` - The token to check
    ///
    /// # Returns
    ///
    /// `true` if the token is a special token, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.add_token("hello");
    /// vocab.add_special_token("[PAD]");
    ///
    /// assert!(vocab.is_special_token("[PAD]"));
    /// assert!(!vocab.is_special_token("hello"));
    /// ```
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains(token)
    }
    
    /// Returns the size of the vocabulary
    ///
    /// # Returns
    ///
    /// The number of tokens in the vocabulary (including special tokens)
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// assert_eq!(vocab.len(), 0);
    ///
    /// vocab.add_token("hello");
    /// vocab.add_token("world");
    /// vocab.add_special_token("[PAD]");
    ///
    /// assert_eq!(vocab.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }
    
    /// Checks if the vocabulary is empty
    ///
    /// # Returns
    ///
    /// `true` if the vocabulary is empty, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use wall_e1::tokenizer::Vocab;
    ///
    /// let mut vocab = Vocab::new();
    /// assert!(vocab.is_empty());
    ///
    /// vocab.add_token("hello");
    /// assert!(!vocab.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }
    
    /// Saves the vocabulary to a file
    ///
    /// The vocabulary is saved in a format where each line contains a token and its ID.
    /// Special tokens are marked with a `<special>` prefix.
    ///
    /// # Arguments
    ///
    /// * `path` - The path where to save the vocabulary
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wall_e1::tokenizer::Vocab;
    /// use std::path::Path;
    ///
    /// let mut vocab = Vocab::new();
    /// vocab.add_token("hello");
    /// vocab.add_special_token("[PAD]");
    ///
    /// // Save to a file
    /// let path = Path::new("vocab.txt");
    /// vocab.save(path).expect("Failed to save vocabulary");
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        
        // Write regular tokens
        for (token, id) in &self.token_to_id {
            if !self.special_tokens.contains(token) {
                writeln!(file, "{}\t{}", token, id)?;
            }
        }
        
        // Write special tokens
        for token in &self.special_tokens {
            let id = self.token_to_id.get(token).unwrap();
            writeln!(file, "<special>\t{}\t{}", token, id)?;
        }
        
        Ok(())
    }
    
    /// Loads a vocabulary from a file
    ///
    /// Loads a vocabulary previously saved with the `save` method.
    ///
    /// # Arguments
    ///
    /// * `path` - The path from which to load the vocabulary
    ///
    /// # Returns
    ///
    /// A `Result` containing the loaded vocabulary or an error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wall_e1::tokenizer::Vocab;
    /// use std::path::Path;
    ///
    /// // Load from a file
    /// let path = Path::new("vocab.txt");
    /// let vocab = Vocab::load(path).expect("Failed to load vocabulary");
    ///
    /// // Check if a token exists
    /// assert!(vocab.token_to_id("hello").is_some());
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut vocab = Vocab {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashSet::new(),
        };
        
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            
            if parts.len() == 2 {
                // Regular token
                let token = parts[0];
                let id = parts[1].parse::<usize>().unwrap_or_default();
                vocab.token_to_id.insert(token.to_string(), id);
                vocab.id_to_token.insert(id, token.to_string());
            } else if parts.len() == 3 && parts[0] == "<special>" {
                // Special token
                let token = parts[1];
                let id = parts[2].parse::<usize>().unwrap_or_default();
                vocab.token_to_id.insert(token.to_string(), id);
                vocab.id_to_token.insert(id, token.to_string());
                vocab.special_tokens.insert(token.to_string());
            }
        }
        
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::env::temp_dir;
    
    #[test]
    fn test_vocab_add_token() {
        let mut vocab = Vocab::new();
        
        // Adding tokens
        assert_eq!(vocab.add_token("hello"), 0);
        assert_eq!(vocab.add_token("world"), 1);
        
        // Verify token -> ID mapping
        assert_eq!(vocab.token_to_id("hello"), Some(0));
        assert_eq!(vocab.token_to_id("world"), Some(1));
        assert_eq!(vocab.token_to_id("unknown"), None);
        
        // Verify ID -> token mapping
        assert_eq!(vocab.id_to_token(0), Some("hello"));
        assert_eq!(vocab.id_to_token(1), Some("world"));
        assert_eq!(vocab.id_to_token(2), None);
        
        // Verify vocabulary size
        assert_eq!(vocab.len(), 2);
    }
    
    #[test]
    fn test_vocab_special_tokens() {
        let mut vocab = Vocab::new();
        
        // Adding special tokens
        assert_eq!(vocab.add_special_token("[PAD]"), 0);
        assert_eq!(vocab.add_special_token("[UNK]"), 1);
        
        // Verify special tokens
        assert!(vocab.is_special_token("[PAD]"));
        assert!(vocab.is_special_token("[UNK]"));
        assert!(!vocab.is_special_token("hello"));
        
        // Adding regular tokens
        assert_eq!(vocab.add_token("hello"), 2);
        
        // Verify vocabulary size
        assert_eq!(vocab.len(), 3);
    }
    
    #[test]
    fn test_vocab_save_load() -> std::io::Result<()> {
        let mut vocab = Vocab::new();
        
        // Add tokens in the correct order for the test
        vocab.add_special_token("[PAD]"); // ID 0
        vocab.add_special_token("[UNK]"); // ID 1
        vocab.add_token("hello");         // ID 2
        vocab.add_token("world");         // ID 3
        
        // Verify order before saving
        assert_eq!(vocab.token_to_id("[PAD]"), Some(0));
        assert_eq!(vocab.token_to_id("[UNK]"), Some(1));
        assert_eq!(vocab.token_to_id("hello"), Some(2));
        
        // Create a temporary path for the file
        let mut temp_path = temp_dir();
        temp_path.push("vocab_test.txt");
        
        // Save the vocabulary
        vocab.save(&temp_path)?;
        
        // Load the vocabulary from the file
        let loaded_vocab = Vocab::load(&temp_path)?;
        
        // Clean up after the test
        fs::remove_file(&temp_path)?;
        
        // Verify that the loaded vocabulary is identical to the original
        assert_eq!(loaded_vocab.len(), vocab.len());
        assert_eq!(loaded_vocab.token_to_id("[PAD]"), vocab.token_to_id("[PAD]"));
        assert_eq!(loaded_vocab.token_to_id("[UNK]"), vocab.token_to_id("[UNK]"));
        assert_eq!(loaded_vocab.token_to_id("hello"), vocab.token_to_id("hello"));
        assert!(loaded_vocab.is_special_token("[PAD]"));
        
        Ok(())
    }
} 