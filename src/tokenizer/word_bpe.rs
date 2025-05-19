use std::collections::{HashMap, HashSet};
use crate::tokenizer::{Tokenizer, Vocab};

/// An improved tokenizer based on WordPiece with better word boundary handling
///
/// This tokenizer extends the standard BPE approach with improved handling of
/// word boundaries, spaces, and punctuation to better preserve word structure
/// during tokenization.
#[derive(Debug, Clone)]
pub struct WordPieceBPETokenizer {
    vocab: Vocab,
    merges: Vec<(String, String, String)>, // (first, second, merged)
    unk_token: String,
    end_token: String,
    space_token: String,
    punctuation: HashSet<String>,
    repetition_penalty: f32,
}

impl WordPieceBPETokenizer {
    /// Creates a new WordPiece BPE tokenizer with an empty vocabulary
    pub fn new() -> Self {
        let mut vocab = Vocab::new();
        
        // Add default special tokens
        vocab.add_special_token("[PAD]"); // Padding token 
        let _unk_id = vocab.add_special_token("[UNK]"); // Unknown token
        vocab.add_special_token("[BOS]"); // Beginning of sequence
        vocab.add_special_token("[EOS]"); // End of sequence
        
        // Add special token for space
        vocab.add_special_token("[SPACE]");
        
        // Add common punctuation as special tokens
        let punctuation_list = vec![
            ".", ",", "!", "?", ":", ";", "'", "\"", "(", ")", "[", "]", "{", "}", "-", "_", 
            "+", "=", "/", "\\", "|", "<", ">", "@", "#", "$", "%", "^", "&", "*"
        ];
        
        let mut punctuation = HashSet::new();
        for p in punctuation_list {
            let token = format!("[{}]", p);
            vocab.add_special_token(&token);
            punctuation.insert(p.to_string());
        }
        
        // Add all ASCII characters as base tokens
        for c in (32..127).map(char::from) {
            if !punctuation.contains(&c.to_string()) {
                vocab.add_token(&c.to_string());
            }
        }
        
        // Add the word-end token
        vocab.add_token("</w>");
        
        WordPieceBPETokenizer {
            vocab,
            merges: Vec::new(),
            unk_token: "[UNK]".to_string(),
            end_token: "</w>".to_string(),
            space_token: "[SPACE]".to_string(),
            punctuation,
            repetition_penalty: 1.2, // Default repetition penalty
        }
    }
    
    /// Sets the repetition penalty for text generation
    pub fn set_repetition_penalty(&mut self, penalty: f32) {
        self.repetition_penalty = penalty;
    }
    
    /// Pre-tokenizes text into words, spaces, and punctuation
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        
        for c in text.chars() {
            if c.is_whitespace() {
                // Handle whitespace
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                tokens.push(self.space_token.clone());
            } else if self.punctuation.contains(&c.to_string()) {
                // Handle punctuation
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                tokens.push(format!("[{}]", c));
            } else {
                // Part of a word - filter out invalid characters
                // Only add alphanumeric and common word characters
                if c.is_alphanumeric() || c == '\'' || c == '-' || c == '_' {
                    current_word.push(c.to_lowercase().next().unwrap_or(c));
                }
            }
        }
        
        // Add any remaining word
        if !current_word.is_empty() {
            tokens.push(current_word);
        }
        
        tokens
    }
    
    /// Learns BPE merge rules from a text with improved word boundary handling
    #[allow(unused_variables)]
    pub fn learn_bpe(&mut self, text: &str, vocab_size: usize, min_frequency: usize) {
        // Pre-tokenize text to preserve word boundaries
        let pre_tokens = self.pre_tokenize(text);
        
        // Prepare the text: extract words
        let words: Vec<String> = pre_tokens.iter()
            .filter(|&token| !token.starts_with('[') && !token.ends_with(']'))
            .map(|token| token.clone() + &self.end_token)
            .collect();
        
        // Count word frequencies
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }
        
        // Initialize word representations as characters
        let mut word_parts: HashMap<String, Vec<String>> = HashMap::new();
        
        for (word, count) in word_counts.iter().filter(|&(_, count)| *count >= min_frequency) {
            let parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            word_parts.insert(word.clone(), parts);
        }
        
        // Learn BPE rules until reaching the desired vocabulary size
        let current_vocab_size = self.vocab.len();
        let max_merges = vocab_size.saturating_sub(current_vocab_size);
        
        for _ in 0..max_merges {
            // Count pair frequencies
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            
            for (word, count) in word_counts.iter().filter(|&(_, count)| *count >= min_frequency) {
                let parts = word_parts.get(word).unwrap();
                
                if parts.len() < 2 {
                    continue;
                }
                
                for i in 0..parts.len() - 1 {
                    let pair = (parts[i].clone(), parts[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += count;
                }
            }
            
            // Find the most frequent pair
            if pair_counts.is_empty() {
                break;
            }
            
            let best_pair = pair_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|((first, second), _)| (first.clone(), second.clone()))
                .unwrap();
            
            // Create the new merged token
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            self.merges.push((best_pair.0.clone(), best_pair.1.clone(), new_token.clone()));
            
            // Update the vocabulary
            self.vocab.add_token(&new_token);
            
            // Update word representations
            for parts in word_parts.values_mut() {
                let mut i = 0;
                while i < parts.len() - 1 {
                    if parts[i] == best_pair.0 && parts[i + 1] == best_pair.1 {
                        parts[i] = new_token.clone();
                        parts.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }
    
    /// Applies BPE rules to a word
    fn apply_bpe(&self, word: &str) -> Vec<String> {
        // Special token handling
        if word.starts_with('[') && word.ends_with(']') {
            return vec![word.to_string()];
        }
        
        // Filter the word to only include valid characters
        let filtered_word: String = word.chars()
            .filter(|&c| c.is_alphanumeric() || c == '\'' || c == '-' || c == '_')
            .collect();
        
        // Skip empty words
        if filtered_word.is_empty() {
            return vec![];
        }
        
        // Add end-of-word token for regular words
        let word_with_end = filtered_word.to_lowercase() + &self.end_token;
        
        // Initialize the word as a sequence of characters
        let mut parts: Vec<String> = word_with_end.chars().map(|c| c.to_string()).collect();
        
        // Apply BPE rules in order
        for (first, second, merged) in &self.merges {
            let mut i = 0;
            while i < parts.len() - 1 {
                if parts[i] == *first && parts[i + 1] == *second {
                    parts[i] = merged.clone();
                    parts.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        
        // Do a final verification that all tokens are valid
        parts.into_iter()
            .filter(|token| {
                // Keep special tokens and valid character sequences
                token.starts_with('[') && token.ends_with(']') || 
                token.ends_with(&self.end_token) ||
                token.chars().all(|c| c.is_alphanumeric() || c == '\'' || c == '-' || c == '_')
            })
            .collect()
    }
    
    /// Apply repetition penalty to discourage repeating tokens
    pub fn apply_repetition_penalty(&self, logits: &mut [f32], previous_tokens: &[usize]) {
        // Create a frequency counter for previous tokens
        let mut token_counts = HashMap::new();
        for &token_id in previous_tokens {
            *token_counts.entry(token_id).or_insert(0) += 1;
        }
        
        // Apply penalty to tokens that have appeared before
        for (token_id, &count) in token_counts.iter() {
            if count > 0 && *token_id < logits.len() {
                // If token appeared multiple times, apply stronger penalty
                let penalty = if count > 1 {
                    self.repetition_penalty * count as f32
                } else {
                    self.repetition_penalty
                };
                
                // Penalize the token by dividing its probability
                logits[*token_id] /= penalty;
            }
        }
    }

    /// Debug helper for tokenization
    pub fn debug_tokenize(&self, text: &str) {
        println!("\n--- DEBUG TOKENIZATION ---");
        println!("Original text: {}", text);
        
        // Pre-tokenize
        let pre_tokens = self.pre_tokenize(text);
        println!("Pre-tokenized: {:?}", pre_tokens);
        
        // Full tokenize
        let tokens = self.tokenize(text);
        println!("Tokenized: {:?}", tokens);
        
        // Encode to IDs
        let ids = self.encode(text);
        println!("Encoded IDs: {:?}", ids);
        
        // Decode back
        let decoded = self.decode(&ids);
        println!("Decoded text: {}", decoded);
        println!("Does it match? {}", text.to_lowercase() == decoded);
        println!("------------------------\n");
    }

    /// Update vocabulary size when loading a model
    pub fn update_vocab_size(&mut self, vocab_size: usize) {
        println!("Updating tokenizer vocabulary size to {}", vocab_size);
        
        // Get current vocabulary size
        let current_size = self.vocab.len();
        println!("Current vocabulary size: {}", current_size);
        
        if current_size == vocab_size {
            println!("No vocabulary size adjustment needed");
            return;
        }
        
        if current_size < vocab_size {
            // Need to add placeholder tokens to reach the required size
            let tokens_to_add = vocab_size - current_size;
            println!("Adding {} placeholder tokens to match model's vocabulary size", tokens_to_add);
            
            for i in 0..tokens_to_add {
                // Add placeholder tokens with a special prefix to distinguish them
                let token = format!("[PLACEHOLDER_{}]", i);
                self.vocab.add_token(&token);
            }
        } else {
            // Need to reduce vocabulary size
            // This is more complex as we need to ensure we keep special tokens
            println!("WARNING: Model expects smaller vocabulary ({}) than tokenizer has ({})", 
                   vocab_size, current_size);
            println!("This may cause issues with token mapping. Consider retraining the model.");
            
            // For now, we'll keep using our vocabulary but log the warning
            // A proper implementation would involve carefully pruning the vocabulary
            // while maintaining token ID consistency for critical tokens
        }
        
        // Verify the new size
        println!("Updated vocabulary size: {}", self.vocab.len());
    }

    /// Returns the current vocabulary size
    pub fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    /// Learns BPE merge rules from multiple text chunks in parallel
    pub fn learn_bpe_parallel(&mut self, text_chunks: &[&str], vocab_size: usize, min_frequency: usize) {
        use rayon::prelude::*;
        use std::sync::{Arc, Mutex};
        use crate::utils::thread_pool::get_global_thread_pool;
        
        println!("Starting parallel vocabulary learning with {} chunks", text_chunks.len());
        let start = std::time::Instant::now();
        
        // Get access to the global thread pool
        let thread_pool_manager = get_global_thread_pool();
        println!("Using global thread pool with {} threads", thread_pool_manager.get_num_threads());
        
        // Step 1: Pre-tokenize all chunks in parallel and gather word frequencies
        let word_count_mutex = Arc::new(Mutex::new(HashMap::<String, usize>::new()));
        
        // Clone necessary data for parallel processing
        let end_token = self.end_token.clone();
        let space_token = self.space_token.clone();
        let punctuation = self.punctuation.clone();
        
        // Define a pre-tokenize function that doesn't capture self
        let pre_tokenize_fn = move |chunk: &str| -> Vec<(String, usize)> {
            // Pre-tokenize this chunk (reimplementing pre_tokenize to avoid borrowing self)
            let mut tokens = Vec::new();
            let mut current_word = String::new();
            
            for c in chunk.chars() {
                if c.is_whitespace() {
                    // Handle whitespace
                    if !current_word.is_empty() {
                        tokens.push(current_word.clone());
                        current_word.clear();
                    }
                    tokens.push(space_token.clone());
                } else if punctuation.contains(&c.to_string()) {
                    // Handle punctuation
                    if !current_word.is_empty() {
                        tokens.push(current_word.clone());
                        current_word.clear();
                    }
                    tokens.push(format!("[{}]", c));
                } else {
                    // Part of a word - filter out invalid characters
                    // Only add alphanumeric and common word characters
                    if c.is_alphanumeric() || c == '\'' || c == '-' || c == '_' {
                        current_word.push(c.to_lowercase().next().unwrap_or(c));
                    }
                }
            }
            
            // Add any remaining word
            if !current_word.is_empty() {
                tokens.push(current_word);
            }
            
            // Extract words
            let words: Vec<String> = tokens.iter()
                .filter(|&token| !token.starts_with('[') && !token.ends_with(']'))
                .map(|token| token.clone() + &end_token)
                .collect();
            
            // Count word frequencies for this chunk
            let mut local_counts = HashMap::new();
            for word in words {
                *local_counts.entry(word.clone()).or_insert(0) += 1;
            }
            
            // Convert to vec of tuples for easier return
            local_counts.into_iter().collect()
        };
        
        // Process chunks in parallel with rayon
        text_chunks.par_iter().for_each(|&chunk| {
            let chunk_counts = pre_tokenize_fn(chunk);
            
            // Combine results into the global counter
            let mut global_counts = word_count_mutex.lock().unwrap();
            for (word, count) in chunk_counts {
                *global_counts.entry(word).or_insert(0) += count;
            }
        });
        
        // Get the final word counts
        let word_counts = Arc::try_unwrap(word_count_mutex).unwrap().into_inner().unwrap();
        println!("Word counting completed in {:.2?}, found {} unique words", 
                 start.elapsed(), word_counts.len());
        
        // Step 2: Initialize word representations as characters
        let mut word_parts: HashMap<String, Vec<String>> = HashMap::new();
        
        for (word, count) in word_counts.iter().filter(|&(_, count)| *count >= min_frequency) {
            let parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            word_parts.insert(word.clone(), parts);
        }
        
        // Step 3: Learn BPE rules iteratively
        let current_vocab_size = self.vocab.len();
        let max_merges = vocab_size.saturating_sub(current_vocab_size);
        println!("Learning up to {} merges to reach vocab size {}", max_merges, vocab_size);
        
        let bpe_start = std::time::Instant::now();
        let mut merges_learned = 0;
        
        for i in 0..max_merges {
            // Count pair frequencies 
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            
            // Process all words to count pairs
            for (word, count) in word_counts.iter().filter(|&(_, count)| *count >= min_frequency) {
                // Get the parts for this word
                if let Some(parts) = word_parts.get(word) {
                    if parts.len() < 2 {
                        continue;
                    }
                    
                    // Count pairs in this word
                    for j in 0..parts.len() - 1 {
                        let pair = (parts[j].clone(), parts[j + 1].clone());
                        *pair_counts.entry(pair).or_insert(0) += count;
                    }
                }
            }
            
            // Find the most frequent pair
            if pair_counts.is_empty() {
                break;
            }
            
            let best_pair = pair_counts
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|((first, second), _)| (first.clone(), second.clone()))
                .unwrap();
            
            // Create the new merged token
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            self.merges.push((best_pair.0.clone(), best_pair.1.clone(), new_token.clone()));
            
            // Update the vocabulary
            self.vocab.add_token(&new_token);
            merges_learned += 1;
            
            // Progress reporting
            if i % 1000 == 0 || i == max_merges - 1 {
                println!("  Learned {} merges ({:.1}%) in {:.2?}...", 
                         i + 1, (i as f32 + 1.0) * 100.0 / max_merges as f32, bpe_start.elapsed());
            }
            
            // Update word representations
            for parts in word_parts.values_mut() {
                let mut i = 0;
                while i < parts.len() - 1 {
                    if parts[i] == best_pair.0 && parts[i + 1] == best_pair.1 {
                        parts[i] = new_token.clone();
                        parts.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        
        println!("Parallel BPE learning completed in {:.2?}, vocabulary size: {}", 
                 start.elapsed(), self.vocab.len());
        println!("Learned {} merge operations", merges_learned);
    }

    /// Resize vocabulary to a new size
    pub fn resize_vocabulary(&mut self, new_size: usize) {
        // Get current vocabulary size
        let current_size = self.vocab.len();
        println!("Resizing tokenizer vocabulary from {} to {}", current_size, new_size);
        
        if current_size == new_size {
            println!("No vocabulary size adjustment needed");
            return;
        }
        
        if current_size < new_size {
            // Need to add placeholder tokens to reach the required size
            let tokens_to_add = new_size - current_size;
            println!("Adding {} placeholder tokens to reach new vocabulary size", tokens_to_add);
            
            for i in 0..tokens_to_add {
                // Add placeholder tokens with a special prefix to distinguish them
                let token = format!("[PLACEHOLDER_{}]", i);
                self.vocab.add_token(&token);
            }
        } else {
            // Need to reduce vocabulary size - this is more complex
            println!("WARNING: Requested smaller vocabulary ({}) than tokenizer currently has ({})", 
                   new_size, current_size);
            println!("Currently only vocabulary expansion is fully supported");
        }
        
        // Verify the new size
        let final_size = self.vocab.len();
        println!("Final vocabulary size: {}", final_size);
        
        // Ensure we actually reached the target size
        if final_size != new_size {
            println!("⚠️ Warning: Could not resize vocabulary exactly to {}. New size is {}", 
                     new_size, final_size);
        }
    }
}

impl Tokenizer for WordPieceBPETokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        
        // Pre-tokenize text to handle word boundaries
        let pre_tokens = self.pre_tokenize(text);
        
        // Apply BPE to each token
        for token in pre_tokens {
            if token.starts_with('[') && token.ends_with(']') {
                // Special tokens (including space) are kept as-is
                result.push(token);
            } else if !token.is_empty() {
                // Apply BPE to regular words
                let bpe_tokens = self.apply_bpe(&token);
                result.extend(bpe_tokens);
            }
        }
        
        result
    }
    
    fn encode(&self, text: &str) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let unk_id = self.vocab.token_to_id(&self.unk_token).unwrap();
        
        tokens.iter()
              .map(|token| self.vocab.token_to_id(token).unwrap_or(unk_id))
              .collect()
    }
    
    fn decode(&self, ids: &[usize]) -> String {
        let tokens: Vec<String> = ids.iter()
            .filter_map(|&id| self.vocab.id_to_token(id).map(|s| s.to_string()))
            .collect();
            
        // Debug token info
        if cfg!(debug_assertions) {
            println!("Decoding {} tokens: {:?}", tokens.len(), tokens);
        }
        
        // Reconstruct the text with proper handling of special tokens
        let mut result = String::new();
        let mut last_was_space = false;
        let mut current_word = String::new();
        
        for (i, token) in tokens.iter().enumerate() {
            let next_token = if i + 1 < tokens.len() { Some(&tokens[i + 1]) } else { None };
            
            // Check if the next token is punctuation
            let next_is_punct = next_token
                .map(|t| t.starts_with('[') && t.ends_with(']') && t.len() > 2)
                .unwrap_or(false);
            
            if token == &self.space_token {
                // Process accumulated word if any
                if !current_word.is_empty() {
                    result.push_str(&current_word);
                    current_word.clear();
                }
                
                // Space token
                // Only add space if there wasn't one already
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else if token.starts_with('[') && token.ends_with(']') && token.len() > 2 {
                // Process accumulated word if any
                if !current_word.is_empty() {
                    result.push_str(&current_word);
                    current_word.clear();
                }
                
                // Punctuation token
                if let Some(punct) = token.get(1..token.len()-1) {
                    result.push_str(punct);
                    last_was_space = false;
                }
            } else if token.ends_with(&self.end_token) {
                // Complete word token with end marker
                let word = token.trim_end_matches(&self.end_token);
                
                // Add to current word
                if !word.is_empty() {
                    current_word.push_str(word);
                }
                
                // Output completed word
                result.push_str(&current_word);
                current_word.clear();
                
                // Add space after word if needed
                if !next_is_punct && next_token.map(|t| t != &self.space_token).unwrap_or(true) {
                    result.push(' ');
                    last_was_space = true;
                } else {
                    last_was_space = false;
                }
            } else if self.vocab.is_special_token(token) {
                // Skip other special tokens
            } else {
                // Regular token (individual characters or subwords)
                
                // Only add valid content to current word
                if token.chars().all(|c| c.is_alphanumeric() || c == '\'' || c == '-' || c == '_') {
                    current_word.push_str(token);
                }
            }
        }
        
        // Process any remaining word
        if !current_word.is_empty() {
            result.push_str(&current_word);
        }
        
        // Trim trailing space if any
        let final_result = result.trim_end().to_string();
        
        // Debug final result
        if cfg!(debug_assertions) {
            println!("Final decoded text: {}", final_result);
        }
        
        final_result
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
    fn test_word_bpe_tokenizer_simple() {
        let mut tokenizer = WordPieceBPETokenizer::new();
        
        // Test pre-tokenization
        let tokens = tokenizer.pre_tokenize("Hello, world!");
        assert_eq!(tokens, vec!["hello", "[,]", "[SPACE]", "world", "[!]"]);
        
        // For the decoder test, we'll manually create a token sequence
        // that represents what we want to decode, rather than using the tokenizer
        let test_tokens = vec!["hello</w>", "[,]", "[SPACE]", "world</w>", "[!]"];
        
        // Convert to token IDs
        let test_ids: Vec<usize> = test_tokens.iter()
            .map(|token| {
                // Add token to vocab to ensure we have IDs for our test tokens
                let id = tokenizer.as_vocab_mut().unwrap().add_token(token);
                id
            })
            .collect();
        
        // Test decoding directly
        let decoded = tokenizer.decode(&test_ids);
        assert_eq!(decoded, "hello, world!");
    }
    
    #[test]
    fn test_word_bpe_learn() {
        let mut tokenizer = WordPieceBPETokenizer::new();
        
        // Example text with word boundaries
        let text = "Hello, hello! This is a test. Hello again.";
        
        // Learn BPE rules
        tokenizer.learn_bpe(text, 200, 1);
        
        // Verify that rules have been learned
        assert!(!tokenizer.merges.is_empty());
        
        // Tokenize a word present in the training text
        let tokens = tokenizer.tokenize("Hello, test!");
        assert!(!tokens.is_empty());
        
        // Verify that tokenization and decoding are consistent
        let ids = tokenizer.encode("Hello, test!");
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hello, test!");
    }
    
    #[test]
    fn test_word_boundaries() {
        let mut tokenizer = WordPieceBPETokenizer::new();
        
        // Example text with word boundaries
        let text = "Hello, world! How are you? I'm fine, thank you.";
        
        // Learn BPE rules
        tokenizer.learn_bpe(text, 200, 1);
        
        // Verify tokenization and decoding of a sentence
        let tokens = tokenizer.tokenize(text);
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        
        // Check if the decoded text has proper word boundaries
        assert_eq!(decoded, "hello, world! how are you? i'm fine, thank you.");
        
        // Test with some problematic input
        let strange_text = "Hello@world#with$strange^chars&and*weird(symbols)!";
        let ids = tokenizer.encode(strange_text);
        let decoded = tokenizer.decode(&ids);
        
        // Should properly filter invalid characters
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
        assert!(decoded.contains("with"));
        assert!(decoded.contains("strange"));
        assert!(decoded.contains("chars"));
        assert!(decoded.contains("and"));
        assert!(decoded.contains("weird"));
        assert!(decoded.contains("symbols"));
    }
} 