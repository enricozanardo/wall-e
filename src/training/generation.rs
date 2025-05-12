use std::collections::{HashMap, HashSet, VecDeque};
use ndarray::s;
use crate::tokenizer::Tokenizer;
use rand::prelude::*;
use rand::Rng;

/// Configuration for text generation
pub struct TextGenerator {
    /// Temperature for controlling randomness (higher = more random)
    temperature: f32,
    /// Whether to dynamically adjust temperature
    dynamic_temperature: bool,
    /// Repetition penalty to apply
    repetition_penalty: f32,
    /// Presence penalty to apply for tokens that have appeared
    presence_penalty: f32,
    /// Frequency penalty to apply based on token frequency
    frequency_penalty: f32,
    /// Maximum length for n-gram repetition detection
    ngram_size: usize,
    /// Size of window to check for local repetitions
    local_window_size: usize,
    /// Entropy threshold for dynamic temperature
    entropy_threshold: f32,
    /// Common words that should have additional handling
    common_words: HashSet<String>,
    /// Whether to use an adaptive temperature that changes based on context
    adaptive_temperature: bool,
    /// Maximum number of tokens to generate by default
    default_max_tokens: usize,
}

impl TextGenerator {
    /// Create a new text generator with default settings
    pub fn new() -> Self {
        let common_words = HashSet::from([
            "the".to_string(), 
            "a".to_string(), 
            "an".to_string(), 
            "and".to_string(),
            "in".to_string(),
            "of".to_string(),
            "to".to_string(),
            "is".to_string(),
            "was".to_string(),
        ]);
        
        Self {
            temperature: 1.0,
            dynamic_temperature: false,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            ngram_size: 3,
            local_window_size: 20,
            entropy_threshold: 2.0,
            common_words,
            adaptive_temperature: false,
            default_max_tokens: 50,
        }
    }
    
    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    /// Enable or disable dynamic temperature
    pub fn with_dynamic_temperature(mut self, enable: bool) -> Self {
        self.dynamic_temperature = enable;
        self
    }
    
    /// Set the repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }
    
    /// Set the presence penalty
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }
    
    /// Set the frequency penalty
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }
    
    /// Set the entropy threshold for dynamic temperature
    pub fn with_entropy_threshold(mut self, threshold: f32) -> Self {
        self.entropy_threshold = threshold;
        self
    }
    
    /// Set the n-gram size for repetition detection
    pub fn with_ngram_size(mut self, size: usize) -> Self {
        self.ngram_size = size;
        self
    }
    
    /// Set the local window size for repetition detection
    pub fn with_local_window_size(mut self, size: usize) -> Self {
        self.local_window_size = size;
        self
    }
    
    /// Enable or disable adaptive temperature
    pub fn with_adaptive_temperature(mut self, enable: bool) -> Self {
        self.adaptive_temperature = enable;
        self
    }
    
    /// Generate text using the model
    pub fn generate<T: Tokenizer>(
        &self,
        model: &dyn crate::training::TextGenerationModel,
        tokenizer: &T,
        prompt: &str,
        max_tokens: Option<usize>
    ) -> String {
        // Set maximum tokens to generate
        let max_tokens = max_tokens.unwrap_or(self.default_max_tokens);
        
        // Tokenize the prompt
        let prompt_tokens = tokenizer.encode(prompt);
        
        if prompt_tokens.is_empty() {
            return String::new();
        }
        
        // Track generated tokens
        let mut tokens = prompt_tokens.clone();
        
        // Track frequency of each token
        let mut token_freq = HashMap::new();
        for &token in &tokens {
            *token_freq.entry(token).or_insert(0) += 1;
        }
        
        // Track n-grams to detect repetition patterns
        let mut recent_ngrams: VecDeque<Vec<usize>> = VecDeque::new();
        
        // Initialize n-grams from prompt
        if self.ngram_size > 0 && tokens.len() >= self.ngram_size {
            for i in 0..=tokens.len() - self.ngram_size {
                let ngram = tokens[i..i + self.ngram_size].to_vec();
                recent_ngrams.push_back(ngram);
                if recent_ngrams.len() > self.local_window_size {
                    recent_ngrams.pop_front();
                }
            }
        }
        
        // Generate tokens
        for _ in 0..max_tokens {
            let input = vec![tokens.clone()];
            
            // Get logits from model
            let output = model.forward(&input, None);
            let logits = output.logits.data.slice(s![0, tokens.len() - 1, ..]);
            
            // Apply temperature and penalties
            let mut scaled_logits = Vec::with_capacity(logits.len());
            
            // Calculate entropy for dynamic temperature
            let mut entropy = 0.0;
            if self.dynamic_temperature {
                let logits_vec: Vec<f32> = logits.iter().cloned().collect();
                let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let scaled = logits_vec.iter().map(|&l| (l - max_logit).exp()).collect::<Vec<_>>();
                let sum: f32 = scaled.iter().sum();
                
                if sum > 0.0 {
                    let probs = scaled.iter().map(|&s| s / sum);
                    for p in probs {
                        if p > 0.0 {
                            entropy -= p * p.ln();
                        }
                    }
                }
            }
            
            // Adaptively adjust temperature based on entropy
            let effective_temp = if self.dynamic_temperature {
                if entropy > self.entropy_threshold {
                    // High entropy means uncertain prediction - lower temperature
                    self.temperature * 0.8
                } else {
                    // Low entropy means confident prediction - use normal temperature
                    self.temperature
                }
            } else {
                self.temperature
            };
            
            // Calculate penalties for each token
            for (idx, &logit) in logits.iter().enumerate() {
                let token_id = idx;
                
                // Calculate repetition penalty
                let base_penalty = if token_freq.contains_key(&token_id) {
                    // This token has been generated before - apply repetition penalty
                    let freq = *token_freq.get(&token_id).unwrap_or(&0);
                    
                    // Increased penalty for common words that appear too often
                    let token_str = tokenizer.get_vocab().id_to_token(token_id);
                    let common_word_mult = if let Some(token_str) = token_str {
                        if self.common_words.contains(&token_str.to_lowercase()) {
                            1.5 // Increased from 1.2 to 1.5 (50% stronger penalty for common words)
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };
                    
                    let presence_component = self.presence_penalty;
                    let frequency_component = self.frequency_penalty * freq as f32;
                    let repetition_component = self.repetition_penalty * common_word_mult;
                    
                    presence_component + frequency_component + repetition_component - 1.0
                } else {
                    0.0
                };
                
                // Additional n-gram repetition penalty
                let mut ngram_penalty = 0.0;
                if self.ngram_size > 0 && !tokens.is_empty() {
                    // Look for repetition of n-grams
                    let window_start = tokens.len().saturating_sub(self.local_window_size);
                    let window_tokens = &tokens[window_start..];
                    
                    // Create potential n-grams ending with the candidate token
                    let min_ngram_size = 2.min(self.ngram_size);
                    
                    for n in 1..=min_ngram_size {
                        if window_tokens.len() >= n - 1 {
                            let mut potential_ngram = window_tokens[window_tokens.len() - (n - 1)..].to_vec();
                            potential_ngram.push(token_id);
                            
                            // Check if this ngram already exists in the recent text
                            for existing_ngram in &recent_ngrams {
                                if existing_ngram == &potential_ngram {
                                    // Penalize ngram repetition more strongly for longer ngrams
                                    ngram_penalty += (n as f32) * 0.5;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // More sophisticated logit adjustment
                let penalty = base_penalty + ngram_penalty;
                let adjusted_logit = if logit < 0.0 {
                    logit * (1.0 + penalty)
                } else {
                    logit / (1.0 + penalty)
                };
                
                // Apply temperature
                scaled_logits.push(adjusted_logit / effective_temp);
            }
            
            // Convert to probabilities
            let max_logit = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = scaled_logits.iter().map(|&l| (l - max_logit).exp()).collect::<Vec<_>>();
            let sum: f32 = probs.iter().sum();
            
            if sum <= 0.0 {
                break;
            }
            
            probs.iter_mut().for_each(|p| *p /= sum);
            
            // Sample from distribution
            let mut rng = thread_rng();
            let mut cumsum = 0.0;
            let sample: f32 = rng.gen_range(0.0..1.0);
            
            let mut next_token = 0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if sample < cumsum {
                    next_token = i;
                    break;
                }
            }
            
            // Add token to the sequence
            tokens.push(next_token);
            
            // Update token frequency
            *token_freq.entry(next_token).or_insert(0) += 1;
            
            // Update n-grams for repetition detection
            if self.ngram_size > 0 && tokens.len() >= self.ngram_size {
                let ngram = tokens[tokens.len() - self.ngram_size..].to_vec();
                recent_ngrams.push_back(ngram);
                if recent_ngrams.len() > self.local_window_size {
                    recent_ngrams.pop_front();
                }
            }
        }
        
        // Decode tokens
        tokenizer.decode(&tokens)
    }
}

/// Default for TextGenerator
impl Default for TextGenerator {
    fn default() -> Self {
        Self::new()
    }
} 