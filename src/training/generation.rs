use std::collections::HashMap;
use ndarray::{s, Array1, Array2};
use crate::tokenizer::Tokenizer;
use crate::nabla::tensor::Tensor;
use rand::prelude::*;
use rand::SeedableRng;
use crate::training::ModelOutput;
use crate::training::Trainer;

/// Enhanced text generation utility that prevents repetition loops
pub struct TextGenerator {
    /// Repetition penalty for discouraging repeated tokens
    repetition_penalty: f32,
    /// Presence penalty (as in OpenAI models) for discouraging repeating already seen tokens
    presence_penalty: f32,
    /// Frequency penalty for discouraging frequently used tokens 
    frequency_penalty: f32,
    /// Minimum token probability threshold (exclusive, 0.0 means no filtering)
    min_probability: f32,
    /// Entropy threshold for detecting repetitive patterns (higher means more sensitivity)
    entropy_threshold: f32,
    /// Minimum generated sequence length
    min_length: usize,
    /// Maximum generated sequence length
    max_length: usize,
    /// Dynamic temperature adjustment (escalation factor when repetition detected)
    dynamic_temperature: bool,
    /// Base temperature value
    temperature: f32,
    /// Tokens to use for sentence endings for early stopping
    stop_tokens: Vec<String>,
}

impl TextGenerator {
    /// Creates a new text generator with default settings
    pub fn new() -> Self {
        Self {
            repetition_penalty: 1.2,
            presence_penalty: 0.1,
            frequency_penalty: 0.1,
            min_probability: 0.0,
            entropy_threshold: 1.5,
            min_length: 10,
            max_length: 100,
            dynamic_temperature: true,
            temperature: 0.8,
            stop_tokens: vec![".".to_string(), "!".to_string(), "?".to_string()],
        }
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
    
    /// Set the minimum probability threshold
    pub fn with_min_probability(mut self, threshold: f32) -> Self {
        self.min_probability = threshold;
        self
    }
    
    /// Set the entropy threshold for repetition detection
    pub fn with_entropy_threshold(mut self, threshold: f32) -> Self {
        self.entropy_threshold = threshold;
        self
    }
    
    /// Set the generation length limits
    pub fn with_length_constraints(mut self, min_length: usize, max_length: usize) -> Self {
        self.min_length = min_length;
        self.max_length = max_length;
        self
    }
    
    /// Set whether to use dynamic temperature adjustments
    pub fn with_dynamic_temperature(mut self, enable: bool) -> Self {
        self.dynamic_temperature = enable;
        self
    }
    
    /// Set the base temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    /// Set custom stop tokens
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<String>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }
    
    /// Calculate the entropy of a probability distribution
    fn calculate_entropy(&self, probs: &[f32]) -> f32 {
        let mut entropy = 0.0;
        for &p in probs {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }
    
    /// Apply penalties to logits based on token history
    fn apply_penalties(&self, logits: &mut [f32], previous_tokens: &[usize]) {
        // Count token frequencies
        let mut token_counts: HashMap<usize, usize> = HashMap::new();
        for &token_id in previous_tokens {
            *token_counts.entry(token_id).or_insert(0) += 1;
        }
        
        for (token_id, &count) in token_counts.iter() {
            if *token_id < logits.len() {
                // Apply repetition penalty (scales more for repeated tokens)
                if count > 0 {
                    let penalty = self.repetition_penalty.powf(count as f32);
                    if logits[*token_id] > 0.0 {
                        logits[*token_id] /= penalty;
                    } else {
                        logits[*token_id] *= penalty;
                    }
                }
                
                // Apply presence penalty (flat penalty for tokens that have appeared)
                logits[*token_id] -= self.presence_penalty;
                
                // Apply frequency penalty (scales with token frequency)
                logits[*token_id] -= self.frequency_penalty * count as f32;
            }
        }
    }
    
    /// Check if the sequence has a repeating pattern
    fn detect_repetition(&self, tokens: &[usize], min_pattern_length: usize, max_pattern_length: usize) -> bool {
        if tokens.len() < min_pattern_length * 2 {
            return false;
        }
        
        // Check for patterns of increasing length
        for pattern_len in min_pattern_length..=max_pattern_length {
            if tokens.len() < pattern_len * 2 {
                continue;
            }
            
            // Get the last 'pattern_len' tokens
            let pattern = &tokens[tokens.len() - pattern_len..];
            
            // Get the tokens before that to check for a match
            let prev_segment = &tokens[tokens.len() - pattern_len * 2..tokens.len() - pattern_len];
            
            // Check if they match
            if pattern == prev_segment {
                return true;
            }
        }
        
        false
    }
    
    /// Apply temperature to logits
    fn apply_temperature(&self, logits: &mut [f32], current_temp: f32) {
        for logit in logits.iter_mut() {
            *logit /= current_temp;
        }
    }
    
    /// Convert logits to probabilities using softmax
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Find the maximum logit for numerical stability
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Calculate exponents
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        
        // Sum of exponents for normalization
        let sum_exp: f32 = exp_logits.iter().sum();
        
        // Normalize to get probabilities
        exp_logits.iter().map(|&x| x / sum_exp).collect()
    }
    
    /// Sample a token from probability distribution 
    fn sample_token(&self, probs: &[f32]) -> usize {
        // Generate a random number between 0 and 1
        let mut rng = rand::thread_rng();
        let rand_val: f32 = rand::Rng::gen_range(&mut rng, 0.0..1.0);
        
        // Cumulative probability for sampling
        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if rand_val < cumulative {
                return i;
            }
        }
        
        // Fallback (should rarely happen due to floating-point precision)
        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Generate text using a model with enhanced mechanisms to prevent repetition
    pub fn generate<T: Tokenizer + ?Sized>(
        &self,
        model: &impl TextGenerationModel,
        tokenizer: &T,
        prompt: &str,
        max_tokens: Option<usize>
    ) -> String {
        // Encode the prompt to token IDs
        let mut tokens = tokenizer.encode(prompt);
        let prompt_length = tokens.len();
        
        // Use the provided max_tokens or default to the instance's max_length
        let max_new_tokens = max_tokens.unwrap_or(self.max_length);
        let max_total_tokens = prompt_length + max_new_tokens;
        
        // Current temperature (may change if dynamic_temperature is enabled)
        let mut current_temp = self.temperature;
        
        // Main generation loop
        for _ in 0..max_new_tokens {
            // Prepare input shape: [1, sequence_length]
            let input = vec![tokens.clone()];
            
            // Get logits from the model
            let output = model.forward(&input, None);
            
            // Extract logits for the last token
            let mut last_token_logits = output.logits.data
                .slice(s![0, tokens.len() - 1, ..])
                .to_owned()
                .into_raw_vec();
            
            // Apply penalties to discourage repetition
            self.apply_penalties(&mut last_token_logits, &tokens);
            
            // Check for repetition patterns that may indicate a loop
            let has_repetition = self.detect_repetition(&tokens, 3, 10);
            
            // Dynamically adjust temperature if repetition detected
            if self.dynamic_temperature && has_repetition {
                // Increase temperature to encourage diversity
                current_temp = (current_temp * 1.2).min(2.0);
            } else if self.dynamic_temperature {
                // Gradually cool down temperature
                current_temp = (current_temp * 0.98).max(self.temperature);
            }
            
            // Apply temperature to logits
            self.apply_temperature(&mut last_token_logits, current_temp);
            
            // Convert to probabilities
            let probs = self.softmax(&last_token_logits);
            
            // Filter out low probability tokens (optional)
            let filtered_probs = if self.min_probability > 0.0 {
                let mut filtered = probs.clone();
                for p in &mut filtered {
                    if *p < self.min_probability {
                        *p = 0.0;
                    }
                }
                
                // Renormalize
                let sum: f32 = filtered.iter().sum();
                if sum > 0.0 {
                    filtered.iter().map(|&p| p / sum).collect()
                } else {
                    probs
                }
            } else {
                probs
            };
            
            // Calculate entropy of the distribution to detect low-diversity outputs
            let entropy = self.calculate_entropy(&filtered_probs);
            
            // Sample next token
            let next_token = self.sample_token(&filtered_probs);
            
            // Add the new token
            tokens.push(next_token);
            
            // Check if we should stop generation
            if tokens.len() > self.min_length {
                // Check for stop tokens
                if let Some(token_text) = tokenizer.get_vocab().id_to_token(next_token) {
                    if self.stop_tokens.contains(&token_text.to_string()) {
                        break;
                    }
                }
                
                // Stop if entropy is very low (likely stuck in a repetition loop)
                if entropy < self.entropy_threshold {
                    break;
                }
                
                // Stop if we've reached the maximum length
                if tokens.len() >= max_total_tokens {
                    break;
                }
            }
        }
        
        // Decode tokens back to text
        tokenizer.decode(&tokens)
    }
}

/// Default for TextGenerator
impl Default for TextGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for models that can generate text
pub trait TextGenerationModel {
    /// Forward pass of the model, producing logits for the next token
    fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput;
}

/// Implement TextGenerationModel for Trainer
impl TextGenerationModel for Trainer {
    fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput {
        // This simply delegates to the existing forward method
        self.forward(input, target)
    }
} 