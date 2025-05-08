use std::collections::HashMap;
use ndarray::Array2;
use crate::tokenizer::Tokenizer;
use crate::tokenizer::WordPieceBPETokenizer;
use crate::training::{Trainer, ModelOutput, TextGenerationModel};
use crate::training::curriculum::{CurriculumScheduler, DifficultyLevel, CurriculumExample};
use crate::training::generation::TextGenerator;
use rand::prelude::*;

/// Enhanced trainer extending the original Trainer with advanced features
/// 
/// This trainer adds curriculum learning and improvements for handling
/// word boundaries and preventing repetitive text generation.
pub struct EnhancedTrainer {
    /// Base trainer
    pub trainer: Trainer,
    /// Advanced tokenizer
    tokenizer: WordPieceBPETokenizer,
    /// Curriculum scheduler for organizing training progression
    curriculum: CurriculumScheduler,
    /// Text generator with repetition prevention
    generator: TextGenerator,
    /// Whether to use curriculum learning
    use_curriculum: bool,
    /// Whether to dynamically adjust learning rate
    dynamic_lr: bool,
    /// Base learning rate
    learning_rate: f32,
    /// Current epoch
    current_epoch: usize,
    /// Stats from training
    stats: HashMap<String, Vec<f32>>,
}

impl EnhancedTrainer {
    /// Create a new enhanced trainer
    pub fn new(
        model_dim: usize,
        ff_dim: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_rate: f32,
        learning_rate: f32,
    ) -> Self {
        // Initialize the tokenizer
        let tokenizer = WordPieceBPETokenizer::new();
        
        // Create a base trainer
        let trainer = Trainer::new(
            Box::new(tokenizer.clone()),
            model_dim,
            ff_dim,
            num_heads,
            num_layers,
            dropout_rate,
            learning_rate
        );
        
        // Create a curriculum scheduler
        let curriculum = CurriculumScheduler::new();
        
        // Create a text generator
        let generator = TextGenerator::new()
            .with_repetition_penalty(1.2)
            .with_temperature(0.8)
            .with_dynamic_temperature(true);
        
        Self {
            trainer,
            tokenizer,
            curriculum,
            generator,
            use_curriculum: true,
            dynamic_lr: true,
            learning_rate,
            current_epoch: 0,
            stats: HashMap::new(),
        }
    }
    
    /// Enable or disable curriculum learning
    pub fn with_curriculum_learning(mut self, enable: bool) -> Self {
        self.use_curriculum = enable;
        self
    }
    
    /// Configure the curriculum scheduler
    pub fn with_curriculum_scheduler(mut self, scheduler: CurriculumScheduler) -> Self {
        self.curriculum = scheduler;
        self
    }
    
    /// Configure the text generator
    pub fn with_text_generator(mut self, generator: TextGenerator) -> Self {
        self.generator = generator;
        self
    }
    
    /// Enable or disable dynamic learning rate
    pub fn with_dynamic_learning_rate(mut self, enable: bool) -> Self {
        self.dynamic_lr = enable;
        self
    }
    
    /// Configure gradient clipping
    pub fn with_gradient_clipping(mut self, threshold: Option<f32>) -> Self {
        self.trainer.with_gradient_clipping(threshold);
        self
    }
    
    /// Get the tokenizer
    pub fn get_tokenizer(&self) -> &WordPieceBPETokenizer {
        &self.tokenizer
    }
    
    /// Get a mutable reference to the tokenizer
    pub fn get_tokenizer_mut(&mut self) -> &mut WordPieceBPETokenizer {
        &mut self.tokenizer
    }
    
    /// Learn tokenizer vocabulary from text
    pub fn learn_tokenizer_from_text(&mut self, text: &str, vocab_size: usize, min_frequency: usize) {
        self.tokenizer.learn_bpe(text, vocab_size, min_frequency);
        
        // Update the tokenizer in the base trainer
        // Note: This is a hack since we can't directly update the tokenizer in the trainer
        self.trainer = Trainer::new(
            Box::new(self.tokenizer.clone()),
            self.trainer.get_model_dim(),
            self.trainer.get_ff_dim(),
            self.trainer.get_num_heads(),
            self.trainer.get_num_layers(),
            self.trainer.get_dropout_rate(),
            self.learning_rate
        );
    }
    
    /// Convert examples to curriculum format and add to scheduler
    pub fn add_examples_to_curriculum(&mut self, inputs: &[Vec<usize>], targets: &Array2<usize>) {
        self.curriculum.add_examples(inputs, targets);
    }
    
    /// Get batch size based on current curriculum level
    pub fn get_batch_size(&self) -> usize {
        if self.use_curriculum {
            self.curriculum.get_batch_size()
        } else {
            32 // Default batch size
        }
    }
    
    /// Get the current maximum sequence length
    pub fn get_max_seq_len(&self) -> usize {
        if self.use_curriculum {
            self.curriculum.get_current_max_length()
        } else {
            self.trainer.get_max_seq_len()
        }
    }
    
    /// Calculate learning rate based on epoch and curriculum level
    fn calculate_learning_rate(&self) -> f32 {
        if !self.dynamic_lr {
            return self.learning_rate;
        }
        
        // Get the current curriculum level
        let level = if self.use_curriculum {
            self.curriculum.get_current_level() as usize
        } else {
            DifficultyLevel::Medium as usize
        };
        
        // Calculate a level factor (higher levels get lower learning rates)
        let level_factor = match level {
            0 => 1.2, // Very easy - higher learning rate
            1 => 1.0, // Easy - base learning rate
            2 => 0.8, // Medium - lower learning rate
            3 => 0.6, // Hard - much lower learning rate
            _ => 0.4, // Very hard - lowest learning rate
        };
        
        // Calculate an epoch factor (later epochs get lower learning rates)
        let epoch = self.current_epoch;
        let epoch_factor = 1.0 / (1.0 + (epoch as f32 * 0.1));
        
        // Combine factors
        self.learning_rate * level_factor * epoch_factor
    }
    
    /// Train on a single batch with repetition penalties
    pub fn train_step_with_penalties(&mut self, batch: &Vec<Vec<usize>>, targets: &Array2<usize>) -> f32 {
        // First, perform the regular training step
        let loss = self.trainer.train_step(batch, targets);
        
        // Record the loss for stats
        self.stats.entry("loss".to_string())
            .or_insert_with(Vec::new)
            .push(loss);
        
        loss
    }
    
    /// Train for one epoch with curriculum learning and repetition penalties
    pub fn train_epoch(&mut self, inputs: &[Vec<Vec<usize>>], targets: &[Array2<usize>]) -> f32 {
        if self.use_curriculum {
            self.train_epoch_with_curriculum(inputs, targets)
        } else {
            self.train_epoch_standard(inputs, targets)
        }
    }
    
    /// Train for one epoch using standard approach (no curriculum)
    fn train_epoch_standard(&mut self, inputs: &[Vec<Vec<usize>>], targets: &[Array2<usize>]) -> f32 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Calculate the current learning rate
        let lr = self.calculate_learning_rate();
        
        // Set the learning rate if dynamic
        if self.dynamic_lr {
            // Directly update the learning rate in the optimizer
            self.trainer.set_learning_rate(lr);
        }
        
        // Process each batch
        for (batch, target) in inputs.iter().zip(targets.iter()) {
            let loss = self.train_step_with_penalties(batch, target);
            total_loss += loss;
            num_batches += 1;
        }
        
        // Increment epoch counter
        self.current_epoch += 1;
        
        // Return average loss
        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }
    
    /// Train for one epoch using curriculum learning
    fn train_epoch_with_curriculum(&mut self, inputs: &[Vec<Vec<usize>>], targets: &[Array2<usize>]) -> f32 {
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        
        // Calculate the current learning rate based on curriculum level
        let lr = self.calculate_learning_rate();
        
        // Set the learning rate if dynamic
        if self.dynamic_lr {
            // Directly update the learning rate in the optimizer
            self.trainer.set_learning_rate(lr);
        }
        
        // Get examples for the current curriculum level
        let mut examples = self.curriculum.get_training_examples();
        
        // Shuffle examples
        examples.shuffle(&mut thread_rng());
        
        // Get the current batch size
        let batch_size = self.get_batch_size();
        
        // Process examples in batches
        for chunk in examples.chunks(batch_size) {
            // Create batch
            let mut batch_inputs = Vec::new();
            let mut batch_targets_vec = Vec::new();
            
            for example in chunk {
                batch_inputs.push(example.input.clone());
                batch_targets_vec.push(example.target.clone());
            }
            
            // Convert targets to Array2
            let max_target_len = batch_targets_vec.iter().map(|t| t.len()).max().unwrap_or(1);
            let mut batch_targets = Array2::zeros((batch_targets_vec.len(), max_target_len));
            
            for (i, target) in batch_targets_vec.iter().enumerate() {
                for (j, &token) in target.iter().enumerate() {
                    batch_targets[[i, j]] = token;
                }
            }
            
            // Train on this batch
            let loss = self.train_step_with_penalties(&batch_inputs, &batch_targets);
            total_loss += loss;
            num_batches += 1;
        }
        
        // Advance to the next epoch in the curriculum
        let level_changed = self.curriculum.next_epoch();
        if level_changed {
            println!("Advancing to curriculum level: {:?}", self.curriculum.get_current_level());
        }
        
        // Increment epoch counter
        self.current_epoch += 1;
        
        // Return average loss
        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }
    
    /// Generate text with improved anti-repetition mechanisms
    pub fn generate_text(&self, prompt: &str, max_tokens: Option<usize>) -> String {
        self.generator.generate(self, &self.tokenizer, prompt, max_tokens)
    }
    
    /// Get training statistics
    pub fn get_stats(&self) -> &HashMap<String, Vec<f32>> {
        &self.stats
    }
    
    /// Save the model
    pub fn save_model(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Convert ModelError to Box<dyn std::error::Error>
        self.trainer.save_model(path).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
    
    /// Debug tokenization of a text sample
    pub fn debug_tokenize(&self, text: &str) {
        // Use the debug_tokenize method from WordPieceBPETokenizer
        self.tokenizer.debug_tokenize(text);
    }
    
    /// Set the repetition penalty for the tokenizer
    pub fn set_tokenizer_repetition_penalty(&mut self, penalty: f32) {
        self.tokenizer.set_repetition_penalty(penalty);
    }
    
    /// Configure the text generator with stronger anti-repetition settings
    pub fn configure_anti_repetition(&mut self, repetition_penalty: f32, presence_penalty: f32, frequency_penalty: f32) {
        // Create a new generator with stronger anti-repetition settings
        let new_generator = TextGenerator::new()
            .with_repetition_penalty(repetition_penalty)
            .with_presence_penalty(presence_penalty)
            .with_frequency_penalty(frequency_penalty)
            .with_temperature(0.8)  // Standard temperature
            .with_dynamic_temperature(true)
            .with_entropy_threshold(1.0);
        
        self.generator = new_generator;
    }
}

/// Implement TextGenerationModel for EnhancedTrainer
impl TextGenerationModel for EnhancedTrainer {
    fn forward(&self, input: &Vec<Vec<usize>>, target: Option<&Array2<usize>>) -> ModelOutput {
        // Delegate to the base trainer
        self.trainer.forward(input, target)
    }
} 