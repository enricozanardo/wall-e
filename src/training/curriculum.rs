use std::collections::HashMap;
use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Difficulty level for curriculum learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DifficultyLevel {
    VeryEasy = 0,
    Easy = 1,
    Medium = 2,
    Hard = 3,
    VeryHard = 4,
}

impl DifficultyLevel {
    /// Convert from integer to DifficultyLevel
    pub fn from_int(value: usize) -> Self {
        match value {
            0 => DifficultyLevel::VeryEasy,
            1 => DifficultyLevel::Easy,
            2 => DifficultyLevel::Medium,
            3 => DifficultyLevel::Hard,
            _ => DifficultyLevel::VeryHard,
        }
    }
    
    /// Get the next difficulty level
    pub fn next(&self) -> Self {
        match self {
            DifficultyLevel::VeryEasy => DifficultyLevel::Easy,
            DifficultyLevel::Easy => DifficultyLevel::Medium,
            DifficultyLevel::Medium => DifficultyLevel::Hard,
            DifficultyLevel::Hard => DifficultyLevel::VeryHard,
            DifficultyLevel::VeryHard => DifficultyLevel::VeryHard,
        }
    }
}

/// Represents an example for curriculum learning
#[derive(Clone)]
pub struct CurriculumExample {
    /// Index in the original dataset
    pub index: usize,
    /// Input token sequences
    pub input: Vec<usize>,
    /// Target token sequences
    pub target: Vec<usize>,
    /// Difficulty level
    pub difficulty: DifficultyLevel,
    /// Length of the example
    pub length: usize,
}

/// Manages curriculum learning by organizing training progression
pub struct CurriculumScheduler {
    /// Current difficulty level
    current_level: DifficultyLevel,
    /// Examples organized by difficulty level
    examples_by_level: HashMap<DifficultyLevel, Vec<CurriculumExample>>,
    /// Current epoch
    current_epoch: usize,
    /// Epochs per level
    epochs_per_level: usize,
    /// Maximum sequence length per level
    max_length_per_level: HashMap<DifficultyLevel, usize>,
    /// Mix-in percentage of harder examples (0.0 - 1.0)
    harder_examples_ratio: f32,
    /// Mix-in percentage of easier examples (0.0 - 1.0)
    easier_examples_ratio: f32,
    /// Random seed for shuffling
    seed: u64,
}

impl CurriculumScheduler {
    /// Create a new curriculum scheduler with default settings
    pub fn new() -> Self {
        let mut max_length_per_level = HashMap::new();
        max_length_per_level.insert(DifficultyLevel::VeryEasy, 16);
        max_length_per_level.insert(DifficultyLevel::Easy, 32);
        max_length_per_level.insert(DifficultyLevel::Medium, 64);
        max_length_per_level.insert(DifficultyLevel::Hard, 128);
        max_length_per_level.insert(DifficultyLevel::VeryHard, 256);
        
        Self {
            current_level: DifficultyLevel::VeryEasy,
            examples_by_level: HashMap::new(),
            current_epoch: 0,
            epochs_per_level: 1,
            max_length_per_level,
            harder_examples_ratio: 0.1,
            easier_examples_ratio: 0.05,
            seed: 42,
        }
    }
    
    /// Set the epochs to spend at each difficulty level
    pub fn with_epochs_per_level(mut self, epochs: usize) -> Self {
        self.epochs_per_level = epochs;
        self
    }
    
    /// Set the maximum sequence length for each difficulty level
    pub fn with_max_length_per_level(mut self, lengths: HashMap<DifficultyLevel, usize>) -> Self {
        self.max_length_per_level = lengths;
        self
    }
    
    /// Set the ratio of harder examples to mix in
    pub fn with_harder_examples_ratio(mut self, ratio: f32) -> Self {
        self.harder_examples_ratio = ratio.max(0.0).min(1.0);
        self
    }
    
    /// Set the ratio of easier examples to mix in
    pub fn with_easier_examples_ratio(mut self, ratio: f32) -> Self {
        self.easier_examples_ratio = ratio.max(0.0).min(1.0);
        self
    }
    
    /// Set the random seed for shuffling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    
    /// Calculate the difficulty of a text example based on various factors
    pub fn calculate_difficulty(&self, input: &[usize], target: &[usize]) -> DifficultyLevel {
        // Primary factor: sequence length
        let length = input.len().max(target.len());
        
        // Determine difficulty level based on length
        if length <= self.max_length_per_level[&DifficultyLevel::VeryEasy] {
            DifficultyLevel::VeryEasy
        } else if length <= self.max_length_per_level[&DifficultyLevel::Easy] {
            DifficultyLevel::Easy
        } else if length <= self.max_length_per_level[&DifficultyLevel::Medium] {
            DifficultyLevel::Medium
        } else if length <= self.max_length_per_level[&DifficultyLevel::Hard] {
            DifficultyLevel::Hard
        } else {
            DifficultyLevel::VeryHard
        }
    }
    
    /// Add examples to the curriculum
    pub fn add_examples(&mut self, inputs: &[Vec<usize>], targets: &Array2<usize>) {
        for (idx, input) in inputs.iter().enumerate() {
            let target_row: Vec<usize> = targets.row(idx).iter().cloned().collect();
            
            // Calculate the difficulty
            let difficulty = self.calculate_difficulty(input, &target_row);
            
            // Create a curriculum example
            let example = CurriculumExample {
                index: idx,
                input: input.clone(),
                target: target_row.clone(),
                difficulty,
                length: input.len().max(target_row.len()),
            };
            
            // Add to the appropriate difficulty level
            self.examples_by_level
                .entry(difficulty)
                .or_insert_with(Vec::new)
                .push(example);
        }
    }
    
    /// Get the current maximum sequence length based on difficulty
    pub fn get_current_max_length(&self) -> usize {
        *self.max_length_per_level.get(&self.current_level).unwrap_or(&64)
    }
    
    /// Advance to the next epoch, potentially increasing difficulty
    pub fn next_epoch(&mut self) -> bool {
        self.current_epoch += 1;
        
        // Check if we should advance to the next difficulty level
        if self.current_epoch % self.epochs_per_level == 0 {
            let old_level = self.current_level;
            self.current_level = self.current_level.next();
            
            // Return true if the level changed
            old_level != self.current_level
        } else {
            false
        }
    }
    
    /// Get training examples for the current epoch, incorporating curriculum learning
    pub fn get_training_examples(&self) -> Vec<CurriculumExample> {
        let mut examples = Vec::new();
        
        // Helper function to get a subset of examples from a level
        let get_examples_from_level = |level: DifficultyLevel, count: usize| -> Vec<CurriculumExample> {
            if let Some(level_examples) = self.examples_by_level.get(&level) {
                if level_examples.is_empty() {
                    return Vec::new();
                }
                
                // Use a deterministic random selection based on seed and epoch
                let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + self.current_epoch as u64);
                let indices: Vec<usize> = (0..level_examples.len()).collect();
                let selected_indices = indices.choose_multiple(&mut rng, count.min(level_examples.len()));
                
                selected_indices.map(|&idx| level_examples[idx].clone()).collect()
            } else {
                Vec::new()
            }
        };
        
        // Get examples from the current level
        if let Some(current_examples) = self.examples_by_level.get(&self.current_level) {
            examples.extend(current_examples.iter().cloned());
        }
        
        // Mix in some harder examples if available
        if self.harder_examples_ratio > 0.0 {
            let next_level = self.current_level.next();
            if next_level != self.current_level {
                let harder_count = (examples.len() as f32 * self.harder_examples_ratio) as usize;
                let harder_examples = get_examples_from_level(next_level, harder_count);
                examples.extend(harder_examples);
            }
        }
        
        // Mix in some easier examples if available and not at the easiest level
        if self.easier_examples_ratio > 0.0 && self.current_level != DifficultyLevel::VeryEasy {
            let prev_level = match self.current_level {
                DifficultyLevel::Easy => DifficultyLevel::VeryEasy,
                DifficultyLevel::Medium => DifficultyLevel::Easy,
                DifficultyLevel::Hard => DifficultyLevel::Medium,
                DifficultyLevel::VeryHard => DifficultyLevel::Hard,
                _ => self.current_level,
            };
            
            let easier_count = (examples.len() as f32 * self.easier_examples_ratio) as usize;
            let easier_examples = get_examples_from_level(prev_level, easier_count);
            examples.extend(easier_examples);
        }
        
        // Shuffle the examples
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + self.current_epoch as u64);
        examples.shuffle(&mut rng);
        
        examples
    }
    
    /// Get the batch size appropriate for the current difficulty level
    pub fn get_batch_size(&self) -> usize {
        match self.current_level {
            DifficultyLevel::VeryEasy => 64,
            DifficultyLevel::Easy => 48,
            DifficultyLevel::Medium => 32,
            DifficultyLevel::Hard => 16,
            DifficultyLevel::VeryHard => 8,
        }
    }
    
    /// Get the current difficulty level
    pub fn get_current_level(&self) -> DifficultyLevel {
        self.current_level
    }
    
    /// Get the current epoch
    pub fn get_current_epoch(&self) -> usize {
        self.current_epoch
    }
}

impl Default for CurriculumScheduler {
    fn default() -> Self {
        Self::new()
    }
} 