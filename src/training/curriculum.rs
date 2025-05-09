use std::collections::HashMap;
use ndarray::Array2;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Difficulty level for curriculum learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Epochs spent in current level
    epochs_in_current_level: usize,
    /// Default batch size
    batch_size: usize,
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
        max_length_per_level.insert(DifficultyLevel::Easy, 24);
        max_length_per_level.insert(DifficultyLevel::Medium, 32);
        max_length_per_level.insert(DifficultyLevel::Hard, 48);
        max_length_per_level.insert(DifficultyLevel::VeryHard, 64);
        
        Self {
            current_level: DifficultyLevel::VeryEasy,
            examples_by_level: HashMap::new(),
            current_epoch: 0,
            epochs_per_level: 3,  // Increased from default of 2 to allow more time per level
            epochs_in_current_level: 0,
            batch_size: 32,
            max_length_per_level,
            harder_examples_ratio: 0.2,
            easier_examples_ratio: 0.1,
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
        
        // More nuanced difficulty assessment
        // 1. Consider sequence length as primary factor
        // 2. Consider amount of rare tokens as secondary factor
        
        // Count rare tokens (tokens with high IDs are typically rarer)
        let rare_token_threshold = 1000; // Tokens with IDs above this are considered rare
        let rare_tokens = input.iter().chain(target.iter())
            .filter(|&&token| token > rare_token_threshold)
            .count();
        
        // Adjust difficulty based on both length and rare token count
        if length <= self.max_length_per_level[&DifficultyLevel::VeryEasy] && rare_tokens < 2 {
            DifficultyLevel::VeryEasy
        } else if length <= self.max_length_per_level[&DifficultyLevel::Easy] && rare_tokens < 5 {
            DifficultyLevel::Easy
        } else if length <= self.max_length_per_level[&DifficultyLevel::Medium] && rare_tokens < 10 {
            DifficultyLevel::Medium
        } else if length <= self.max_length_per_level[&DifficultyLevel::Hard] && rare_tokens < 20 {
            DifficultyLevel::Hard
        } else {
            DifficultyLevel::VeryHard
        }
    }
    
    /// Add examples to the curriculum
    pub fn add_examples(&mut self, inputs: &[Vec<usize>], targets: &Array2<usize>) {
        let mut level_counts = HashMap::new();
        
        for (idx, input) in inputs.iter().enumerate() {
            let target_row: Vec<usize> = targets.row(idx).iter().cloned().collect();
            
            // Calculate the difficulty
            let difficulty = self.calculate_difficulty(input, &target_row);
            
            // Update counts for debugging
            *level_counts.entry(difficulty).or_insert(0) += 1;
            
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
        
        // Force distribute some examples to ensure we have training material for all levels
        self.ensure_examples_at_all_levels(inputs, targets);
        
        // Print debug info about level distribution
        println!("Curriculum example distribution:");
        for level in [DifficultyLevel::VeryEasy, DifficultyLevel::Easy, 
                     DifficultyLevel::Medium, DifficultyLevel::Hard, DifficultyLevel::VeryHard] {
            let count = self.examples_by_level.get(&level).map_or(0, |v| v.len());
            println!("  Level {:?}: {} examples", level, count);
        }
    }
    
    /// Ensure we have examples at all difficulty levels by artificially distributing
    /// some examples across levels where we have none
    fn ensure_examples_at_all_levels(&mut self, inputs: &[Vec<usize>], targets: &Array2<usize>) {
        // Check which levels need examples
        let mut levels_needing_examples = Vec::new();
        
        for level in [DifficultyLevel::VeryEasy, DifficultyLevel::Easy, 
                     DifficultyLevel::Medium, DifficultyLevel::Hard, DifficultyLevel::VeryHard] {
            if self.examples_by_level.get(&level).map_or(0, |v| v.len()) < 100 {
                levels_needing_examples.push(level);
            }
        }
        
        if levels_needing_examples.is_empty() {
            return; // All levels have enough examples
        }
        
        // Find the level with the most examples to redistribute from
        let mut max_level = DifficultyLevel::VeryEasy;
        let mut max_count = 0;
        
        for level in [DifficultyLevel::VeryEasy, DifficultyLevel::Easy, 
                     DifficultyLevel::Medium, DifficultyLevel::Hard, DifficultyLevel::VeryHard] {
            let count = self.examples_by_level.get(&level).map_or(0, |v| v.len());
            if count > max_count {
                max_count = count;
                max_level = level;
            }
        }
        
        // Only redistribute if we have enough examples to share
        if max_count < 300 {
            println!("Warning: Not enough examples to redistribute (max count: {})", max_count);
            return;
        }
        
        // Redistribute examples to ensure all levels have at least some training material
        let to_redistribute = (max_count / 5).min(200); // Take at most 200 examples
        
        if let Some(source_examples) = self.examples_by_level.get(&max_level) {
            if source_examples.is_empty() {
                return;
            }
            
            let mut rng = thread_rng();
            let mut indices: Vec<usize> = (0..source_examples.len()).collect();
            indices.shuffle(&mut rng);
            
            // Take only a subset of the indices to redistribute
            let indices_to_use = indices.into_iter().take(to_redistribute).collect::<Vec<_>>();
            
            // Clone examples we'll redistribute
            let examples_to_redistribute: Vec<CurriculumExample> = indices_to_use.iter()
                .map(|&idx| source_examples[idx].clone())
                .collect();
            
            // Distribute examples evenly among levels that need them
            let per_level = to_redistribute / levels_needing_examples.len();
            
            for (i, &level) in levels_needing_examples.iter().enumerate() {
                let start = i * per_level;
                let end = if i == levels_needing_examples.len() - 1 {
                    examples_to_redistribute.len()
                } else {
                    (i + 1) * per_level
                };
                
                if start < examples_to_redistribute.len() {
                    // Create modified examples for this level
                    let mut modified_examples = Vec::new();
                    
                    for example in &examples_to_redistribute[start..end.min(examples_to_redistribute.len())] {
                        // Create a new example with the target difficulty level
                        let mut modified = example.clone();
                        modified.difficulty = level;
                        modified_examples.push(modified);
                    }
                    
                    // Add to level
                    self.examples_by_level
                        .entry(level)
                        .or_insert_with(|| Vec::new())
                        .extend(modified_examples);
                }
            }
            
            println!("Redistributed {} examples from level {:?} to ensure training material for all levels", 
                     to_redistribute, max_level);
        }
    }
    
    /// Get the current maximum sequence length based on difficulty
    pub fn get_current_max_length(&self) -> usize {
        match self.current_level {
            DifficultyLevel::VeryEasy => 16,  // Shorter sequences for easier learning
            DifficultyLevel::Easy => 24,
            DifficultyLevel::Medium => 32,
            DifficultyLevel::Hard => 48,
            DifficultyLevel::VeryHard => 64,
        }
    }
    
    /// Advance to the next epoch and check if it's time to move to the next level
    pub fn next_epoch(&mut self) -> bool {
        self.current_epoch += 1;
        self.epochs_in_current_level += 1;
        
        // Check if it's time to advance to the next level
        if self.epochs_in_current_level >= self.epochs_per_level {
            // Only advance if we're not already at the maximum level
            if self.current_level != DifficultyLevel::VeryHard {
                let next_level = match self.current_level {
                    DifficultyLevel::VeryEasy => DifficultyLevel::Easy,
                    DifficultyLevel::Easy => DifficultyLevel::Medium,
                    DifficultyLevel::Medium => DifficultyLevel::Hard,
                    DifficultyLevel::Hard => DifficultyLevel::VeryHard,
                    DifficultyLevel::VeryHard => DifficultyLevel::VeryHard, // Can't go higher
                };
                self.current_level = next_level;
                self.epochs_in_current_level = 0;
                return true;
            } else {
                // At max level, just reset the epochs counter but stay at the same level
                self.epochs_in_current_level = 0;
            }
        }
        
        false
    }
    
    /// Get training examples for the current difficulty level
    /// Include a mix of examples from lower difficulty levels to prevent overfitting 
    /// and maintain knowledge of easier tasks
    pub fn get_training_examples(&self) -> Vec<CurriculumExample> {
        // First, get examples for the current level
        let current_level_examples = match self.examples_by_level.get(&self.current_level) {
            Some(examples) => examples,
            None => return Vec::new(), // No examples for this level
        };
        
        // Start with current level examples
        let mut selected_examples = current_level_examples.clone();

        // Add a mix of easier examples for knowledge reinforcement (unless at the easiest level)
        if self.current_level != DifficultyLevel::VeryEasy {
            // For each level below the current, add some examples
            for level in 0..self.current_level as usize {
                let level_enum = match level {
                    0 => DifficultyLevel::VeryEasy,
                    1 => DifficultyLevel::Easy,
                    2 => DifficultyLevel::Medium,
                    3 => DifficultyLevel::Hard,
                    _ => DifficultyLevel::VeryEasy,
                };
                
                // Get examples from this easier level
                if let Some(examples) = self.examples_by_level.get(&level_enum) {
                    if !examples.is_empty() {
                        // Determine how many examples to take based on the gap between levels
                        // The closer the level is to current, the more examples we take
                        let level_distance = (self.current_level as usize) - level;
                        let ratio = self.easier_examples_ratio / level_distance as f32;
                        let count = (examples.len() as f32 * ratio) as usize;
                        
                        // Select random examples from this level
                        let mut rng = thread_rng();
                        let mut indices = (0..examples.len()).collect::<Vec<_>>();
                        indices.shuffle(&mut rng);
                        
                        for &idx in indices.iter().take(count) {
                            selected_examples.push(examples[idx].clone());
                        }
                    }
                }
            }
        }
        
        // If not at the hardest level, add a smaller proportion of harder examples
        if self.current_level != DifficultyLevel::VeryHard {
            // Add examples from one level above current
            let next_level = match self.current_level {
                DifficultyLevel::VeryEasy => DifficultyLevel::Easy,
                DifficultyLevel::Easy => DifficultyLevel::Medium,
                DifficultyLevel::Medium => DifficultyLevel::Hard,
                DifficultyLevel::Hard => DifficultyLevel::VeryHard,
                DifficultyLevel::VeryHard => DifficultyLevel::VeryHard, // Shouldn't happen
            };
            
            if let Some(examples) = self.examples_by_level.get(&next_level) {
                if !examples.is_empty() {
                    // Take a smaller ratio of harder examples
                    let reduced_ratio = self.harder_examples_ratio * 0.5; // Half the normal ratio
                    let count = (examples.len() as f32 * reduced_ratio) as usize;
                    
                    // Select random examples from the harder level
                    let mut rng = thread_rng();
                    let mut indices = (0..examples.len()).collect::<Vec<_>>();
                    indices.shuffle(&mut rng);
                    
                    for &idx in indices.iter().take(count) {
                        selected_examples.push(examples[idx].clone());
                    }
                }
            }
        }
        
        selected_examples
    }
    
    /// Get the batch size based on current difficulty level
    pub fn get_batch_size(&self) -> usize {
        // For more difficult examples, use smaller batches
        match self.current_level {
            DifficultyLevel::VeryEasy => self.batch_size,
            DifficultyLevel::Easy => self.batch_size,
            DifficultyLevel::Medium => self.batch_size / 2 + self.batch_size / 4, // 75% of original
            DifficultyLevel::Hard => self.batch_size / 2,  // Half the base batch size
            DifficultyLevel::VeryHard => self.batch_size / 4, // Quarter the base batch size
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