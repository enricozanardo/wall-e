use std::collections::HashMap;
use ndarray::Array2;
use rand::prelude::*;

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

/// A curriculum example with difficulty information
#[derive(Clone, Debug)]
pub struct CurriculumExample {
    /// Index in the original batch
    pub index: usize,
    /// Input sequence
    pub input: Vec<usize>,
    /// Target sequence
    pub target: Vec<usize>,
    /// Assessed difficulty level
    pub difficulty: DifficultyLevel,
    /// Length of the sequence
    pub length: usize,
    /// Loss value for this example
    pub loss: f32,
}


/// Curriculum learning scheduler
#[allow(dead_code)]
pub struct CurriculumScheduler {
    /// Current difficulty level
    current_level: DifficultyLevel,
    /// Examples organized by difficulty level
    examples_by_level: HashMap<DifficultyLevel, Vec<CurriculumExample>>,
    /// Current training epoch
    current_epoch: usize,
    /// Number of epochs to spend at each level
    epochs_per_level: usize,
    /// Epochs spent in the current level
    epochs_in_current_level: usize,
    /// Whether to auto-advance to the next level
    auto_advance: bool,
    /// Whether we've advanced beyond the hardest standard level
    beyond_standard_levels: bool,
    /// Batch size for training
    batch_size: usize,
    /// Whether to use a fixed batch size
    fixed_batch_size: bool,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Counter for tracking updates
    update_count: usize,
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
            auto_advance: false,
            beyond_standard_levels: false,
            max_seq_len: 0,
            update_count: 0,
            fixed_batch_size: false,
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
    
    /// Assess the difficulty of a single example based on its loss value
    fn assess_difficulty(&self, loss: f32) -> DifficultyLevel {
        // Make difficulty criteria more lenient to ensure examples get categorized
        if loss < 2.0 {
            DifficultyLevel::VeryEasy
        } else if loss < 3.5 {
            DifficultyLevel::Easy
        } else if loss < 5.0 {
            DifficultyLevel::Medium
        } else if loss < 6.5 {
            DifficultyLevel::Hard
        } else {
            DifficultyLevel::VeryHard
        }
    }
    
    /// Redistribute examples across difficulty levels
    pub fn redistribute_examples(&mut self) {
        // Create a copy of all examples
        let mut all_examples = Vec::new();
        for examples in self.examples_by_level.values() {
            all_examples.extend(examples.clone());
        }
        
        // Sort examples by loss (ascending)
        all_examples.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal));
        
        // Clear existing categories
        self.examples_by_level.clear();
        
        // Ensure we have at least some examples in each level
        if !all_examples.is_empty() {
            let levels = [
                DifficultyLevel::VeryEasy,
                DifficultyLevel::Easy,
                DifficultyLevel::Medium,
                DifficultyLevel::Hard,
                DifficultyLevel::VeryHard,
            ];
            
            // Always guarantee some examples in each level
            let total = all_examples.len();
            let examples_per_level = std::cmp::max(10, total / 5); // At least 10 examples per level
            
            // Collect previous level examples for possible reuse
            let mut prev_level_examples: Option<Vec<CurriculumExample>> = None;
            
            // Distribute examples evenly across levels
            for (i, level) in levels.iter().enumerate() {
                let start = i * examples_per_level;
                let end = if i == levels.len() - 1 { total } else { (i + 1) * examples_per_level };
                
                if start < total {
                    let end = std::cmp::min(end, total);
                    let examples_for_level: Vec<CurriculumExample> = all_examples[start..end].to_vec();
                    
                    self.examples_by_level.insert(*level, examples_for_level.clone());
                    prev_level_examples = Some(examples_for_level);
                    
                    println!("Redistributed {} examples to level {:?}", end - start, level);
                } else {
                    // If we run out of examples, copy some from the previous level
                    if let Some(examples_to_copy) = prev_level_examples.clone() {
                        if !examples_to_copy.is_empty() {
                            self.examples_by_level.insert(*level, examples_to_copy.clone());
                            println!("Copied {} examples to level {:?} from previous level", examples_to_copy.len(), level);
                        }
                    }
                }
            }
        }
        
        // Print example counts per level for debugging
        for level in [DifficultyLevel::VeryEasy, DifficultyLevel::Easy, 
                     DifficultyLevel::Medium, DifficultyLevel::Hard, 
                     DifficultyLevel::VeryHard].iter() {
            let count = self.examples_by_level.get(level).map_or(0, |v| v.len());
            println!("After redistribution: {} examples at level {:?}", count, level);
        }
    }
    
    /// Add examples to the curriculum using old add_examples method
    pub fn add_examples(&mut self, inputs: &[Vec<usize>], targets: &Array2<usize>) {
        let mut level_counts = HashMap::new();
        
        for (idx, input) in inputs.iter().enumerate() {
            let target_row: Vec<usize> = targets.row(idx).iter().cloned().collect();
            
            // Calculate the difficulty
            let difficulty = self.calculate_difficulty(input, &target_row);
            
            // Estimate a loss value based on difficulty
            let estimated_loss = match difficulty {
                DifficultyLevel::VeryEasy => 1.5,
                DifficultyLevel::Easy => 3.0,
                DifficultyLevel::Medium => 4.5,
                DifficultyLevel::Hard => 6.0,
                DifficultyLevel::VeryHard => 7.5,
            };
            
            // Update counts for debugging
            *level_counts.entry(difficulty).or_insert(0) += 1;
            
            // Create a curriculum example
            let example = CurriculumExample {
                index: idx,
                input: input.clone(),
                target: target_row.clone(),
                difficulty,
                length: input.len().max(target_row.len()),
                loss: estimated_loss,
            };
            
            // Add to the appropriate difficulty level
            self.examples_by_level
                .entry(difficulty)
                .or_insert_with(|| Vec::new())
                .push(example);
        }
        
        // Force redistribute examples to ensure we have training material for all levels
        self.update_count += 1;
        if self.update_count % 5 == 0 {
            println!("Performing example redistribution across difficulty levels...");
            self.redistribute_examples();
        } else {
            // Print debug info about level distribution
            println!("Curriculum example distribution:");
            for level in [DifficultyLevel::VeryEasy, DifficultyLevel::Easy, 
                        DifficultyLevel::Medium, DifficultyLevel::Hard, DifficultyLevel::VeryHard] {
                let count = self.examples_by_level.get(&level).map_or(0, |v| v.len());
                println!("  Level {:?}: {} examples", level, count);
            }
        }
    }
    
    /// Add examples to the curriculum with the new approach using loss values
    pub fn update_examples(&mut self, examples: &Vec<CurriculumExample>) {
        // If no examples are provided, don't update
        if examples.is_empty() {
            return;
        }
        
        // Categorize by difficulty
        for example in examples {
            let level = self.assess_difficulty(example.loss);
            
            // Add to level
            self.examples_by_level
                .entry(level)
                .or_insert_with(|| Vec::new())
                .push(example.clone());
        }
        
        // Increment update counter
        self.update_count += 1;
        
        // Log the current counts
        for level in [DifficultyLevel::VeryEasy, DifficultyLevel::Easy, 
                    DifficultyLevel::Medium, DifficultyLevel::Hard, 
                    DifficultyLevel::VeryHard].iter() {
            let count = self.examples_by_level.get(level).map_or(0, |v| v.len());
            println!("Before redistribution: {} examples at level {:?}", count, level);
        }
        
        // Every N updates, redistribute examples to maintain balance
        if self.update_count % 5 == 0 {
            println!("Performing example redistribution across difficulty levels...");
            self.redistribute_examples();
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