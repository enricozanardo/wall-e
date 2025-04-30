use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::min;
use std::error::Error;

/// Structure containing dataset splits
pub struct DatasetSplit<T> {
    pub train: Vec<T>,
    pub validation: Vec<T>,
    pub test: Vec<T>,
}

/// Splits a dataset into training, validation and test sets
/// 
/// # Arguments
/// 
/// * `data` - The dataset to split
/// * `train_ratio` - The proportion of the dataset to use for training (0.0-1.0)
/// * `validation_ratio` - The proportion of the dataset to use for validation (0.0-1.0)
/// * `test_ratio` - The proportion of the dataset to use for testing (0.0-1.0)
/// * `shuffle` - Whether to shuffle the data before splitting
///
/// # Returns
/// 
/// * `DatasetSplit` - Structure containing the dataset splits
///
/// # Note
/// 
/// The proportions must sum to 1.0, otherwise an error is returned
pub fn split_dataset<T: Clone>(
    data: &[T],
    train_ratio: f32,
    validation_ratio: f32,
    test_ratio: f32,
    shuffle: bool,
) -> Result<DatasetSplit<T>, Box<dyn Error>> {
    // Verify that the proportions are valid
    if (train_ratio + validation_ratio + test_ratio - 1.0).abs() > 1e-6 {
        return Err("The proportions of train, validation and test must sum to 1.0".into());
    }
    
    if train_ratio < 0.0 || validation_ratio < 0.0 || test_ratio < 0.0 {
        return Err("The proportions cannot be negative".into());
    }
    
    let total_size = data.len();
    let mut indices: Vec<usize> = (0..total_size).collect();
    
    // Shuffle the indices if requested
    if shuffle {
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
    }
    
    // Calculate the size of each set
    let train_size = (total_size as f32 * train_ratio).round() as usize;
    let validation_size = (total_size as f32 * validation_ratio).round() as usize;
    let test_size = min(total_size - train_size - validation_size, (total_size as f32 * test_ratio).round() as usize);
    
    // Extract the elements based on the indices
    let mut train = Vec::with_capacity(train_size);
    let mut validation = Vec::with_capacity(validation_size);
    let mut test = Vec::with_capacity(test_size);
    
    for i in 0..train_size {
        train.push(data[indices[i]].clone());
    }
    
    for i in train_size..(train_size + validation_size) {
        validation.push(data[indices[i]].clone());
    }
    
    for i in (train_size + validation_size)..(train_size + validation_size + test_size) {
        test.push(data[indices[i]].clone());
    }
    
    Ok(DatasetSplit {
        train,
        validation,
        test,
    })
}

/// Splits a dataset into k subsets for cross-validation
/// 
/// # Arguments
/// 
/// * `data` - The dataset to split
/// * `k` - The number of folds for cross-validation
/// * `shuffle` - Whether to shuffle the data before splitting
///
/// # Returns
/// 
/// * Vec<(Vec<T>, Vec<T>)> - A vector of tuples (training_set, validation_set) for each fold
pub fn k_fold_split<T: Clone>(
    data: &[T],
    k: usize,
    shuffle: bool,
) -> Vec<(Vec<T>, Vec<T>)> {
    if k <= 1 {
        panic!("The number of folds must be greater than 1");
    }
    
    let total_size = data.len();
    let mut indices: Vec<usize> = (0..total_size).collect();
    
    // Shuffle the indices if requested
    if shuffle {
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
    }
    
    // Calculate the size of each fold
    let fold_size = total_size / k;
    
    let mut folds = Vec::with_capacity(k);
    
    for fold_idx in 0..k {
        let val_start = fold_idx * fold_size;
        let val_end = if fold_idx == k - 1 {
            total_size
        } else {
            (fold_idx + 1) * fold_size
        };
        
        let mut train_data = Vec::with_capacity(total_size - (val_end - val_start));
        let mut val_data = Vec::with_capacity(val_end - val_start);
        
        for i in 0..total_size {
            if i >= val_start && i < val_end {
                val_data.push(data[indices[i]].clone());
            } else {
                train_data.push(data[indices[i]].clone());
            }
        }
        
        folds.push((train_data, val_data));
    }
    
    folds
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_split_dataset() {
        let data: Vec<i32> = (0..100).collect();
        
        let split = split_dataset(&data, 0.7, 0.15, 0.15, false).unwrap();
        
        assert_eq!(split.train.len(), 70);
        assert_eq!(split.validation.len(), 15);
        assert_eq!(split.test.len(), 15);
        
        // Verify that the data has been split correctly
        for i in 0..70 {
            assert_eq!(split.train[i], i as i32);
        }
        
        for i in 0..15 {
            assert_eq!(split.validation[i], (i + 70) as i32);
        }
        
        for i in 0..15 {
            assert_eq!(split.test[i], (i + 85) as i32);
        }
    }
    
    #[test]
    fn test_k_fold_split() {
        let data: Vec<i32> = (0..100).collect();
        
        let folds = k_fold_split(&data, 5, false);
        
        assert_eq!(folds.len(), 5);
        
        for (fold_idx, (train, val)) in folds.iter().enumerate() {
            assert_eq!(train.len(), 80);
            assert_eq!(val.len(), if fold_idx == 4 { 20 } else { 20 });
        }
    }
    
    #[test]
    #[should_panic]
    fn test_k_fold_invalid_k() {
        let data: Vec<i32> = (0..100).collect();
        let _ = k_fold_split(&data, 0, false);
    }
    
    #[test]
    fn test_split_dataset_with_invalid_ratios() {
        let data: Vec<i32> = (0..100).collect();
        
        let result = split_dataset(&data, 0.8, 0.8, 0.2, false);
        assert!(result.is_err());
        
        let result = split_dataset(&data, -0.1, 0.5, 0.6, false);
        assert!(result.is_err());
    }
} 