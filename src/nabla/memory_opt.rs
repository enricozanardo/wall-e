use ndarray::{Array, Array2, ArrayD, Ix2, Ix3};
use rayon::prelude::*;
use std::cmp;
use std::time::{Instant, Duration};
use std::collections::HashMap;
use super::tensor::Tensor;

/// Contains the determined hardware parameters
pub struct CacheParameters {
    /// L1 cache size in bytes
    pub l1_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// CPU cache line size in bytes
    pub line_size: usize,
}

impl Default for CacheParameters {
    fn default() -> Self {
        // Default conservative cache parameters
        // These will work reasonably well on most modern CPUs
        CacheParameters {
            l1_size: 32 * 1024,      // 32 KB
            l2_size: 256 * 1024,     // 256 KB
            l3_size: 8 * 1024 * 1024, // 8 MB
            line_size: 64,           // 64 bytes (typical cache line)
        }
    }
}

/// Detect cache parameters from the system if possible
pub fn detect_cache_parameters() -> CacheParameters {
    // Try to read from sysfs on Linux
    #[cfg(target_os = "linux")]
    {
        let l1_size = read_sysfs_value("/sys/devices/system/cpu/cpu0/cache/index0/size")
            .unwrap_or(32 * 1024);
        let l2_size = read_sysfs_value("/sys/devices/system/cpu/cpu0/cache/index2/size")
            .unwrap_or(256 * 1024);
        let l3_size = read_sysfs_value("/sys/devices/system/cpu/cpu0/cache/index3/size")
            .unwrap_or(8 * 1024 * 1024);
        let line_size = read_sysfs_value("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
            .unwrap_or(64);
        
        return CacheParameters {
            l1_size,
            l2_size,
            l3_size,
            line_size,
        };
    }
    
    // Default for other platforms
    #[cfg(not(target_os = "linux"))]
    CacheParameters::default()
}

/// Helper function to read cache information from sysfs on Linux
#[cfg(target_os = "linux")]
fn read_sysfs_value(path: &str) -> Option<usize> {
    use std::fs::File;
    use std::io::Read;
    
    let mut file = File::open(path).ok()?;
    let mut contents = String::new();
    file.read_to_string(&mut contents).ok()?;
    
    // Parse value, handling K and M suffixes
    let contents = contents.trim();
    if contents.ends_with('K') {
        contents[..contents.len()-1].parse::<usize>().ok().map(|v| v * 1024)
    } else if contents.ends_with('M') {
        contents[..contents.len()-1].parse::<usize>().ok().map(|v| v * 1024 * 1024)
    } else {
        contents.parse::<usize>().ok()
    }
}

/// Calculate optimal block sizes for cache-blocked matrix multiplication
pub fn calculate_block_sizes(cache_params: &CacheParameters, element_size: usize) -> (usize, usize, usize) {
    // Calculate block sizes to fit in L1, L2, and L3 caches
    
    // For cache blocking, we want to ensure blocks fit in L1 cache
    // A block of size B×B requires B²*element_size bytes
    // We leave some room for other data by using 80% of cache
    let l1_capacity = (cache_params.l1_size as f64 * 0.8) as usize / element_size;
    let l1_block = (l1_capacity as f64).sqrt() as usize;
    
    // Similar calculations for L2 and L3
    let l2_capacity = (cache_params.l2_size as f64 * 0.8) as usize / element_size;
    let l2_block = (l2_capacity as f64).sqrt() as usize;
    
    let l3_capacity = (cache_params.l3_size as f64 * 0.8) as usize / element_size;
    let l3_block = (l3_capacity as f64).sqrt() as usize;
    
    // Ensure block sizes are multiples of cache line size / element_size
    // for better alignment
    let line_elements = cache_params.line_size / element_size;
    let l1_block = (l1_block / line_elements) * line_elements;
    let l2_block = (l2_block / line_elements) * line_elements;
    let l3_block = (l3_block / line_elements) * line_elements;
    
    // Ensure a minimum size even for small caches
    let l1_block = cmp::max(l1_block, 16);
    
    (l1_block, l2_block, l3_block)
}

/// Calculate the optimal batch size for model dimensions and hardware
pub fn calculate_optimal_batch_size(cache_params: &CacheParameters, model_dim: usize, seq_len: usize, element_size: usize) -> usize {
    // Calculate memory requirements per sequence
    let seq_memory = model_dim * seq_len * element_size;
    println!("Memory per sequence: {} bytes ({:.2} MB)", 
            seq_memory, seq_memory as f64 / (1024.0 * 1024.0));
    
    // Calculate how many sequences fit in L3 cache, using 80% of the cache
    let cache_usage_percent = 0.8;
    let available_cache = (cache_params.l3_size as f64 * cache_usage_percent) as usize;
    println!("Available L3 cache ({}%): {} bytes ({:.2} MB)", 
            cache_usage_percent * 100.0, available_cache, available_cache as f64 / (1024.0 * 1024.0));
    
    // Calculate maximum batch size that fits in L3 cache with 80% utilization
    let max_batch_size = available_cache / seq_memory;
    println!("Maximum batch size that fits in L3 cache: {}", max_batch_size);
    
    // Apply practical constraints
    let min_batch_size = 16; // Ensure there's enough parallelism
    let max_practical_batch = 1024; // Avoid excessive memory use
    
    let optimal_batch_size = max_batch_size.clamp(min_batch_size, max_practical_batch);
    println!("Final optimal batch size (after constraints): {}", optimal_batch_size);
    
    optimal_batch_size
}

/// Perform cache-blocked matrix multiplication on 2D arrays
pub fn cache_blocked_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    assert_eq!(a.shape()[1], b.shape()[0], "Incompatible matrix dimensions for multiplication");
    
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    
    let cache_params = detect_cache_parameters();
    let (block_size, _, _) = calculate_block_sizes(&cache_params, std::mem::size_of::<f32>());
    
    // Initialize result matrix with zeros
    let mut result = Array2::<f32>::zeros((m, n));
    
    // Cache-blocked matrix multiplication
    for i_block in (0..m).step_by(block_size) {
        let i_end = cmp::min(i_block + block_size, m);
        
        for j_block in (0..n).step_by(block_size) {
            let j_end = cmp::min(j_block + block_size, n);
            
            for k_block in (0..k).step_by(block_size) {
                let k_end = cmp::min(k_block + block_size, k);
                
                // Process the current block
                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let mut sum = 0.0;
                        
                        // Inner loop processes elements within the block
                        // This exploits L1 cache locality
                        for kk in k_block..k_end {
                            sum += a[[i, kk]] * b[[kk, j]];
                        }
                        
                        result[[i, j]] += sum;
                    }
                }
            }
        }
    }
    
    result
}

/// Perform cache-efficient matrix multiplication on 3D tensors
/// [batch, seq_len, features] × [features, output_dim] -> [batch, seq_len, output_dim]
pub fn batch_matmul_3d_2d(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    // Convert dimensions
    let a_3d = a.clone().into_dimensionality::<Ix3>().unwrap();
    let b_2d = b.clone().into_dimensionality::<Ix2>().unwrap();
    
    let batch_size = a_3d.shape()[0];
    let seq_len = a_3d.shape()[1];
    let features = a_3d.shape()[2];
    let output_dim = b_2d.shape()[1];
    
    assert_eq!(features, b_2d.shape()[0], "Incompatible matrix dimensions for multiplication");
    
    // Process in parallel batches by creating a vector of partial results
    let results: Vec<_> = (0..batch_size).into_par_iter().map(|batch| {
        // Create a result array for this batch
        let mut batch_result = Array::zeros((seq_len, output_dim));
        
        // Process sequence items
        for seq in 0..seq_len {
            // Extract the vector for this position
            let vec = a_3d.slice(ndarray::s![batch, seq, ..]);
            
            // Perform efficient matrix-vector multiplication
            for out in 0..output_dim {
                let mut sum = 0.0;
                
                // Manually unrolled inner loop in chunks for better cache utilization
                const CHUNK_SIZE: usize = 16; // Adjust based on cache line size
                let feature_chunks = features / CHUNK_SIZE;
                let remainder = features % CHUNK_SIZE;
                
                // Process chunks of 16 features at a time
                for chunk in 0..feature_chunks {
                    let offset = chunk * CHUNK_SIZE;
                    
                    // Accumulate dot product in chunks
                    let mut chunk_sum = 0.0;
                    for i in 0..CHUNK_SIZE {
                        chunk_sum += vec[offset + i] * b_2d[[offset + i, out]];
                    }
                    
                    sum += chunk_sum;
                }
                
                // Handle remaining elements
                for i in 0..remainder {
                    sum += vec[feature_chunks * CHUNK_SIZE + i] * b_2d[[feature_chunks * CHUNK_SIZE + i, out]];
                }
                
                // Set result for this batch
                batch_result[[seq, out]] = sum;
            }
        }
        
        batch_result
    }).collect();
    
    // Combine the results into a single 3D array
    let mut result = Array::zeros((batch_size, seq_len, output_dim));
    for (batch, batch_result) in results.iter().enumerate() {
        for seq in 0..seq_len {
            for out in 0..output_dim {
                result[[batch, seq, out]] = batch_result[[seq, out]];
            }
        }
    }
    
    result.into_dyn()
}

/// Prefetch data into cache
#[cfg(target_arch = "x86_64")]
pub fn prefetch<T>(data: &[T], offset: usize) {
    use std::arch::x86_64::_mm_prefetch;
    use std::arch::x86_64::_MM_HINT_T0;
    
    unsafe {
        if offset < data.len() {
            let ptr = data.as_ptr().add(offset) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }
}

/// No-op implementation for non-x86 platforms
#[cfg(not(target_arch = "x86_64"))]
pub fn prefetch<T>(_data: &[T], _offset: usize) {
    // No-op
}

/// Strategy for selecting which layers to checkpoint during backpropagation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointStrategy {
    /// Only checkpoint the boundaries (input and output of layer blocks)
    Boundary,
    /// Checkpoint layers at uniform intervals
    Uniform,
    /// Adaptively select checkpoints based on memory usage
    Adaptive,
    /// No checkpointing (store all intermediate activations)
    None,
}

/// Memory tracker for gradient checkpointing
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Memory usage snapshots at different points
    pub snapshots: HashMap<String, usize>,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            snapshots: HashMap::new(),
        }
    }

    /// Track memory allocation
    pub fn allocate(&mut self, bytes: usize) {
        self.current_usage += bytes;
        self.peak_usage = self.peak_usage.max(self.current_usage);
    }

    /// Track memory deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
    }

    /// Take a memory snapshot with a label
    pub fn snapshot(&mut self, label: &str) {
        self.snapshots.insert(label.to_string(), self.current_usage);
    }

    /// Get current memory usage in MB
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage as f64 / (1024.0 * 1024.0)
    }
}

/// Gradient checkpointing for memory-efficient backpropagation
pub struct GradientCheckpointer {
    /// The chosen checkpointing strategy
    strategy: CheckpointStrategy,
    /// Number of model layers
    num_layers: usize,
    /// Memory tracker
    memory_tracker: MemoryTracker,
    /// Checkpoint interval for uniform strategy
    checkpoint_interval: usize,
    /// Checkpointed tensors by layer index
    checkpoints: HashMap<usize, Vec<Tensor>>,
    /// Flag to track whether forward pass is active
    in_forward_pass: bool,
}

impl GradientCheckpointer {
    /// Create a new gradient checkpointer with the specified strategy
    pub fn new(strategy: CheckpointStrategy, num_layers: usize) -> Self {
        // Calculate a reasonable default checkpoint interval based on layers
        let default_interval = match num_layers {
            0..=4 => 1,    // For very small models, checkpoint everything
            5..=10 => 2,   // For small models
            11..=20 => 3,  // For medium models
            21..=40 => 4,  // For large models
            _ => 5,        // For very large models
        };

        Self {
            strategy,
            num_layers,
            memory_tracker: MemoryTracker::new(),
            checkpoint_interval: default_interval,
            checkpoints: HashMap::new(),
            in_forward_pass: false,
        }
    }

    /// Begin the forward pass
    pub fn begin_forward(&mut self) {
        self.in_forward_pass = true;
        self.checkpoints.clear();
        self.memory_tracker = MemoryTracker::new();
    }

    /// End the forward pass
    pub fn end_forward(&mut self) {
        self.in_forward_pass = false;
        self.memory_tracker.snapshot("end_forward");
    }

    /// Determine if a layer's activations should be checkpointed
    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        if !self.in_forward_pass {
            return false;
        }

        match self.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::Boundary => {
                // Only checkpoint the first and last layers
                layer_idx == 0 || layer_idx == self.num_layers - 1
            },
            CheckpointStrategy::Uniform => {
                // Checkpoint at regular intervals
                layer_idx % self.checkpoint_interval == 0 || layer_idx == self.num_layers - 1
            },
            CheckpointStrategy::Adaptive => {
                // Adaptive checkpointing based on available memory and layer characteristics
                // This is a simplified version - a real implementation would consider tensor sizes
                if layer_idx == 0 || layer_idx == self.num_layers - 1 {
                    return true;
                }
                
                // Check if memory usage is high
                let current_mb = self.memory_tracker.current_usage_mb();
                let threshold_mb = 1000.0; // 1GB threshold
                
                if current_mb > threshold_mb {
                    // If memory usage is high, checkpoint more aggressively
                    layer_idx % 2 == 0
                } else {
                    // Otherwise use a more relaxed interval
                    layer_idx % 3 == 0
                }
            }
        }
    }

    /// Store a checkpoint for a layer
    pub fn store_checkpoint(&mut self, layer_idx: usize, tensors: Vec<Tensor>) {
        if !self.in_forward_pass {
            return;
        }

        if self.should_checkpoint(layer_idx) {
            // Calculate size of tensors for memory tracking
            let size_bytes = tensors.iter()
                .map(|t| t.data.len() * std::mem::size_of::<f32>())
                .sum();
            
            self.memory_tracker.allocate(size_bytes);
            self.checkpoints.insert(layer_idx, tensors);
            
            // Take a snapshot at this checkpoint
            self.memory_tracker.snapshot(&format!("checkpoint_layer_{}", layer_idx));
        }
    }

    /// Retrieve a checkpoint for a layer
    pub fn get_checkpoint(&self, layer_idx: usize) -> Option<&Vec<Tensor>> {
        self.checkpoints.get(&layer_idx)
    }

    /// Check if a layer has a stored checkpoint
    pub fn has_checkpoint(&self, layer_idx: usize) -> bool {
        self.checkpoints.contains_key(&layer_idx)
    }

    /// Recompute activations for a layer that wasn't checkpointed
    pub fn recompute_activations(&self, layer_idx: usize, 
                                input_tensors: Vec<Tensor>, 
                                compute_fn: &dyn Fn(Vec<Tensor>) -> Vec<Tensor>) -> Vec<Tensor> {
        // If we have a checkpoint, return it
        if let Some(checkpoint) = self.get_checkpoint(layer_idx) {
            return checkpoint.clone();
        }
        
        // Otherwise, recompute using the provided function
        compute_fn(input_tensors)
    }

    /// Set the checkpoint interval for uniform strategy
    pub fn set_checkpoint_interval(&mut self, interval: usize) {
        self.checkpoint_interval = interval.max(1); // Ensure at least 1
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> (f64, f64) {
        (
            self.memory_tracker.current_usage_mb(),
            self.memory_tracker.peak_usage_mb()
        )
    }

    /// Free a specific checkpoint to reclaim memory
    pub fn free_checkpoint(&mut self, layer_idx: usize) {
        if let Some(tensors) = self.checkpoints.remove(&layer_idx) {
            // Calculate size of tensors for memory tracking
            let size_bytes = tensors.iter()
                .map(|t| t.data.len() * std::mem::size_of::<f32>())
                .sum();
            
            self.memory_tracker.deallocate(size_bytes);
        }
    }

    /// Estimate memory savings from current checkpointing strategy
    pub fn estimate_memory_savings(&self) -> (f64, f64) {
        // Calculate current memory usage with checkpointing
        let with_checkpointing = self.memory_tracker.peak_usage_mb();
        
        // Estimate memory usage without checkpointing
        // (Assuming each layer has similar memory requirements)
        let avg_checkpoint_size = if !self.checkpoints.is_empty() {
            let total_size: usize = self.checkpoints.values()
                .map(|tensors| tensors.iter()
                    .map(|t| t.data.len() * std::mem::size_of::<f32>())
                    .sum::<usize>())
                .sum();
            total_size as f64 / self.checkpoints.len() as f64
        } else {
            0.0
        };
        
        let without_checkpointing = avg_checkpoint_size * self.num_layers as f64 / (1024.0 * 1024.0);
        
        (with_checkpointing, without_checkpointing)
    }
}

/// Helper function to measure memory bandwidth
pub fn measure_memory_bandwidth() -> f64 {
    // Use a fixed size for memory bandwidth testing to avoid sys_info dependency
    let memory_size = 500_000_000; // 500MB, a reasonable size for most systems
    let num_elements = memory_size / std::mem::size_of::<f32>();
    
    // Allocate memory
    let mut data = vec![0.0f32; num_elements];
    
    // Warm-up
    for i in 0..num_elements {
        data[i] = i as f32;
    }
    
    // Measure read bandwidth
    let read_start = Instant::now();
    let mut sum = 0.0f32;
    for _ in 0..3 {
        for i in 0..num_elements {
            sum += data[i];
        }
    }
    let read_time = read_start.elapsed();
    
    // Measure write bandwidth
    let write_start = Instant::now();
    for _ in 0..3 {
        for i in 0..num_elements {
            data[i] = i as f32 * 0.5;
        }
    }
    let write_time = write_start.elapsed();
    
    // Prevent compiler from optimizing away the operations
    if sum < 0.0 {
        println!("Sum: {}", sum);
    }
    
    // Calculate bandwidth in GB/s
    let total_bytes = (num_elements * std::mem::size_of::<f32>() * 3) as f64;
    let read_bandwidth = total_bytes / read_time.as_secs_f64() / 1_000_000_000.0;
    let write_bandwidth = total_bytes / write_time.as_secs_f64() / 1_000_000_000.0;
    
    // Return average of read and write bandwidth
    (read_bandwidth + write_bandwidth) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array3};
    
    #[test]
    fn test_cache_blocked_matmul() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        
        let result = cache_blocked_matmul(&a, &b);
        
        let expected = arr2(&[[19.0, 22.0], [43.0, 50.0]]);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_batch_matmul_3d_2d() {
        let a = Array3::<f32>::from_shape_vec(
            (2, 3, 2), // batch_size=2, seq_len=3, features=2
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        ).unwrap().into_dyn();
        
        let b = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        
        let result = batch_matmul_3d_2d(&a, &b);
        let result_3d = result.into_dimensionality::<Ix3>().unwrap();
        
        // Expected:
        // For batch 0:
        //   [1.0, 2.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] = [9.0, 12.0, 15.0]
        //   [3.0, 4.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] = [19.0, 26.0, 33.0]
        //   [5.0, 6.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] = [29.0, 40.0, 51.0]
        // For batch 1:
        //   [7.0, 8.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] = [39.0, 54.0, 69.0]
        //   [9.0, 10.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] = [49.0, 68.0, 87.0]
        //   [11.0, 12.0] * [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] = [59.0, 82.0, 105.0]
        
        assert_eq!(result_3d.shape(), &[2, 3, 3]);
        
        // Check a few values
        assert!((result_3d[[0, 0, 0]] - 9.0).abs() < 1e-5);
        assert!((result_3d[[0, 1, 1]] - 26.0).abs() < 1e-5);
        assert!((result_3d[[1, 2, 2]] - 105.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_measure_memory_bandwidth() {
        // This is just a sanity check to ensure the function runs
        let bandwidth = measure_memory_bandwidth();
        println!("Measured memory bandwidth: {:.2} GB/s", bandwidth);
        assert!(bandwidth > 0.0);
    }

    #[test]
    fn test_gradient_checkpointer() {
        // Create a gradient checkpointer with uniform strategy
        let mut checkpointer = GradientCheckpointer::new(CheckpointStrategy::Uniform, 10);
        
        // Begin forward pass
        checkpointer.begin_forward();
        
        // Check which layers should be checkpointed
        for i in 0..10 {
            let should_store = checkpointer.should_checkpoint(i);
            println!("Layer {}: checkpoint={}", i, should_store);
            
            if should_store {
                // Create a mock tensor for testing
                let tensor = Tensor::new(Array2::<f32>::zeros((2, 2)));
                checkpointer.store_checkpoint(i, vec![tensor]);
            }
        }
        
        // End forward pass
        checkpointer.end_forward();
        
        // Check which layers were checkpointed
        for i in 0..10 {
            println!("Layer {}: has_checkpoint={}", i, checkpointer.has_checkpoint(i));
        }
        
        // Get memory stats
        let (current, peak) = checkpointer.get_memory_stats();
        println!("Memory usage: current={:.2} MB, peak={:.2} MB", current, peak);
    }
} 