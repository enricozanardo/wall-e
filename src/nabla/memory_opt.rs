use ndarray::{Array, Array2, ArrayD, Ix2, Ix3};
use rayon::prelude::*;
use std::cmp;

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

/// Measure memory bandwidth
pub fn measure_memory_bandwidth() -> f64 {
    // Create large arrays to ensure we're measuring memory bandwidth not cache
    let size = 100 * 1024 * 1024; // 100 MB
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let mut c = vec![0.0f32; size];
    
    // Warm up
    for i in 0..1000 {
        c[i] = a[i] + b[i];
    }
    
    // Measure bandwidth
    let start = std::time::Instant::now();
    
    for i in 0..size {
        c[i] = a[i] + b[i];
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    let bytes_processed = (size * 3 * std::mem::size_of::<f32>()) as f64;
    let bandwidth_gb_per_sec = bytes_processed / (elapsed * 1_000_000_000.0);
    
    bandwidth_gb_per_sec
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
} 