use rayon::ThreadPool;
use std::env;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex, Once};

// Use lazy_static for safe global initialization
lazy_static! {
    static ref GLOBAL_POOL: Mutex<Option<Arc<ThreadPoolManager>>> = Mutex::new(None);
    static ref OPERATION_POOLS: Mutex<std::collections::HashMap<String, Arc<ThreadPoolManager>>> = Mutex::new(std::collections::HashMap::new());
}

// One-time initialization flag
static INIT: Once = Once::new();

/// Thread pool manager that handles central thread pool access
pub struct ThreadPoolManager {
    pool: Mutex<ThreadPool>,
    num_threads: usize,
    operation_type: Option<String>,
}

impl ThreadPoolManager {
    /// Create a new thread pool manager with the specified number of threads
    fn new(num_threads: usize, operation_type: Option<&str>) -> Self {
        let operation_name = operation_type.unwrap_or("general");
        println!("Creating thread pool with {} threads for {} operations", num_threads, operation_name);
        
        // Create the thread pool with the specified configuration
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool");
            
        ThreadPoolManager {
            pool: Mutex::new(pool),
            num_threads,
            operation_type: operation_type.map(|s| s.to_string()),
        }
    }
    
    /// Get the underlying thread pool
    pub fn get_pool(&self) -> &Mutex<ThreadPool> {
        &self.pool
    }
    
    /// Get the number of threads in the pool
    pub fn get_num_threads(&self) -> usize {
        self.num_threads
    }
    
    /// Get the operation type this pool is optimized for
    pub fn get_operation_type(&self) -> Option<&str> {
        self.operation_type.as_deref()
    }
    
    /// Install this thread pool as the global Rayon thread pool
    pub fn install_as_global_pool(&self) {
        let pool_guard = self.pool.lock().unwrap();
        pool_guard.install(|| {
            println!("Thread pool with {} threads installed for global use", self.num_threads);
        });
    }
}

/// Get or initialize the global thread pool manager
pub fn get_global_thread_pool() -> Arc<ThreadPoolManager> {
    INIT.call_once(|| {
        // Determine the number of threads to use
        let num_threads = determine_thread_count();
        
        // Create the thread pool manager and store it in the global variable
        let mut pool_guard = GLOBAL_POOL.lock().unwrap();
        *pool_guard = Some(Arc::new(ThreadPoolManager::new(num_threads, None)));
    });
    
    // Return a clone of the global pool
    GLOBAL_POOL.lock().unwrap().clone().unwrap()
}

/// Get a thread pool optimized for a specific operation type
pub fn get_thread_pool_for_operation(operation: &str) -> Arc<ThreadPoolManager> {
    // Try to get the pool with a non-blocking approach first
    match OPERATION_POOLS.try_lock() {
        Ok(mut pools) => {
            // Successfully acquired lock
            if !pools.contains_key(operation) {
                // Create a new pool for this operation type with optimal thread count
                let num_threads = get_optimal_thread_count_for_operation(operation);
                println!("🔹 Creating dedicated thread pool for '{}' operation with {} threads", operation, num_threads);
                let pool = Arc::new(ThreadPoolManager::new(num_threads, Some(operation)));
                pools.insert(operation.to_string(), pool.clone());
                return pool;
            } else {
                // Return existing pool
                println!("🔸 Reusing existing thread pool for '{}' operation", operation);
                return pools.get(operation).unwrap().clone();
            }
        },
        Err(_) => {
            // Couldn't get the lock - this might be a deadlock situation
            // Fall back to the global thread pool
            println!("⚠️ Warning: Could not acquire lock for operation pools, using global pool for '{}' to avoid deadlock", operation);
            return get_global_thread_pool();
        }
    }
}

/// Determine the optimal thread count based on environment and system capabilities
fn determine_thread_count() -> usize {
    // First check if the user has specified RAYON_NUM_THREADS
    if let Ok(threads) = env::var("RAYON_NUM_THREADS") {
        if let Ok(num) = threads.parse::<usize>() {
            if num > 0 {
                return num;
            }
        }
    }
    
    // Next check for environment variable specifically for our application
    if let Ok(threads) = env::var("WALL_E_THREADS") {
        if let Ok(num) = threads.parse::<usize>() {
            if num > 0 {
                return num;
            }
        }
    }
    
    // Otherwise use available parallelism, but leave one CPU core free
    // to avoid completely overwhelming the system
    let available_parallelism = num_cpus::get();
    if available_parallelism > 1 {
        available_parallelism - 1
    } else {
        1
    }
}

/// Get the optimal thread count for specific operation types
fn get_optimal_thread_count_for_operation(operation: &str) -> usize {
    let available_parallelism = num_cpus::get();
    let physical_cores = num_cpus::get_physical();
    
    match operation {
        "matrix_multiply" => available_parallelism.saturating_sub(1), // Compute intensive
        "gradient_update" => (available_parallelism as f32 * 0.75).max(1.0) as usize, // Memory intensive
        "attention" => available_parallelism.saturating_sub(1), // Compute intensive  
        "tokenization" => 4.min(available_parallelism), // I/O bound, doesn't need all cores
        "data_loading" => {
            // Check for environment variable specifically for data loading operations
            if let Ok(threads) = env::var("WALL_E_DATA_THREADS") {
                if let Ok(num) = threads.parse::<usize>() {
                    println!("Using WALL_E_DATA_THREADS={} from environment", num);
                    return num.min(available_parallelism);
                }
            }
            
            // Default: More aggressive thread allocation for data loading
            // At least 6 threads or 50% of available cores (up from 20%), whichever is greater
            // This helps ensure enough CPU utilization for data loading
            let min_data_threads = std::cmp::max(6, (available_parallelism as f32 * 0.5) as usize);
            
            println!("Data loading threads: min={}, available={}, using={}",
                     min_data_threads, available_parallelism, min_data_threads.min(available_parallelism));
            
            // Log warning if we're using too few threads
            if min_data_threads < 4 {
                println!("⚠️ Warning: Using only {} threads for data loading, may cause low CPU utilization", 
                         min_data_threads);
            }
            
            min_data_threads.min(available_parallelism)
        },
        _ => physical_cores.max(1), // Default to physical core count
    }
}

/// Execute a job on the global thread pool
pub fn execute_in_pool<F>(job: F)
where
    F: FnOnce() + Send + 'static,
{
    let manager = get_global_thread_pool();
    let pool = manager.get_pool().lock().unwrap();
    pool.spawn(job);
}

/// Execute a job on a thread pool optimized for a specific operation
pub fn execute_in_pool_for_operation<F>(operation: &str, job: F)
where
    F: FnOnce() + Send + 'static,
{
    let manager = get_thread_pool_for_operation(operation);
    let pool = manager.get_pool().lock().unwrap();
    pool.spawn(job);
}

/// Execute something in parallel using the global thread pool
/// 
/// This is an improved version that uses proper parallelism
pub fn parallel_execute<T, F, R>(items: Vec<T>, f: F) -> Vec<R>
where
    T: Send + 'static,
    F: Fn(T) -> R + Send + Sync + 'static,
    R: Send + 'static,
{
    use rayon::prelude::*;
    
    let manager = get_global_thread_pool();
    let pool = manager.get_pool().lock().unwrap();
    
    // Convert the items to a parallel iterator and collect the results directly
    pool.install(|| {
        items.into_par_iter()
            .map(f)
            .collect()
    })
}

/// Execute something in parallel using a thread pool optimized for a specific operation
pub fn parallel_execute_for_operation<T, F, R>(operation: &str, items: Vec<T>, f: F) -> Vec<R>
where
    T: Send + 'static,
    F: Fn(T) -> R + Send + Sync + 'static,
    R: Send + 'static,
{
    use rayon::prelude::*;
    
    let manager = get_thread_pool_for_operation(operation);
    let pool = manager.get_pool().lock().unwrap();
    
    // Convert the items to a parallel iterator and collect the results directly
    pool.install(|| {
        items.into_par_iter()
            .map(f)
            .collect()
    })
} 