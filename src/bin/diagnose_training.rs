use std::time::Instant;
use std::env;
use wall_e1::training::batch_dispatcher::BatchDispatcher;
use wall_e1::EnhancedTrainer;
use wall_e1::tokenizer::Tokenizer;
use ndarray::Array2;
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::sync::atomic::{AtomicBool, Ordering};

/// Simulate a stuck thread scenario for testing deadlock detection and recovery
fn simulate_stuck_thread(
    dispatcher: Arc<Mutex<BatchDispatcher>>,
    thread_id: usize,
    should_exit: Arc<AtomicBool>,
) {
    let thread_name = format!("worker_{}", thread_id);
    println!("🧵 Thread {} starting", thread_name);
    
    // Get and process batches until we should exit
    while !should_exit.load(Ordering::SeqCst) {
        // Acquire the lock
        let mut dispatcher_guard = match dispatcher.lock() {
            Ok(guard) => guard,
            Err(e) => {
                println!("❌ Thread {} failed to acquire lock: {:?}", thread_name, e);
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        
        // Let the dispatcher know we've acquired the lock
        dispatcher_guard.acquire_lock(&thread_name);
        
        // Get a batch to process
        let batch_idx = match dispatcher_guard.get_next_batch(&thread_name) {
            Some(idx) => idx,
            None => {
                // No batches left, release lock and exit
                dispatcher_guard.release_lock(&thread_name);
                break;
            }
        };
        
        // Release the lock while we "process" the batch
        dispatcher_guard.release_lock(&thread_name);
        
        // Simulate batch processing
        println!("🧵 Thread {} processing batch {}", thread_name, batch_idx);
        
        // Simulate a stuck thread if this is the thread we want to get stuck
        if thread_id == 1 && batch_idx % 3 == 0 {
            println!("🔄 Thread {} simulating a stuck thread on batch {}", thread_name, batch_idx);
            // Simulate being stuck for a long time
            let stuck_duration = Duration::from_secs(90);
            let start = Instant::now();
            
            while start.elapsed() < stuck_duration && !should_exit.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(500));
            }
            
            println!("🔄 Thread {} unstuck after {:?}", thread_name, start.elapsed());
        } else {
            // Normal processing time
            thread::sleep(Duration::from_millis(thread_id as u64 * 100 + 100));
        }
        
        // Reacquire the lock to mark batch as complete
        let mut dispatcher_guard = match dispatcher.lock() {
            Ok(guard) => guard,
            Err(e) => {
                println!("❌ Thread {} failed to reacquire lock: {:?}", thread_name, e);
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        
        // Let the dispatcher know we've acquired the lock
        dispatcher_guard.acquire_lock(&thread_name);
        
        // Mark batch as complete
        dispatcher_guard.complete_batch(batch_idx, &thread_name);
        
        // Release the lock
        dispatcher_guard.release_lock(&thread_name);
    }
    
    println!("🧵 Thread {} exiting", thread_name);
}

/// Start a watchdog thread to detect and recover from deadlocks
fn start_watchdog_thread(
    dispatcher: Arc<Mutex<BatchDispatcher>>,
    should_exit: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let thread_name = "watchdog".to_string();
        println!("🔍 Watchdog thread starting");
        
        // Check every 30 seconds
        let check_interval = Duration::from_secs(30);
        
        while !should_exit.load(Ordering::SeqCst) {
            thread::sleep(check_interval);
            
            println!("🔍 Watchdog checking for stuck threads");
            
            // Try to acquire the lock
            match dispatcher.lock() {
                Ok(mut guard) => {
                    // Let the dispatcher know we've acquired the lock
                    guard.acquire_lock(&thread_name);
                    
                    // Check for stuck threads
                    let stuck_threads = guard.check_for_stuck_threads();
                    
                    if !stuck_threads.is_empty() {
                        println!("🔍 Watchdog detected {} stuck threads: {:?}", 
                                 stuck_threads.len(), stuck_threads);
                        
                        // Attempt to recover
                        let recovered = guard.recover_from_deadlocks();
                        println!("🔄 Watchdog recovered {} batches", recovered);
                    } else {
                        println!("✅ No stuck threads detected");
                    }
                    
                    // Print status report
                    let status = guard.get_status_report();
                    println!("{}", status);
                    
                    // Release the lock
                    guard.release_lock(&thread_name);
                },
                Err(e) => {
                    println!("❌ Watchdog failed to acquire lock: {:?}", e);
                    println!("⚠️ This may indicate a deadlock on the lock itself!");
                    
                    // TODO: Implement more advanced recovery for lock deadlocks
                    // For now, we just log the issue and continue
                }
            }
        }
        
        println!("🔍 Watchdog thread exiting");
    })
}

/// Simulate a deadlock scenario for diagnostics
fn simulate_deadlock_scenario() {
    println!("🔄 Simulating deadlock scenario...");
    
    let total_batches = 20;
    
    // Create shared thread-safe batch dispatcher
    let dispatcher = Arc::new(Mutex::new(BatchDispatcher::new(total_batches)));
    
    // Exit flag for clean shutdown
    let should_exit = Arc::new(AtomicBool::new(false));
    
    // Start the watchdog thread
    let watchdog = start_watchdog_thread(dispatcher.clone(), should_exit.clone());
    
    // Start worker threads
    let mut workers = Vec::new();
    for thread_id in 0..4 {
        let dispatcher_clone = dispatcher.clone();
        let should_exit_clone = should_exit.clone();
        
        let handle = thread::spawn(move || {
            simulate_stuck_thread(dispatcher_clone, thread_id, should_exit_clone);
        });
        
        workers.push(handle);
    }
    
    // Let the simulation run for 3 minutes
    println!("🔄 Simulation will run for 3 minutes...");
    thread::sleep(Duration::from_secs(180));
    
    // Signal all threads to exit
    println!("🛑 Signaling threads to exit...");
    should_exit.store(true, Ordering::SeqCst);
    
    // Wait for all threads to finish
    for (i, worker) in workers.into_iter().enumerate() {
        match worker.join() {
            Ok(_) => println!("✅ Worker {} exited cleanly", i),
            Err(e) => println!("❌ Worker {} panicked: {:?}", i, e),
        }
    }
    
    // Wait for watchdog to finish
    match watchdog.join() {
        Ok(_) => println!("✅ Watchdog exited cleanly"),
        Err(e) => println!("❌ Watchdog panicked: {:?}", e),
    }
    
    // Print final status
    if let Ok(dispatcher_guard) = dispatcher.lock() {
        let status = dispatcher_guard.get_status_report();
        println!("Final status:\n{}", status);
    }
    
    println!("✅ Deadlock simulation completed");
}

/// Simulate target_id out of range errors for testing auto-resize capability
fn simulate_target_id_errors() {
    println!("🔄 Simulating target_id out of range errors...");
    
    // Create a trainer with a small vocabulary size
    let mut trainer = EnhancedTrainer::new(128, 512, 4, 2, 0.1, 0.001);
    
    // Create a batch with target IDs that exceed the vocabulary size
    let batch = vec![vec![1, 2, 3, 4, 5]];
    
    // Create targets with IDs outside the range
    let mut targets = Array2::zeros((1, 5));
    targets[[0, 0]] = 50;
    targets[[0, 1]] = 101;  // Just above the default max
    targets[[0, 2]] = 200;  // Well beyond default max
    targets[[0, 3]] = 500;  // Very large ID
    targets[[0, 4]] = 1000; // Extremely large ID
    
    // Train step should auto-resize
    println!("💡 Testing auto-resize with out-of-range target IDs...");
    match trainer.auto_resize_for_targets(&targets) {
        Ok(()) => println!("✅ Auto-resize successful"),
        Err(e) => println!("❌ Auto-resize failed: {}", e),
    }
    
    // Check the new vocabulary size
    println!("📊 New vocabulary size: {}", trainer.trainer.get_vocab_size());
    
    // Now try a training step
    println!("🔄 Attempting training step with resized vocabulary...");
    let loss = trainer.train_step_with_penalties(&batch, &targets);
    println!("📊 Training step completed with loss: {}", loss);
    
    println!("✅ Target ID simulation completed");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Wall-E1 Training Diagnostics Tool");
    println!("===================================");
    
    // Check for command line arguments
    let args: Vec<String> = std::env::args().collect();
    let target_id_only = args.iter().any(|arg| arg == "--target-id-only");
    
    if !target_id_only {
        // Run the deadlock simulation
        println!("\n🧪 Test 1: Deadlock Detection and Recovery");
        simulate_deadlock_scenario();
    }
    
    // Run the target ID out of range simulation
    println!("\n🧪 Test 2: Target ID Out of Range Handling");
    simulate_target_id_errors();
    
    println!("\n✅ All diagnostic tests completed");
    
    Ok(())
} 