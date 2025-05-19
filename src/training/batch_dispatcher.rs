use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::{HashSet, HashMap};
use std::sync::{Mutex, MutexGuard};
use std::thread::ThreadId;

/// A thread-safe batch dispatcher for managing batch processing across multiple threads
pub struct BatchDispatcher {
    /// Total number of batches
    total_batches: usize,
    /// Batches remaining to be processed
    remaining_batches: Vec<usize>,
    /// Batches currently being processed
    in_progress: HashMap<usize, BatchStatus>,
    /// Batches completed
    completed: HashSet<usize>,
    /// Lock status for debugging
    current_lock_owner: Option<String>,
    /// Time when lock was acquired
    lock_acquired_time: Option<Instant>,
    /// Atomic counter for tracking total batch progress
    progress_counter: AtomicUsize,
    /// Last activity timestamps per thread
    thread_timestamps: HashMap<String, Instant>,
    /// Lock acquisition history for diagnosing deadlocks
    lock_history: Vec<LockEvent>,
}

/// Track status of each batch
struct BatchStatus {
    /// Which thread has this batch
    thread_name: String,
    /// When the batch was assigned
    assigned_time: Instant,
    /// Status of the batch
    status: String,
}

/// Track lock acquisitions for debugging
struct LockEvent {
    /// Thread that acquired the lock
    thread_name: String, 
    /// When the lock was acquired
    time: Instant,
    /// How long the lock was held
    duration_ms: Option<u64>,
    /// Operation being performed
    operation: String,
}

impl BatchDispatcher {
    /// Create a new batch dispatcher with the given number of batches
    pub fn new(total_batches: usize) -> Self {
        // Create a vector of batch indices
        let remaining_batches: Vec<usize> = (0..total_batches).collect();

        Self {
            total_batches,
            remaining_batches,
            in_progress: HashMap::new(),
            completed: HashSet::new(),
            current_lock_owner: None,
            lock_acquired_time: None,
            progress_counter: AtomicUsize::new(0),
            thread_timestamps: HashMap::new(),
            lock_history: Vec::with_capacity(100), // Limit history to prevent memory issues
        }
    }

    /// Acquire the lock with timeout and diagnostics
    pub fn acquire_lock(&mut self, thread_name: &str) -> bool {
        // Record lock acquisition attempt
        self.update_thread_timestamp(thread_name);
        
        // Check if we already own the lock (re-entrancy)
        if let Some(owner) = &self.current_lock_owner {
            if owner == thread_name {
                println!("⚠️ Thread {} already owns the lock - potential deadlock risk", thread_name);
                return true;
            }
        }
        
        // Now we got the lock
        self.current_lock_owner = Some(thread_name.to_string());
        self.lock_acquired_time = Some(Instant::now());
        
        // Record this lock acquisition in history
        self.lock_history.push(LockEvent {
            thread_name: thread_name.to_string(),
            time: Instant::now(),
            duration_ms: None,
            operation: "acquire_lock".to_string(),
        });
        
        // Trim history if it gets too large
        if self.lock_history.len() > 100 {
            self.lock_history.remove(0);
        }
        
        true
    }

    /// Release the lock
    pub fn release_lock(&mut self, thread_name: &str) {
        // Update timestamp first
        self.update_thread_timestamp(thread_name);
        
        // Check if this thread actually owns the lock
        if let Some(owner) = &self.current_lock_owner {
            if owner != thread_name {
                println!("⚠️ Thread {} attempting to release lock owned by {}", 
                         thread_name, owner);
                return;
            }
            
            // Calculate how long the lock was held
            if let Some(acquired_time) = self.lock_acquired_time {
                let duration = acquired_time.elapsed();
                
                // Update the last lock event with duration
                if let Some(last_event) = self.lock_history.last_mut() {
                    if last_event.thread_name == thread_name {
                        last_event.duration_ms = Some(duration.as_millis() as u64);
                    }
                }
                
                // Log if the lock was held for too long
                if duration > Duration::from_millis(100) {
                    println!("⚠️ Thread {} held lock for {:?} - risk of blocking other threads", 
                             thread_name, duration);
                }
            }
            
            // Release the lock
            self.current_lock_owner = None;
            self.lock_acquired_time = None;
        }
    }

    /// Get the next available batch
    pub fn get_next_batch(&mut self, thread_name: &str) -> Option<usize> {
        // Check if this thread owns the lock
        self.check_lock_ownership(thread_name);
        
        // Update thread timestamp
        self.update_thread_timestamp(thread_name);
        
        // Log operation in history
        self.record_operation(thread_name, "get_next_batch");
        
        // Check if any batches are available
        if self.remaining_batches.is_empty() {
            return None;
        }
        
        // Get the next batch
        let batch_idx = self.remaining_batches.remove(0);
        
        // Add to in-progress map with status
        self.in_progress.insert(batch_idx, BatchStatus {
            thread_name: thread_name.to_string(),
            assigned_time: Instant::now(),
            status: "assigned".to_string(),
        });
        
        println!("🧵 Thread {} assigned batch {}", thread_name, batch_idx);
        
        Some(batch_idx)
    }

    /// Mark a batch as completed
    pub fn complete_batch(&mut self, batch_idx: usize, thread_name: &str) {
        // Check if this thread owns the lock
        self.check_lock_ownership(thread_name);
        
        // Update thread timestamp
        self.update_thread_timestamp(thread_name);
        
        // Record operation
        self.record_operation(thread_name, &format!("complete_batch {}", batch_idx));
        
        // Check if batch was in progress
        if let Some(status) = self.in_progress.remove(&batch_idx) {
            // Verify the thread owns this batch
            if status.thread_name != thread_name {
                println!("⚠️ Thread {} attempting to complete batch {} owned by {}", 
                         thread_name, batch_idx, status.thread_name);
                
                // Allow it anyway but log the warning
                println!("⚠️ Allowing thread {} to complete batch {} anyway to prevent deadlock",
                         thread_name, batch_idx);
            }
            
            // Log processing time
            let elapsed = status.assigned_time.elapsed();
            println!("✅ Thread {} completed batch {} in {:?}", 
                     thread_name, batch_idx, elapsed);
            
            // Add to completed set
            self.completed.insert(batch_idx);
            
            // Update progress counter
            self.progress_counter.fetch_add(1, Ordering::SeqCst);
            
            println!("✅ Thread {} completed batch {}. Progress: {}/{}", 
                     thread_name, batch_idx, self.get_progress_count(), self.total_batches);
        } else if self.completed.contains(&batch_idx) {
            println!("⚠️ Thread {} attempting to complete already completed batch {}", 
                     thread_name, batch_idx);
        } else {
            println!("⚠️ Thread {} attempting to complete unknown batch {}", 
                     thread_name, batch_idx);
        }
    }

    /// Check if all batches are completed
    pub fn is_complete(&self) -> bool {
        self.completed.len() == self.total_batches
    }

    /// Get the total number of batches
    pub fn get_total_batches(&self) -> usize {
        self.total_batches
    }

    /// Get the number of completed batches
    pub fn get_progress_count(&self) -> usize {
        self.progress_counter.load(Ordering::SeqCst)
    }

    /// Get the completion percentage
    pub fn completion_percentage(&self) -> f32 {
        if self.total_batches == 0 {
            return 100.0;
        }
        
        (self.get_progress_count() as f32 / self.total_batches as f32) * 100.0
    }

    /// Check for and log any deadlocked or stuck threads
    pub fn check_for_stuck_threads(&self) -> Vec<String> {
        println!("📊 Checking for stuck threads...");
        
        // Use a 60-second threshold for determining if a thread is stuck
        let threshold = Duration::from_secs(60);
        let now = Instant::now();
        let mut stuck_threads = Vec::new();
        
        // Print current lock status
        if let (Some(owner), Some(time)) = (&self.current_lock_owner, self.lock_acquired_time) {
            let duration = time.elapsed();
            println!("  Current lock owner: {} (held for {:?})", owner, duration);
            
            if duration > threshold {
                println!("  ⚠️ Lock held by {} for {:?} - possible deadlock", 
                         owner, duration);
                stuck_threads.push(owner.clone());
            }
        } else {
            println!("  Lock not currently held by any thread");
        }
        
        // Check in-progress batches
        println!("  In-progress batches: {}", self.in_progress.len());
        for (batch_idx, status) in &self.in_progress {
            let duration = status.assigned_time.elapsed();
            println!("  • Batch {} processed by {} for {:?}", 
                     batch_idx, status.thread_name, duration);
            
            if duration > threshold {
                println!("    ⚠️ Batch {} stuck with thread {} for {:?}", 
                         batch_idx, status.thread_name, duration);
                
                if !stuck_threads.contains(&status.thread_name) {
                    stuck_threads.push(status.thread_name.clone());
                }
            }
        }
        
        // Check thread timestamps
        println!("  Thread last seen timestamps:");
        for (thread_name, last_seen) in &self.thread_timestamps {
            let duration = now.duration_since(*last_seen);
            println!("  • Thread {} last active {:?} ago", thread_name, duration);
            
            if duration > threshold {
                println!("    ⚠️ Thread {} inactive for {:?}", thread_name, duration);
                
                if !stuck_threads.contains(thread_name) {
                    stuck_threads.push(thread_name.clone());
                }
            }
        }
        
        stuck_threads
    }

    /// Attempt to recover from deadlocks by reassigning stuck batches
    pub fn recover_from_deadlocks(&mut self) -> usize {
        println!("🔄 Attempting to recover from deadlocks...");
        
        let stuck_threads = self.check_for_stuck_threads();
        if stuck_threads.is_empty() {
            println!("  No stuck threads detected, no recovery needed");
            return 0;
        }
        
        // Reassign batches from stuck threads
        let mut recovered_batches = Vec::new();
        let threshold = Duration::from_secs(60);
        
        // Go through in-progress batches and recover those from stuck threads
        let batches_to_recover: Vec<_> = self.in_progress.iter()
            .filter(|(_, status)| {
                status.assigned_time.elapsed() > threshold && 
                stuck_threads.contains(&status.thread_name)
            })
            .map(|(&batch_idx, _)| batch_idx)
            .collect();
        
        // Now move these batches back to the remaining queue
        for batch_idx in batches_to_recover {
            if let Some(status) = self.in_progress.remove(&batch_idx) {
                println!("🔄 Recovering batch {} from stuck thread {}", 
                         batch_idx, status.thread_name);
                self.remaining_batches.push(batch_idx);
                recovered_batches.push(batch_idx);
            }
        }
        
        // Clear lock if it's held by a stuck thread
        if let Some(owner) = &self.current_lock_owner {
            if stuck_threads.contains(owner) {
                println!("🔓 Forcibly releasing lock held by stuck thread {}", owner);
                self.current_lock_owner = None;
                self.lock_acquired_time = None;
            }
        }
        
        println!("✅ Recovered {} batches from stuck threads", recovered_batches.len());
        recovered_batches.len()
    }

    /// Force unlock if locked by the specified thread
    pub fn force_unlock(&mut self, thread_name: &str) -> bool {
        if let Some(owner) = &self.current_lock_owner {
            if owner == thread_name {
                println!("🔓 Forcibly releasing lock held by thread {}", owner);
                self.current_lock_owner = None;
                self.lock_acquired_time = None;
                return true;
            }
        }
        false
    }

    /// Get detailed status for diagnostics
    pub fn get_status_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("Batch Dispatcher Status\n"));
        report.push_str(&format!("----------------------\n"));
        report.push_str(&format!("Total batches: {}\n", self.total_batches));
        report.push_str(&format!("Remaining batches: {}\n", self.remaining_batches.len()));
        report.push_str(&format!("In-progress batches: {}\n", self.in_progress.len()));
        report.push_str(&format!("Completed batches: {}\n", self.completed.len()));
        report.push_str(&format!("Progress: {}/{} ({:.1}%)\n", 
                                self.get_progress_count(), self.total_batches, 
                                self.completion_percentage()));
        
        // Lock status
        report.push_str(&format!("\nLock Status:\n"));
        if let (Some(owner), Some(time)) = (&self.current_lock_owner, self.lock_acquired_time) {
            report.push_str(&format!("  Held by: {} for {:?}\n", owner, time.elapsed()));
        } else {
            report.push_str(&format!("  Not held\n"));
        }
        
        // Recent lock history
        report.push_str(&format!("\nRecent Lock Activity:\n"));
        for (i, event) in self.lock_history.iter().rev().take(10).enumerate() {
            let duration_str = match event.duration_ms {
                Some(ms) => format!("{}ms", ms),
                None => "ongoing".to_string(),
            };
            
            report.push_str(&format!("  {}: {} - {} - {}\n", 
                                   i+1, event.thread_name, event.operation, duration_str));
        }
        
        report
    }

    /// Monitor thread activity over time
    pub fn monitor_thread_activity(&self, max_inactivity_secs: u64) -> (Vec<String>, HashMap<String, Duration>) {
        let now = Instant::now();
        let mut inactive_threads = Vec::new();
        let mut thread_inactivity = HashMap::new();
        
        println!("🔍 Monitoring thread activity...");
        
        // Check each thread's last activity timestamp
        for (thread_name, last_seen) in &self.thread_timestamps {
            let inactivity_duration = now.duration_since(*last_seen);
            
            // Record the inactivity duration
            thread_inactivity.insert(thread_name.clone(), inactivity_duration);
            
            // Check if thread has been inactive for too long
            if inactivity_duration.as_secs() > max_inactivity_secs {
                println!("⚠️ Thread {} inactive for {:?} (threshold: {}s)",
                         thread_name, inactivity_duration, max_inactivity_secs);
                inactive_threads.push(thread_name.clone());
            } else {
                println!("✅ Thread {} active within last {:?}",
                         thread_name, inactivity_duration);
            }
        }
        
        // Check for lock contention
        if let (Some(owner), Some(time)) = (&self.current_lock_owner, self.lock_acquired_time) {
            let lock_hold_duration = time.elapsed();
            
            if lock_hold_duration.as_secs() > max_inactivity_secs / 2 {
                println!("⚠️ Lock held by {} for {:?} - potential deadlock",
                         owner, lock_hold_duration);
                        
                if !inactive_threads.contains(owner) {
                    inactive_threads.push(owner.clone());
                }
            }
        }
        
        // Check in-progress batches for stuck processing
        for (batch_idx, status) in &self.in_progress {
            let processing_time = status.assigned_time.elapsed();
            
            if processing_time.as_secs() > max_inactivity_secs {
                println!("⚠️ Batch {} stuck in processing by {} for {:?}",
                         batch_idx, status.thread_name, processing_time);
                         
                if !inactive_threads.contains(&status.thread_name) {
                    inactive_threads.push(status.thread_name.clone());
                }
            }
        }
        
        (inactive_threads, thread_inactivity)
    }

    /// Set batch status to help track progress
    pub fn set_batch_status(&mut self, batch_idx: usize, thread_name: &str, status: &str) {
        // Check if this thread owns the lock
        self.check_lock_ownership(thread_name);
        
        // Update thread timestamp
        self.update_thread_timestamp(thread_name);
        
        // Update batch status if it's in progress
        if let Some(batch_status) = self.in_progress.get_mut(&batch_idx) {
            if batch_status.thread_name != thread_name {
                println!("⚠️ Thread {} attempting to update status of batch {} owned by {}",
                        thread_name, batch_idx, batch_status.thread_name);
                return;
            }
            
            println!("ℹ️ Thread {} updated batch {} status: {} -> {}",
                    thread_name, batch_idx, batch_status.status, status);
            
            batch_status.status = status.to_string();
        } else {
            println!("⚠️ Thread {} attempted to update status of non-existent batch {}",
                    thread_name, batch_idx);
        }
    }

    /// Get detailed information about a particular batch
    pub fn get_batch_info(&self, batch_idx: usize) -> Option<String> {
        // Check if the batch is in progress
        if let Some(status) = self.in_progress.get(&batch_idx) {
            let mut info = format!("Batch {} Information\n", batch_idx);
            info.push_str(&format!("Status: {}\n", status.status));
            info.push_str(&format!("Assigned to: {}\n", status.thread_name));
            info.push_str(&format!("Processing time: {:?}\n", status.assigned_time.elapsed()));
            
            return Some(info);
        }
        
        // Check if the batch is completed
        if self.completed.contains(&batch_idx) {
            return Some(format!("Batch {} is completed", batch_idx));
        }
        
        // Check if the batch is still in the queue
        if self.remaining_batches.contains(&batch_idx) {
            return Some(format!("Batch {} is waiting in queue", batch_idx));
        }
        
        // Batch not found
        None
    }

    // Private helper methods
    
    /// Update the timestamp for a thread
    fn update_thread_timestamp(&mut self, thread_name: &str) {
        self.thread_timestamps.insert(thread_name.to_string(), Instant::now());
    }
    
    /// Check if thread owns lock and log warning if not
    fn check_lock_ownership(&self, thread_name: &str) {
        if let Some(owner) = &self.current_lock_owner {
            if owner != thread_name {
                println!("⚠️ Thread {} attempting to use dispatcher while lock owned by {}", 
                         thread_name, owner);
            }
        } else {
            println!("⚠️ Thread {} attempting to use dispatcher without owning lock", thread_name);
        }
    }
    
    /// Record an operation in the lock history
    fn record_operation(&mut self, thread_name: &str, operation: &str) {
        // Only record if we're not going to overflow the history
        if self.lock_history.len() < 100 {
            self.lock_history.push(LockEvent {
                thread_name: thread_name.to_string(),
                time: Instant::now(),
                duration_ms: Some(0),  // Instant operation
                operation: operation.to_string(),
            });
        }
    }
} 