mod e_tensor;

use ndarray::{Array, s};
use e_tensor::Tensor;


fn main() {
    println!("Hello, world!");
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
      
    // More complete example showing model training
    train_simple_model();
}

// Example of training a simple model over multiple steps
fn train_simple_model() {
    println!("\n--- Training a simple model ---");
    
    // Create a model with parameters to be learned
    let x = Tensor::new(Array::from_shape_fn((1, 2), |_| 0.5));
    let mut w = Tensor::new(Array::from_shape_fn((2, 1), |_| 1.0));
    
    // Target: what we want our model to produce
    let target = Tensor::new(Array::from_elem((1, 1), 0.8));
    
    // Learning rate - use a smaller value to prevent overshooting
    let lr = 0.02;
    
    // Train for several steps
    println!("\nTraining for 20 steps:");
    for i in 0..20 {
        // Forward pass
        let out = Tensor::matmul(&x, &w);
        
        // Compute loss
        let neg_target = Tensor::new(-1.0 * &target.data);
        let diff = Tensor::add(&out, &neg_target);
        let squared = Tensor::square(&diff);
        let loss = Tensor::sum(&squared);
        
        // Backward pass
        loss.backward(None);
        
        // Get gradient and update weights
        let w_grad = w.grad.borrow().clone().unwrap();
        
        // Create new weights by subtracting gradient * learning rate
        let new_weights = &w.data - &(&w_grad * lr);
        
        // Print progress
        println!("Step {}: loss = {:.6}, prediction = {:.6}, w = [{:.6}, {:.6}]", 
                i, 
                loss.data[[0, 0]], 
                out.data[[0, 0]],
                w.data[[0, 0]],
                w.data[[1, 0]]);
        
        // Re-create weight tensor with new values and reset grad
        w = Tensor::new(new_weights);
    }
    
    // Final forward pass to check result
    let final_out = Tensor::matmul(&x, &w);
    println!("\nFinal prediction: {:.6} (target: 0.8)", final_out.data[[0, 0]]);
    println!("Final weights: w = [{:.6}, {:.6}]", w.data[[0, 0]], w.data[[1, 0]]);
    
    // For our simple network, both weights should converge to 0.8 / 2 = 0.4
    // since x = [0.5, 0.5] and we want out = 0.8
    println!("Expected optimal weights = [0.8, 0.8] for single input or [0.4, 0.4] for both inputs summing to 0.8");
}

