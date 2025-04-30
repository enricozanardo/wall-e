use ndarray::{Array, Array2, Array3, ArrayD, Ix2, IxDyn};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

// autograd engine for gradient calculation


/// Represents a tensor with autograd capabilities
#[derive(Clone)]
pub struct Tensor {
    /// Tensor data (values)
    pub data: ArrayD<f32>,
    /// Gradients calculated during backpropagation
    pub grad: Arc<Mutex<Option<ArrayD<f32>>>>,
    /// Function to calculate gradient during backprop
    pub grad_fn: Option<Arc<dyn Fn(&Tensor, &ArrayD<f32>) + Send + Sync>>,
    /// Parent tensors from which this tensor was created
    pub parents: Vec<Tensor>,
}

impl Tensor {
    /// Creates a new tensor with the specified data (Array2)
    pub fn new(data: Array2<f32>) -> Self {
        Tensor { 
            data: data.into_dyn(), 
            grad: Arc::new(Mutex::new(None)), 
            grad_fn: None, 
            parents: vec![] 
        }
    }
    
    /// Creates a new tensor with the specified data (Array3)
    pub fn new_3d(data: Array3<f32>) -> Self {
        Tensor { 
            data: data.into_dyn(), 
            grad: Arc::new(Mutex::new(None)), 
            grad_fn: None, 
            parents: vec![] 
        }
    }

    /// Creates a tensor with a gradient function
    pub fn with_grad_fn(data: ArrayD<f32>, parents: Vec<Tensor>, grad_fn: Arc<dyn Fn(&Tensor, &ArrayD<f32>) + Send + Sync>) -> Self {
        Self {
            data,
            grad: Arc::new(Mutex::new(None)),
            grad_fn: Some(grad_fn),
            parents,
        }
    }

    /// Performs backpropagation starting from this tensor
    pub fn backward(&self, grad_output: Option<ArrayD<f32>>) {
        // If no gradient is provided, use a tensor of all 1s
        let grad = grad_output.unwrap_or_else(|| Array::ones(self.data.raw_dim()));
        
        // Update the gradient of this tensor
        self.update_grad(&grad);
        
        // Propagate the gradient to parent tensors, if there's a grad_fn
        if let Some(ref grad_fn) = self.grad_fn {
            // Apply the gradient function
            grad_fn(self, &grad);
        }
    }
    
    /// Updates the accumulated gradient for this tensor
    fn update_grad(&self, grad: &ArrayD<f32>) {
        let mut locked_grad = self.grad.lock().unwrap();
        if let Some(ref existing_grad) = *locked_grad {
            // Accumulate existing gradient
            *locked_grad = Some(existing_grad + grad);
        } else {
            // Set the gradient if it doesn't exist
            *locked_grad = Some(grad.clone());
        }
    }

    // ===== TENSOR OPERATIONS =====

    /// Adds two tensors element-wise
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum
    ///
    /// # Panics
    ///
    /// Panics if the tensors have incompatible shapes
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    /// let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
    /// let c = Tensor::add(&a, &b);
    ///
    /// // c now contains [[6.0, 8.0], [10.0, 12.0]]
    /// ```
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        // Verify that dimensions are compatible
        assert_eq!(a.data.shape(), b.data.shape(), "Tensors with incompatible shapes for addition");
        
        // Calculate the forward result using Rayon
        let a_vec: Vec<f32> = a.data.iter().cloned().collect();
        let b_vec: Vec<f32> = b.data.iter().cloned().collect();
        
        // Parallelize the element-wise addition
        let result_vec: Vec<f32> = a_vec.par_iter()
            .zip(b_vec.par_iter())
            .map(|(&a_val, &b_val)| a_val + b_val)
            .collect();
        
        // Convert the result to Array with the same shape
        let data = Array::from_shape_vec(a.data.raw_dim(), result_vec).unwrap();
        
        // Clone the parent tensors for the closure
        let a_clone = a.clone();
        let b_clone = b.clone();

        // Create a new tensor with the gradient function
        Tensor::with_grad_fn(data, vec![a.clone(), b.clone()], Arc::new(move |_, grad| {
            // The gradient of addition propagates identically to both inputs
            a_clone.update_grad(grad);
            b_clone.update_grad(grad);
        }))
    }

    /// Multiplies two tensors (matrix multiplication)
    ///
    /// # Arguments
    ///
    /// * `a` - First tensor
    /// * `b` - Second tensor
    ///
    /// # Returns
    ///
    /// A new tensor containing the matrix multiplication result
    ///
    /// # Panics
    ///
    /// Panics if the tensors have incompatible shapes for matrix multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    /// let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
    /// let c = Tensor::matmul(&a, &b);
    ///
    /// // c now contains [[19.0, 22.0], [43.0, 50.0]]
    /// ```
    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        // Convert to Array2 for matrix multiplication
        let a_2d = a.data.clone().into_dimensionality::<Ix2>().unwrap();
        let b_2d = b.data.clone().into_dimensionality::<Ix2>().unwrap();
        
        // Matrix multiplication uses BLAS when available
        let data = a_2d.dot(&b_2d).into_dyn();
        
        // Clone the parent tensors for the closure
        let a_clone = a.clone();
        let b_clone = b.clone();
    
        Tensor::with_grad_fn(
            data,
            vec![a.clone(), b.clone()],
            Arc::new(move |_, grad| {
                // Convert to Array2 for matrix multiplication
                let grad_2d = grad.clone().into_dimensionality::<Ix2>().unwrap();
                let a_2d = a_clone.data.clone().into_dimensionality::<Ix2>().unwrap();
                let b_2d = b_clone.data.clone().into_dimensionality::<Ix2>().unwrap();
                
                // For matrix multiplication:
                // dL/dA = dL/dZ · B^T
                // dL/dB = A^T · dL/dZ
                let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();
                let grad_b = a_2d.t().dot(&grad_2d).into_dyn();
                
                // Update the gradients of A and B
                a_clone.update_grad(&grad_a);
                b_clone.update_grad(&grad_b);
            }),
        )
    }
    
    /// Computes the transpose of a tensor
    ///
    /// # Returns
    ///
    /// A new tensor that is the transpose of the input tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let a = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    /// let at = a.transpose();
    ///
    /// // at now contains [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    /// ```
    pub fn transpose(&self) -> Tensor {
        // Convert to Array2 for transposition
        let data_2d = self.data.clone().into_dimensionality::<Ix2>().unwrap();
        
        // Get the transpose of the data array
        let transposed_data = data_2d.t().to_owned().into_dyn();
        
        // Clone the parent tensor for the closure
        let self_clone = self.clone();
        
        Tensor::with_grad_fn(
            transposed_data,
            vec![self.clone()],
            Arc::new(move |_, grad| {
                // Convert to Array2 for transposition
                let grad_2d = grad.clone().into_dimensionality::<Ix2>().unwrap();
                
                // The gradient of transpose is the transpose of the gradient
                let grad_input = grad_2d.t().to_owned().into_dyn();
                
                // Update the gradient
                self_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Multiplies this tensor by another tensor (matrix multiplication)
    ///
    /// This is an instance method that provides a more convenient syntax for 
    /// matrix multiplication operations.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply with
    ///
    /// # Returns
    ///
    /// A new tensor containing the matrix multiplication result
    ///
    /// # Panics
    ///
    /// Panics if the tensors have incompatible shapes for matrix multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    /// let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
    /// let c = a.matmul_with(&b);
    ///
    /// // c now contains [[19.0, 22.0], [43.0, 50.0]]
    /// ```
    pub fn matmul_with(&self, other: &Tensor) -> Tensor {
        // Debug logging of shapes
        // println!("Debug: matmul_with - self shape: {:?}, other shape: {:?}", self.data.shape(), other.data.shape());
        
        // Special case support for 3D tensors [batch, seq_len, feature_dim] 
        // multiplied by a 2D tensor [feature_dim, output_dim]
        if self.data.ndim() == 3 && other.data.ndim() == 2 {
            let self_shape = self.data.shape();
            let other_shape = other.data.shape();
            
            // Verify dimension compatibility
            if self_shape[2] != other_shape[0] {
                panic!("Incompatible dimensions for matmul_with: {:?} and {:?}", self_shape, other_shape);
            }
            
            let batch_size = self_shape[0];
            let seq_len = self_shape[1];
            let feature_dim = self_shape[2];
            let output_dim = other_shape[1];
            
            // println!("Debug: matmul_with - special case 3D x 2D");
            
            // Result: [batch_size, seq_len, output_dim]
            let mut result = Array3::<f32>::zeros((batch_size, seq_len, output_dim));
            
            // Perform matrix multiplication for each batch and each position in the sequence
            for b in 0..batch_size {
                for s in 0..seq_len {
                    for o in 0..output_dim {
                        let mut sum = 0.0;
                        for f in 0..feature_dim {
                            sum += self.data[[b, s, f]] * other.data[[f, o]];
                        }
                        result[[b, s, o]] = sum;
                    }
                }
            }
            
            // Return a 3D tensor
            return Tensor::new_3d(result);
        }
        
        // General case - use standard matmul
        Tensor::matmul(self, other)
    }

    /// Applies the ReLU (Rectified Linear Unit) activation function to the tensor
    ///
    /// This applies the function f(x) = max(0, x) to each element of the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with ReLU applied element-wise
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let x = Tensor::new(arr2(&[[-1.0, 2.0], [-3.0, 4.0]]));
    /// let y = x.relu();
    /// // y now contains [[0.0, 2.0], [0.0, 4.0]]
    /// ```
    pub fn relu(&self) -> Self {
        let result = self.data.mapv(|x| x.max(0.0));
        
        // Clone the parent tensor for the closure
        let self_clone = self.clone();
        
        Tensor::with_grad_fn(
            result,
            vec![self.clone()],
            Arc::new(move |_, grad| {
                // The gradient of ReLU is 1 if the input is > 0, otherwise 0
                let input_data = &self_clone.data;
                let mut grad_input = grad.clone();
                
                // Apply the mask to the gradient
                for (i, &val) in input_data.iter().enumerate() {
                    if val <= 0.0 {
                        grad_input.as_slice_mut().unwrap()[i] = 0.0;
                    }
                }
                
                // Update the gradient
                self_clone.update_grad(&grad_input);
            }),
        )
    }

    /// Squares each element of the tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    ///
    /// # Returns
    ///
    /// A new tensor with each element squared
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let x = Tensor::new(arr2(&[[1.0, 2.0], [3.0, -4.0]]));
    /// let y = Tensor::square(&x);
    /// // y now contains [[1.0, 4.0], [9.0, 16.0]]
    /// ```
    pub fn square(x: &Tensor) -> Tensor {
        // x² is calculated as x * x using Rayon for parallelization
        // Convert to vector to use Rayon, then back to Array
        let data_vec: Vec<f32> = x.data.iter().cloned().collect();
        let result_vec: Vec<f32> = data_vec.par_iter()
            .map(|&v| v * v)
            .collect();
        
        // Convert the result to Array with the same shape
        let result_data = Array::from_shape_vec(x.data.raw_dim(), result_vec).unwrap();
        
        // Clone the parent tensor for the closure
        let x_clone = x.clone();
        
        Tensor::with_grad_fn(
            result_data,
            vec![x.clone()],
            Arc::new(move |_, grad| {
                // The gradient of x² is 2x
                let mut grad_input = grad.clone();
                let x_data = &x_clone.data;
                
                for (i, &x_val) in x_data.iter().enumerate() {
                    grad_input.as_slice_mut().unwrap()[i] *= 2.0 * x_val;
                }
                
                // Update the gradient
                x_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Computes the sum of all elements in the tensor
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor
    ///
    /// # Returns
    ///
    /// A new scalar tensor (1x1) containing the sum of all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let x = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    /// let sum = Tensor::sum(&x);
    /// // sum now contains a 1x1 tensor with value 10.0
    /// ```
    pub fn sum(x: &Tensor) -> Tensor {
        // Calculate the scalar sum of all elements
        let sum_val = x.data.sum();
        let mut data = Array::zeros(IxDyn(&[1, 1]));
        data[IxDyn(&[0, 0])] = sum_val;
        
        // Clone the parent tensor for the closure
        let x_clone = x.clone();
        
        Tensor::with_grad_fn(
            data,
            vec![x.clone()],
            Arc::new(move |_, grad| {
                // The gradient of sum is an array of all 1s with the same shape as the input
                let grad_val = grad[IxDyn(&[0, 0])];
                let grad_input = Array::from_elem(x_clone.data.raw_dim(), grad_val);
                
                // Update the gradient
                x_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Computes the Mean Squared Error (MSE) between output and target tensors
    ///
    /// # Arguments
    ///
    /// * `output` - The predicted tensor
    /// * `target` - The target tensor
    ///
    /// # Returns
    ///
    /// A scalar tensor (1x1) containing the MSE value
    ///
    /// # Panics
    ///
    /// Panics if the tensors have incompatible shapes
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::arr2;
    ///
    /// let output = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
    /// let target = Tensor::new(arr2(&[[0.0, 0.0], [0.0, 0.0]]));
    /// let loss = Tensor::mse(&output, &target);
    /// // loss now contains the mean squared error: 7.5
    /// ```
    pub fn mse(output: &Tensor, target: &Tensor) -> Tensor {
        // Verify that dimensions are compatible
        assert_eq!(output.data.shape(), target.data.shape(), "Tensors with incompatible shapes for MSE");
        
        // Calculate the difference
        let diff_vec: Vec<f32> = output.data.iter()
            .zip(target.data.iter())
            .map(|(&o, &t)| o - t)
            .collect();
        
        // Convert the result to Array with the same shape
        let diff_data = Array::from_shape_vec(output.data.raw_dim(), diff_vec).unwrap();
        
        // Calculate the squared differences
        let squared_diff: Vec<f32> = diff_data.iter().map(|&d| d * d).collect();
        
        // Calculate the mean
        let n = squared_diff.len() as f32;
        let mse_val = squared_diff.iter().sum::<f32>() / n;
        
        // Create a scalar Tensor
        let mut data = Array::zeros(IxDyn(&[1, 1]));
        data[IxDyn(&[0, 0])] = mse_val;
        
        // Clone the parent tensors for the closure
        let output_clone = output.clone();
        let target_clone = target.clone();
        
        Tensor::with_grad_fn(
            data,
            vec![output.clone(), target.clone()],
            Arc::new(move |_, grad| {
                // The gradient of MSE with respect to the output is 2(output - target) / n
                let grad_val = grad[IxDyn(&[0, 0])];
                let n = output_clone.data.len() as f32;
                
                let grad_vec: Vec<f32> = output_clone.data.iter()
                    .zip(target_clone.data.iter())
                    .map(|(&o, &t)| 2.0 * (o - t) * grad_val / n)
                    .collect();
                
                // Convert the gradient to Array with the same shape as the output
                let grad_output = Array::from_shape_vec(output_clone.data.raw_dim(), grad_vec).unwrap();
                
                // Update the gradient
                output_clone.update_grad(&grad_output);
                
                // The gradient with respect to the target is -grad(output)
                let grad_target_vec: Vec<f32> = grad_output.iter().map(|&g| -g).collect();
                let grad_target = Array::from_shape_vec(target_clone.data.raw_dim(), grad_target_vec).unwrap();
                
                // Update the gradient
                target_clone.update_grad(&grad_target);
            }),
        )
    }

    /// Creates a new tensor from an ArrayD
    ///
    /// This allows creating a tensor from an n-dimensional array directly.
    ///
    /// # Arguments
    ///
    /// * `data` - The n-dimensional array
    ///
    /// # Returns
    ///
    /// A new tensor containing the input data
    ///
    /// # Examples
    ///
    /// ```
    /// use nabla::tensor::Tensor;
    /// use ndarray::{Array, IxDyn};
    ///
    /// // Create a 2x3 array
    /// let data = Array::from_shape_vec(
    ///     IxDyn(&[2, 3]),
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    /// ).unwrap();
    /// 
    /// let tensor = Tensor::new_from_array(data);
    /// ```
    pub fn new_from_array(data: ArrayD<f32>) -> Self {
        Self {
            data,
            grad: Arc::new(Mutex::new(None)),
            grad_fn: None,
            parents: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    
    #[test]
    fn test_new_tensor() {
        let data: Array2<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let t = Tensor::new(data.clone());
        assert_eq!(t.data, data.into_dyn());
        assert!(t.grad.lock().unwrap().is_none());
    }
    
    #[test]
    fn test_add_forward_and_backward() {
        // Initialize input tensors
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Test forward operation
        let z = Tensor::add(&a, &b);
        assert_eq!(z.data, arr2(&[[6.0, 8.0], [10.0, 12.0]]).into_dyn());
        
        // Test backpropagation
        z.backward(None);
        
        // The gradient should be 1.0 everywhere
        let expected_grad: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> = Array2::ones((2, 2)).into_dyn();
        assert_eq!(a.grad.lock().unwrap().clone().unwrap(), expected_grad);
        assert_eq!(b.grad.lock().unwrap().clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_matmul_forward_and_backward() {
        // Initialize input tensors
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Test forward operation
        let z = Tensor::matmul(&a, &b);
        assert_eq!(z.data, arr2(&[[19.0, 22.0], [43.0, 50.0]]).into_dyn());
        
        // Test backpropagation
        z.backward(None);
        
        // Get the gradients
        let a_grad = a.grad.lock().unwrap().clone().unwrap();
        let b_grad = b.grad.lock().unwrap().clone().unwrap();
        
        // For matrix multiplication with gradient of all 1s:
        // dA = dZ · B^T = [1 1; 1 1] · [5 7; 6 8]^T = [1 1; 1 1] · [5 6; 7 8] = [11 15; 11 15]
        let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]).into_dyn();
        assert_eq!(a_grad, expected_grad_a);
        
        // dB = A^T · dZ = [1 3; 2 4]^T · [1 1; 1 1] = [1 2; 3 4] · [1 1; 1 1] = [4 4; 6 6]
        let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]).into_dyn();
        assert_eq!(b_grad, expected_grad_b);
    }
    
    #[test]
    fn test_relu_forward_and_backward() {
        // Initialize input tensor
        let x = Tensor::new(arr2(&[[-1.0, 2.0], [-3.0, 4.0]]));
        
        // Test forward operation
        let y = x.relu();
        assert_eq!(y.data, arr2(&[[0.0, 2.0], [0.0, 4.0]]).into_dyn());
        
        // Test backpropagation
        y.backward(None);
        
        // For ReLU the gradient should be 1 where the input is positive, 0 otherwise
        let expected_grad = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();
        assert_eq!(x.grad.lock().unwrap().clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_transpose() {
        // Initialize input tensor
        let x = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        
        // Test forward operation
        let y = x.transpose();
        
        // Verify that the matrix is actually transposed
        assert_eq!(y.data, arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).into_dyn());
        
        // Test backpropagation
        y.backward(None);
        
        // The gradient should be 1.0 everywhere, but with the original shape
        let expected_grad: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> = Array2::ones((2, 3)).into_dyn();
        assert_eq!(x.grad.lock().unwrap().clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_matmul_with() {
        // Test for the matmul_with instance method
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Using the instance method
        let z1 = a.matmul_with(&b);
        
        // Using the static function
        let z2 = Tensor::matmul(&a, &b);
        
        // The results should be identical
        assert_eq!(z1.data, z2.data);
    }
    
    #[test]
    fn test_thread_safety() {
        // Test to verify safety in multi-thread contexts
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let tensor = Tensor::new(data.clone());
        
        // Clone the tensor to use it in another thread
        let tensor_clone = tensor.clone();
        
        // Create a new thread that accesses the tensor
        let handle = std::thread::spawn(move || {
            // Access the tensor in another thread
            tensor_clone.data.clone()
        });
        
        // Wait for the thread to complete and verify the result
        let result = handle.join().unwrap();
        assert_eq!(result, data.into_dyn());
    }
}

