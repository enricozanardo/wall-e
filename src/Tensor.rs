use ndarray::Array2;
use std::rc::Rc;
use std::cell::RefCell;
use rayon::prelude::*;
use ndarray::{Zip, Axis};
use ndarray_parallel::prelude::*;


#[derive(Clone)]
pub struct Tensor {
    pub data: Array2<f32>,
    pub grad: Rc<RefCell<Option<Array2<f32>>>>,
    pub grad_fn: Option<Rc<dyn Fn(&Tensor, &Array2<f32>)>>,
    pub parents: Vec<Tensor>,
}

impl Tensor {
    pub fn new(data: Array2<f32>) -> Self {
        Tensor { data, grad: Rc::new(RefCell::new(None)), grad_fn: None, parents: vec![] }
    }

    pub fn with_grad_fn(data: Array2<f32>, parents: Vec<Tensor>, grad_fn: Rc<dyn Fn(&Tensor, &Array2<f32>)>) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: Some(grad_fn),
            parents,
        }
    }

    pub fn backward(&self, grad_output: Option<Array2<f32>>) {
        let grad = grad_output.unwrap_or_else(|| Array2::ones(self.data.raw_dim()));
        
        // Update gradient without nested borrows
        let mut borrowed_grad = self.grad.borrow_mut();
        match *borrowed_grad {
            Some(ref grad_val) => {
                // Create a new array with the sum
                let result = grad_val.clone() + &grad;
                *borrowed_grad = Some(result);
            },
            None => {
                *borrowed_grad = Some(grad.clone());
            }
        }
        drop(borrowed_grad); // Explicitly drop the borrow before calling grad_fn
        
        if let Some(ref grad_fn) = self.grad_fn {
            grad_fn(self, &grad);
        }
    }

    // Basic operations
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
       // Use parallel addition with operators which use SIMD under the hood
       let data = &a.data + &b.data;
       
       let a_ = a.clone();
       let b_ = b.clone();

       Tensor::with_grad_fn(data, vec![a_.clone(), b_.clone()], Rc::new(move |_, grad| {
           // Update a's gradient
           {
               let mut a_grad = a_.grad.borrow_mut();
               match *a_grad {
                   Some(ref existing) => {
                       // Addition is vectorized by ndarray
                       *a_grad = Some(existing + grad);
                   },
                   None => *a_grad = Some(grad.clone()),
               }
           }
           
           // Update b's gradient
           {
               let mut b_grad = b_.grad.borrow_mut();
               match *b_grad {
                   Some(ref existing) => {
                       // Addition is vectorized by ndarray
                       *b_grad = Some(existing + grad);
                   },
                   None => *b_grad = Some(grad.clone()),
               }
           }
       }))
    }

    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        // ndarray's dot product uses BLAS under the hood when available
        let data = a.data.dot(&b.data);
        let a_ = a.clone();
        let b_ = b.clone();
    
        Tensor::with_grad_fn(
            data,
            vec![a.clone(), b.clone()],
            Rc::new(move |_, grad| {
                let grad_a = grad.dot(&b_.data.t());
                let grad_b = a_.data.t().dot(grad);
                
                // Update a's gradient
                {
                    let mut a_grad = a_.grad.borrow_mut();
                    match *a_grad {
                        Some(ref existing) => {
                            *a_grad = Some(existing + &grad_a);
                        },
                        None => *a_grad = Some(grad_a),
                    }
                }
                
                // Update b's gradient
                {
                    let mut b_grad = b_.grad.borrow_mut();
                    match *b_grad {
                        Some(ref existing) => {
                            *b_grad = Some(existing + &grad_b);
                        },
                        None => *b_grad = Some(grad_b),
                    }
                }
            }),
        )
    }

    pub fn relu(x: &Tensor) -> Tensor {
        // Use parallelism through Rayon
        // Create a vectorized version using standard operations and Rayon
        let data_vec: Vec<f32> = x.data.iter().cloned().collect();
        let result_vec: Vec<f32> = data_vec.par_iter()
            .map(|&v| v.max(0.0))
            .collect();
        
        // Convert back to Array2 with the same shape
        let shape = x.data.raw_dim();
        let data = Array2::from_shape_vec(shape, result_vec).unwrap();
        
        let x_ = x.clone();
    
        Tensor::with_grad_fn(
            data,
            vec![x.clone()],
            Rc::new(move |_, grad| {
                // Parallel mask computation
                let data_vec: Vec<f32> = x_.data.iter().cloned().collect();
                let mask_vec: Vec<f32> = data_vec.par_iter()
                    .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                
                let mask = Array2::from_shape_vec(x_.data.raw_dim(), mask_vec).unwrap();
                
                // Element-wise multiplication uses SIMD
                let grad_input = grad * &mask;
                
                // Update gradient with proper scoping
                let mut x_grad = x_.grad.borrow_mut();
                match *x_grad {
                    Some(ref existing) => {
                        // Addition uses SIMD
                        *x_grad = Some(existing + &grad_input);
                    },
                    None => *x_grad = Some(grad_input),
                }
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn test_new_tensor() {
        let data: Array2<f32> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let t = Tensor::new(data.clone());
        assert_eq!(t.data, data);
        assert!(t.grad.borrow().is_none());
    }

    #[test]
    fn test_add_forward_and_backward() {
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        let z = Tensor::add(&a, &b);
        assert_eq!(z.data, arr2(&[[6.0, 8.0], [10.0, 12.0]]));
        z.backward(None);
        let expected_grad: Array2<f32> = Array2::ones((2, 2));
        assert_eq!(a.grad.borrow().as_ref().unwrap(), &expected_grad);
        assert_eq!(b.grad.borrow().as_ref().unwrap(), &expected_grad);
    }

    #[test]
    fn test_matmul_forward_and_backward() {
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        let z = Tensor::matmul(&a, &b);
        assert_eq!(z.data, arr2(&[[19.0, 22.0], [43.0, 50.0]]));
        z.backward(None);
        
        // Get the actual gradients and verify they're correct
        let a_grad = a.grad.borrow().clone().unwrap();
        let b_grad = b.grad.borrow().clone().unwrap();
        
        // Verify a_grad = grad.dot(b.T)
        // For a 2x2 ones matrix, dotting with b.T should give [[11, 15], [11, 15]]
        let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]);
        assert_eq!(a_grad, expected_grad_a);
        
        // Verify b_grad = a.T.dot(grad)
        // For a 2x2 ones matrix, a.T.dot should give [[4, 4], [6, 6]]
        let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]);
        assert_eq!(b_grad, expected_grad_b);
    }

    #[test]
    fn test_relu_forward_and_backward() {
        let x = Tensor::new(arr2(&[[-1.0, 2.0], [-3.0, 4.0]]));
        let y = Tensor::relu(&x);
        assert_eq!(y.data, arr2(&[[0.0, 2.0], [0.0, 4.0]]));
        y.backward(None);
        let expected_grad = arr2(&[[0.0, 1.0], [0.0, 1.0]]);
        assert_eq!(x.grad.borrow().as_ref().unwrap(), &expected_grad);
    }
}

