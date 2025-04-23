use ndarray::Array2;
use std::rc::Rc;
use std::cell::RefCell;
use rayon::prelude::*;

// motore di autograd per il calcolo dei gradienti


/// Rappresenta un tensore con capacità di autograd
#[derive(Clone)]
pub struct Tensor {
    /// Dati del tensore (valori)
    pub data: Array2<f32>,
    /// Gradienti calcolati durante il backpropagation
    pub grad: Rc<RefCell<Option<Array2<f32>>>>,
    /// Funzione per calcolare il gradiente durante il backprop
    pub grad_fn: Option<Rc<dyn Fn(&Tensor, &Array2<f32>)>>,
    /// Tensori padre da cui questo tensore è stato creato
    pub parents: Vec<Tensor>,
}

impl Tensor {
    /// Crea un nuovo tensore con i dati specificati
    pub fn new(data: Array2<f32>) -> Self {
        Tensor { 
            data, 
            grad: Rc::new(RefCell::new(None)), 
            grad_fn: None, 
            parents: vec![] 
        }
    }

    /// Crea un tensore con una funzione di gradiente
    pub fn with_grad_fn(data: Array2<f32>, parents: Vec<Tensor>, grad_fn: Rc<dyn Fn(&Tensor, &Array2<f32>)>) -> Self {
        Self {
            data,
            grad: Rc::new(RefCell::new(None)),
            grad_fn: Some(grad_fn),
            parents,
        }
    }

    /// Esegue il backpropagation a partire da questo tensore
    pub fn backward(&self, grad_output: Option<Array2<f32>>) {
        // Se non viene fornito un gradiente, usa un tensore di tutti 1
        let grad = grad_output.unwrap_or_else(|| Array2::ones(self.data.raw_dim()));
        
        // Aggiorna il gradiente di questo tensore
        self.update_grad(&grad);
        
        // Propaga il gradiente ai tensori padre, se presente una grad_fn
        if let Some(ref grad_fn) = self.grad_fn {
            // Applica la funzione di gradiente
            grad_fn(self, &grad);
        }
    }
    
    /// Aggiorna il gradiente accumulato per questo tensore
    fn update_grad(&self, grad: &Array2<f32>) {
        let mut borrowed_grad = self.grad.borrow_mut();
        if let Some(ref existing_grad) = *borrowed_grad {
            // Accumula il gradiente esistente
            *borrowed_grad = Some(existing_grad + grad);
        } else {
            // Imposta il gradiente se non esiste
            *borrowed_grad = Some(grad.clone());
        }
    }

    // ===== OPERAZIONI SUI TENSORI =====

    /// Somma due tensori elemento per elemento
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        // Calcola il risultato forward
        let data = &a.data + &b.data;
        
        // Clona i tensori parent per la chiusura
        let a_clone = a.clone();
        let b_clone = b.clone();

        // Crea un nuovo tensore con la funzione di gradiente
        Tensor::with_grad_fn(data, vec![a.clone(), b.clone()], Rc::new(move |_, grad| {
            // Il gradiente della somma si propaga identico a entrambi gli input
            a_clone.update_grad(grad);
            b_clone.update_grad(grad);
        }))
    }

    /// Moltiplica due tensori (moltiplicazione matriciale)
    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        // La moltiplicazione di matrici utilizza BLAS quando disponibile
        let data = a.data.dot(&b.data);
        
        // Clona i tensori parent per la chiusura
        let a_clone = a.clone();
        let b_clone = b.clone();
    
        Tensor::with_grad_fn(
            data,
            vec![a.clone(), b.clone()],
            Rc::new(move |_, grad| {
                // Per la moltiplicazione matriciale:
                // dL/dA = dL/dZ · B^T
                // dL/dB = A^T · dL/dZ
                
                let grad_a = grad.dot(&b_clone.data.t());
                let grad_b = a_clone.data.t().dot(grad);
                
                // Aggiorna i gradienti di A e B
                a_clone.update_grad(&grad_a);
                b_clone.update_grad(&grad_b);
            }),
        )
    }

    /// Applica la funzione ReLU al tensore
    pub fn relu(x: &Tensor) -> Tensor {
        // Calcola ReLU in modo parallelo utilizzando Rayon
        // Convertiamo in vettore per usare Rayon, poi torniamo ad Array2
        let data_vec: Vec<f32> = x.data.iter().cloned().collect();
        let result_vec: Vec<f32> = data_vec.par_iter()
            .map(|&v| v.max(0.0))
            .collect();
        
        // Converte il risultato in Array2 con la stessa forma
        let result_data = Array2::from_shape_vec(x.data.raw_dim(), result_vec).unwrap();
        
        // Clona il tensore parent per la chiusura
        let x_clone = x.clone();
    
        Tensor::with_grad_fn(
            result_data,
            vec![x.clone()],
            Rc::new(move |_, grad| {
                // Crea una maschera per ReLU: 1 dove input > 0, 0 altrimenti
                let mask = x_clone.data.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                
                // Moltiplica il gradiente per la maschera
                let grad_input = grad * mask;
                
                // Aggiorna il gradiente
                x_clone.update_grad(&grad_input);
            }),
        )
    }

    /// Eleva al quadrato ogni elemento del tensore
    pub fn square(x: &Tensor) -> Tensor {
        // x² è calcolato come x * x
        let data = &x.data * &x.data;
        
        // Clona il tensore parent per la chiusura
        let x_clone = x.clone();
        
        Tensor::with_grad_fn(
            data,
            vec![x.clone()],
            Rc::new(move |_, grad| {
                // Il gradiente di x² è 2x
                let two_x = &x_clone.data * 2.0;
                let grad_input = grad * two_x;
                
                // Aggiorna il gradiente
                x_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Somma tutti gli elementi del tensore
    pub fn sum(x: &Tensor) -> Tensor {
        // Calcola la somma scalare di tutti gli elementi
        let sum_val = x.data.sum();
        let data = Array2::from_elem((1, 1), sum_val);
        
        // Clona il tensore parent per la chiusura
        let x_clone = x.clone();
        
        Tensor::with_grad_fn(
            data,
            vec![x.clone()],
            Rc::new(move |_, grad| {
                // Il gradiente della somma è un tensore di 1 moltiplicato per il gradiente in arrivo
                let grad_val = grad[[0, 0]];
                let grad_input = Array2::ones(x_clone.data.raw_dim()) * grad_val;
                
                // Aggiorna il gradiente
                x_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Calcola l'errore quadratico medio tra due tensori
    pub fn mse(output: &Tensor, target: &Tensor) -> Tensor {
        // Calcola la differenza
        let neg_target = Tensor::new(-1.0 * &target.data);
        let diff = Tensor::add(output, &neg_target);
        
        // Eleva al quadrato ogni elemento
        let squared = Tensor::square(&diff);
        
        // Somma e divide per il numero di elementi per ottenere la media
        let sum_squared = Tensor::sum(&squared);
        let n = (output.data.shape()[0] * output.data.shape()[1]) as f32;
        let mse_data = sum_squared.data / n;
        
        Tensor::new(mse_data)
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
        
        // Test operazione forward
        let z = Tensor::add(&a, &b);
        assert_eq!(z.data, arr2(&[[6.0, 8.0], [10.0, 12.0]]));
        
        // Test backpropagation
        z.backward(None);
        
        // Il gradiente dovrebbe essere 1.0 ovunque
        let expected_grad: Array2<f32> = Array2::ones((2, 2));
        assert_eq!(a.grad.borrow().as_ref().unwrap(), &expected_grad);
        assert_eq!(b.grad.borrow().as_ref().unwrap(), &expected_grad);
    }

    #[test]
    fn test_matmul_forward_and_backward() {
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Test operazione forward
        let z = Tensor::matmul(&a, &b);
        assert_eq!(z.data, arr2(&[[19.0, 22.0], [43.0, 50.0]]));
        
        // Test backpropagation
        z.backward(None);
        
        // Ottieni e verifica i gradienti
        let a_grad = a.grad.borrow().clone().unwrap();
        let b_grad = b.grad.borrow().clone().unwrap();
        
        // Per la moltiplicazione di matrici con gradiente di tutti 1:
        // dA = dZ · B^T = [1 1; 1 1] · [5 7; 6 8]^T = [1 1; 1 1] · [5 6; 7 8] = [11 15; 11 15]
        let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]);
        assert_eq!(a_grad, expected_grad_a);
        
        // dB = A^T · dZ = [1 3; 2 4]^T · [1 1; 1 1] = [1 2; 3 4] · [1 1; 1 1] = [4 4; 6 6]
        let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]);
        assert_eq!(b_grad, expected_grad_b);
    }

    #[test]
    fn test_relu_forward_and_backward() {
        let x = Tensor::new(arr2(&[[-1.0, 2.0], [-3.0, 4.0]]));
        
        // Test operazione forward
        let y = Tensor::relu(&x);
        assert_eq!(y.data, arr2(&[[0.0, 2.0], [0.0, 4.0]]));
        
        // Test backpropagation
        y.backward(None);
        
        // Per ReLU il gradiente dovrebbe essere 1 dove l'input è positivo, 0 altrimenti
        let expected_grad = arr2(&[[0.0, 1.0], [0.0, 1.0]]);
        assert_eq!(x.grad.borrow().as_ref().unwrap(), &expected_grad);
    }
}

