use ndarray::{Array, Array2, Array3, ArrayD, Dimension, Ix2, Ix3, IxDyn};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

// motore di autograd per il calcolo dei gradienti


/// Rappresenta un tensore con capacità di autograd
#[derive(Clone)]
pub struct Tensor {
    /// Dati del tensore (valori)
    pub data: ArrayD<f32>,
    /// Gradienti calcolati durante il backpropagation
    pub grad: Arc<Mutex<Option<ArrayD<f32>>>>,
    /// Funzione per calcolare il gradiente durante il backprop
    pub grad_fn: Option<Arc<dyn Fn(&Tensor, &ArrayD<f32>) + Send + Sync>>,
    /// Tensori padre da cui questo tensore è stato creato
    pub parents: Vec<Tensor>,
}

impl Tensor {
    /// Crea un nuovo tensore con i dati specificati (Array2)
    pub fn new(data: Array2<f32>) -> Self {
        Tensor { 
            data: data.into_dyn(), 
            grad: Arc::new(Mutex::new(None)), 
            grad_fn: None, 
            parents: vec![] 
        }
    }
    
    /// Crea un nuovo tensore con i dati specificati (Array3)
    pub fn new_3d(data: Array3<f32>) -> Self {
        Tensor { 
            data: data.into_dyn(), 
            grad: Arc::new(Mutex::new(None)), 
            grad_fn: None, 
            parents: vec![] 
        }
    }

    /// Crea un tensore con una funzione di gradiente
    pub fn with_grad_fn(data: ArrayD<f32>, parents: Vec<Tensor>, grad_fn: Arc<dyn Fn(&Tensor, &ArrayD<f32>) + Send + Sync>) -> Self {
        Self {
            data,
            grad: Arc::new(Mutex::new(None)),
            grad_fn: Some(grad_fn),
            parents,
        }
    }

    /// Esegue il backpropagation a partire da questo tensore
    pub fn backward(&self, grad_output: Option<ArrayD<f32>>) {
        // Se non viene fornito un gradiente, usa un tensore di tutti 1
        let grad = grad_output.unwrap_or_else(|| Array::ones(self.data.raw_dim()));
        
        // Aggiorna il gradiente di questo tensore
        self.update_grad(&grad);
        
        // Propaga il gradiente ai tensori padre, se presente una grad_fn
        if let Some(ref grad_fn) = self.grad_fn {
            // Applica la funzione di gradiente
            grad_fn(self, &grad);
        }
    }
    
    /// Aggiorna il gradiente accumulato per questo tensore
    fn update_grad(&self, grad: &ArrayD<f32>) {
        let mut locked_grad = self.grad.lock().unwrap();
        if let Some(ref existing_grad) = *locked_grad {
            // Accumula il gradiente esistente
            *locked_grad = Some(existing_grad + grad);
        } else {
            // Imposta il gradiente se non esiste
            *locked_grad = Some(grad.clone());
        }
    }

    // ===== OPERAZIONI SUI TENSORI =====

    /// Somma due tensori elemento per elemento
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        // Verifica che le dimensioni siano compatibili
        assert_eq!(a.data.shape(), b.data.shape(), "Tensori di forme incompatibili per l'addizione");
        
        // Calcola il risultato forward usando Rayon
        let a_vec: Vec<f32> = a.data.iter().cloned().collect();
        let b_vec: Vec<f32> = b.data.iter().cloned().collect();
        
        // Parallelizza la somma elemento per elemento
        let result_vec: Vec<f32> = a_vec.par_iter()
            .zip(b_vec.par_iter())
            .map(|(&a_val, &b_val)| a_val + b_val)
            .collect();
        
        // Converte il risultato in Array con la stessa forma
        let data = Array::from_shape_vec(a.data.raw_dim(), result_vec).unwrap();
        
        // Clona i tensori parent per la chiusura
        let a_clone = a.clone();
        let b_clone = b.clone();

        // Crea un nuovo tensore con la funzione di gradiente
        Tensor::with_grad_fn(data, vec![a.clone(), b.clone()], Arc::new(move |_, grad| {
            // Il gradiente della somma si propaga identico a entrambi gli input
            a_clone.update_grad(grad);
            b_clone.update_grad(grad);
        }))
    }

    /// Moltiplica due tensori (moltiplicazione matriciale)
    pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
        // Convertiamo a Array2 per la moltiplicazione matriciale
        let a_2d = a.data.clone().into_dimensionality::<Ix2>().unwrap();
        let b_2d = b.data.clone().into_dimensionality::<Ix2>().unwrap();
        
        // La moltiplicazione di matrici utilizza BLAS quando disponibile
        let data = a_2d.dot(&b_2d).into_dyn();
        
        // Clona i tensori parent per la chiusura
        let a_clone = a.clone();
        let b_clone = b.clone();
    
        Tensor::with_grad_fn(
            data,
            vec![a.clone(), b.clone()],
            Arc::new(move |_, grad| {
                // Convertiamo a Array2 per la moltiplicazione matriciale
                let grad_2d = grad.clone().into_dimensionality::<Ix2>().unwrap();
                let a_2d = a_clone.data.clone().into_dimensionality::<Ix2>().unwrap();
                let b_2d = b_clone.data.clone().into_dimensionality::<Ix2>().unwrap();
                
                // Per la moltiplicazione matriciale:
                // dL/dA = dL/dZ · B^T
                // dL/dB = A^T · dL/dZ
                let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();
                let grad_b = a_2d.t().dot(&grad_2d).into_dyn();
                
                // Aggiorna i gradienti di A e B
                a_clone.update_grad(&grad_a);
                b_clone.update_grad(&grad_b);
            }),
        )
    }
    
    /// Calcola la trasposta di un Tensor
    pub fn transpose(&self) -> Tensor {
        // Convertiamo a Array2 per la trasposizione
        let data_2d = self.data.clone().into_dimensionality::<Ix2>().unwrap();
        
        // Ottiene la trasposta dell'array di dati
        let transposed_data = data_2d.t().to_owned().into_dyn();
        
        // Clona il tensore parent per la chiusura
        let self_clone = self.clone();
        
        Tensor::with_grad_fn(
            transposed_data,
            vec![self.clone()],
            Arc::new(move |_, grad| {
                // Convertiamo a Array2 per la trasposizione
                let grad_2d = grad.clone().into_dimensionality::<Ix2>().unwrap();
                
                // Il gradiente della trasposta è la trasposta del gradiente
                let grad_input = grad_2d.t().to_owned().into_dyn();
                
                // Aggiorna il gradiente
                self_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Moltiplica un tensore per un altro (moltiplicazione matriciale)
    /// Metodo di istanza per migliorare l'usabilità
    pub fn matmul_with(&self, other: &Tensor) -> Tensor {
        Tensor::matmul(self, other)
    }

    /// Applica la funzione ReLU al tensore
    pub fn relu(&self) -> Self {
        let result = self.data.mapv(|x| x.max(0.0));
        
        // Clona il tensore parent per la chiusura
        let self_clone = self.clone();
        
        Tensor::with_grad_fn(
            result,
            vec![self.clone()],
            Arc::new(move |_, grad| {
                // Il gradiente di ReLU è 1 se l'input è > 0, altrimenti 0
                let input_data = &self_clone.data;
                let mut grad_input = grad.clone();
                
                // Applicare la maschera al gradiente
                for (i, &val) in input_data.iter().enumerate() {
                    if val <= 0.0 {
                        grad_input.as_slice_mut().unwrap()[i] = 0.0;
                    }
                }
                
                // Aggiorna il gradiente
                self_clone.update_grad(&grad_input);
            }),
        )
    }

    /// Eleva al quadrato ogni elemento del tensore
    pub fn square(x: &Tensor) -> Tensor {
        // x² è calcolato come x * x utilizzando Rayon per parallelizzare
        // Convertiamo in vettore per usare Rayon, poi torniamo ad Array
        let data_vec: Vec<f32> = x.data.iter().cloned().collect();
        let result_vec: Vec<f32> = data_vec.par_iter()
            .map(|&v| v * v)
            .collect();
        
        // Converte il risultato in Array con la stessa forma
        let result_data = Array::from_shape_vec(x.data.raw_dim(), result_vec).unwrap();
        
        // Clona il tensore parent per la chiusura
        let x_clone = x.clone();
        
        Tensor::with_grad_fn(
            result_data,
            vec![x.clone()],
            Arc::new(move |_, grad| {
                // Il gradiente di x² è 2x
                let mut grad_input = grad.clone();
                let x_data = &x_clone.data;
                
                for (i, &x_val) in x_data.iter().enumerate() {
                    grad_input.as_slice_mut().unwrap()[i] *= 2.0 * x_val;
                }
                
                // Aggiorna il gradiente
                x_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Somma tutti gli elementi del tensore
    pub fn sum(x: &Tensor) -> Tensor {
        // Calcola la somma scalare di tutti gli elementi
        let sum_val = x.data.sum();
        let mut data = Array::zeros(IxDyn(&[1, 1]));
        data[IxDyn(&[0, 0])] = sum_val;
        
        // Clona il tensore parent per la chiusura
        let x_clone = x.clone();
        
        Tensor::with_grad_fn(
            data,
            vec![x.clone()],
            Arc::new(move |_, grad| {
                // Il gradiente della somma è un array di tutti 1 con la stessa forma dell'input
                let grad_val = grad[IxDyn(&[0, 0])];
                let grad_input = Array::from_elem(x_clone.data.raw_dim(), grad_val);
                
                // Aggiorna il gradiente
                x_clone.update_grad(&grad_input);
            }),
        )
    }
    
    /// Calcola l'errore quadratico medio (MSE) tra output e target
    pub fn mse(output: &Tensor, target: &Tensor) -> Tensor {
        // Verifica che le dimensioni siano compatibili
        assert_eq!(output.data.shape(), target.data.shape(), "Tensori di forme incompatibili per MSE");
        
        // Calcola la differenza
        let diff_vec: Vec<f32> = output.data.iter()
            .zip(target.data.iter())
            .map(|(&o, &t)| o - t)
            .collect();
        
        // Converte il risultato in Array con la stessa forma
        let diff_data = Array::from_shape_vec(output.data.raw_dim(), diff_vec).unwrap();
        
        // Calcola il quadrato delle differenze
        let squared_diff: Vec<f32> = diff_data.iter().map(|&d| d * d).collect();
        
        // Calcola la media
        let n = squared_diff.len() as f32;
        let mse_val = squared_diff.iter().sum::<f32>() / n;
        
        // Crea un Tensor scalare
        let mut data = Array::zeros(IxDyn(&[1, 1]));
        data[IxDyn(&[0, 0])] = mse_val;
        
        // Cloniamo i tensori parent per la chiusura
        let output_clone = output.clone();
        let target_clone = target.clone();
        
        Tensor::with_grad_fn(
            data,
            vec![output.clone(), target.clone()],
            Arc::new(move |_, grad| {
                // Il gradiente dell'MSE rispetto all'output è 2(output - target) / n
                let grad_val = grad[IxDyn(&[0, 0])];
                let n = output_clone.data.len() as f32;
                
                let grad_vec: Vec<f32> = output_clone.data.iter()
                    .zip(target_clone.data.iter())
                    .map(|(&o, &t)| 2.0 * (o - t) * grad_val / n)
                    .collect();
                
                // Converte il gradiente in Array con la stessa forma dell'output
                let grad_output = Array::from_shape_vec(output_clone.data.raw_dim(), grad_vec).unwrap();
                
                // Aggiorna il gradiente
                output_clone.update_grad(&grad_output);
                
                // Il gradiente rispetto al target è -grad(output)
                let grad_target_vec: Vec<f32> = grad_output.iter().map(|&g| -g).collect();
                let grad_target = Array::from_shape_vec(target_clone.data.raw_dim(), grad_target_vec).unwrap();
                
                // Aggiorna il gradiente
                target_clone.update_grad(&grad_target);
            }),
        )
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
        // Inizializza i tensori di input
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Test operazione forward
        let z = Tensor::add(&a, &b);
        assert_eq!(z.data, arr2(&[[6.0, 8.0], [10.0, 12.0]]).into_dyn());
        
        // Test backpropagation
        z.backward(None);
        
        // Il gradiente dovrebbe essere 1.0 ovunque
        let expected_grad = Array2::ones((2, 2)).into_dyn();
        assert_eq!(a.grad.lock().unwrap().clone().unwrap(), expected_grad);
        assert_eq!(b.grad.lock().unwrap().clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_matmul_forward_and_backward() {
        // Inizializza i tensori di input
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Test operazione forward
        let z = Tensor::matmul(&a, &b);
        assert_eq!(z.data, arr2(&[[19.0, 22.0], [43.0, 50.0]]).into_dyn());
        
        // Test backpropagation
        z.backward(None);
        
        // Ottieni i gradienti
        let a_grad = a.grad.lock().unwrap().clone().unwrap();
        let b_grad = b.grad.lock().unwrap().clone().unwrap();
        
        // Per la moltiplicazione di matrici con gradiente di tutti 1:
        // dA = dZ · B^T = [1 1; 1 1] · [5 7; 6 8]^T = [1 1; 1 1] · [5 6; 7 8] = [11 15; 11 15]
        let expected_grad_a = arr2(&[[11.0, 15.0], [11.0, 15.0]]).into_dyn();
        assert_eq!(a_grad, expected_grad_a);
        
        // dB = A^T · dZ = [1 3; 2 4]^T · [1 1; 1 1] = [1 2; 3 4] · [1 1; 1 1] = [4 4; 6 6]
        let expected_grad_b = arr2(&[[4.0, 4.0], [6.0, 6.0]]).into_dyn();
        assert_eq!(b_grad, expected_grad_b);
    }
    
    #[test]
    fn test_relu_forward_and_backward() {
        // Inizializza il tensore di input
        let x = Tensor::new(arr2(&[[-1.0, 2.0], [-3.0, 4.0]]));
        
        // Test operazione forward
        let y = x.relu();
        assert_eq!(y.data, arr2(&[[0.0, 2.0], [0.0, 4.0]]).into_dyn());
        
        // Test backpropagation
        y.backward(None);
        
        // Per ReLU il gradiente dovrebbe essere 1 dove l'input è positivo, 0 altrimenti
        let expected_grad = arr2(&[[0.0, 1.0], [0.0, 1.0]]).into_dyn();
        assert_eq!(x.grad.lock().unwrap().clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_transpose() {
        // Inizializza il tensore di input
        let x = Tensor::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        
        // Test operazione forward
        let y = x.transpose();
        
        // Verifica che la matrice sia effettivamente trasposta
        assert_eq!(y.data, arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).into_dyn());
        
        // Test backpropagation
        y.backward(None);
        
        // Il gradiente dovrebbe essere 1.0 ovunque, ma con la forma originale
        let expected_grad = Array2::ones((2, 3)).into_dyn();
        assert_eq!(x.grad.lock().unwrap().clone().unwrap(), expected_grad);
    }
    
    #[test]
    fn test_matmul_with() {
        // Test per il metodo di istanza matmul_with
        let a = Tensor::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        let b = Tensor::new(arr2(&[[5.0, 6.0], [7.0, 8.0]]));
        
        // Utilizzo del metodo di istanza
        let z1 = a.matmul_with(&b);
        
        // Utilizzo della funzione statica
        let z2 = Tensor::matmul(&a, &b);
        
        // I risultati dovrebbero essere identici
        assert_eq!(z1.data, z2.data);
    }
    
    #[test]
    fn test_thread_safety() {
        // Test per verificare la sicurezza in contesti multi-thread
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let tensor = Tensor::new(data.clone());
        
        // Clona il tensore per usarlo in un altro thread
        let tensor_clone = tensor.clone();
        
        // Crea un nuovo thread che accede al tensore
        let handle = std::thread::spawn(move || {
            // Accedi al tensore in un altro thread
            tensor_clone.data.clone()
        });
        
        // Attendi il completamento del thread e verifica il risultato
        let result = handle.join().unwrap();
        assert_eq!(result, data.into_dyn());
    }
}

