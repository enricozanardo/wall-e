/// Module for automatic differentiation and tensor operations.
///
/// This module provides a lightweight tensor library with automatic differentiation
/// capabilities similar to PyTorch or TensorFlow, but implemented natively in Rust.
/// It supports basic tensor operations and calculating gradients for backpropagation.
pub mod tensor;

/// Module for memory-optimized operations.
///
/// This module provides cache-efficient implementations of tensor operations
/// and utilities for optimizing memory access patterns to improve performance.
pub mod memory_opt;

// Re-export key components
pub use tensor::Tensor;