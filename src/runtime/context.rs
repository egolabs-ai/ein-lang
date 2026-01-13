//! Runtime context for Ein evaluation.

use crate::syntax::ast::{BinOp, Expr, ParamInit, Statement, TensorRef, TypeSpec};
use crate::tensor::einsum;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Optimizer types supported by Ein.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    Sgd,
    /// AdamW (Adam with decoupled weight decay)
    AdamW,
}

/// Runtime context holding tensors and state.
pub struct Runtime {
    device: Device,
    tensors: IndexMap<String, Tensor>,
    /// Variables (learnable parameters with gradients)
    params: IndexMap<String, Var>,
    /// Set of tensor names that are parameters
    param_names: HashSet<String>,
    /// Forward pass statements (for training)
    forward_statements: Vec<String>,
}

impl Runtime {
    /// Create a new runtime, using Metal GPU if available, otherwise CPU.
    pub fn new() -> Self {
        // Try Metal first (Apple Silicon), fall back to CPU
        #[cfg(feature = "metal")]
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;

        Self {
            device,
            tensors: IndexMap::new(),
            params: IndexMap::new(),
            param_names: HashSet::new(),
            forward_statements: Vec::new(),
        }
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Store a tensor.
    pub fn set_tensor(&mut self, name: &str, tensor: Tensor) {
        self.tensors.insert(name.to_string(), tensor);
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Check if a tensor is a learnable parameter.
    pub fn is_param(&self, name: &str) -> bool {
        self.param_names.contains(name)
    }

    /// Get all parameter names.
    pub fn param_names(&self) -> Vec<&String> {
        self.params.keys().collect()
    }

    /// Get a parameter variable by name.
    pub fn get_param(&self, name: &str) -> Option<&Var> {
        self.params.get(name)
    }

    /// Get all parameters as a Vec for optimization.
    pub fn all_params(&self) -> Vec<Var> {
        self.params.values().cloned().collect()
    }

    /// Create a parameter tensor with Xavier/Glorot initialization.
    fn init_xavier(&self, shape: &[usize]) -> Result<Tensor> {
        // Xavier initialization: stddev = sqrt(2 / (fan_in + fan_out))
        let fan_in = if shape.len() >= 2 { shape[shape.len() - 2] } else { shape[0] };
        let fan_out = if shape.len() >= 2 { shape[shape.len() - 1] } else { shape[0] };
        let stddev = (2.0 / (fan_in + fan_out) as f64).sqrt();

        // Generate random values with normal distribution
        let n_elements: usize = shape.iter().product();
        let data: Vec<f32> = (0..n_elements)
            .map(|i| {
                // Simple pseudo-random using deterministic sequence
                // For production, use rand crate
                let x = ((i as f64 * 0.1234567) % 1.0 - 0.5) * 2.0;
                (x * stddev) as f32
            })
            .collect();

        Tensor::new(data, &self.device)?.reshape(shape)
    }

    /// Create and register an embedding lookup table.
    /// Shape is [vocab_size, embed_dim] with Xavier initialization.
    fn create_embedding(&mut self, name: &str, vocab_size: usize, embed_dim: usize) -> Result<()> {
        let shape = [vocab_size, embed_dim];
        let tensor = self.init_xavier(&shape)?;

        // Create a Var for gradient tracking
        let var = Var::from_tensor(&tensor)?;

        // Store in both tensors and params
        self.tensors.insert(name.to_string(), var.as_tensor().clone());
        self.params.insert(name.to_string(), var);
        self.param_names.insert(name.to_string());

        Ok(())
    }

    /// Create and register a parameter.
    fn create_param(&mut self, name: &str, type_spec: &TypeSpec, init: &ParamInit) -> Result<()> {
        let shape: Vec<usize> = type_spec.shape.clone();

        // Create initial tensor based on initialization strategy
        let tensor = match init {
            ParamInit::Xavier => self.init_xavier(&shape)?,
            ParamInit::Zeros => Tensor::zeros(shape.as_slice(), DType::F32, &self.device)?,
            ParamInit::Normal(stddev) => {
                let n_elements: usize = shape.iter().product();
                let data: Vec<f32> = (0..n_elements)
                    .map(|i| {
                        let x = ((i as f64 * 0.1234567) % 1.0 - 0.5) * 2.0;
                        (x * *stddev) as f32
                    })
                    .collect();
                Tensor::new(data, &self.device)?.reshape(shape.as_slice())?
            }
            ParamInit::Uniform(low, high) => {
                let n_elements: usize = shape.iter().product();
                let range = high - low;
                let data: Vec<f32> = (0..n_elements)
                    .map(|i| {
                        let x = (i as f64 * 0.1234567) % 1.0;
                        (*low + x * range) as f32
                    })
                    .collect();
                Tensor::new(data, &self.device)?.reshape(shape.as_slice())?
            }
        };

        // Create a Var for gradient tracking
        let var = Var::from_tensor(&tensor)?;

        // Store in both tensors (for evaluation) and params (for optimization)
        self.tensors.insert(name.to_string(), var.as_tensor().clone());
        self.params.insert(name.to_string(), var);
        self.param_names.insert(name.to_string());

        Ok(())
    }

    /// Evaluate an expression and return the result tensor.
    pub fn eval_expr(&self, expr: &Expr) -> Result<Tensor> {
        match expr {
            Expr::Ref(tensor_ref) => self.eval_tensor_ref(tensor_ref),

            Expr::Join(terms) => self.eval_join(terms),

            Expr::Apply { func, args } => {
                // Evaluate all arguments
                let arg_tensors: Vec<Tensor> = args
                    .iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<Vec<_>>>()?;
                self.apply_function(func, &arg_tensors)
            }

            Expr::Literal(val) => Tensor::new(&[*val as f32], &self.device),

            Expr::BinOp { op, lhs, rhs } => {
                let left = self.eval_expr(lhs)?;
                let right = self.eval_expr(rhs)?;
                self.eval_binop(*op, &left, &right)
            }

            Expr::Array(elements) => self.eval_array(elements),
        }
    }

    /// Evaluate a binary operation.
    fn eval_binop(&self, op: BinOp, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        match op {
            BinOp::Add => lhs.broadcast_add(rhs),
            BinOp::Sub => lhs.broadcast_sub(rhs),
            BinOp::Mul => lhs.broadcast_mul(rhs),
            BinOp::Div => lhs.broadcast_div(rhs),
        }
    }

    /// Evaluate an array literal to a tensor.
    fn eval_array(&self, elements: &[Expr]) -> Result<Tensor> {
        if elements.is_empty() {
            return Tensor::new(&[] as &[f32], &self.device);
        }

        // Check if this is a nested array (2D+) or flat array (1D)
        let first = &elements[0];
        if matches!(first, Expr::Array(_)) {
            // Nested array - evaluate each row and stack
            let mut rows = Vec::new();
            for elem in elements {
                let row = self.eval_expr(elem)?;
                rows.push(row);
            }
            Tensor::stack(&rows, 0)
        } else {
            // Flat array of literals
            let mut values = Vec::new();
            for elem in elements {
                match elem {
                    Expr::Literal(v) => values.push(*v as f32),
                    _ => {
                        // Evaluate non-literal element
                        let t = self.eval_expr(elem)?;
                        let v: f32 = t.to_scalar()?;
                        values.push(v);
                    }
                }
            }
            Tensor::new(&values[..], &self.device)
        }
    }

    /// Evaluate a tensor reference.
    fn eval_tensor_ref(&self, tensor_ref: &TensorRef) -> Result<Tensor> {
        self.tensors.get(&tensor_ref.name).cloned().ok_or_else(|| {
            candle_core::Error::Msg(format!("Unknown tensor: {}", tensor_ref.name))
        })
    }

    /// Evaluate a join (product) of tensor references.
    fn eval_join(&self, terms: &[Expr]) -> Result<Tensor> {
        // Collect tensor refs and their indices
        let mut refs: Vec<(&TensorRef, &Tensor)> = Vec::new();

        for term in terms {
            match term {
                Expr::Ref(r) => {
                    let t = self.tensors.get(&r.name).ok_or_else(|| {
                        candle_core::Error::Msg(format!("Unknown tensor: {}", r.name))
                    })?;
                    refs.push((r, t));
                }
                _ => {
                    return Err(candle_core::Error::Msg(
                        "Join currently only supports tensor refs".into(),
                    ))
                }
            }
        }

        if refs.len() == 1 {
            return Ok(refs[0].1.clone());
        }

        if refs.len() == 2 {
            // Build einsum notation from indices
            let (ref1, tensor1) = refs[0];
            let (ref2, tensor2) = refs[1];

            let notation = build_einsum_notation_2(ref1, ref2);
            einsum(&notation, &[tensor1, tensor2])
        } else if refs.len() == 3 {
            // 3-tensor join: chain two 2-tensor contractions
            // Example: Derived[x,r,z] = State[x,r1,y] State[y,r2,z] Rules[r,r1,r2]
            // Step 1: Temp = State @ State (contract shared indices)
            // Step 2: Result = Temp @ Rules (contract remaining shared indices)
            let (ref1, tensor1) = refs[0];
            let (ref2, tensor2) = refs[1];
            let (ref3, tensor3) = refs[2];

            let (notation1, _intermediate_indices, notation2) =
                build_einsum_notation_3(ref1, ref2, ref3);

            // Step 1: Contract first two tensors
            let intermediate = einsum(&notation1, &[tensor1, tensor2])?;

            // Step 2: Contract intermediate with third tensor
            einsum(&notation2, &[&intermediate, tensor3])
        } else {
            Err(candle_core::Error::Msg(
                "Join of more than 3 tensors not yet supported".into(),
            ))
        }
    }

    /// Apply a built-in function.
    fn apply_function(&self, func: &str, args: &[Tensor]) -> Result<Tensor> {
        // Helper to get single argument
        let arg = || -> Result<&Tensor> {
            if args.len() != 1 {
                return Err(candle_core::Error::Msg(format!(
                    "{} expects 1 argument, got {}",
                    func,
                    args.len()
                )));
            }
            Ok(&args[0])
        };

        // Helper to get two arguments
        let args2 = || -> Result<(&Tensor, &Tensor)> {
            if args.len() != 2 {
                return Err(candle_core::Error::Msg(format!(
                    "{} expects 2 arguments, got {}",
                    func,
                    args.len()
                )));
            }
            Ok((&args[0], &args[1]))
        };

        match func {
            // Activations (single argument)
            "sigmoid" => candle_nn::ops::sigmoid(arg()?),
            "relu" => arg()?.relu(),
            "gelu" => arg()?.gelu_erf(),
            "tanh" => arg()?.tanh(),
            "step" => {
                let a = arg()?;
                let zero = Tensor::zeros_like(a)?;
                let cmp = a.gt(&zero)?;
                cmp.to_dtype(DType::F32)
            }

            // Softmax (single argument, applies on last axis)
            "softmax" => candle_nn::ops::softmax(arg()?, candle_core::D::Minus1),

            // Math functions (single argument)
            "sqrt" => arg()?.sqrt(),
            "exp" => arg()?.exp(),
            "log" => arg()?.log(),

            // Reductions
            // sum(x) - full reduction to scalar
            // sum(x, dim) - sum along dimension, keeping dims
            "sum" => {
                if args.len() == 1 {
                    args[0].sum_all()
                } else if args.len() == 2 {
                    let dim = if args[1].dims().is_empty() {
                        args[1].to_scalar::<f32>()? as usize
                    } else {
                        args[1].flatten_all()?.to_vec1::<f32>()?[0] as usize
                    };
                    args[0].sum_keepdim(dim)
                } else {
                    Err(candle_core::Error::Msg(
                        "sum expects 1 or 2 arguments".into()
                    ))
                }
            }

            // mean(x) - full reduction to scalar
            // mean(x, dim) - mean along dimension, keeping dims
            "mean" => {
                if args.len() == 1 {
                    args[0].mean_all()
                } else if args.len() == 2 {
                    let dim = if args[1].dims().is_empty() {
                        args[1].to_scalar::<f32>()? as usize
                    } else {
                        args[1].flatten_all()?.to_vec1::<f32>()?[0] as usize
                    };
                    args[0].mean_keepdim(dim)
                } else {
                    Err(candle_core::Error::Msg(
                        "mean expects 1 or 2 arguments".into()
                    ))
                }
            }

            // Trace: sum of diagonal elements for 2D matrix
            // trace(A) where A is [n, n] -> scalar
            "trace" => {
                let a = arg()?;
                let dims = a.dims();
                if dims.len() != 2 || dims[0] != dims[1] {
                    return Err(candle_core::Error::Msg(
                        format!("trace requires square matrix, got shape {:?}", dims),
                    ));
                }
                let n = dims[0];
                // Extract diagonal elements
                let mut diag_values: Vec<f32> = Vec::with_capacity(n);
                for i in 0..n {
                    let row = a.get(i)?;
                    let val: f32 = row.get(i)?.to_scalar()?;
                    diag_values.push(val);
                }
                let diag_tensor = Tensor::new(diag_values, a.device())?;
                diag_tensor.sum_all()
            }

            // Diagonal extraction: diag(A) where A is [n, m] -> [min(n,m)]
            // Returns the main diagonal of a matrix
            "diag" => {
                let a = arg()?;
                let dims = a.dims();
                if dims.len() != 2 {
                    return Err(candle_core::Error::Msg(
                        format!("diag requires 2D matrix, got shape {:?}", dims),
                    ));
                }
                let n = dims[0].min(dims[1]);
                let mut diag_values: Vec<f32> = Vec::with_capacity(n);
                for i in 0..n {
                    let row = a.get(i)?;
                    let val: f32 = row.get(i)?.to_scalar()?;
                    diag_values.push(val);
                }
                Tensor::new(diag_values, a.device())
            }

            // Layer normalization (single argument, normalizes last dimension)
            "lnorm" => {
                let a = arg()?;
                let mean = a.mean_keepdim(candle_core::D::Minus1)?;
                let centered = a.broadcast_sub(&mean)?;
                let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
                let std = (var + 1e-5)?.sqrt()?;
                centered.broadcast_div(&std)
            }

            // Loss functions (two arguments)
            "mse" => {
                // Mean squared error: mean((pred - target)^2)
                let (pred, target) = args2()?;
                let diff = pred.broadcast_sub(target)?;
                diff.sqr()?.mean_all()
            }

            "cross_entropy" => {
                // Cross-entropy loss for classification
                // Args: logits [*, classes], targets [*] (as indices)
                // Handles multi-dimensional inputs by flattening to 2D/1D
                let (logits, targets) = args2()?;

                // Get number of classes from last dim of logits
                let logits_dims = logits.dims();
                let n_classes = logits_dims[logits_dims.len() - 1];

                // Flatten logits to [batch, classes] where batch = product of all dims except last
                let batch_size: usize = logits_dims[..logits_dims.len() - 1].iter().product();
                let logits_2d = logits.reshape(&[batch_size, n_classes])?;

                // Flatten targets to [batch]
                let targets_flat = targets.flatten_all()?;
                let targets_i64 = targets_flat.to_dtype(DType::I64)?;
                let targets_u32 = targets_i64.to_dtype(DType::U32)?;

                candle_nn::loss::cross_entropy(&logits_2d, &targets_u32)
            }

            // Embedding lookup: embed(embedding_table, indices)
            // Args: embedding_table [vocab_size, embed_dim], indices [seq_len] or [batch, seq_len]
            // Returns: [seq_len, embed_dim] or [batch, seq_len, embed_dim]
            "embed" => {
                let (table, indices) = args2()?;
                // Convert indices to U32 (via I64 since F32->U32 is not allowed)
                let indices_i64 = indices.to_dtype(DType::I64)?;
                let indices_u32 = indices_i64.to_dtype(DType::U32)?;

                // Get embedding dimension from table
                let embed_dim = table.dims()[1];
                let orig_shape = indices_u32.dims().to_vec();

                // Flatten indices to 1D for embedding lookup
                let flat_indices = indices_u32.flatten_all()?;
                let flat_result = table.embedding(&flat_indices)?;

                // Reshape result back to original shape + embed_dim
                let mut result_shape = orig_shape;
                result_shape.push(embed_dim);
                flat_result.reshape(&result_shape[..])
            }

            // Causal mask: causal_mask(x)
            // Takes any tensor and returns a lower-triangular mask of shape [n, n]
            // For 3D tensors [batch, seq, embed]: n = seq (second dimension)
            // For 2D tensors [seq, embed]: n = seq (first dimension)
            // Returns 1s on and below diagonal, 0s above diagonal.
            "causal_mask" => {
                let x = arg()?;
                let dims = x.dims();
                let n = if dims.len() >= 3 {
                    dims[1] // For [batch, seq, embed], use seq dimension
                } else if dims.len() == 2 {
                    dims[0] // For [seq, embed] or [seq, seq], use first dimension
                } else {
                    dims[0] // For 1D, use the only dimension
                };
                // Create lower triangular mask [n, n]
                let mut mask_data = vec![0.0f32; n * n];
                for i in 0..n {
                    for j in 0..=i {
                        mask_data[i * n + j] = 1.0;
                    }
                }
                Tensor::new(mask_data, &self.device)?.reshape(&[n, n])
            }

            // Mask fill: mask_fill(scores, mask, fill_value)
            // Where mask==0, replace scores with fill_value (typically -inf for attention)
            // mask should be 0/1 values, scores and mask should be broadcastable
            "mask_fill" => {
                if args.len() != 3 {
                    return Err(candle_core::Error::Msg(format!(
                        "mask_fill expects 3 arguments, got {}",
                        args.len()
                    )));
                }
                let scores = &args[0];
                let mask = &args[1];
                let fill_value = &args[2];

                // fill_value can be a scalar or 1-element tensor
                let fill: f32 = if fill_value.dims().is_empty() {
                    fill_value.to_scalar()?
                } else {
                    // Handle 1-element tensors
                    let flat = fill_value.flatten_all()?;
                    flat.to_vec1::<f32>()?[0]
                };

                // Use where_cond to avoid inf * 0 = NaN issue
                // Broadcast mask to match scores shape
                let mask_bool = mask.broadcast_as(scores.shape())?;
                let mask_bool = mask_bool.gt(&Tensor::zeros_like(&mask_bool)?)?;
                let fill_tensor = Tensor::full(fill, scores.shape(), &self.device)?;
                mask_bool.where_cond(scores, &fill_tensor)
            }

            // Negative infinity constant: neg_inf()
            // Returns a scalar -inf value for masking attention scores
            "neg_inf" => {
                Tensor::new(&[f32::NEG_INFINITY], &self.device)
            }

            // Size: size(x, dim)
            // Returns the size of tensor x along dimension dim.
            // Usage: B = size(X, 0) -- get batch size (dim 0)
            //        S = size(X, 1) -- get sequence length (dim 1)
            "size" => {
                let (x, dim_tensor) = args2()?;
                let dim = if dim_tensor.dims().is_empty() {
                    dim_tensor.to_scalar::<f32>()? as usize
                } else {
                    dim_tensor.flatten_all()?.to_vec1::<f32>()?[0] as usize
                };
                let size = x.dims()[dim];
                Tensor::new(&[size as f32], &self.device)
            }

            // Arange: arange(x)
            // Creates position indices [0, 1, 2, ..., n-1] for positional embeddings.
            // For 3D tensors [batch, seq, embed]: n = seq (second dimension)
            // For 2D tensors [seq, embed]: n = seq (first dimension)
            // For 1D tensors [n]: n = first dimension
            "arange" => {
                let x = arg()?;
                let dims = x.dims();
                let n = if dims.len() >= 3 {
                    dims[1] // For [batch, seq, embed], use seq dimension
                } else {
                    dims[0] // For [seq, embed] or [n], use first dimension
                };
                let positions: Vec<f32> = (0..n).map(|i| i as f32).collect();
                Tensor::new(positions, &self.device)
            }

            // Reshape: reshape(x, dim1, dim2, ...)
            // Reshape tensor to new dimensions. Total elements must match.
            // Usage: Y = reshape(X, 2, 4, 8) -- reshape X to [2, 4, 8]
            "reshape" => {
                if args.len() < 2 {
                    return Err(candle_core::Error::Msg(
                        "reshape requires tensor and at least one dimension".into()
                    ));
                }
                let x = &args[0];
                let mut new_shape: Vec<i64> = Vec::new();
                for t in &args[1..] {
                    // Try to extract scalar value (works for 0-dim or 1-element tensors)
                    let val = if t.dims().is_empty() {
                        t.to_scalar::<f32>()?
                    } else {
                        t.flatten_all()?.to_vec1::<f32>()?[0]
                    };
                    new_shape.push(val as i64);
                }

                // Handle -1 dimension (infer from total elements)
                let neg_count = new_shape.iter().filter(|&&d| d < 0).count();
                if neg_count > 1 {
                    return Err(candle_core::Error::Msg(
                        "reshape can only have one inferred dimension (-1)".into()
                    ));
                }

                let final_shape: Vec<usize> = if neg_count == 1 {
                    let total_elements: usize = x.elem_count();
                    let known_product: i64 = new_shape.iter().filter(|&&d| d > 0).product();
                    let inferred = total_elements as i64 / known_product;
                    new_shape.iter().map(|&d| {
                        if d < 0 { inferred as usize } else { d as usize }
                    }).collect()
                } else {
                    new_shape.iter().map(|&d| d as usize).collect()
                };

                x.reshape(final_shape.as_slice())
            }

            // Transpose: transpose(x, dim1, dim2)
            // Swap two dimensions of a tensor.
            // Usage: Y = transpose(X, 1, 2) -- swap dims 1 and 2
            "transpose" => {
                if args.len() != 3 {
                    return Err(candle_core::Error::Msg(
                        "transpose requires tensor and two dimension indices".into()
                    ));
                }
                let x = &args[0];
                // Handle both scalar and 1-element tensors for dims
                let get_dim = |t: &Tensor| -> Result<usize> {
                    let val = if t.dims().is_empty() {
                        t.to_scalar::<f32>()?
                    } else {
                        t.flatten_all()?.to_vec1::<f32>()?[0]
                    };
                    Ok(val as usize)
                };
                let dim1 = get_dim(&args[1])?;
                let dim2 = get_dim(&args[2])?;
                x.transpose(dim1, dim2)
            }

            // View: view(x, dim1, dim2, ...) - alias for reshape
            // Provided for PyTorch compatibility.
            "view" => {
                if args.len() < 2 {
                    return Err(candle_core::Error::Msg(
                        "view requires tensor and at least one dimension".into()
                    ));
                }
                let x = &args[0];
                let new_shape: Vec<usize> = args[1..].iter()
                    .filter_map(|t| t.to_scalar::<f32>().ok().map(|v| v as usize))
                    .collect();
                if new_shape.len() != args.len() - 1 {
                    return Err(candle_core::Error::Msg(
                        "view dimensions must be scalar integers".into()
                    ));
                }
                x.reshape(new_shape.as_slice())
            }

            // Sinusoidal positional encoding: sin_pos(positions, dim)
            // positions: [seq_len] tensor of position indices (from arange)
            // dim: [embed_dim] tensor (only its size is used)
            // Returns: [seq_len, embed_dim] sinusoidal positional encodings
            // Uses the formula from "Attention is All You Need":
            //   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            //   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
            "sin_pos" => {
                let (positions, dim_tensor) = args2()?;
                let seq_len = positions.dims()[0];
                let embed_dim = dim_tensor.dims()[dim_tensor.dims().len() - 1];

                // Get position values
                let pos_vec = positions.to_vec1::<f32>()?;

                // Create positional encoding matrix
                let mut pe_data = vec![0.0f32; seq_len * embed_dim];
                for (pos_idx, &pos) in pos_vec.iter().enumerate() {
                    for i in 0..embed_dim {
                        let div_term = (10000.0_f32).powf((i / 2 * 2) as f32 / embed_dim as f32);
                        let angle = pos / div_term;
                        pe_data[pos_idx * embed_dim + i] = if i % 2 == 0 {
                            angle.sin()
                        } else {
                            angle.cos()
                        };
                    }
                }

                Tensor::new(pe_data, &self.device)?.reshape(&[seq_len, embed_dim])
            }

            // Gradient control: detach(x)
            // Detaches tensor from computation graph, preventing gradients from flowing back.
            // Critical for JEPA target encoder and other self-supervised learning methods.
            "detach" => {
                let a = arg()?;
                Ok(a.detach())
            }

            // Element-wise operations
            // max(a, b): element-wise maximum
            "max" => {
                let (a, b) = args2()?;
                a.broadcast_maximum(b)
            }

            // min(a, b): element-wise minimum
            "min" => {
                let (a, b) = args2()?;
                a.broadcast_minimum(b)
            }

            // abs(x): absolute value
            "abs" => {
                arg()?.abs()
            }

            // Selection functions
            // argmax(x, dim): index of maximum value along dimension
            "argmax" => {
                let (x, dim_tensor) = args2()?;
                let dim = if dim_tensor.dims().is_empty() {
                    dim_tensor.to_scalar::<f32>()? as usize
                } else {
                    dim_tensor.flatten_all()?.to_vec1::<f32>()?[0] as usize
                };
                let result = x.argmax(dim)?;
                // Convert to f32 for consistency
                result.to_dtype(DType::F32)
            }

            // argmin(x, dim): index of minimum value along dimension
            "argmin" => {
                let (x, dim_tensor) = args2()?;
                let dim = if dim_tensor.dims().is_empty() {
                    dim_tensor.to_scalar::<f32>()? as usize
                } else {
                    dim_tensor.flatten_all()?.to_vec1::<f32>()?[0] as usize
                };
                let result = x.argmin(dim)?;
                // Convert to f32 for consistency
                result.to_dtype(DType::F32)
            }

            // Comparison functions (return 0/1 tensors)
            // gt(a, b): element-wise greater than
            "gt" => {
                let (a, b) = args2()?;
                let result = a.broadcast_gt(b)?;
                result.to_dtype(DType::F32)
            }

            // lt(a, b): element-wise less than
            "lt" => {
                let (a, b) = args2()?;
                let result = a.broadcast_lt(b)?;
                result.to_dtype(DType::F32)
            }

            // eq(a, b): element-wise equality
            "eq" => {
                let (a, b) = args2()?;
                let result = a.broadcast_eq(b)?;
                result.to_dtype(DType::F32)
            }

            // ge(a, b): element-wise greater than or equal
            "ge" => {
                let (a, b) = args2()?;
                let result = a.broadcast_ge(b)?;
                result.to_dtype(DType::F32)
            }

            // le(a, b): element-wise less than or equal
            "le" => {
                let (a, b) = args2()?;
                let result = a.broadcast_le(b)?;
                result.to_dtype(DType::F32)
            }

            // Conditional selection
            // where(cond, a, b): select from a where cond is true, else b
            "where" => {
                if args.len() != 3 {
                    return Err(candle_core::Error::Msg(format!(
                        "where expects 3 arguments, got {}",
                        args.len()
                    )));
                }
                let cond = &args[0];
                let a = &args[1];
                let b = &args[2];
                // Convert condition to bool (non-zero = true)
                let zero = Tensor::zeros_like(cond)?;
                let cond_bool = cond.broadcast_ne(&zero)?;
                cond_bool.where_cond(a, b)
            }

            _ => Err(candle_core::Error::Msg(format!(
                "Unknown function: {}",
                func
            ))),
        }
    }

    /// Evaluate an expression and store result in tensor with given name.
    /// Used by :forward command to evaluate deferred statements.
    pub fn eval_expr_to(&mut self, lhs: &TensorRef, rhs: &Expr) -> Result<()> {
        let result = self.eval_expr(rhs)?;
        self.set_tensor(&lhs.name, result);
        Ok(())
    }

    /// Add a deferred equation to the forward pass (for training).
    pub fn add_deferred_to_forward(&mut self, stmt_str: &str) {
        self.forward_statements.push(stmt_str.to_string());
    }

    pub fn eval_statement(&mut self, stmt: &Statement) -> Result<()> {
        match stmt {
            Statement::Equation { lhs, rhs } => {
                let result = self.eval_expr(rhs)?;
                self.set_tensor(&lhs.name, result);
                Ok(())
            }
            Statement::Fact(_) => {
                // TODO: Add to sparse relation
                Ok(())
            }
            Statement::Rule { .. } => {
                // TODO: Store rule for forward chaining
                Ok(())
            }
            Statement::Query(_) => {
                // TODO: Return query result
                Ok(())
            }
            Statement::ParamDecl {
                name,
                type_spec,
                init,
            } => {
                self.create_param(name, type_spec, init)
            }
            Statement::EmbeddingDecl {
                name,
                vocab_size,
                embed_dim,
            } => {
                self.create_embedding(name, *vocab_size, *embed_dim)
            }
            Statement::ForwardDecl { .. } => {
                // ForwardDecl is handled at the REPL/file loader level, not here
                Ok(())
            }
        }
    }

    /// Parse and evaluate a string.
    pub fn eval(&mut self, input: &str) -> Result<()> {
        let stmt = crate::syntax::parse(input)
            .map_err(|e| candle_core::Error::Msg(format!("Parse error: {}", e)))?;

        // Track equations for forward pass (not params or other statements)
        if matches!(stmt, Statement::Equation { .. }) {
            self.forward_statements.push(input.to_string());
        }

        self.eval_statement(&stmt)
    }

    /// Add a forward statement without evaluating (for training setup).
    pub fn add_forward_statement(&mut self, stmt: &str) {
        self.forward_statements.push(stmt.to_string());
    }

    /// Get all forward statements.
    pub fn forward_statements(&self) -> &[String] {
        &self.forward_statements
    }

    /// Clear forward statements.
    pub fn clear_forward_statements(&mut self) {
        self.forward_statements.clear();
    }

    /// Re-run the forward pass using current parameter values.
    /// This is used during training to recompute all tensors.
    pub fn run_forward_pass(&mut self) -> Result<()> {
        // First, update tensors with current param values
        for (name, var) in &self.params {
            self.tensors.insert(name.clone(), var.as_tensor().clone());
        }

        // Then re-evaluate all forward statements
        let statements = self.forward_statements.clone();
        for stmt_str in &statements {
            let stmt = crate::syntax::parse(stmt_str)
                .map_err(|e| candle_core::Error::Msg(format!("Parse error: {}", e)))?;
            if let Statement::Equation { lhs, rhs } = &stmt {
                let result = self.eval_expr(rhs)?;
                self.tensors.insert(lhs.name.clone(), result);
            }
        }

        Ok(())
    }

    /// Update parameters after a training step.
    /// Takes gradients and applies SGD update: param = param - lr * grad
    pub fn apply_gradients(&mut self, grads: &candle_core::backprop::GradStore, lr: f64) -> Result<()> {
        for (name, var) in &self.params {
            if let Some(grad) = grads.get(var.as_tensor()) {
                // SGD update: param = param - lr * grad
                let update = (var.as_tensor() - (grad * lr)?)?;
                var.set(&update)?;
                // Update the tensor copy as well
                self.tensors.insert(name.clone(), var.as_tensor().clone());
            }
        }
        Ok(())
    }

    /// Sync tensor copies from Var values after external optimizer updates (e.g., AdamW).
    /// Call this after using candle-nn optimizers that update Vars directly.
    pub fn sync_params_from_vars(&mut self) -> Result<()> {
        for (name, var) in &self.params {
            self.tensors.insert(name.clone(), var.as_tensor().clone());
        }
        Ok(())
    }

    /// Create an AdamW optimizer for the current parameters.
    pub fn create_adamw(&self, lr: f64, weight_decay: f64) -> Result<AdamW> {
        let params = self.all_params();
        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        };
        AdamW::new(params, adamw_params)
    }

    /// Save all tensors to a safetensors file.
    /// Only saves parameters (learnable weights), not intermediate tensors.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        // Collect all parameter tensors for saving
        let mut tensors_to_save: HashMap<String, Tensor> = HashMap::new();

        // Save parameters
        for (name, var) in &self.params {
            tensors_to_save.insert(name.clone(), var.as_tensor().clone());
        }

        if tensors_to_save.is_empty() {
            return Err(candle_core::Error::Msg("No parameters to save".into()));
        }

        // Use candle's safetensors save
        candle_core::safetensors::save(&tensors_to_save, path)?;
        Ok(())
    }

    /// Load tensors from a safetensors file.
    /// Updates existing Vars if they exist, or creates new tensors otherwise.
    pub fn load_checkpoint(&mut self, path: &str) -> Result<usize> {
        use candle_core::safetensors::load;
        use std::path::Path;

        let path = Path::new(path);
        let loaded = load(path, &self.device)?;

        let mut count = 0;
        for (name, tensor) in loaded {
            // If this is an existing parameter, update the Var
            if let Some(var) = self.params.get(&name) {
                var.set(&tensor)?;
                self.tensors.insert(name.clone(), tensor);
            } else {
                // Just set as a regular tensor
                self.tensors.insert(name, tensor);
            }
            count += 1;
        }

        Ok(count)
    }

    /// Get a map of parameter tensors for external saving.
    pub fn get_param_tensors(&self) -> HashMap<String, Tensor> {
        self.params
            .iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect()
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Character-level tokenizer for text processing.
#[derive(Default)]
pub struct Tokenizer {
    /// Character to index mapping
    pub char_to_idx: HashMap<char, usize>,
    /// Index to character mapping
    pub idx_to_char: Vec<char>,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Create a new tokenizer from text, building vocabulary from unique characters.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort(); // Deterministic ordering

        let mut char_to_idx = HashMap::new();
        for (i, c) in chars.iter().enumerate() {
            char_to_idx.insert(*c, i);
        }

        Self {
            char_to_idx,
            idx_to_char: chars.clone(),
            vocab_size: chars.len(),
        }
    }

    /// Encode text to indices.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    /// Decode indices to text.
    pub fn decode(&self, indices: &[usize]) -> String {
        indices.iter()
            .filter_map(|&i| self.idx_to_char.get(i))
            .collect()
    }
}

/// Text data for language modeling.
pub struct TextData {
    /// The tokenizer
    pub tokenizer: Tokenizer,
    /// Full encoded text as indices
    pub data: Vec<usize>,
    /// Sequence length for training
    pub seq_len: usize,
}

impl TextData {
    /// Load text from a string and create training data.
    pub fn from_text(text: &str, seq_len: usize) -> Self {
        let tokenizer = Tokenizer::from_text(text);
        let data = tokenizer.encode(text);
        Self { tokenizer, data, seq_len }
    }

    /// Get a batch of input/target sequences for training.
    /// Returns (inputs, targets) where each is [batch_size, seq_len]
    /// Uses deterministic positions (for initial batch)
    pub fn get_batch(&self, batch_size: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        self.get_batch_at_step(batch_size, 0, device)
    }

    /// Get a batch of input/target sequences at a specific training step.
    /// Different steps produce different random batches for proper training.
    pub fn get_batch_at_step(&self, batch_size: usize, step: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let max_start = self.data.len().saturating_sub(self.seq_len + 1);
        if max_start == 0 {
            return Err(candle_core::Error::Msg("Text too short for sequence length".into()));
        }

        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        // Use step to vary the random seed for each batch
        let seed = step.wrapping_mul(1103515245).wrapping_add(12345);

        for i in 0..batch_size {
            // Pseudo-random start position based on step and batch index
            let mut rng = seed.wrapping_add(i.wrapping_mul(7919));
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let start = rng % max_start;

            for j in 0..self.seq_len {
                inputs.push(self.data[start + j] as f32);
                targets.push(self.data[start + j + 1] as f32);
            }
        }

        let input_tensor = Tensor::new(inputs, device)?
            .reshape(&[batch_size, self.seq_len])?;
        let target_tensor = Tensor::new(targets, device)?
            .reshape(&[batch_size, self.seq_len])?;

        Ok((input_tensor, target_tensor))
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size
    }
}

/// Knowledge base data for link prediction training.
/// Stores triples (head, relation, tail) and provides batching with negative sampling.
pub struct KBData {
    /// Entity name to index mapping
    pub entity_to_idx: IndexMap<String, usize>,
    /// Relation name to index mapping
    pub relation_to_idx: IndexMap<String, usize>,
    /// Training triples as (head_idx, rel_idx, tail_idx)
    pub train_triples: Vec<(usize, usize, usize)>,
    /// Test triples for evaluation
    pub test_triples: Vec<(usize, usize, usize)>,
    /// Set of all true triples for filtered evaluation
    pub all_true_triples: HashSet<(usize, usize, usize)>,
    /// Embedding dimension
    pub dim: usize,
}

impl KBData {
    /// Load KB from TSV file format: head\trelation\ttail
    pub fn from_tsv(train_path: &str, test_path: Option<&str>, dim: usize) -> std::result::Result<Self, String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let mut entity_to_idx: IndexMap<String, usize> = IndexMap::new();
        let mut relation_to_idx: IndexMap<String, usize> = IndexMap::new();
        let mut train_triples = Vec::new();
        let mut all_true_triples = HashSet::new();

        // Load training triples
        let file = File::open(train_path)
            .map_err(|e| format!("Failed to open train file: {}", e))?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                continue;
            }

            let head = parts[0].to_string();
            let rel = parts[1].to_string();
            let tail = parts[2].to_string();

            let num_entities = entity_to_idx.len();
            let head_idx = *entity_to_idx.entry(head).or_insert(num_entities);

            let num_entities = entity_to_idx.len();
            let tail_idx = *entity_to_idx.entry(tail).or_insert(num_entities);

            let num_relations = relation_to_idx.len();
            let rel_idx = *relation_to_idx.entry(rel).or_insert(num_relations);

            train_triples.push((head_idx, rel_idx, tail_idx));
            all_true_triples.insert((head_idx, rel_idx, tail_idx));
        }

        // Load test triples if provided
        let mut test_triples = Vec::new();
        if let Some(test_path) = test_path {
            let file = File::open(test_path)
                .map_err(|e| format!("Failed to open test file: {}", e))?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() != 3 {
                    continue;
                }

                // Only include test triples where all entities/relations are known
                if let (Some(&head_idx), Some(&rel_idx), Some(&tail_idx)) = (
                    entity_to_idx.get(parts[0]),
                    relation_to_idx.get(parts[1]),
                    entity_to_idx.get(parts[2]),
                ) {
                    test_triples.push((head_idx, rel_idx, tail_idx));
                    all_true_triples.insert((head_idx, rel_idx, tail_idx));
                }
            }
        }

        Ok(Self {
            entity_to_idx,
            relation_to_idx,
            train_triples,
            test_triples,
            all_true_triples,
            dim,
        })
    }

    /// Get number of entities
    pub fn num_entities(&self) -> usize {
        self.entity_to_idx.len()
    }

    /// Get number of relations
    pub fn num_relations(&self) -> usize {
        self.relation_to_idx.len()
    }

    /// Get a training batch with negative sampling.
    /// Returns (heads, relations, tails, neg_tails) each of shape [batch_size]
    pub fn get_batch(&self, batch_size: usize, step: usize, neg_ratio: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let num_triples = self.train_triples.len();
        let num_entities = self.num_entities();

        let mut heads = Vec::with_capacity(batch_size);
        let mut relations = Vec::with_capacity(batch_size);
        let mut tails = Vec::with_capacity(batch_size);
        let mut neg_tails = Vec::with_capacity(batch_size * neg_ratio);

        // Pseudo-random sampling based on step
        let seed = step.wrapping_mul(1103515245).wrapping_add(12345);

        for i in 0..batch_size {
            let mut rng = seed.wrapping_add(i.wrapping_mul(7919));
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = rng % num_triples;

            let (h, r, t) = self.train_triples[idx];
            heads.push(h as f32);
            relations.push(r as f32);
            tails.push(t as f32);

            // Generate negative samples by corrupting tail
            for j in 0..neg_ratio {
                rng = rng.wrapping_add(j.wrapping_mul(1009));
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let neg_t = rng % num_entities;
                neg_tails.push(neg_t as f32);
            }
        }

        let heads_tensor = Tensor::new(heads, device)?.reshape(&[batch_size])?;
        let relations_tensor = Tensor::new(relations, device)?.reshape(&[batch_size])?;
        let tails_tensor = Tensor::new(tails, device)?.reshape(&[batch_size])?;
        let neg_tails_tensor = Tensor::new(neg_tails, device)?.reshape(&[batch_size, neg_ratio])?;

        Ok((heads_tensor, relations_tensor, tails_tensor, neg_tails_tensor))
    }

    /// Compute link prediction metrics on test set.
    /// Returns (MRR, Hits@1, Hits@3, Hits@10)
    pub fn evaluate(&self, entity_emb: &Tensor, relation_emb: &Tensor, num_samples: usize) -> Result<(f64, f64, f64, f64)> {
        let num_entities = self.num_entities();
        let test_count = num_samples.min(self.test_triples.len());

        if test_count == 0 {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        let mut sum_rr = 0.0;  // Sum of reciprocal ranks
        let mut hits_1 = 0usize;
        let mut hits_3 = 0usize;
        let mut hits_10 = 0usize;

        for i in 0..test_count {
            let (h, r, t) = self.test_triples[i];

            // Get embeddings for this triple
            let h_emb = entity_emb.get(h)?;    // [dim]
            let r_emb = relation_emb.get(r)?;  // [dim]

            // Score all possible tails: score = h * r * t (DistMult)
            // entity_emb is [num_entities, dim]
            let hr = (&h_emb * &r_emb)?;  // [dim]

            // Compute scores for all entities as tails
            // scores[i] = sum_d(hr[d] * entity_emb[i, d])
            let scores = entity_emb.matmul(&hr.unsqueeze(1)?)?; // [num_entities, 1]
            let scores = scores.squeeze(1)?;  // [num_entities]

            let scores_vec: Vec<f32> = scores.to_vec1()?;

            // Get score of true tail
            let true_score = scores_vec[t];

            // Compute filtered rank (ignoring other true triples)
            let mut rank = 1usize;
            for (ent_idx, &score) in scores_vec.iter().enumerate() {
                if ent_idx == t {
                    continue;
                }
                // Only count entities that aren't known true completions
                if !self.all_true_triples.contains(&(h, r, ent_idx)) {
                    if score > true_score {
                        rank += 1;
                    }
                }
            }

            sum_rr += 1.0 / rank as f64;
            if rank == 1 { hits_1 += 1; }
            if rank <= 3 { hits_3 += 1; }
            if rank <= 10 { hits_10 += 1; }
        }

        let mrr = sum_rr / test_count as f64;
        let h1 = hits_1 as f64 / test_count as f64;
        let h3 = hits_3 as f64 / test_count as f64;
        let h10 = hits_10 as f64 / test_count as f64;

        Ok((mrr, h1, h3, h10))
    }
}

/// Build einsum notation for a 2-tensor join.
///
/// Example: W[i,j] X[j] -> "ij,j->i"
/// - Indices appearing in both inputs are contracted (summed)
/// - Output indices are those appearing in only one input
fn build_einsum_notation_2(ref1: &TensorRef, ref2: &TensorRef) -> String {
    let idx1 = &ref1.indices;
    let idx2 = &ref2.indices;

    // Convert indices to single chars (a-z)
    let mut char_map: IndexMap<String, char> = IndexMap::new();
    let mut next_char = 'a';

    for idx in idx1.iter().chain(idx2.iter()) {
        if !char_map.contains_key(idx) {
            char_map.insert(idx.clone(), next_char);
            next_char = (next_char as u8 + 1) as char;
        }
    }

    // Build input specs
    let spec1: String = idx1.iter().map(|i| char_map[i]).collect();
    let spec2: String = idx2.iter().map(|i| char_map[i]).collect();

    // Find output indices (those not appearing in both)
    let mut output_indices: Vec<char> = Vec::new();
    for idx in idx1 {
        if !idx2.contains(idx) {
            output_indices.push(char_map[idx]);
        }
    }
    for idx in idx2 {
        if !idx1.contains(idx) {
            output_indices.push(char_map[idx]);
        }
    }

    let output_spec: String = output_indices.into_iter().collect();

    format!("{},{}->{}", spec1, spec2, output_spec)
}

/// Build einsum notation for a 2-tensor join with explicit output indices.
/// Returns (notation, output_indices) where output_indices are the symbolic names
/// of indices in the result tensor.
fn build_einsum_with_output_indices(
    idx1: &[String],
    idx2: &[String],
) -> (String, Vec<String>) {
    // Convert indices to single chars (a-z)
    let mut char_map: IndexMap<String, char> = IndexMap::new();
    let mut next_char = 'a';

    for idx in idx1.iter().chain(idx2.iter()) {
        if !char_map.contains_key(idx) {
            char_map.insert(idx.clone(), next_char);
            next_char = (next_char as u8 + 1) as char;
        }
    }

    // Build input specs
    let spec1: String = idx1.iter().map(|i| char_map[i]).collect();
    let spec2: String = idx2.iter().map(|i| char_map[i]).collect();

    // Find output indices (those not appearing in both) - preserve order
    let mut output_indices: Vec<String> = Vec::new();
    let mut output_chars: Vec<char> = Vec::new();
    for idx in idx1 {
        if !idx2.contains(idx) {
            output_indices.push(idx.clone());
            output_chars.push(char_map[idx]);
        }
    }
    for idx in idx2 {
        if !idx1.contains(idx) {
            output_indices.push(idx.clone());
            output_chars.push(char_map[idx]);
        }
    }

    let output_spec: String = output_chars.into_iter().collect();
    let notation = format!("{},{}->{}", spec1, spec2, output_spec);

    (notation, output_indices)
}

/// Build einsum notation for a 3-tensor join by chaining two contractions.
/// Returns the notations for both steps and the intermediate indices.
///
/// For: Derived[x,r,z] = State[x,r1,y] State[y,r2,z] Rules[r,r1,r2]
/// Step 1: Temp[x,r1,r2,z] = State[x,r1,y] @ State[y,r2,z]  (contract y)
/// Step 2: Derived[x,r,z] = Temp[x,r1,r2,z] @ Rules[r,r1,r2]  (contract r1,r2)
fn build_einsum_notation_3(
    ref1: &TensorRef,
    ref2: &TensorRef,
    ref3: &TensorRef,
) -> (String, Vec<String>, String) {
    let idx1 = &ref1.indices;
    let idx2 = &ref2.indices;
    let idx3 = &ref3.indices;

    // Step 1: Contract first two tensors
    let (notation1, intermediate_indices) = build_einsum_with_output_indices(idx1, idx2);

    // Step 2: Contract intermediate with third tensor
    let (notation2, _final_indices) = build_einsum_with_output_indices(&intermediate_indices, idx3);

    (notation1, intermediate_indices, notation2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_runtime() -> Runtime {
        Runtime::new()
    }

    #[test]
    fn test_trace() {
        let rt = create_test_runtime();
        // Create 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[3, 3]).unwrap();

        let result = rt.apply_function("trace", &[a]).unwrap();
        let val: f32 = result.to_scalar().unwrap();

        // trace = 1 + 5 + 9 = 15
        assert!((val - 15.0).abs() < 1e-5, "trace should be 15, got {}", val);
    }

    #[test]
    fn test_diag() {
        let rt = create_test_runtime();
        // Create 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[3, 3]).unwrap();

        let result = rt.apply_function("diag", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // diag = [1, 5, 9]
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 5.0).abs() < 1e-5);
        assert!((vals[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_full() {
        let rt = create_test_runtime();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();

        let result = rt.apply_function("sum", &[a]).unwrap();
        let val: f32 = result.to_scalar().unwrap();

        // sum = 1+2+3+4+5+6 = 21
        assert!((val - 21.0).abs() < 1e-5, "sum should be 21, got {}", val);
    }

    #[test]
    fn test_sum_dim() {
        let rt = create_test_runtime();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();
        let dim0 = Tensor::new(&[0.0f32], rt.device()).unwrap();
        let dim1 = Tensor::new(&[1.0f32], rt.device()).unwrap();

        // Sum along dim 0: [[1,2,3],[4,5,6]] -> [[5,7,9]]
        let result0 = rt.apply_function("sum", &[a.clone(), dim0]).unwrap();
        assert_eq!(result0.dims(), &[1, 3]);
        let vals0: Vec<f32> = result0.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals0[0] - 5.0).abs() < 1e-5);
        assert!((vals0[1] - 7.0).abs() < 1e-5);
        assert!((vals0[2] - 9.0).abs() < 1e-5);

        // Sum along dim 1: [[1,2,3],[4,5,6]] -> [[6],[15]]
        let result1 = rt.apply_function("sum", &[a, dim1]).unwrap();
        assert_eq!(result1.dims(), &[2, 1]);
        let vals1: Vec<f32> = result1.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals1[0] - 6.0).abs() < 1e-5);
        assert!((vals1[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_full() {
        let rt = create_test_runtime();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();

        let result = rt.apply_function("mean", &[a]).unwrap();
        let val: f32 = result.to_scalar().unwrap();

        // mean = 21/6 = 3.5
        assert!((val - 3.5).abs() < 1e-5, "mean should be 3.5, got {}", val);
    }

    #[test]
    fn test_mean_dim() {
        let rt = create_test_runtime();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();
        let dim1 = Tensor::new(&[1.0f32], rt.device()).unwrap();

        // Mean along dim 1: [[1,2,3],[4,5,6]] -> [[2],[5]]
        let result1 = rt.apply_function("mean", &[a, dim1]).unwrap();
        assert_eq!(result1.dims(), &[2, 1]);
        let vals1: Vec<f32> = result1.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals1[0] - 2.0).abs() < 1e-5);
        assert!((vals1[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_elementwise() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 5.0, 3.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 4.0], rt.device()).unwrap();

        let result = rt.apply_function("max", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 2.0).abs() < 1e-5); // max(1,2) = 2
        assert!((vals[1] - 5.0).abs() < 1e-5); // max(5,3) = 5
        assert!((vals[2] - 4.0).abs() < 1e-5); // max(3,4) = 4
    }

    #[test]
    fn test_min_elementwise() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 5.0, 3.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 4.0], rt.device()).unwrap();

        let result = rt.apply_function("min", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 1.0).abs() < 1e-5); // min(1,2) = 1
        assert!((vals[1] - 3.0).abs() < 1e-5); // min(5,3) = 3
        assert!((vals[2] - 3.0).abs() < 1e-5); // min(3,4) = 3
    }

    #[test]
    fn test_abs() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[-1.0f32, 2.0, -3.0], rt.device()).unwrap();

        let result = rt.apply_function("abs", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_trace_non_square_error() {
        let rt = create_test_runtime();
        // Create 2x3 non-square matrix
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();

        let result = rt.apply_function("trace", &[a]);
        assert!(result.is_err(), "trace should fail on non-square matrix");
    }

    #[test]
    fn test_diag_non_2d_error() {
        let rt = create_test_runtime();
        // Create 1D tensor
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("diag", &[a]);
        assert!(result.is_err(), "diag should fail on 1D tensor");
    }

    #[test]
    fn test_diag_rectangular() {
        let rt = create_test_runtime();
        // Create 2x4 rectangular matrix
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 4]).unwrap();

        let result = rt.apply_function("diag", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // diag of 2x4 should be [1, 6] (min(2,4)=2 elements)
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_trace_identity_matrix() {
        let rt = create_test_runtime();
        // Create 4x4 identity matrix
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[4, 4]).unwrap();

        let result = rt.apply_function("trace", &[a]).unwrap();
        let val: f32 = result.to_scalar().unwrap();

        assert!((val - 4.0).abs() < 1e-5, "trace of 4x4 identity should be 4");
    }

    #[test]
    fn test_trace_zeros() {
        let rt = create_test_runtime();
        // Create 3x3 zero matrix
        let a = Tensor::zeros(&[3, 3], candle_core::DType::F32, rt.device()).unwrap();

        let result = rt.apply_function("trace", &[a]).unwrap();
        let val: f32 = result.to_scalar().unwrap();

        assert!((val - 0.0).abs() < 1e-5, "trace of zero matrix should be 0");
    }

    #[test]
    fn test_triangle_counting() {
        let rt = create_test_runtime();
        // Adjacency matrix for graph with 1 triangle: 0-1-2-0
        // Also edge 2-3 (no triangle involving 3)
        //     0 1 2 3
        // 0 [ 0 1 1 0 ]
        // 1 [ 1 0 1 0 ]
        // 2 [ 1 1 0 1 ]
        // 3 [ 0 0 1 0 ]
        let data: Vec<f32> = vec![
            0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[4, 4]).unwrap();

        // A² = A @ A
        let a2 = a.matmul(&a).unwrap();
        // A³ = A² @ A
        let a3 = a2.matmul(&a).unwrap();

        let result = rt.apply_function("trace", &[a3]).unwrap();
        let trace_val: f32 = result.to_scalar().unwrap();

        // trace(A³)/6 = number of triangles
        // Each triangle is counted 6 times (3 vertices × 2 directions)
        let triangle_count = trace_val / 6.0;
        assert!((triangle_count - 1.0).abs() < 1e-5, "should have 1 triangle, got {}", triangle_count);
    }

    #[test]
    fn test_sum_3d_tensor() {
        let rt = create_test_runtime();
        // Create 2x2x2 tensor
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 2, 2]).unwrap();
        let dim0 = Tensor::new(&[0.0f32], rt.device()).unwrap();

        // Sum along dim 0
        let result = rt.apply_function("sum", &[a, dim0]).unwrap();
        assert_eq!(result.dims(), &[1, 2, 2]);

        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
        assert!((vals[0] - 6.0).abs() < 1e-5);
        assert!((vals[1] - 8.0).abs() < 1e-5);
        assert!((vals[2] - 10.0).abs() < 1e-5);
        assert!((vals[3] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_dim0() {
        let rt = create_test_runtime();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();
        let dim0 = Tensor::new(&[0.0f32], rt.device()).unwrap();

        // Mean along dim 0: [[1,2,3],[4,5,6]] -> [[2.5,3.5,4.5]]
        let result = rt.apply_function("mean", &[a, dim0]).unwrap();
        assert_eq!(result.dims(), &[1, 3]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 2.5).abs() < 1e-5);
        assert!((vals[1] - 3.5).abs() < 1e-5);
        assert!((vals[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_detach() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("detach", &[a.clone()]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // detach should return same values
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[0.0f32, 1.0, -1.0], rt.device()).unwrap();

        let result = rt.apply_function("sigmoid", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
        assert!((vals[0] - 0.5).abs() < 1e-5);
        assert!((vals[1] - 0.7310586).abs() < 1e-5);
        assert!((vals[2] - 0.2689414).abs() < 1e-5);
    }

    #[test]
    fn test_relu() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[-2.0f32, 0.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("relu", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5); // relu(-2) = 0
        assert!((vals[1] - 0.0).abs() < 1e-5); // relu(0) = 0
        assert!((vals[2] - 3.0).abs() < 1e-5); // relu(3) = 3
    }

    #[test]
    fn test_exp() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[0.0f32, 1.0, 2.0], rt.device()).unwrap();

        let result = rt.apply_function("exp", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 1.0).abs() < 1e-5);           // exp(0) = 1
        assert!((vals[1] - std::f32::consts::E).abs() < 1e-5); // exp(1) = e
        assert!((vals[2] - std::f32::consts::E.powi(2)).abs() < 1e-4); // exp(2) = e²
    }

    #[test]
    fn test_log() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, std::f32::consts::E, 10.0], rt.device()).unwrap();

        let result = rt.apply_function("log", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5);           // log(1) = 0
        assert!((vals[1] - 1.0).abs() < 1e-5);           // log(e) = 1
        assert!((vals[2] - 2.302585).abs() < 1e-4);      // log(10) ≈ 2.303
    }

    #[test]
    fn test_sqrt() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 4.0, 9.0], rt.device()).unwrap();

        let result = rt.apply_function("sqrt", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_tanh() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[0.0f32, 1.0, -1.0], rt.device()).unwrap();

        let result = rt.apply_function("tanh", &[a]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5);           // tanh(0) = 0
        assert!((vals[1] - 0.7615942).abs() < 1e-5);     // tanh(1) ≈ 0.762
        assert!((vals[2] - (-0.7615942)).abs() < 1e-5);  // tanh(-1) ≈ -0.762
    }

    // Phase 2: Selection & Comparison tests

    #[test]
    fn test_argmax() {
        let rt = create_test_runtime();
        // 2x3 matrix: [[1, 5, 3], [4, 2, 6]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();
        let dim1 = Tensor::new(&[1.0f32], rt.device()).unwrap();

        // argmax along dim 1: indices of max in each row
        // row 0: max at index 1 (value 5)
        // row 1: max at index 2 (value 6)
        let result = rt.apply_function("argmax", &[a, dim1]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-5); // index 1
        assert!((vals[1] - 2.0).abs() < 1e-5); // index 2
    }

    #[test]
    fn test_argmin() {
        let rt = create_test_runtime();
        // 2x3 matrix: [[1, 5, 3], [4, 2, 6]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();
        let dim1 = Tensor::new(&[1.0f32], rt.device()).unwrap();

        // argmin along dim 1: indices of min in each row
        // row 0: min at index 0 (value 1)
        // row 1: min at index 1 (value 2)
        let result = rt.apply_function("argmin", &[a, dim1]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 0.0).abs() < 1e-5); // index 0
        assert!((vals[1] - 1.0).abs() < 1e-5); // index 1
    }

    #[test]
    fn test_gt() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 5.0, 3.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("gt", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5); // 1 > 2 = false
        assert!((vals[1] - 1.0).abs() < 1e-5); // 5 > 3 = true
        assert!((vals[2] - 0.0).abs() < 1e-5); // 3 > 3 = false
    }

    #[test]
    fn test_lt() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 5.0, 3.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("lt", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 1.0).abs() < 1e-5); // 1 < 2 = true
        assert!((vals[1] - 0.0).abs() < 1e-5); // 5 < 3 = false
        assert!((vals[2] - 0.0).abs() < 1e-5); // 3 < 3 = false
    }

    #[test]
    fn test_eq_cmp() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 3.0, 5.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("eq", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5); // 1 == 2 = false
        assert!((vals[1] - 1.0).abs() < 1e-5); // 3 == 3 = true
        assert!((vals[2] - 0.0).abs() < 1e-5); // 5 == 3 = false
    }

    #[test]
    fn test_ge() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 5.0, 3.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("ge", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5); // 1 >= 2 = false
        assert!((vals[1] - 1.0).abs() < 1e-5); // 5 >= 3 = true
        assert!((vals[2] - 1.0).abs() < 1e-5); // 3 >= 3 = true
    }

    #[test]
    fn test_le() {
        let rt = create_test_runtime();
        let a = Tensor::new(&[1.0f32, 5.0, 3.0], rt.device()).unwrap();
        let b = Tensor::new(&[2.0f32, 3.0, 3.0], rt.device()).unwrap();

        let result = rt.apply_function("le", &[a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 1.0).abs() < 1e-5); // 1 <= 2 = true
        assert!((vals[1] - 0.0).abs() < 1e-5); // 5 <= 3 = false
        assert!((vals[2] - 1.0).abs() < 1e-5); // 3 <= 3 = true
    }

    #[test]
    fn test_where_cond() {
        let rt = create_test_runtime();
        let cond = Tensor::new(&[1.0f32, 0.0, 1.0, 0.0], rt.device()).unwrap();
        let a = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], rt.device()).unwrap();
        let b = Tensor::new(&[100.0f32, 200.0, 300.0, 400.0], rt.device()).unwrap();

        let result = rt.apply_function("where", &[cond, a, b]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // where cond=1, take from a; where cond=0, take from b
        assert!((vals[0] - 10.0).abs() < 1e-5);  // cond=1 -> a[0]=10
        assert!((vals[1] - 200.0).abs() < 1e-5); // cond=0 -> b[1]=200
        assert!((vals[2] - 30.0).abs() < 1e-5);  // cond=1 -> a[2]=30
        assert!((vals[3] - 400.0).abs() < 1e-5); // cond=0 -> b[3]=400
    }

    #[test]
    fn test_where_with_comparison() {
        let rt = create_test_runtime();
        // Combine gt with where: threshold operation
        let scores = Tensor::new(&[0.3f32, 0.7, 0.5, 0.9], rt.device()).unwrap();
        let threshold = Tensor::new(&[0.5f32, 0.5, 0.5, 0.5], rt.device()).unwrap();
        let ones = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], rt.device()).unwrap();
        let zeros = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], rt.device()).unwrap();

        // gt(scores, threshold) -> [0, 1, 0, 1]
        let cond = rt.apply_function("gt", &[scores, threshold]).unwrap();
        // where(cond, ones, zeros) -> [0, 1, 0, 1]
        let result = rt.apply_function("where", &[cond, ones, zeros]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-5); // 0.3 > 0.5 = false -> 0
        assert!((vals[1] - 1.0).abs() < 1e-5); // 0.7 > 0.5 = true -> 1
        assert!((vals[2] - 0.0).abs() < 1e-5); // 0.5 > 0.5 = false -> 0
        assert!((vals[3] - 1.0).abs() < 1e-5); // 0.9 > 0.5 = true -> 1
    }

    #[test]
    fn test_argmax_dim0() {
        let rt = create_test_runtime();
        // 2x3 matrix: [[1, 5, 3], [4, 2, 6]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let a = Tensor::new(data, rt.device()).unwrap().reshape(&[2, 3]).unwrap();
        let dim0 = Tensor::new(&[0.0f32], rt.device()).unwrap();

        // argmax along dim 0: indices of max in each column
        // col 0: max at index 1 (value 4 > 1)
        // col 1: max at index 0 (value 5 > 2)
        // col 2: max at index 1 (value 6 > 3)
        let result = rt.apply_function("argmax", &[a, dim0]).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-5); // col 0: row 1
        assert!((vals[1] - 0.0).abs() < 1e-5); // col 1: row 0
        assert!((vals[2] - 1.0).abs() < 1e-5); // col 2: row 1
    }

    #[test]
    fn test_find_closest_node() {
        let rt = create_test_runtime();
        // Proximity scores from node 0: [0, 0.9, 0.3, 0.7]
        // (distance to self is 0, closest other node is 1 with score 0.9)
        let proximity = Tensor::new(&[0.0f32, 0.9, 0.3, 0.7], rt.device()).unwrap();
        let dim0 = Tensor::new(&[0.0f32], rt.device()).unwrap();

        let result = rt.apply_function("argmax", &[proximity, dim0]).unwrap();
        let closest: f32 = result.to_scalar().unwrap();

        // Closest node is index 1 (score 0.9)
        assert!((closest - 1.0).abs() < 1e-5, "closest node should be 1, got {}", closest);
    }
}