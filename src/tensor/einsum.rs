//! Einsum implementation for tensor contractions.
//!
//! Einsum notation: "ij,jk->ik" means contract index j
//! - Left of -> : input tensor indices
//! - Right of -> : output tensor indices
//! - Repeated indices are summed over (contracted)

use candle_core::{Result, Tensor};

/// Parse einsum notation into input specs and output spec.
/// Example: "ij,jk->ik" -> (["ij", "jk"], "ik")
fn parse_notation(notation: &str) -> (Vec<Vec<char>>, Vec<char>) {
    let parts: Vec<&str> = notation.split("->").collect();

    let inputs_str = parts[0];
    let output_str = if parts.len() > 1 { parts[1] } else { "" };

    let input_specs: Vec<Vec<char>> = inputs_str
        .split(',')
        .map(|s| s.chars().collect())
        .collect();

    let output_spec: Vec<char> = output_str.chars().collect();

    (input_specs, output_spec)
}

/// Execute einsum operation.
///
/// Currently supports common patterns by delegating to Candle operations.
/// This is a simplified implementation for learning purposes.
pub fn einsum(notation: &str, inputs: &[&Tensor]) -> Result<Tensor> {
    let (input_specs, output_spec) = parse_notation(notation);

    // For now, handle specific common cases
    // We'll expand this as needed

    match (input_specs.as_slice(), inputs) {
        // Matrix multiply: "ij,jk->ik"
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 2
                && b_spec.len() == 2
                && a_spec[1] == b_spec[0]
                && output_spec == vec![a_spec[0], b_spec[1]] =>
        {
            a.matmul(b)
        }

        // Matrix-vector multiply: "ij,j->i"
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 2
                && b_spec.len() == 1
                && a_spec[1] == b_spec[0]
                && output_spec == vec![a_spec[0]] =>
        {
            // b is [j], need [j, 1] for matmul, then squeeze
            let b_col = b.unsqueeze(1)?;
            let result = a.matmul(&b_col)?;
            result.squeeze(1)
        }

        // Outer product: "i,j->ij"
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 1
                && b_spec.len() == 1
                && a_spec[0] != b_spec[0]
                && output_spec == vec![a_spec[0], b_spec[0]] =>
        {
            // a is [i], b is [j]
            // outer product: a[:, None] * b[None, :]
            let a_col = a.unsqueeze(1)?; // [i, 1]
            let b_row = b.unsqueeze(0)?; // [1, j]
            a_col.broadcast_mul(&b_row)
        }

        // Trace: "ii->"
        ([a_spec], [a]) if a_spec.len() == 2 && a_spec[0] == a_spec[1] && output_spec.is_empty() => {
            // Sum of diagonal elements
            // Candle doesn't have diagonal(), so we implement it manually
            let shape = a.dims();
            let n = shape[0].min(shape[1]);
            let mut sum = 0.0f32;
            let data = a.to_vec2::<f32>()?;
            for i in 0..n {
                sum += data[i][i];
            }
            Tensor::new(&[sum], a.device())?.sum_all()
        }

        // A @ B^T pattern: "ij,kj->ik" (contract over second index of both)
        // Used in attention: Q @ K^T where Q[i,d], K[j,d] -> Scores[i,j]
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 2
                && b_spec.len() == 2
                && a_spec[1] == b_spec[1]  // Second indices match (contraction)
                && output_spec == vec![a_spec[0], b_spec[0]] =>
        {
            // a @ b.T
            let b_t = b.t()?;
            a.matmul(&b_t)
        }

        // A^T @ B pattern: "ji,jk->ik" (contract over first index of both)
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 2
                && b_spec.len() == 2
                && a_spec[0] == b_spec[0]  // First indices match (contraction)
                && output_spec == vec![a_spec[1], b_spec[1]] =>
        {
            // a.T @ b
            let a_t = a.t()?;
            a_t.matmul(b)
        }

        // Batched matmul: "bij,bjk->bik" (3D tensors, batch dimension preserved)
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 3
                && b_spec.len() == 3
                && a_spec[0] == b_spec[0]  // Batch dimension matches
                && a_spec[2] == b_spec[1]  // Inner dimensions match
                && output_spec == vec![a_spec[0], a_spec[1], b_spec[2]] =>
        {
            // Candle's matmul handles batched case
            a.matmul(b)
        }

        // Batched A @ B^T: "bij,bkj->bik" (attention scores in batched form)
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 3
                && b_spec.len() == 3
                && a_spec[0] == b_spec[0]  // Batch dimension matches
                && a_spec[2] == b_spec[2]  // Contract over last dimension
                && output_spec == vec![a_spec[0], a_spec[1], b_spec[1]] =>
        {
            // Transpose last two dims of b, then matmul
            let b_t = b.transpose(1, 2)?;
            a.matmul(&b_t)
        }

        // Batched A @ B^T with wrong output (legacy einsum builder bug):
        // Pattern "abc,adc->bd" should actually be "abc,adc->abd"
        // This happens when batch dim 'a' appears in both inputs but not output
        // Fix: detect this case and produce correct batched result
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 3
                && b_spec.len() == 3
                && a_spec[0] == b_spec[0]  // Batch dimension matches
                && a_spec[2] == b_spec[2]  // Contract over last dimension
                && output_spec.len() == 2
                && output_spec == vec![a_spec[1], b_spec[1]] =>
        {
            // Same as above but output was computed incorrectly (missing batch dim)
            // Transpose last two dims of b, then matmul
            let b_t = b.transpose(1, 2)?;
            a.matmul(&b_t)
        }

        // Batched matmul with wrong output: "abc,acd->bd" should be "abc,acd->abd"
        // This is Attn @ V pattern where Attn[b,s,t], V[b,t,d] -> Out[b,s,d]
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 3
                && b_spec.len() == 3
                && a_spec[0] == b_spec[0]  // Batch dimension matches
                && a_spec[2] == b_spec[1]  // Inner dimensions match (t contracted)
                && output_spec.len() == 2
                && output_spec == vec![a_spec[1], b_spec[2]] =>
        {
            // Batched matmul: a @ b
            a.matmul(b)
        }

        // 4D batched attention scores: "abcd,abed->ce" should be "abcd,abed->abce"
        // This is Q[b,h,i,k] @ K[b,h,j,k] -> Scores[b,h,i,j]
        // Pattern: first two dims match (batch, heads), last dim contracted
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 4
                && b_spec.len() == 4
                && a_spec[0] == b_spec[0]  // Batch dim matches
                && a_spec[1] == b_spec[1]  // Head dim matches
                && a_spec[3] == b_spec[3]  // Contract over last dim (k)
                && output_spec.len() == 2
                && output_spec == vec![a_spec[2], b_spec[2]] =>
        {
            // 4D batched Q @ K^T: transpose last two dims of b, then matmul
            // Need contiguous() after transpose for matmul to work
            let a_c = a.contiguous()?;
            let b_t = b.transpose(2, 3)?.contiguous()?;
            a_c.matmul(&b_t)
        }

        // 4D batched attention output: "abcd,abde->ce" should be "abcd,abde->abce"
        // This is Attn[b,h,s,t] @ V[b,h,t,k] -> Out[b,h,s,k]
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 4
                && b_spec.len() == 4
                && a_spec[0] == b_spec[0]  // Batch dim matches
                && a_spec[1] == b_spec[1]  // Head dim matches
                && a_spec[3] == b_spec[2]  // Contract over t dim
                && output_spec.len() == 2
                && output_spec == vec![a_spec[2], b_spec[3]] =>
        {
            // 4D batched matmul: a @ b
            // Need contiguous() for non-contiguous inputs from transpose
            let a_c = a.contiguous()?;
            let b_c = b.contiguous()?;
            a_c.matmul(&b_c)
        }

        // 3D @ 2D broadcasting: "abc,cd->abd" (e.g., [batch, seq, embed] @ [embed, vocab])
        // Used for language model output projection
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 3
                && b_spec.len() == 2
                && a_spec[2] == b_spec[0]  // Contract over c/first dim of b
                && output_spec == vec![a_spec[0], a_spec[1], b_spec[1]] =>
        {
            // Reshape a from [batch, seq, embed] to [batch*seq, embed]
            // Matmul with [embed, vocab]
            // Reshape result from [batch*seq, vocab] to [batch, seq, vocab]
            let a_dims = a.dims();
            let batch = a_dims[0];
            let seq = a_dims[1];
            let embed = a_dims[2];
            let vocab = b.dims()[1];

            let a_2d = a.reshape(&[batch * seq, embed])?;
            let result_2d = a_2d.matmul(b)?;
            result_2d.reshape(&[batch, seq, vocab])
        }

        // ============================================================
        // N-TENSOR JOIN PATTERNS (for soft rules / differentiable Datalog)
        // ============================================================

        // 3D × 3D → 4D: "abc,cde->abde" (expand via shared middle index)
        // Used for: Temp[x,r1,r2,z] = State[x,r1,y] State[y,r2,z]
        // Contracts over c (the shared index), keeps all others
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 3
                && b_spec.len() == 3
                && a_spec[2] == b_spec[0]  // Last of a == first of b (contraction index)
                && output_spec.len() == 4
                && output_spec == vec![a_spec[0], a_spec[1], b_spec[1], b_spec[2]] =>
        {
            // A[i,j,k] @ B[k,l,m] -> C[i,j,l,m]
            // Reshape A to [i*j, k], B to [k, l*m], matmul, reshape to [i,j,l,m]
            let a_dims = a.dims();
            let b_dims = b.dims();
            let i = a_dims[0];
            let j = a_dims[1];
            let k = a_dims[2];
            let l = b_dims[1];
            let m = b_dims[2];

            let a_2d = a.reshape(&[i * j, k])?;
            let b_2d = b.reshape(&[k, l * m])?;
            let result_2d = a_2d.matmul(&b_2d)?;
            result_2d.reshape(&[i, j, l, m])
        }

        // 4D × 3D → 3D: "abcd,eab->ecd" (contract over shared indices a,b)
        // Used for: Derived[r,x,z] = Temp[r1,r2,x,z] Rules[r,r1,r2]
        // Note: indices reordered to match the contraction pattern
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 4
                && b_spec.len() == 3
                && a_spec[0] == b_spec[1]  // a shared
                && a_spec[1] == b_spec[2]  // b shared
                && output_spec.len() == 3
                && output_spec == vec![b_spec[0], a_spec[2], a_spec[3]] =>
        {
            // A[a,b,c,d] @ B[e,a,b] -> C[e,c,d]
            // Reshape A to [a*b, c*d], B to [e, a*b], do B @ A, reshape
            let a_dims = a.dims();
            let b_dims = b.dims();
            let dim_a = a_dims[0];
            let dim_b = a_dims[1];
            let dim_c = a_dims[2];
            let dim_d = a_dims[3];
            let dim_e = b_dims[0];

            let a_2d = a.reshape(&[dim_a * dim_b, dim_c * dim_d])?;
            let b_2d = b.reshape(&[dim_e, dim_a * dim_b])?;
            let result_2d = b_2d.matmul(&a_2d)?;
            result_2d.reshape(&[dim_e, dim_c, dim_d])
        }

        // 4D × 3D pattern: "abcd,ebc->ade"
        // This is the natural output order from chained contractions:
        // Temp[x,r1,r2,z] @ Rules[r,r1,r2] -> Derived[x,z,r]
        // Output has indices from first tensor (x,z) then second tensor (r)
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 4
                && b_spec.len() == 3
                && a_spec[1] == b_spec[1]  // r1 shared
                && a_spec[2] == b_spec[2]  // r2 shared
                && output_spec.len() == 3
                && output_spec == vec![a_spec[0], a_spec[3], b_spec[0]] =>
        {
            // A[x,r1,r2,z] @ B[r,r1,r2] -> C[x,z,r]
            // Contracts r1,r2, outputs in order: x (from A), z (from A), r (from B)
            let a_dims = a.dims();
            let b_dims = b.dims();
            let dim_x = a_dims[0];
            let dim_r1 = a_dims[1];
            let dim_r2 = a_dims[2];
            let dim_z = a_dims[3];
            let dim_r = b_dims[0];

            // Reshape A from [x, r1, r2, z] to [x, r1*r2, z]
            let a_3d = a.reshape(&[dim_x, dim_r1 * dim_r2, dim_z])?;
            // Reshape B from [r, r1, r2] to [r, r1*r2]
            let b_2d = b.reshape(&[dim_r, dim_r1 * dim_r2])?;

            // We need: C[x,z,r] = sum_{r1,r2} A[x,r1,r2,z] * B[r,r1,r2]
            // = sum_k A[x,k,z] * B[r,k]  where k = r1*r2

            // Transpose A to [x, z, k] then reshape to [x*z, k]
            let a_t = a_3d.transpose(1, 2)?; // [x, z, k]
            let a_2d = a_t.reshape(&[dim_x * dim_z, dim_r1 * dim_r2])?;

            // B is [r, k], we want [x*z, k] @ [k, r] = [x*z, r]
            let b_t = b_2d.t()?; // [k, r]
            let result_2d = a_2d.matmul(&b_t)?; // [x*z, r]

            // Reshape to [x, z, r]
            result_2d.reshape(&[dim_x, dim_z, dim_r])
        }

        // Alternative 4D × 3D pattern: "abcd,ebc->aed" (different output order)
        // Derived[x,r,z] = Temp[x,r1,r2,z] Rules[r,r1,r2]
        // A[x,r1,r2,z] B[r,r1,r2] -> C[x,r,z]
        ([a_spec, b_spec], [a, b])
            if a_spec.len() == 4
                && b_spec.len() == 3
                && a_spec[1] == b_spec[1]  // r1 shared
                && a_spec[2] == b_spec[2]  // r2 shared
                && output_spec.len() == 3
                && output_spec == vec![a_spec[0], b_spec[0], a_spec[3]] =>
        {
            // A[x,r1,r2,z] @ B[r,r1,r2] -> C[x,r,z]
            // Contracts r1,r2, outputs in order: x (from A), r (from B), z (from A)
            let a_dims = a.dims();
            let b_dims = b.dims();
            let dim_x = a_dims[0];
            let dim_r1 = a_dims[1];
            let dim_r2 = a_dims[2];
            let dim_z = a_dims[3];
            let dim_r = b_dims[0];

            // Reshape A from [x, r1, r2, z] to [x, r1*r2, z]
            let a_3d = a.reshape(&[dim_x, dim_r1 * dim_r2, dim_z])?;
            // Reshape B from [r, r1, r2] to [r, r1*r2]
            let b_2d = b.reshape(&[dim_r, dim_r1 * dim_r2])?;

            // We need: C[x,r,z] = sum_{r1,r2} A[x,r1,r2,z] * B[r,r1,r2]

            // Transpose A to [x, z, k] then reshape to [x*z, k]
            let a_t = a_3d.transpose(1, 2)?; // [x, z, k]
            let a_2d = a_t.reshape(&[dim_x * dim_z, dim_r1 * dim_r2])?;

            // B is [r, k], we want [x*z, k] @ [k, r] = [x*z, r]
            let b_t = b_2d.t()?; // [k, r]
            let result_2d = a_2d.matmul(&b_t)?; // [x*z, r]

            // Reshape to [x, z, r] then transpose to [x, r, z]
            let result_3d = result_2d.reshape(&[dim_x, dim_z, dim_r])?;
            result_3d.transpose(1, 2) // [x, r, z]
        }

        _ => {
            panic!(
                "Einsum pattern '{}' not yet implemented. Input specs: {:?}, output: {:?}",
                notation, input_specs, output_spec
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_notation() {
        let (inputs, output) = parse_notation("ij,jk->ik");
        assert_eq!(inputs, vec![vec!['i', 'j'], vec!['j', 'k']]);
        assert_eq!(output, vec!['i', 'k']);
    }

    #[test]
    fn test_parse_notation_trace() {
        let (inputs, output) = parse_notation("ii->");
        assert_eq!(inputs, vec![vec!['i', 'i']]);
        assert_eq!(output, Vec::<char>::new());
    }
}