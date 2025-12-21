//! Benchmarks for tensor operations.

use candle_core::{Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ein::einsum;

fn bench_matrix_multiply(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("matrix_multiply");

    for size in [32, 64, 128, 256].iter() {
        let a = Tensor::rand(0.0f32, 1.0, (*size, *size), &device).unwrap();
        let b = Tensor::rand(0.0f32, 1.0, (*size, *size), &device).unwrap();

        // Native matmul
        group.bench_with_input(
            BenchmarkId::new("native", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| a.matmul(b).unwrap());
            },
        );

        // Einsum matmul
        group.bench_with_input(
            BenchmarkId::new("einsum", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| einsum("ij,jk->ik", &[a, b]).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_matvec(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("matrix_vector");

    for size in [64, 128, 256, 512].iter() {
        let w = Tensor::rand(0.0f32, 1.0, (*size, *size), &device).unwrap();
        let x = Tensor::rand(0.0f32, 1.0, (*size,), &device).unwrap();

        // Native (matmul with unsqueeze/squeeze)
        group.bench_with_input(
            BenchmarkId::new("native", size),
            &(&w, &x),
            |bench, (w, x)| {
                bench.iter(|| {
                    w.matmul(&x.unsqueeze(1).unwrap())
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                });
            },
        );

        // Einsum matvec
        group.bench_with_input(
            BenchmarkId::new("einsum", size),
            &(&w, &x),
            |bench, (w, x)| {
                bench.iter(|| einsum("ij,j->i", &[w, x]).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_outer_product(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("outer_product");

    for size in [64, 128, 256].iter() {
        let a = Tensor::rand(0.0f32, 1.0, (*size,), &device).unwrap();
        let b = Tensor::rand(0.0f32, 1.0, (*size,), &device).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| einsum("i,j->ij", &[a, b]).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matrix_multiply, bench_matvec, bench_outer_product);
criterion_main!(benches);