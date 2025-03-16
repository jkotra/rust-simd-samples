#![feature(portable_simd)]
use core::simd::{f32x4, u16x8, u8x8};
use image::{GrayImage, ImageBuffer, ImageReader};
use std::{
    simd::num::{SimdFloat, SimdUint},
    time::Instant,
};

pub fn hello_simd() {
    let a = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    let result = a + b;
    println!("SIMD result: {:?}", result.to_array())
}

pub fn to_grayscale_simd_u8(width: u32, height: u32, image_data: Vec<u8>) -> Vec<u8> {
    let mut gray_pixels: Vec<u8> = Vec::with_capacity((width * height) as usize);

    let r_coeff = u16x8::splat(77);
    let g_coeff = u16x8::splat(150);
    let b_coeff = u16x8::splat(29);

    for chunk in image_data.chunks(8 * 4) {
        // process 8 RGBA pixels at a time.
        let mut a = [0; 8];
        let mut b = [0; 8];
        let mut c = [0; 8];
        let mut d = [0; 8];

        for (i, pixel) in chunk.chunks_exact(4).enumerate() {
            a[i] = pixel[0];
            b[i] = pixel[1];
            c[i] = pixel[2];
            d[i] = pixel[3];
        }

        let r_arr = u8x8::from_slice(&a).cast::<u16>();
        let g_arr = u8x8::from_slice(&b).cast::<u16>();
        let b_arr = u8x8::from_slice(&c).cast::<u16>();

        let gray = (r_coeff * r_arr) + (g_coeff * g_arr) + (b_coeff * b_arr);
        gray_pixels.extend((gray >> 8).cast::<u8>().to_array());
    }

    gray_pixels
}

pub fn multiply_matrix_simd(
    matrix_a: Vec<Vec<f32>>,
    matrix_b: Vec<Vec<f32>>,
    n: usize,
) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut local_sum = 0.0f32;

            for k in (0..n).step_by(4) {
                let a_chunk = f32x4::from_array([
                    matrix_a[i][k],
                    matrix_a[i][k + 1],
                    matrix_a[i][k + 2],
                    matrix_a[i][k + 3],
                ]);

                let b_chunk = f32x4::from_array([
                    matrix_b[k][j],
                    matrix_b[k + 1][j],
                    matrix_b[k + 2][j],
                    matrix_b[k + 3][j],
                ]);

                local_sum += (a_chunk * b_chunk).reduce_sum();
            }

            result[i][j] = local_sum;
        }
    }

    result
}

fn multiply_matrix_simd_1d(matrix_a: Vec<f32>, matrix_b: Vec<f32>, n: usize) -> Vec<f32> {
    let mut result = [0.0f32; 256 * 256];

    for i in 0..n {
        for j in 0..n {
            let mut local_sum = 0.0;

            for k in (0..n).step_by(4) {
                let a_chunk = f32x4::from_array([
                    matrix_a[i * n + k],
                    matrix_a[i * n + k + 1],
                    matrix_a[i * n + k + 2],
                    matrix_a[i * n + k + 3],
                ]);

                // For matrix B:
                // We need the j-th column. In row-major order, the element in row k and column j
                // is located at index k * n + j.
                let b_chunk = f32x4::from_array([
                    matrix_b[k * n + j],
                    matrix_b[(k + 1) * n + j],
                    matrix_b[(k + 2) * n + j],
                    matrix_b[(k + 3) * n + j],
                ]);

                // Multiply the chunks element-wise and sum their values.
                local_sum += (a_chunk * b_chunk).reduce_sum();
            }

            result[i * n + j] = local_sum;
        }
    }

    result.to_vec()
}

pub fn multiply_matrix(
    matrix_a: Vec<Vec<f32>>,
    matrix_b: Vec<Vec<f32>>,
    n: usize,
) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result[i][j] = sum;
        }
    }

    result
}

pub fn main() {
    hello_simd();

    let mut matrix_a = vec![];
    let mut matrix_b = vec![];

    for _i in 0..256 {
        let data: Vec<f32> = (0..256).map(|v| v as f32).collect();
        matrix_a.push(data.clone());
        matrix_b.push(data.clone());
    }

    let a_1d: Vec<f32> = matrix_a.clone().into_iter().flatten().collect();
    let b_1d: Vec<f32> = matrix_b.clone().into_iter().flatten().collect();

    let timer = Instant::now();
    let scalar_matrix_result = multiply_matrix(matrix_a.clone(), matrix_b.clone(), 256);
    println!(
        "matrix multiplication - scalar - time: {:?}",
        timer.elapsed()
    );

    let timer = Instant::now();
    let simd_matrix_result = multiply_matrix_simd(matrix_a, matrix_b, 256);
    println!(
        "matrix multiplication - SIMD 2D - time: {:?}",
        timer.elapsed()
    );
    assert_eq!(scalar_matrix_result, simd_matrix_result);

    let timer = Instant::now();
    let simd_matrix_result_1d = multiply_matrix_simd_1d(a_1d, b_1d, 256);
    println!(
        "matrix multiplication - SIMD 1D - time: {:?}",
        timer.elapsed()
    );
    let reconstructed_2d_matrix: Vec<Vec<f32>> = simd_matrix_result_1d
        .chunks(256)
        .map(|f| f.to_vec())
        .collect();
    assert_eq!(scalar_matrix_result, reconstructed_2d_matrix);

    let timer = Instant::now();
    let image = ImageReader::open("input.jpg").unwrap().decode().unwrap();
    let _ = image.grayscale();
    println!("Grayscale - scalar - time: {:?}", timer.elapsed());

    let image_data = image.to_rgba8().to_vec();

    let timer = Instant::now();
    let gray_image_data: Vec<u8> = to_grayscale_simd_u8(image.width(), image.height(), image_data);
    println!("Grayscale - SIMD - time: {:?}", timer.elapsed());

    let img: GrayImage = ImageBuffer::from_raw(image.width(), image.height(), gray_image_data)
        .expect("Unable to create image from u8 array!");
    img.save("output.jpg").unwrap()
}
