#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>

namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm {

namespace {

void FillMatrixWithOnes(DenseMatrix &mat, int n) {
  mat.rows = n;
  mat.cols = n;
  mat.values.assign(static_cast<std::size_t>(n) * n, 1.0);
}

void FillMatrixWithZeros(DenseMatrix &mat, int r, int c) {
  mat.rows = r;
  mat.cols = c;
  mat.values.assign(static_cast<std::size_t>(r) * c, 0.0);
}

void NaiveMultiplyOMP(const DenseMatrix &a, const DenseMatrix &b, DenseMatrix &res) {
  FillMatrixWithZeros(res, a.rows, b.cols);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < a.rows; ++i) {
    for (int k = 0; k < a.cols; ++k) {
      double factor = a.At(i, k);
      if (factor == 0.0) {
        continue;
      }
      for (int j = 0; j < b.cols; ++j) {
#pragma omp atomic
        res.At(i, j) += factor * b.At(k, j);
      }
    }
  }
}

void MultiplyBlocksOMP(const DenseMatrix &a, const DenseMatrix &b, DenseMatrix &res, int blk, int stage) {
  int n = a.rows;
  int num_blocks = n / blk;

#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < num_blocks; ++i) {
    for (int j = 0; j < num_blocks; ++j) {
      int broadcast_idx = (i + stage) % num_blocks;
      int row_off = i * blk;
      int col_off = j * blk;
      int a_col_shift = broadcast_idx * blk;
      int b_row_shift = broadcast_idx * blk;

      for (int bi = 0; bi < blk; ++bi) {
        for (int bj = 0; bj < blk; ++bj) {
          double acc = 0.0;
          for (int bk = 0; bk < blk; ++bk) {
            acc += a.At(row_off + bi, a_col_shift + bk) * b.At(b_row_shift + bk, col_off + bj);
          }
#pragma omp atomic
          res.At(row_off + bi, col_off + bj) += acc;
        }
      }
    }
  }
}

int DetermineBlockSize(int n) {
  int blk = n / 4;
  if (blk <= 0) {
    blk = 1;
  }
  return std::min(blk, 128);
}

void ExecuteFoxAlgorithmOMP(const DenseMatrix &a, const DenseMatrix &b, DenseMatrix &res, int blk) {
  if (!a.IsSquare() || !b.IsSquare() || a.rows != b.rows || blk <= 0 || a.rows % blk != 0) {
    NaiveMultiplyOMP(a, b, res);
    return;
  }

  int n = a.rows;
  int stages = n / blk;
  FillMatrixWithZeros(res, n, n);

  for (int stage = 0; stage < stages; ++stage) {
    MultiplyBlocksOMP(a, b, res, blk, stage);
  }
}

}  // namespace

LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP::LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP::ValidationImpl() {
  return GetInput() > 0 && GetOutput() == 0.0;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP::PreProcessingImpl() {
  int n = GetInput();
  FillMatrixWithOnes(matrix_a_, n);
  FillMatrixWithOnes(matrix_b_, n);
  block_size_ = DetermineBlockSize(n);
  return true;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP::RunImpl() {
  ExecuteFoxAlgorithmOMP(matrix_a_, matrix_b_, result_matrix_, block_size_);
  return true;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP::PostProcessingImpl() {
  double total = 0.0;
#pragma omp parallel for reduction(+ : total)
  for (size_t i = 0; i < result_matrix_.values.size(); ++i) {
    total += result_matrix_.values[i];
  }
  GetOutput() = total;
  return std::isfinite(total);
}

}  // namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm
