#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm_seq/stl/include/ops_stl.hpp"
#include <algorithm>

namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_seq {

namespace {

struct FoxRowWorker {
    const int* a;
    const int* b;
    int* c;
    int n;
    int blk;
    int stage;
    int num_blks;

    void operator()(int i) const {
        int broadcast_idx = (i + stage) % num_blks;
        int row_off = i * blk;
        int a_col_shift = broadcast_idx * blk;
        int b_row_shift = broadcast_idx * blk;

        for (int j = 0; j < num_blks; ++j) {
            int col_off = j * blk;
            for (int bi = 0; bi < blk; ++bi) {
                for (int bj = 0; bj < blk; ++bj) {
                    int acc = 0;
                    for (int bk = 0; bk < blk; ++bk) {
                        acc += a[((row_off + bi) * n) + a_col_shift + bk] *
                               b[((b_row_shift + bk) * n) + col_off + bj];
                    }
                    c[((row_off + bi) * n) + col_off + bj] += acc;
                }
            }
        }
    }
};

int DetermineBlockSize(int n) {
    int blk = n / 4;
    return (blk <= 0) ? 1 : std::min(blk, 128);
}

}  // namespace

LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL::LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL::ValidationImpl() {
    return (GetInput() > 0) && (GetOutput() == 0);
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL::PreProcessingImpl() {
    int n = GetInput();
    matrix_a_.assign(static_cast<size_t>(n) * n, 1);
    matrix_b_.assign(static_cast<size_t>(n) * n, 1);
    matrix_c_.assign(static_cast<size_t>(n) * n, 0);
    block_size_ = DetermineBlockSize(n);
    return true;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL::RunImpl() {
    int n = GetInput();
    if (n <= 0) return false;

    int num_blks = n / block_size_;
    if (n % block_size_ != 0 || num_blks == 0) {
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                int factor = matrix_a_[i * n + k];
                if (factor == 0) continue;
                for (int j = 0; j < n; ++j) {
                    matrix_c_[i * n + j] += factor * matrix_b_[k * n + j];
                }
            }
        }
        return true;
    }

    std::vector<int> block_indices(num_blks);
    std::iota(block_indices.begin(), block_indices.end(), 0);

    for (int stage = 0; stage < num_blks; ++stage) {
        std::vector<std::thread> threads;
        threads.reserve(num_blks);

        for (int i : block_indices) {
            FoxRowWorker worker{
                matrix_a_.data(),
                matrix_b_.data(),
                matrix_c_.data(),
                n,
                block_size_,
                stage,
                num_blks
            };
            threads.emplace_back(worker, i);
        }

        for (auto& t : threads) {
            t.join();
        }
    }
    return true;
}

bool LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL::PostProcessingImpl() {
    int total = 0;
    for (int val : matrix_c_) {
        total += val;
    }
    GetOutput() = total;
    return true;
}

}  // namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_seq