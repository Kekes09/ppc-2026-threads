#pragma once
#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm {

class LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit LuchnikovEMultOfDenseMatrixFoxAlgoritmOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  DenseMatrix matrix_a_;
  DenseMatrix matrix_b_;
  DenseMatrix result_matrix_;
  int block_size_ = 32;
};

}  // namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm
