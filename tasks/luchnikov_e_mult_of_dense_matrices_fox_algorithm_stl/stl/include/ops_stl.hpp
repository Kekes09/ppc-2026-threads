#pragma once
#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl {

class LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }

  explicit LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> matrix_a_;
  std::vector<int> matrix_b_;
  std::vector<int> matrix_c_;
  int block_size_ = 32;
};

}  // namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl
