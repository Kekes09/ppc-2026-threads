#include <gtest/gtest.h>

#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl/common/include/common.hpp"
#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl/stl/include/ops_stl.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl {

class LuchnikovEMultOfDenseMatrixFoxAlgoritmPerfTestSTL : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 200;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuchnikovEMultOfDenseMatrixFoxAlgoritmPerfTestSTL, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL>(
    PPC_SETTINGS_luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = LuchnikovEMultOfDenseMatrixFoxAlgoritmPerfTestSTL::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(StlPerfTests, LuchnikovEMultOfDenseMatrixFoxAlgoritmPerfTestSTL, kGtestValues, kPerfTestName);
}  // namespace
}  // namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl
