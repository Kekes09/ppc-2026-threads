#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl/common/include/common.hpp"
#include "luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl/stl/include/ops_stl.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl {

class LuchnikovEMultOfDenseMatrixFoxAlgoritmFuncTestsSTL
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    int n = GetTestInputData();
    int expected = n * n * n;
    return (output_data == expected);
  }

  InType GetTestInputData() final {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }
};

namespace {

TEST_P(LuchnikovEMultOfDenseMatrixFoxAlgoritmFuncTestsSTL, StlMatmulTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestCases = {
    std::make_tuple(2, "small_2"), std::make_tuple(4, "medium_4"),  std::make_tuple(6, "medium_6"),
    std::make_tuple(8, "large_8"), std::make_tuple(10, "large_10"), std::make_tuple(12, "large_12"),
    std::make_tuple(16, "xl_16"),  std::make_tuple(20, "xl_20"),    std::make_tuple(24, "xl_24")};

const auto kTestTasksList = ppc::util::AddFuncTask<LuchnikovEMultOfDenseMatrixFoxAlgoritmSTL, InType>(
    kTestCases, PPC_SETTINGS_luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPrinter = LuchnikovEMultOfDenseMatrixFoxAlgoritmFuncTestsSTL::PrintFuncTestName<
    LuchnikovEMultOfDenseMatrixFoxAlgoritmFuncTestsSTL>;

INSTANTIATE_TEST_SUITE_P(StlMatrixTests, LuchnikovEMultOfDenseMatrixFoxAlgoritmFuncTestsSTL, kGtestValues, kPrinter);

}  // namespace
}  // namespace luchnikov_e_mult_of_dense_matrices_fox_algorithm_stl
