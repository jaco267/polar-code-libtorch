#pragma once  
#include <torch/torch.h>

torch::Tensor gen_arikan(const torch::Tensor& F2, int lay) {
    // Clone F2 to initialize FN
    auto FN = F2.clone();
    // Perform Kronecker product lay-1 times
    for (int i = 0; i < lay - 1; ++i) {
        FN = torch::kron(F2, FN);
    }
    return FN;
}
torch::Tensor F2 = torch::tensor({
    {1, 0},
    {1, 1}},
    torch::TensorOptions().dtype(torch::kInt8)
);

auto F4 = gen_arikan(F2, 2);