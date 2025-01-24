#include <iostream>
#include <torch/torch.h>
#include "torch_utils.h"
using namespace std;
torch::Tensor setdiff1d(const torch::Tensor& a, const torch::Tensor& b) {
    // Create a boolean mask for elements in `a` not in `b`
    //ex a [01234567]  b0124  frozenpos  
    auto mask = torch::ones_like(a, torch::kBool);
    //mask = 11111111
    for (int64_t i = 0; i < b.size(0); ++i) {
        mask = mask & (a != b[i]); //a!=b0
        //             a!=0  FTTT TTTT
        //             a!=1  FFTT TTTT
        //             a!=2  FFFT TTTT
        //             a!=4  FFFT FTTT 
    }
    return a.masked_select(mask);//3,5,6,7
}

torch::Tensor binary_repr(int x, int bits){
    torch::Tensor mask = torch::pow(2,torch::arange(bits-1,-1,-1));
    torch::Tensor out = torch::tensor({x}).unsqueeze(-1).bitwise_and(mask).ne(0).reshape(-1);
    return out;
}