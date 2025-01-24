#include "sn_utils.h"
#include <torch/torch.h>
#include <iostream>
using namespace std;  

torch::Tensor complex_normal(c10::ArrayRef<int64_t> shape){
    float var = 1.0;
    float stddev = std::sqrt(var/2);
    // cout<<"var"<<var<<" stddev"<<stddev<<endl;
    torch::Tensor xr = torch::normal(/*mean*/0.,
      /*std*/stddev,/*size*/shape).to(torch::kFloat);
    torch::Tensor xi = torch::normal(/*mean*/0.,
      /*std*/stddev,/*size*/shape).to(torch::kFloat);
    torch::Tensor x = torch::complex(xr,xi);
    return x;
}
torch::Tensor expand_to_rank(torch::Tensor tensor, int target_rank,int axis){
  int num_dims = std::max((int)target_rank - (int)tensor.dim(),0);
//   cout<<"num_dims"<<num_dims<<endl;
  // cout<<"num_dims"<<tensor.dim()<<endl;
//   int num_dims = 2;
  torch::Tensor out;  out = tensor;
  for (int i = 0; i<num_dims;i++){out = out.unsqueeze(axis);}
  return out;  
}

