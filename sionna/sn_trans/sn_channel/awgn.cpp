
#include <iostream>
#include <torch/torch.h>
#include "awgn.h"
#include "sn_utils.h"
using namespace std;
AWGN::AWGN(){}
torch::Tensor AWGN::forward(
    torch::Tensor x,float no
){ //todo complex normal 
  // cout<<"test awgn forawrd"<<endl;
   // cout<<"hey"<<torch::normal(0,1,{3,2})<<endl;
  torch::Tensor noise = complex_normal(x.sizes());
  torch::Tensor no_ten = expand_to_rank(
     torch::tensor(no),/*target_rank*/x.dim(),/*axis*/-1);
//   cout<<"no_ten"<<no_ten.sizes()<<endl; 
//   cout<<"noise xr"<<torch::real(noise)<<endl;
//   cout<<"noise xi"<<torch::imag(noise)<<endl;
//   cout<<"dtype"<<noise.dtype()<<endl;
  noise = noise*torch::sqrt(no_ten.to(torch::kFloat)).to(noise.dtype());
//   cout<<"noise dtype"<<noise.dtype()<<"x dtype"<<x.dtype()<<endl;
  torch::Tensor y = x+noise;  
//   cout<<"y dtype"<<y.dtype()<<endl;
  return y;
}