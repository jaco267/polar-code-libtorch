#include "sim.h"
#include "awgn_model.h"
#include <torch/torch.h>
#include <iostream>
using namespace std;
using namespace torch;
Tensor count_errors(Tensor& b, Tensor& b_hat){
  Tensor errors = torch::not_equal(b,b_hat).to(torch::kInt64);
  return torch::sum(errors);
}
Tensor count_block_errors(Tensor& b, Tensor& b_hat){
  Tensor errors = torch::not_equal(b,b_hat).to(torch::kInt64);
  auto block_errs = torch::any(errors,/*dim*/-1);
  return torch::sum(block_errs); 
}
void sim_ber(System_Awgn_model* mc_fun,
  Tensor ebno_dbs,int batch_size,
  int max_mc_iter,Tensor& ber,Tensor& bler){
  // cout<<"sim_ber"<<endl;
  int num_points = ebno_dbs.sizes()[0];  
  // cout<<"num_points"<<num_points<<endl;
  auto bit_errors = torch::zeros({num_points},torch::kInt64);
  auto block_errors = torch::zeros({num_points},torch::kInt64);
  auto nb_bits  = torch::zeros({num_points},torch::kInt64);
  auto nb_blocks = torch::zeros({num_points},torch::kInt64);
  for (int i =0; i<num_points;i++){
    // cout<<i<<endl;
    int iter_count = -1;  
    for (int ii=0; ii<max_mc_iter;ii++){
      iter_count+=1;  
      Tensor b;
      Tensor b_hat = mc_fun->forward(batch_size,ebno_dbs[i].item<float>(),b);
      // cout<<"b"<<b<<endl;cout<<"b_hat"<<b_hat<<endl;
      Tensor bit_e = count_errors(b,b_hat);
      Tensor block_e = count_block_errors(b,b_hat);
      int bit_n = torch::numel(b);
      int last_shape = b.sizes()[b.dim()-1];
      int block_n = bit_n/last_shape;
      // cout<<"be"<<bit_e<<endl;
      // cout<<"ble"<<block_e<<endl;
      // cout<<"block_n"<<block_n<<endl;
      bit_errors[i]+= bit_e;  
      block_errors[i]+=block_e;
      nb_bits[i]+= bit_n;  
      nb_blocks[i]+=block_n;
    }
  }
  ber = bit_errors / nb_bits;
  bler = block_errors / nb_blocks;
  ber = torch::where(torch::isnan(ber),torch::zeros_like(ber),ber);

  bler = torch::where(torch::isnan(bler),torch::zeros_like(bler),bler);
  
}