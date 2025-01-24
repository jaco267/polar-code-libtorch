#include "froze.h"
#include <torch/torch.h>
using namespace std;
torch::Tensor get_Kern_frozen_bits(
  int n,int f_num,
  torch::Tensor kern,
  torch::Tensor &frozen_pos
  ){
  assert(kern.dim()==2 && "kernel dim != 2");
  int base = kern.sizes()[0];
  int _nb_stages = std::log2(n)/std::log2(base);
  // cout<<"dim"<<kern.dim()<<endl<<"base"<<base<<endl<<"stages"<<_nb_stages<<endl;
  assert(std::pow(base,_nb_stages)==n && "n is not power of kern size");
  torch::Tensor m = kern.clone();
  for (int i = 0; i < _nb_stages - 1; ++i) {
      m = torch::kron(kern, m);
  }
  torch::Tensor G = m;

  // Compute G_weights as the sum of G along dimension 1
  torch::Tensor G_weights = torch::sum(G,/*dim*/1);

  // Find the frozen positions (indices of the smallest `f_num` elements)
  // torch::Tensor sorted_indices = std::get<0>(torch::sort(torch::argsort(G_weights), 0));
  torch::Tensor f_unsort = torch::argsort(G_weights).slice(0, 0, f_num);
  frozen_pos = std::get<0>(torch::sort(f_unsort));
  return G;
}