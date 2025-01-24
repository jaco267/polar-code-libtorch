#include <iostream>
#include <stdio.h> 
#include <unistd.h> 
#include <bits/getopt_core.h>
#include <torch/torch.h>
#include "polar/froze.h" //polar/froze.h
#include "polar/enc.h"
#include "polar/polar_sc.h"
// #include "torch_utils/torch_utils.h"
#include "d_kernels.h" //F2,F8...
#include "sys_model/awgn_model.h"
#include "sim.h"
#include "awgn.h"
// https://www.geeksforgeeks.org/getopt-function-in-c-to-parse-command-line-arguments/
using namespace std;
using namespace torch;
void  gen_code(int n,int k,int bs,torch::Tensor kernel,
  System_Awgn_model* rm_scl_awgn_raw
){
  int a = std::log2(n);
  
  assert(std::pow(2,a) == n && "N!= 2^n");
  torch::Tensor frozen_pos;//* pass by ref
  torch::Tensor G = get_Kern_frozen_bits(n,n-k,kernel,frozen_pos);
  PolarEncoder encoder(frozen_pos,n,G);
  //unique_ptr<SC_Dec>
  SC_Dec dec(frozen_pos,n);
  rm_scl_awgn_raw->set_awgn(n,k,encoder,dec);//encoder copy by value so it wont get deleted out of this scope  
  return;
}

//*./main -i -f file.txt -lr -x 'hero'
int main(int argc, char *argv[]) { 
  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
  torch::manual_seed(42);
  int opt; //test
  int k;  int n; int bs; int max_iter; 
  while((opt = getopt(argc, argv, ":k:n:b:i:rx")) != -1) { 
    switch(opt) { 
      case 'r': printf("option: %c\n", opt); break; 
      case 'b': bs = atoi(optarg);break;//batchsize
      case 'n': n = atoi(optarg); break;//codeword len 
      case 'k': k = atoi(optarg); break;//info len  
      case 'i': max_iter = atoi(optarg); break;
      case ':': printf("option needs a value\n"); break; 
      case '?': printf("unknown option: %c\n", optopt); break; 
    } 
  } 
  // optind is for the extra arg which are not parsed 
  for(; optind < argc; optind++){	 printf("extra arguments: %s\n", argv[optind]); } 
  torch::Tensor ebno_db = torch::arange(0,5,0.5); 
  // torch::Tensor ebno_db = torch::arange(0,1,0.5); 
  torch::Tensor kernel = F2;
  unique_ptr<System_Awgn_model> code = make_unique<System_Awgn_model>();
  gen_code(n,k,bs,kernel,code.get());
  // cout<<"end of gen code"<<endl;
  // cout<<"n"<<n<<"k"<<k<<"bs"<<bs<<" max iter"<<max_iter<< endl;
  Tensor ber;
  Tensor bler;
  sim_ber(code.get(),ebno_db,bs,max_iter,ber,bler);// code->forward(bs,3.);
  // cout<<"ebno"<<ebno_db.reshape({1,-1})<<endl;
  cout<<"ber"<<ber<<endl;
  cout<<"bler"<<bler<<endl;
  cout<<"end of program"<<endl;
  return 0;
}
