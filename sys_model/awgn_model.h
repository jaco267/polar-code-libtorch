#pragma once  
#include "enc.h"          //polar/enc
#include "binary_source.h"//sionna/trans
#include "mapping.h"      //sionna/trans
#include "awgn.h"  //sionna/trans/models
#include "polar_sc.h"   

#include <torch/torch.h>
#include <memory>   //for smart pointer  
class System_Awgn_model{
public:
  System_Awgn_model(){}; //default const
  void set_awgn(int n, int k, PolarEncoder enc, SC_Dec dec);
  torch::Tensor forward(int batch_size,
    float ebno_db,torch::Tensor & bits);
  ~System_Awgn_model();
private:
    /* data */
  int n_bits_per_sym = 2; 
  int n_;  int k_;  
  float coderate;
  unique_ptr<QamConstell> constell_;   
  unique_ptr<Mapper> mapper_;
  unique_ptr<Demapper> demapper_; 
  unique_ptr<BinarySource> binary_src_;
  unique_ptr<AWGN> awgn_channel_;
  PolarEncoder encoder_;
  SC_Dec dec_;
};


