#pragma once
#include <iostream>
#include <torch/torch.h>
#include "sn_utils.h"
class AWGN{
public:
    AWGN();
    ~AWGN(){};
    torch::Tensor forward(torch::Tensor input,
       float no);
private:
    /* data */
};


