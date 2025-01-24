#pragma once
using namespace std;
float ebnodb2no(float ebno_db,int n_bits_per_sym,float coderate){
  float ebno = std::pow(10.,ebno_db/10.);
  int energy_per_symbol = 1;
  float no = 1/(ebno*coderate*n_bits_per_sym/energy_per_symbol);

  return no;
}