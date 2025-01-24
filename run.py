import os
import pyrallis
from dataclasses import dataclass
def c(cmd_str):
    os.system(cmd_str)
@dataclass
class TrainConfig:
  #* python run.py --n 8 --k 4 --bs 3 --iter 1
  n:int = 8
  k:int = 4
  bs:int = 3
  iter:int = 1
  libtorch_path:str = "~/Downloads/libtorch" #* this should be your path to libtorch
@pyrallis.wrap()    
def main(cfg: TrainConfig):
  c("rm ./build")
  c("mkdir ./build")
  os.chdir("./build")
  c(f"""cmake -DCMAKE_PREFIX_PATH={cfg.libtorch_path} .. """)
  c("cmake  --build . --config Release")
  run_cmd = f"./main -n {cfg.n} -k {cfg.k} -b {cfg.bs} -i {cfg.iter}"
  c(run_cmd)
if __name__ == '__main__':
   main()