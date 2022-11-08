# tvm-utils
Utilities for Apache TVM

## Model tuning

### CPU

```sh
python -u tvm_tune.py \
  -t llvm \
  -O3 \
  -k grid_search \
  --measure-num 1 \
  --measure-repeats 10 \
  --flush-cpu
  model.onnx \
  tune.log
```

### GPU

```sh
python -u tvm_tune.py \
  -t cuda \
  -O3 \
  -k xgb \
  -n 2000 \
  -s 750 \
  --measure-num 20 \
  --measure-repeats 3 \
  --measure-min-time 200 \
  --timeout-builder 10 \
  --timeout-runner 20 \
  --enable-transfer-learning \
  model.onnx \
  tune.log
```

## Model compilation

```sh
python -u tvm_compile.py \
  --tuner-log tune.log \
  -t cuda \
  -O3 \
  model.onnx \
  model-cuda.so
```
