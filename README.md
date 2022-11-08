# tvm-utils
Utilities for Apache TVM

## Model tuning

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
