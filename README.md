# Custom PyTorch CUDA layer

# Usage
Install python module: `./setup.py`

Example:
```python
import cuda_pytorch_layer
import torch

device = torch.device('cuda:0')
mult_layer = cuda_pytorch_layer.Mult()

a = torch.rand(15, 10, device=device)
b = torch.rand(15, 10, device=device)
result = mult_layer(a, b)
```

[Example](./example/check.ipynb) where backprob is tested. 

