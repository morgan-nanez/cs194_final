# Final Project 1



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install matplotlib
pip install numpy
pip install skimage
pip install PyTorch
pip install torchvision
```

## Usage

```python
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt

plt.imshow(image)
```



```python
import skimage.transform as sktr
import skimage.draw as draw
import skimage.io as io

#loading images
image = io.imread(image_path)
```

```python
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

gram = torch.mm(tensor,tensor.t())  
loss = torch.mul(total_style_loss , style_weight)
optimizer = optim.Adam(params=[white_noise.requires_grad_(True)], lr=0.01)
F.mse_loss()
```

```python
import torchvision
from torchvision import transforms, utils
from torchvision.utils import save_image
import torchvision.models as models

vgg = models.vgg19(pretrained=True).features
save_image(white_noise.cpu().detach())
```
