1. Pad简介
Pad操作可以用来对一个Tensor四周进行填充。为Pytorch版的Pad为例：
`torch.nn.functional.pad(input, pad, mode='constant', value=0)`
```python
import torch
import torch.nn.functional as tf
torch.manual_seed(123)
img = torch.randint(0, 100, (3, 3))
print('Original:')
print(img)
# Pad left
img_pd_left = tf.pad(img, (1, 0, 0, 0), mode='constant')
print('Pad Left:')
print(img_pd_left)
# Pad right
img_pd_right = tf.pad(img, (0, 1, 0, 0), mode='constant')
print('Pad Right:')
print(img_pd_right)
# Pad top
img_pd_top = tf.pad(img, (0, 0, 1, 0), mode='constant')
print('Pad Top:')
print(img_pd_top)
img_pd_bottom = tf.pad(img, (0, 0, 0, 1), mode='constant')
print('Pad Bottom:')
print(img_pd_bottom)
```
输出:
```
Original:
tensor([[82, 89,  2],
        [10, 80, 42],
        [86, 57, 99]])
Pad Left:
tensor([[ 0, 82, 89,  2],
        [ 0, 10, 80, 42],
        [ 0, 86, 57, 99]])
Pad Right:
tensor([[82, 89,  2,  0],
        [10, 80, 42,  0],
        [86, 57, 99,  0]])
Pad Top:
tensor([[ 0,  0,  0],
        [82, 89,  2],
        [10, 80, 42],
        [86, 57, 99]])
Pad Bottom:
tensor([[82, 89,  2],
        [10, 80, 42],
        [86, 57, 99],
        [ 0,  0,  0]])
```
当参数为负数时，Pad可以达到对Tensor裁剪的目的：
```python
import torch
import torch.nn.functional as tf
torch.manual_seed(123)
img = torch.randint(0, 100, (3, 3))
print('Original:')
print(img)
# Crop left
img_cp_left = tf.pad(img, (-1, 0, 0, 0), mode='constant')
print('Crop Left:')
print(img_cp_left)
# Crop right
img_cp_right = tf.pad(img, (0, -1, 0, 0), mode='constant')
print('Crop Right:')
print(img_cp_right)
# Crop top
img_cp_top = tf.pad(img, (0, 0, -1, 0), mode='constant')
print('Crop Top:')
print(img_cp_top)
img_cp_bottom = tf.pad(img, (0, 0, 0, -1), mode='constant')
print('Crop Bottom:')
print(img_cp_bottom)
```
输出:
```
Original:
tensor([[82, 89,  2],
        [10, 80, 42],
        [86, 57, 99]])
Crop Left:
tensor([[89,  2],
        [80, 42],
        [57, 99]])
Crop Right:
tensor([[82, 89],
        [10, 80],
        [86, 57]])
Crop Top:
tensor([[10, 80, 42],
        [86, 57, 99]])
Crop Bottom:
tensor([[82, 89,  2],
        [10, 80, 42]])
```
2. 问题描述
现在需要将一个Pytorch模型转换成OpenVino支持的模型，而且这个模型里用到了带负参数的Pad操作。不管是将Pytorch模型转换成onnx格式还是IR格式，在调用`InferenceEngine::Core::LoadNetwork`对网络进行检查时会报错，提示Pad操作的负参数不能转换成无符号整型。

3. 解决办法
查看Openvino的[文档](https://docs.openvinotoolkit.org/latest/openvino_docs_ops_movement_Pad_1.html)后发现OpenVino版的Pad操作不支持负参数。解决方法就是避免使用带有负参数的Pad,因为使用Tensor的切片操作可以达到同样的目的。代码如下：
```python
import torch
torch.manual_seed(123)
img = torch.randint(0, 100, (3, 3))
print('Original:')
print(img)
# Crop left
img_cp_left = img[:, 1:]
print('Crop Left:')
print(img_cp_left)
# Crop right
img_cp_right = img[:, :-1]
print('Crop Right:')
print(img_cp_right)
# Crop top
img_cp_top = img[1:, :]
print('Crop Top:')
print(img_cp_top)
img_cp_bottom = img[:-1, :]
print('Crop Bottom:')
print(img_cp_bottom)
```
输出:
```
Original:
tensor([[82, 89,  2],
        [10, 80, 42],
        [86, 57, 99]])
Crop Left:
tensor([[89,  2],
        [80, 42],
        [57, 99]])
Crop Right:
tensor([[82, 89],
        [10, 80],
        [86, 57]])
Crop Top:
tensor([[10, 80, 42],
        [86, 57, 99]])
Crop Bottom:
tensor([[82, 89,  2],
        [10, 80, 42]])
```