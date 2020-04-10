# Pytorch

## 数据并行处理

在这个教程里，我们将学习如何使用数据并行(`DataParallel`）来使用多GPU。

PyTorch非常容易的就可以使用GPU，可以用如下方式把一个模型放到GPU上:

```python
device = torch.device("cuda: 0")
model.to(device)Copy

# 然后可以复制所有的张量到GPU上:
mytensor = my_tensor.to(device)
```

请注意，调用`my_tensor.to(device)`返回一个GPU上的`my_tensor`副本，而不是重写`my_tensor`。我们需要把它赋值给一个新的张量并在GPU上使用这个张量。

在多GPU上执行前向和反向传播是自然而然的事。然而，PyTorch默认将只是用一个GPU。你可以使用`DataParallel`让模型并行运行来轻易的让你的操作在多个GPU上运行。

```python
model = nn.DataParallel(model)
```



首先，我们需要创建一个模型实例和检测我们是否有多个GPU。如果我们有多个GPU，我们使用`nn.DataParallel`来包装我们的模型。然后通过`model.to(device)`把模型放到GPU上。

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1: 
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```



`DataParallel`自动的划分数据，并将作业发送到多个GPU上的多个模型。`DataParallel`会在每个模型完成作业后，收集与合并结果然后返回给你。

```python
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        
Let's use 3 GPUs!
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])        
```







## 保存和加载模型

关于保存和加载模型，有三个核心功能需要熟悉：

1. [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save)：将序列化的对象保存到磁盘。此函数使用Python的 pickle实用程序进行序列化。可以保存各种对象的模型，张量和字典。
2. [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch load#torch.load)：使用pickle将目标文件反序列化到内存中。
3. [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/nn.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)：使用反序列化的*state_dict*加载模型的参数字典 。



### 什么是state_dict？

在PyTorch中，模型的可学习参数(即权重和偏差） `torch.nn.Module` 包含在模型的参数中 (通过访问`model.parameters()`）。**state_dict是一个简单的Python字典对象**，每个层映射到其参数张量。请注意，只有具有可学习参数的层(卷积层，线性层等）和已注册的缓冲区(batchnorm的running_mean）才在模型的state_dict中具有条目。优化器对象(`torch.optim`）还具有state_dict，其中包含有关优化器状态以及所用超参数的信息。

由于 *state_dict* 对象是Python词典，因此可以轻松地保存，更新，更改和还原它们，从而为PyTorch模型和优化器增加了很多模块化。

```python
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])    
```

```python
Model's state_dict:
conv1.weight     torch.Size([6, 3, 5, 5])
conv1.bias   torch.Size([6])
conv2.weight     torch.Size([16, 6, 5, 5])
conv2.bias   torch.Size([16])
fc1.weight   torch.Size([120, 400])
fc1.bias     torch.Size([120])
fc2.weight   torch.Size([84, 120])
fc2.bias     torch.Size([84])
fc3.weight   torch.Size([10, 84])
fc3.bias     torch.Size([10])

Optimizer's state_dict:
state    {}
param_groups     [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4675713712, 4675713784, 4675714000, 4675714072, 4675714216, 4675714288, 4675714432, 4675714504, 4675714648, 4675714720]}]
```



### 推理模型的保存和加载

#### 保存/加载`state_dict`(推荐）

```python
# save
torch.save(model.state_dict(), PATH)

# load
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

保存模型以进行推理时，仅需要保存训练后的模型的学习参数。使用 `torch.save()` 函数保存模型的state_dict将为您提供最大的灵活性，以便以后恢复模型，这就是为什么推荐使用此方法来保存模型。

常见的PyTorch约定是使用`.pt`或 `.pth`文件扩展名保存模型。

请记住，`model.eval()`在运行推理之前，必须先调用以将退出和批处理规范化层设置为评估模式。不这样做将产生不一致的推断结果。

`load_state_dict()`函数使用字典对象，而不是保存对象的路径。这意味着，在将保存的state_dict传递给`load_state_dict()`函数之前 ，必须对其进行反序列化。例如，您无法使用加载 `model.load_state_dict(PATH)`。



#### 整个模型的保存和加载

```python
torch.save(model, PATH)

model = torch.load(PATH)
model.eval()
```

此保存/加载过程使用最直观的语法，并且涉及最少的代码。以这种方式保存模型将使用Python的[pickle](https://docs.python.org/3/library/pickle.html)模块保存整个 模块。

这种方法的缺点是序列化的数据绑定到特定的类，并且在保存模型时使用确切的目录结构。这样做的原因是因为pickle不会保存模型类本身。而是将其保存到包含类的文件的路径，该路径在加载时使用。因此，在其他项目中使用或重构后，您的代码可能会以各种方式中断。



#### 保存和加载用于推理和/或继续训练的常规检查点

```python
# save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    ...
    }, PATH)

# load
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

保存用于检查或继续训练的常规检查点时，您必须保存的不仅仅是模型的 *state_dict*。保存优化器的state_dict也是很重要的，因为它包含随着模型训练而更新的缓冲区和参数。您可能要保存的其他项目包括您未启用的时期，最新记录的训练损失，外部`torch.nn.Embedding` 图层等。

要保存多个组件，请将它们组织在字典中并用于 torch.save()序列化字典。常见的PyTorch约定是使用`.tar`文件扩展名保存这些检查点。



#### 将多个模型保存在一个文件中

```python
torch.save({
    'modelA_state_dict': modelA.state_dict(),
    'modelB_state_dict': modelB.state_dict(),
    'optimizerA_state_dict': optimizerA.state_dict(),
    'optimizerB_state_dict': optimizerB.state_dict(),
    ...
    }, PATH)

modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```



####　使用来自不同模型的参数进行热启动模型

```python
torch.save(modelA.state_dict(), PATH)

modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

在转移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见方案。利用经过训练的参数，即使只有少数几个可用的参数，也将有助于热启动训练过程，并希望与从头开始训练相比，可以更快地收敛模型。

无论是从缺少某些键的部分state_dict加载，还是要加载比要加载的模型更多的key 的`state_dict`，都可以在函数中将strict参数设置为**False**，`load_state_dict()`以忽略不匹配的键。

如果要将参数从一层加载到另一层，但是某些键不匹配，只需更改要加载的`state_dict`中参数键的名称， 以匹配要加载到的模型中的键。



### 跨设备保存和加载模型







# API

## torch

***



### Tensor

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| torch.**is_tensor**(obj)                                     |                                                              |
| torch.**is_storage**(obj)                                    |                                                              |
| torch.**is_floating_point**(input) -> (bool)                 | 如果输入的数据类型是浮点数据类型，即Torch.Float 64、Torch.Float 32和Torch.Float 16之一，则返回True。 |
| torch.**set_default_dtype**(d)                               | 将默认的浮点dtype设置为d，此类型将用作torch.tensor()中类型推断的默认浮点类型。 |
| torch.**get_default_dtype**() → torch.dtype                  |                                                              |
| torch.**set_default_tensor_type**(t)                         |                                                              |
| torch.**numel**(input) → int                                 | 返回输入张量中的元素总数。                                   |
| torch.**set_printoptions**(precision=None, threshold=None, <br />edgeitems=None, linewidth=None, profile=None, sci_mode=None) |                                                              |
| torch.**set_flush_denormal**(mode) → bool                    | 禁用CPU上的正常浮点数。                                      |



#### Creation Ops

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| torch.**tensor**(data, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor |                                                              |
| torch.**sparse_coo_tensor**(indices, values, size=None, dtype=None, device=None, requires_grad=False) → Tensor | 构造coo(Rdinate)格式的稀疏张量，在给定值的索引处使用非零元素构造稀疏张量。稀疏张量可以不合并，在这种情况下，索引中有重复坐标，该索引的值是所有重复值条目的之和：torch.sparse。 |
| torch.**as_tensor**(data, dtype=None, device=None) → Tensor  | 把数据转换成`torch.Tensor`。如果数据已经是具有相同dtype和设备的张量，则不执行复制，否则，如果数据张量`requires_grad=True`，则将返回一个新的张量，并保留计算图。同样，如果数据是对应的dtype的ndarray，并且设备是CPU，则不执行复制。 |
| torch.**as_strided**(input, size, stride, storage_offset=0) → Tensor | Create a view of an existing torch.Tensor `input` with specified `size`, `stride` and `storage_offset`. |
| torch.**from_numpy**(ndarray) → Tensor                       | Creates a [`Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) from a `numpy.ndarray`. |
| torch.**zeros**(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**zeros_like**(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**ones**(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**ones_like**(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**arange**(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**linspace**(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**logspace**(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**eye**(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**empty**(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor |                                                              |
| torch.**empty_like**(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**empty_strided**(size, stride, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) → Tensor |                                                              |
| torch.**full**(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor | Returns a tensor of size `size` filled with `fill_value`.    |
| torch.**full_like**(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                                              |
| torch.**quantize_per_tensor**(input, scale, zero_point, dtype) → Tensor | 将浮点张量转换为具有给定标度和零点的量子化张量。             |
| torch.**quantize_per_channel**(input, scales, zero_points, axis, dtype) → Tensor |                                                              |
|                                                              |                                                              |



#### Indexing, Slicing, Joining, Mutating Ops



|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| torch.**cat**(tensors, dim=0, out=None) → Tensor             |                                                              |
| torch.**chunk**(input, chunks, dim=0) → List of Tensors      | 将张量分成一定数量的块。如果给定维DIM的张量大小不能被块整除，那么最后的块将更小。 |
| torch.**gather**(input, dim, index, out=None, sparse_grad=False) → Tensor | 沿dim指定的轴收集值。                                        |
| torch.**index_select**(input, dim, index, out=None) → Tensor | 返回一个新的张量，它使用索引中的条目来索引输入张量，该张量是一个*LongTensor*。 |
| torch.**masked_select**(input, mask, out=None) → Tensor      | 返回一个新的一维张量，它根据布尔掩码掩码来索引输入张量，这是一个*BoolTensor*。 |
| torch.**narrow**(input, dim, start, length) → Tensor         | 返回一个新的张量，即输入张量的缩小版本。从开始到开始+长度输入维度DIM。返回的张量和输入张量共享相同的底层存储。 |
| torch.**nonzero**(input, *, out=None, as_tuple=False) → LongTensor or tuple of LongTensors |                                                              |
| torch.**reshape**(input, shape) → Tensor                     |                                                              |
| torch.**split**(tensor, split_size_or_sections, dim=0)       |                                                              |
| torch.**squeeze**(input, dim=None, out=None) → Tensor        |                                                              |
| torch.**stack**(tensors, dim=0, out=None) → Tensor           |                                                              |
| torch.**t**(input) → Tensor                                  |                                                              |
| torch.**take**(input, index) → Tensor                        | 返回在给定索引处输入元素的新张量。输入张量被视为一维张量.结果与指标的形状相同。 |
| torch.**transpose**(input, dim0, dim1) → Tensor              | 返回一个张量，该张量是输入的转置版本。给出的维数DINE 0和DIND 1被交换。 |
| torch.**unbind**(input, dim=0) → seq                         | Removes a tensor dimension.                                  |
| torch.**unsqueeze**(input, dim, out=None) → Tensor           |                                                              |
| torch.**where**(condition, x, y) → Tensor<br />torch.**where**(condition) → tuple of LongTensor | torch.where(condition) is identical to torch.nonzero(condition, as_tuple=True). |
|                                                              |                                                              |
|                                                              |                                                              |



###　Random sampling

> torch.**seed**()

> torch.**manual_seed**(seed)

> torch.**initial_seed**()

> torch.**get_rng_state**()

> torch.**set_rng_state**(new_state)

> torch.**default_generator**

> torch.**bernoulli**(input, *, generator=None, out=None) → Tensor

> torch.**multinomial**(input, num_samples, replacement=False, *, generator=None, out=None) → LongTensor

> torch.**normal**()
>
> torch.**normal**(mean, std, *, generator=None, out=None) → Tensor
>
> torch.**normal**(mean=0.0, std, out=None) → Tensor
>
> torch.**normal**(mean, std=1.0, out=None) → Tensor



#### rand

> torch.**rand**(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

> torch.**rand_like**(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor

返回区间［０，１］上均匀分布的随机数填充的张量。



#### randint

> torch.**randint**(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

> torch.**randint_like**(input, low=0, high, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor



#### randn

> torch.**randn**(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

> torch.**randn_like**(input, dtype=None, layout=None, device=None, requires_grad=False) → Tensor

从均值为0和方差为1的正态分布中返回一个由随机数填充的张量(也称为标准正态分布)。



####  randperm

> torch.**randperm**(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → LongTensor

返回整数从0到n-1的随机排列。

```python
>>> torch.randperm(4)
tensor([2, 1, 0, 3])
```



#### In-place random sampling

- [`torch.Tensor.bernoulli_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.bernoulli_) - in-place version of [`torch.bernoulli()`](https://pytorch.org/docs/stable/torch.html#torch.bernoulli)
- [`torch.Tensor.cauchy_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.cauchy_) - numbers drawn from the Cauchy distribution
- [`torch.Tensor.exponential_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.exponential_) - numbers drawn from the exponential distribution
- [`torch.Tensor.geometric_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.geometric_) - elements drawn from the geometric distribution
- [`torch.Tensor.log_normal_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.log_normal_) - samples from the log-normal distribution
- [`torch.Tensor.normal_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.normal_) - in-place version of [`torch.normal()`](https://pytorch.org/docs/stable/torch.html#torch.normal)
- [`torch.Tensor.random_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.random_) - numbers sampled from the discrete uniform distribution
- [`torch.Tensor.uniform_()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.uniform_) - numbers sampled from the continuous uniform distribution







### Serialzation

#### Save

> torch.**save**(obj, f, pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>, pickle_protocol=2, _use_new_zipfile_serialization=False)

Parameters

- **obj** – saved object
- **f** – a file-like object (has to implement write and flush) or a string containing a file name
- **pickle_module** – module used for pickling metadata and objects
- **pickle_protocol** – can be specified to override the default protocol



#### Load

> torch.**load**(f, map_location=None, pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>, **pickle_load_args)

Parameters

- **f** – a file-like object (has to implement `read()`, :meth`readline`, :meth`tell`, and :meth`seek`), or a string containing a file name
- **map_location** – a function, [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device), string or a dict specifying how to remap storage locations
- **pickle_module** – module used for unpickling metadata and objects (has to match the `pickle_module` used to serialize file)
- **pickle_load_args** – (Python 3 only) optional keyword arguments passed over to `pickle_module.load()`and `pickle_module.Unpickler()`, e.g., `errors=...`.



### Math operations



#### Pointwise Ops



|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| torch.**abs**(input, out=None) → Tensor                      |                                                              |
| torch.**acos**(input, out=None) → Tensor                     |                                                              |
| torch.**add**()<br />torch.**add**(input, other, out=None)<br />torch.**add**(input, alpha=1, other, out=None) | 3、out=input+alpha×other                                     |
| torch.**addcdiv**(input, value=1, tensor1, tensor2, out=None) → Tensor | 执行按元素对tensor1除以tensor2，将结果乘以标量值并将其添加到输入中。 |
| torch.**addcmul**(input, value=1, tensor1, tensor2, out=None) → Tensor | out*i*=input*i*+value×tensor1*i*×tensor2*i*                  |
| torch.**angle**(input, out=None) → Tensor                    | 计算给定输入张量的单元角度(以弧度表示)。<br />torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))*180/3.14159<br />tensor([ 135.,  135,  -45]) |
| torch.**asin**(input, out=None) → Tensor                     |                                                              |
| torch.**atan**(input, out=None) → Tensor                     |                                                              |
| torch.**atan2**(input, other, out=None) → Tensor             |                                                              |
| torch.**bitwise_not**(input, out=None) → Tensor              | 计算按位计算，而不是按给定输入张量计算。输入张量必须是整型或布尔型。对于bool张量，它计算逻辑NOT。 |
| torch.**bitwise_xor**(input, other, out=None) → Tensor       |                                                              |
| torch.**ceil**(input, out=None) → Tensor                     | 返回一个新的张量，该张量包含输入元素的单元，最小整数大于或等于每个元素。 |
| torch.**clamp**(input, min, max, out=None) → Tensor          | 将输入中的所有元素夹紧到[min，max]范围内，并返回一个由此产生的张量： |
| torch.**conj**(input, out=None) → Tensor                     | 计算给定输入张量的单元共轭。                                 |
| torch.**cos**(input, out=None) → Tensor                      |                                                              |
| torch.**cosh**(input, out=None) → Tensor                     | 用输入元素的双曲余弦返回一个新的张量。                       |
| torch.**div**()<br />torch.**div**(input, other, out=None) → Tensor |                                                              |
| torch.**digamma**(input, out=None) → Tensor                  | 计算输入的伽马函数的对数导数。                               |
| torch.**erf**(input, out=None) → Tensor                      | 计算每个元素的错误函数。错误函数定义如下：                   |
| torch.**erfc**(input, out=None) → Tensor                     |                                                              |
| torch.**erfinv**(input, out=None) → Tensor                   |                                                              |
| torch.**exp**(input, out=None) → Tensor                      |                                                              |
| torch.**expm1**(input, out=None) → Tensor                    | 返回一个新的张量，其中元素的指数减去输入的1。                |
| torch.**floor**(input, out=None) → Tensor                    |                                                              |
| torch.**fmod**(input, other, out=None) → Tensor              | 计算除法的元素级余数。                                       |
| torch.**frac**(input, out=None) → Tensor                     | 计算输入中每个元素的小数部分。                               |
| torch.**imag**(input, out=None) → Tensor                     | 计算给定输入张量的元素级映射值。                             |
| torch.**lerp**(input, end, weight, out=None)                 | 两个张量的线性插值开始(输入)和结束基于一个标量或张量的权重，并返回结果的张量。 |
| torch.**lgamma**(input, out=None) → Tensor                   | 计算输入的伽马函数的对数。                                   |
| torch.**log**(input, out=None) → Tensor                      | 返回输入元素的自然对数(e)的新张量。                          |
| torch.**log10**(input, out=None) → Tensor                    |                                                              |
| torch.**log1p**(input, out=None) → Tensor                    | 返回一个新的张量，其自然对数为(1+输入)。                     |
| torch.**log2**(input, out=None) → Tensor                     |                                                              |
| torch.**logical_not**(input, out=None) → Tensor              |                                                              |
| torch.**logical_xor**(input, other, out=None) → Tensor       |                                                              |
| torch.**mul**()<br />torch.**mul**(input, other, out=None)   |                                                              |
| torch.**mvlgamma**(input, p) → Tensor                        |                                                              |
| torch.**neg**(input, out=None) → Tensor                      | 返回输入元素负数的新张量。                                   |
| torch.**polygamma**(n, input, out=None) → Tensor             |                                                              |
| torch.**pow**()                                              |                                                              |
| torch.**real**(input, out=None) → Tensor                     | 计算给定输入张量的元素实数。                                 |
| torch.**reciprocal**(input, out=None) → Tensor               | 返回一个具有输入元素倒数的新张量。                           |
| torch.**remainder**(input, other, out=None) → Tensor         | 计算除法的元素级余数。                                       |
| torch.**round**(input, out=None) → Tensor                    | 返回一个新的张量，每个输入元素四舍五入到最接近的整数。       |
| torch.**rsqrt**(input, out=None) → Tensor                    | 返回一个新的张量与每个输入元素的平方根的倒数。               |
| torch.**sigmoid**(input, out=None) → Tensor                  | $out_i = \frac{1}{1+e^{-input_i}}$                           |
| torch.**sign**(input, out=None) → Tensor                     | 返回一个带有输入元素符号的新张量。<br />tensor([ 0.7000, -1.2000,  0.0000,  2.3000])<br />tensor([ 1., -1.,  0.,  1.]) |
| torch.**sin**(input, out=None) → Tensor                      |                                                              |
| torch.**sinh**(input, out=None) → Tensor                     |                                                              |
| torch.**sqrt**(input, out=None) → Tensor                     |                                                              |
| torch.**tan**(input, out=None) → Tensor                      |                                                              |
| torch.**tanh**(input, out=None) → Tensor                     |                                                              |
| torch.**trunc**(input, out=None) → Tensor                    | 返回一个新的张量，其中包含输入元素的截断整数值。             |
|                                                              |                                                              |
|                                                              |                                                              |



#### Reduction Ops



|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| torch.**argmax**(input) → LongTensor<br />torch.**argmax**(input, dim, keepdim=False) → LongTensor |                                                              |
| torch.**argmin**(input) → LongTensor<br />torch.**argmin**(input, dim, keepdim=False, out=None) → LongTensor |                                                              |
| torch.**dist**(input, other, p=2) → Tensor                   | 返回(输入-其他)的p-范数。                                    |
| torch.**logsumexp**(input, dim, keepdim=False, out=None)     |                                                              |
| torch.**mean**(input) → Tensor<br />torch.**mean**(input, dim, keepdim=False, out=None) → Tensor |                                                              |
| torch.**median**(input) → Tensor<br />torch.**median**(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor) | 中位数                                                       |
| torch.**mode**(input, dim=-1, keepdim=False, values=None, indices=None) -> (Tensor, LongTensor) | 返回一个命名元组(值、索引)，其中值是给定维DIM中输入张量的每一行的模式值，即最常出现在该行中的值，索引是找到的每个模式值的索引位置。 |
| torch.**norm**(input, p='fro', dim=None, keepdim=False, out=None, dtype=None) | 返回给定张量的矩阵范数或向量范数。                           |
| torch.**prod**(input, dtype=None) → Tensor<br />torch.**prod**(input, dim, keepdim=False, dtype=None) → Tensor | 返回给定维DIM中输入张量的每一行的乘积。                      |
| torch.**std**(input, unbiased=True) → Tensor<br />torch.**std**(input, dim, keepdim=False, unbiased=True, out=None) → Tensor |                                                              |
| torch.**std_mean**(input, unbiased=True) -> (Tensor, Tensor)<br />torch.**std_mean**(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor) | 返回输入张量中所有元素的标准差和平均值。                     |
| torch.**sum**(input, dtype=None) → Tensor<br />torch.**sum**(input, dim, keepdim=False, dtype=None) → Tensor |                                                              |
| torch.**unique**(input, sorted=True, return_inverse=False, return_counts=False, dim=None) | 返回输入张量的唯一元素。                                     |
| torch.**unique_consecutive**(input, return_inverse=False, return_counts=False, dim=None) | 从每个连续的等效元素组中删除除第一个元素之外的所有元素。     |
| torch.**var**(input, unbiased=True) → Tensor<br />torch.**var**(input, dim, keepdim=False, unbiased=True, out=None) → Tensor | 返回输入张量中所有元素的方差。                               |
| torch.**var_mean**(input, unbiased=True) -> (Tensor, Tensor)<br />torch.**var_mean**(input, dim, keepdim=False, unbiased=True) -> (Tensor, Tensor) |                                                              |
|                                                              |                                                              |



#### Comparison Ops



|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| torch.**allclose**(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) → bool | 该函数检查所有输入和其他输入是否满足以下条件：∣input−other∣≤atol+rtol×∣other∣ |
| torch.**argsort**(input, dim=-1, descending=False, out=None) → LongTensor | 返回按值按升序对张量进行排序的索引。                         |
| torch.**eq**(input, other, out=None) → Tensor                | 计算按元素相等。第二个参数可以是数字或张量，其形状可与第一个参数一起广播。<br />返回一个 `torch.BoolTensor` 在比较为真的每个位置包含True |
| torch.**equal**(input, other) → bool                         | 如果两个张量具有相同的大小和元素，则为true，否则为false。    |
| torch.**ge**(input, other, out=None) → Tensor                | input 大于等于 other                                         |
| torch.**gt**(input, other, out=None) → Tensor                | input 大于 other                                             |
| torch.**isfinite**()                                         | 返回一个新的张量，如果每个单元都是有限的，则用布尔元素表示。 |
| torch.**isinf**(tensor)                                      | 返回一个新的张量，其布尔元素表示每个元素是否为+/-INF。       |
| torch.**isnan**()                                            |                                                              |
| torch.**kthvalue**(input, k, dim=None, keepdim=False, out=None) -> (Tensor, LongTensor) | 返回一个名称元组(值、索引)，其中值是给定维DIM中输入张量的每一行的k个最小元素。索引是找到的每个元素的索引位置。 |
| torch.**le**(input, other, out=None) → Tensor                |                                                              |
| torch.**lt**(input, other, out=None) → Tensor                |                                                              |
| torch.**max**(input) → Tensor<br />torch.**max**(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)<br />torch.**max**(input, other, out=None) → Tensor |                                                              |
| torch.**min**(input) → Tensor<br />torch.**min**(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)<br />torch.**min**(input, other, out=None) → Tensor |                                                              |
| torch.**ne**(input, other, out=None) → Tensor                |                                                              |
| torch.**sort**(input, dim=-1, descending=False, out=None) -> (Tensor, LongTensor) |                                                              |
| torch.**topk**(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor) |                                                              |



#### Spectral Ops

|                                                              |                                               |
| ------------------------------------------------------------ | --------------------------------------------- |
| torch.**fft**(input, signal_ndim, normalized=False) → Tensor | Complex-to-complex Discrete Fourier Transform |
| torch.**ifft**(input, signal_ndim, normalized=False) → Tensor |                                               |
| torch.**rfft**(input, signal_ndim, normalized=False, onesided=True) → Tensor |                                               |
| torch.**irfft**(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None) → Tensor |                                               |
| torch.**stft**(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True) |                                               |
| torch.**bartlett_window**(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                               |
| torch.**blackman_window**(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                               |
| torch.**hamming_window**(window_length, periodic=True, alpha=0.54, beta=0.46, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                               |
| torch.**hann_window**(window_length, periodic=True, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor |                                               |



#### Other Operations

|                                                              |                                 |
| ------------------------------------------------------------ | ------------------------------- |
| torch.**bincount**(input, weights=None, minlength=0) → Tensor | 计算非负INT数组中每个值的频率。 |
| torch.**broadcast_tensors**(*tensors) → List of Tensors      |                                 |
| torch.**cartesian_prod**(*tensors)                           |                                 |
| torch.**cdist**(x1, x2, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary') |                                 |
| torch.**combinations**(input, r=2, with_replacement=False) → seq |                                 |
| torch.**cross**(input, other, dim=-1, out=None) → Tensor     |                                 |
| torch.**cumprod**(input, dim, out=None, dtype=None) → Tensor |                                 |
| torch.**cumsum**(input, dim, out=None, dtype=None) → Tensor  | 返回维度DIM中输入元素的累积和。 |
| torch.**diag**(input, diagonal=0, out=None) → Tensor         |                                 |
| torch.**diag_embed**(input, offset=0, dim1=-2, dim2=-1) → Tensor |                                 |
| torch.**diagflat**(input, offset=0) → Tensor                 |                                 |
| torch.**diagonal**(input, offset=0, dim1=0, dim2=1) → Tensor |                                 |
| torch.**einsum**(equation, *operands) → Tensor               |                                 |
| torch.**flatten**(input, start_dim=0, end_dim=-1) → Tensor   |                                 |
| torch.**flip**(input, dims) → Tensor                         |                                 |
| torch.**rot90**(input, k, dims) → Tensor                     |                                 |
| torch.**histc**(input, bins=100, min=0, max=0, out=None) → Tensor |                                 |
| torch.**meshgrid**(*tensors, **kwargs)                       |                                 |
| torch.**renorm**(input, p, dim, maxnorm, out=None) → Tensor  |                                 |
| torch.**repeat_interleave**(input, repeats, dim=None) → Tensor |                                 |
| torch.**roll**(input, shifts, dims=None) → Tensor            |                                 |
| torch.**tensordot**(a, b, dims=2)                            |                                 |
| torch.**trace**(input) → Tensor                              |                                 |
| torch.**tril**(input, diagonal=0, out=None) → Tensor         |                                 |
| torch.**tril_indices**(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) → Tensor |                                 |
| torch.**triu**(input, diagonal=0, out=None) → Tensor         |                                 |
| torch.**triu_indices**(row, col, offset=0, dtype=torch.long, device='cpu', layout=torch.strided) → Tensor |                                 |
|                                                              |                                 |





## torch.nn

***

### Parameters

> CLASS torch.nn.**Parameter**

一种可视为模块参数的张量。

Parameters是`Tensor`的子类，当与`Module`一起使用时具有非常特殊的属性--当它们被赋值为Module属性时，它们会自动添加到参数列表中，并且会出现在`parameters()`迭代器中。指定一个张量没有这样的效果。这是因为您可能希望在模型中缓存一些临时状态，比如RNN的最后一个隐藏状态。如果没有Parameters这样的类，这些临时人员也会被注册。

**Args:**

- **data** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – parameter tensor.
- **requires_grad** (*bool**,* *optional*) – if the parameter requires gradient. See [Excluding subgraphs from backward](https://pytorch.org/docs/stable/notes/autograd.html#excluding-subgraphs) for more details. Default: True



###　Containers

#### Module

> CLASS torch.nn.**Module**

所有神经网络模块的基类。

模块还可以包含其他模块，允许将它们嵌套在树结构中。您可以将子模块分配为常规属性：



| Methods                                                      |      |
| ------------------------------------------------------------ | ---- |
| `add_module`(*name*, *module*)                               |      |
| `apply`(*fn*)                                                |      |
| `buffers`(*recurse=True*)                                    |      |
| `children`()                                                 |      |
| `cpu`()                                                      |      |
| `cuda`(*device=None*)                                        |      |
| `double`()                                                   |      |
| `dump_patches` *= FALSE*                                     |      |
| `eval`()                                                     |      |
| `extra_repr`()                                               |      |
| `float`()                                                    |      |
| `forward`(**input*)                                          |      |
| `half`()                                                     |      |
| `load_state_dict`(*state_dict*, *strict=True*)               |      |
| `modules`()                                                  |      |
| `named_buffers`(*prefix=''*, *recurse=True*)                 |      |
| `named_children`()                                           |      |
| `named_modules`(*memo=None*, *prefix=''*)                    |      |
| `named_parameters`(*prefix=''*, *recurse=True*)              |      |
| `parameters`(*recurse=True*)                                 |      |
| `register_backward_hook`(*hook*)                             |      |
| `register_buffer`(*name*, *tensor*)                          |      |
| `register_forward_hook`(*hook*)                              |      |
| `register_forward_pre_hook`(*hook*)                          |      |
| `register_parameter`(*name*, *param*)                        |      |
| `requires_grad_`(*requires_grad=True*)                       |      |
| `state_dict`(*destination=None*, *prefix=''*, *keep_vars=False*) |      |
| `to`(**args*, ***kwargs*)                                    |      |
| `train`(*mode=True*)                                         |      |
| `type`(*dst_type*)                                           |      |
| `zero_grad`()                                                |      |
|                                                              |      |



### Convolution layers

#### Conv2d

> torch.nn.**Conv2d**(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')





### Pooling layers



#### AvgPool2d

> torch.nn.**AvgPool2d**(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
>

在由多个输入平面组成的输入信号上应用2D平均池。















#### ConvTranspose2d

> torch.nn.**ConvTranspose2d**(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

在由多个输入平面组成的输入图像上应用2D转置卷积算子。

该模块可以看作是Conv2d相对于其输入的梯度。它也被称为分步卷积或反褶积(虽然它不是实际的反褶积操作)。

$H_{out} = (H_{in}-1)\times stride[0]-2\times padding[0]+dilation[0]\times(kernel\_size[0]-1)+output\_padding[0]+1$



### Normalization layers



#### BatchNorm2d

> torch.nn.**BatchNorm2d**(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

在4D输入上应用批归一化.
$$
\begin{equation}
y=\frac{x-\mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}} * \gamma+\beta
\end{equation}
$$
期望和标准差是在小批次上按维数计算的。$\gamma$ 和 $\beta$ 是大小为C的可学习参数向量(其中C是输入大小)。默认情况下，$\gamma$ 的元素设置为1，$\beta$的元素设置为0。

此外，在默认情况下，在培训期间，该层会运行其计算平均值和方差的估计值，然后在评估期间使用这些估计值进行规范化。运行中的估计值保持在默认动量为0.1的情况下。

如果`track_running_stats`设置为false，则此层将不再保持运行估计，而批处理统计信息也将在评估期间使用。

由于批归一化是在C维上进行的，计算(N，H，W)片上的统计数据，所以通常称为空间批规范化(Spatial Batch Normalization)。

**Args:**

- **num_features** – C from an expected input of size (N, C, H, W)
- **eps** – a value added to the denominator for numerical stability. Default: 1e-5
- **momentum** – the value used for the running_mean and running_var computation. Can be set to `None` for cumulative moving average (i.e. simple average). Default: 0.1
- **affine** – a boolean value that when set to `True`, this module has learnable affine parameters. Default: `True`
- **track_running_stats** – a boolean value that when set to `True`, this module tracks the running mean and variance, and when set to `False`, this module does not track such statistics and always uses batch statistics in both training and eval modes. Default: `True`



### Recurrent layers

#### LSTM

> CLASS torch.nn.**LSTM**(*args, **kwargs)



**Parameters**

- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature). Default: `False`
- **dropout** – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`. Default: 0
- **bidirectional** – If `True`, becomes a bidirectional LSTM. Default: `False`



Inputs: input, (h_0, c_0)

- **input** of shape `(seq_len, batch, input_size)`: tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See [`torch.nn.utils.rnn.pack_padded_sequence()`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence) or[`torch.nn.utils.rnn.pack_sequence()`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence) for details.

- **h_0** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.

- **c_0** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.

  If (h_0, c_0) is not provided, both **h_0** and **c_0** default to zero.



Outputs: output, (h_n, c_n)

- **output** of shape (seq_len, batch, num_directions * hidden_size): tensor containing the output features (h_t)from the last layer of the LSTM, for each t. If a [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence) has been given as the input, the output will also be a packed sequence.

  For the unpacked case, the directions can be separated using `output.view(seq_len, batch,num_directions, hidden_size)`, with forward and backward being direction 0 and 1 respectively. Similarly, the directions can be separated in the packed case.

- **h_n** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len.

  Like *output*, the layers can be separated using `h_n.view(num_layers, num_directions, batch,hidden_size)` and similarly for *c_n*.

- **c_n** of shape (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t = seq_len.



Variables

- **~LSTM.weight_ih_l[k]** – the learnable input-hidden weights of the \text{k}^{th}k*t**h* layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) for k = 0. Otherwise, the shape is (4*hidden_size, num_directions * hidden_size)
- **~LSTM.weight_hh_l[k]** – the learnable hidden-hidden weights of the \text{k}^{th}k*t**h* layer (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size)
- **~LSTM.bias_ih_l[k]** – the learnable input-hidden bias of the \text{k}^{th}k*t**h* layer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)
- **~LSTM.bias_hh_l[k]** – the learnable hidden-hidden bias of the \text{k}^{th}k*t**h* layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size)





### Loss

> CLASS torch.nn.**L1Loss**(size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**MSELoss**(size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**CrossEntropyLoss**(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

> CLASS torch.nn.**CTCLoss**(blank=0, reduction='mean', zero_infinity=False)

> CLASS torch.nn.**NLLLoss**(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

> CLASS torch.nn.**PoissonNLLLoss**(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')

> CLASS torch.nn.**KLDivLoss**(size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**BCELoss**(weight=None, size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**BCEWithLogitsLoss**(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

> CLASS torch.nn.**MarginRankingLoss**(margin=0.0, size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**HingeEmbeddingLoss**(margin=1.0, size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**MultiLabelMarginLoss**(size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**SmoothL1Loss**(size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**SoftMarginLoss**(size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**MultiLabelSoftMarginLoss**(weight=None, size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**CosineEmbeddingLoss**(margin=0.0, size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**MultiMarginLoss**(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')

> CLASS torch.nn.**TripletMarginLoss**(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')



#### L1Loss

> CLASS torch.nn.**L1Loss**(size_average=None, reduce=None, reduction='mean')



####  CTCLoss

> CLASS torch.nn.**CTCLoss**(blank=0, reduction='mean', zero_infinity=False)

The Connectionist Temporal Classification loss.

计算连续(未分段)时间序列与目标序列之间的损失。CTCLoss对输入到目标的可能对齐概率进行求和，产生相对于每个输入节点可微的损失值。输入到目标的对齐被假定为“多对一”，这限制了目标序列的长度，因此它必须是小于等于输入长度。



**Parameters**：

- **blank** (`python:int,` `optional`) – 空白标签所在的label值，默认为0
- **reduction** (`string`, `optional`) – 处理output losses的方式，string类型，可选'none' 、 'mean' 及 'sum'，'none'表示对output losses不做任何处理，'mean' 则对output losses取平均值处理，'sum'则是对output losses求和处理，默认为'mean' 。
- **zero_infinity** (`bool`, `optional`) – 是否为零的无限损耗和相关的梯度。默认值:“假”的无限损失主要发生在输入太短而不能对准目标的时候。



Shape:

- **Log_probs**: Tensor of size$ (T, N, C)$, where $T = \text{input length}$ , $N = \text{batch size}$  , and $C = \text{number of classes (including blank)}$ . The logarithmized probabilities of the outputs e.g. obtained with `torch.nn.functional.log_softmax()`

- **Targets**: 大小为$（N，S）$或$（\operatorname {sum}（\text {target_lengths}））$ 的张量，

  其中$ N = \text{batch_size} $ 和 $ S = \text {最大目标长度}$ ，如果shape为$（N，S）$。 它代表目标序列。 目标序列中的每个元素都是一个类索引。 并且目标索引不能为空（默认= 0）。 在（N，S）形式中，将目标填充到最长序列的长度并进行堆叠。 在$（\operatorname {sum}（\text {target_lengths}））$格式中，假定目标未填充并且在1维内串联。

- **Input_lengths**: Tuple or tensor of size $(N)$ , 

  $N = \text{batch size}$ . 它代表输入的长度（每个必须为$ \leq T $）。 并且在序列被填充为相等长度的假设下，为每个序列指定长度以实现屏蔽。

- **Target_lengths**: Tuple or tensor of size $(N)$, where $N = \text{batch size}$  . 

  代表目标的长度。 在将序列填充为相等长度的假设下，为每个序列指定长度以实现屏蔽。如果目标形状为$（N,S）$，则target_lengths实际上是每个目标序列的停止索引$ S_n $，这样批次中每个目标的 $target_n =targets[n，0：s_n]$ 。 长度必须分别为$\leq S$。如果目标是作为单个目标的并置的1d张量给出的，则target_lengths必须加起来为张量的总长度。

- **Output**: scalar. If `reduction` is `'none'`, then $(N)$, where $N = \text{batch size}$ 



```python
>>> T = 50      # Input sequence length
>>> C = 20      # Number of classes (including blank)
>>> N = 16      # Batch size
>>> S = 30      # Target sequence length of longest target in batch
>>> S_min = 10  # Minimum target length, for demonstration purposes
>>>
>>> # Initialize random batch of input vectors, for *size = (T,N,C)
>>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
>>>
>>> # Initialize random batch of targets (0 = blank, 1:C = classes)
>>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
>>>
>>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
>>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
>>> ctc_loss = nn.CTCLoss()
>>> loss = ctc_loss(input, target, input_lengths, target_lengths)
>>> loss.backward()
```











### Utilities

#### clip_grad_norm_

> torch.nn.utils.**clip_grad_norm_**(parameters, max_norm, norm_type=2)

裁剪参数可迭代的梯度范数Clips gradient norm of an iterable of parameters.

范数是在所有梯度上一起计算的，就好像它们被串联到单个矢量中一样。渐变就地修改



#### clip_grad_value_

> torch.nn.utils.**clip_grad_value_**(parameters, clip_value)
>

将可迭代参数的梯度剪切为指定值



#### parameters_to_vector

> torch.nn.utils.**parameters_to_vector**(parameters)

Convert parameters to one vector



#### vector_to_parameters

> torch.nn.utils.**vector_to_parameters**(vec, parameters)



#### ...



#### PackedSequence

> torch.nn.utils.rnn.**PackedSequence**(data, batch_sizes=None, sorted_indices=None, unsorted_indices=None)

保有打包序列的数据和batch_sizes列表。所有RNN模块都将打包序列作为输入。

此类的实例永远不要手动创建。它们应通过`pack_padded_sequence（）`之类的函数实例化。

批次大小代表批次中每个序列步骤的数量元素，而不是传递给pack_padded_sequence（）的不同序列长度。例如，给定数据`abc`和`x`，PackedSequence将包含`batch_sizes = [2,1,1]`的数据axbc。



**Variables：**

- **~PackedSequence.data** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor containing packed sequence
- **~PackedSequence.batch_sizes** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of integers holding information about the batch size at each sequence step
- **~PackedSequence.sorted_indices** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – Tensor of integers holding how this[`PackedSequence`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence) is constructed from sequences.
- **~PackedSequence.unsorted_indices** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)*,* *optional*) – Tensor of integers holding how this to recover the original sequences with correct order.



#### pack_padded_sequence

> torch.nn.utils.rnn.**pack_padded_sequence**(input, lengths, batch_first=False, enforce_sorted=True)

打包一个Tensor，其中包含可变长度的填充序列。

输入的大小可以为T x B x \*，其中T是最长序列的长度（等于lengths [0]），B是批处理大小，*是任意数量的尺寸（包括0）。如果batch_first为True，则需要输入B x T x *。

对于未排序的序列，请使用enforce_sorted= False。 如果enforce_sorted为True，则序列应按长度降序排列，即，input [：，0]应该是最长的序列，而input [：，B-1]应该是最短的序列。 enforce_sorted= True仅对于ONNX导出是必需的。

此函数接受至少具有二维的任何输入。 您可以将其应用于包装标签，并与它们一起使用RNN的输出直接计算损失。 可以通过访问PackedSequence对象的.data属性来获取Tensor。



**Parameters**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – padded batch of variable length sequences.
- **lengths** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – list of sequences lengths of each batch element.
- **batch_first** (*bool**,* *optional*) – if `True`, the input is expected in `B x T x *` format.
- **enforce_sorted** (*bool**,* *optional*) – if `True`, the input is expected to contain sequences sorted by length in a decreasing order. If `False`, this condition is not checked. Default: `True`.

**Returns**

a [`PackedSequence`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.PackedSequence) object



#### pad_packed_sequence

> torch.nn.utils.rnn.**pad_packed_sequence**(sequence, batch_first=False, padding_value=0.0, total_length=None)

填充打包的可变长度序列批次



#### pad_sequence

> torch.nn.utils.rnn.**pad_sequence**(sequences, batch_first=False, padding_value=0)

用padding_value填充可变长度张量的列表

`pad_sequence`沿新维度堆叠张量列表，并将它们填充为相等的长度。例如，如果输入是大小为`L x *`的序列列表，并且batch_first为False，否则为`T x B x *`。

`B`是批量大小。它等于序列中元素的数量。 `T`是最长序列的长度。 `L`是序列的长度。 `*`是任意数量的尾随尺寸，包括无。

此函数返回张量为`T x B x *`或`B x T x *`的张量，其中T是最长序列的长度。此函数假定序列中所有张量的尾随尺寸和类型相同。

```python
>>> from torch.nn.utils.rnn import pad_sequence
>>> a = torch.ones(25, 300)
>>> b = torch.ones(22, 300)
>>> c = torch.ones(15, 300)
>>> pad_sequence([a, b, c]).size()
torch.Size([25, 3, 300])
```



Parameters

- **sequences** (list[Tensor]) – list of variable length sequences.
- **batch_first** (bool, optional) – output will be in `B x T x *` if True, or in `T x B x *` otherwise
- **padding_value** (python:float, optional) – value for padded elements. Default: 0.

Returns

- Tensor of size `T x B x *` if `batch_first` is `False`. Tensor of size `B x T x *` otherwise





#### pack_sequence

> torch.nn.utils.rnn.**pack_sequence**(sequences, enforce_sorted=True)

打包可变长度张量的列表

`sequences` should be a list of Tensors of size `L x *`, where L is the length of a sequence and * is any number of trailing dimensions, including zero.

```python
>>> from torch.nn.utils.rnn import pack_sequence
>>> a = torch.tensor([1,2,3])
>>> b = torch.tensor([4,5])
>>> c = torch.tensor([6])
>>> pack_sequence([a, b, c])
PackedSequence(data=tensor([ 1,  4,  6,  2,  5,  3]), batch_sizes=tensor([ 3,  2,  1]))
```





## torch.cuda

该软件包增加了对CUDA张量类型的支持，该类型实现与CPU张量相同的功能，但是它们利用GPU进行计算。

它是延迟初始化的，因此您始终可以导入它，并使用`is_available（）`确定您的系统是否支持CUDA。

***

















## torch.nn.functional

***



### interpolate

> torch.nn.functional.**interpolate**(*input*, *size=None*, *scale_factor=None*, *mode='nearest'*, *align_corners=None*)

向下/向上采样输入到给定的`size`或给定的比例因子`scale_factor`.

插值算法是由`mode`决定的。

目前支持时间、空间和空间采样，即期望输入为三维、四维或五维空间形状.

输入尺寸以如下形式解释：`mini-batch x channels x [optional depth] x [optional height] x width.`

适用于调整大小的模式有：`nearest`，`linear` (3d-only)，`bilinear`，`bicubic`(4D-only)，`trilinear`(5D-only)，`area`。

**参数：**

- *align_corners ( bool )*：几何上，我们把输入和输出的像素看作是正方形而不是点。如果设置为`True`，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。如果设置为`False`，则输入和输出张量根据其角点像素的角点对齐，并且插值使用边界外值填充，使得当缩放因子保持不变时，此操作与输入大小无关。这只有在模式为“线性”、“双线性”、“双三线性”或“三线性”时才有效果。默认值：false



### grid_sample

> torch.nn.functional.**grid_sample**(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)

给定`输入`和流场`网格(grid)`，使用输入值和网格中的像素位置来计算输出。当前，仅支持空间（4-D）和体积（5-D）输入。

在4-D中，`input`大小$(N,C,H_{in},W_{in})$ ，`grid`大小$(N,H_{out},W_{out},2)$ ，那么输出大小为$(N,C,H_{out},W_{out})$ 

对于每个输出位置`output [n，：，h，w]`，size-2的矢量`grid[n，h，w]`指定输入像素位置x和y，这些像素位置用于插值输出值`output [n，：,, h，w]`。

网格指定通过输入空间尺寸归一化的采样像素位置。 因此，它应具有`[-1，1]`范围内的大多数值。 例如，值x = -1，y = -1是输入的左上像素，值`x = 1，y = 1`是输入的右下像素。

如果grid的值超出[-1，1]的范围，则将按照padding_mode的定义处理相应的输出。 选项是

`padding_mode="zeros"` ：使用0表示超出范围的网格位置

`padding_mode="border"`：将边界值用于出站网格位置

`padding_mode="reflection"`：将边界所反映的位置的值用于边界外的网格位置。 对于远离边界的位置，它将一直被反射直到成为边界，例如，（标准化的）像素位置x = -3.5被边界-1反射并变为x'= 1.5，然后被边界1反射并变为x' '= -0.5。



**Args：**

- **input** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input of shape $(N,C,H_{in},W_{in})$ (4-D case) 
- **grid** ([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – flow-field of shape $(N,H_{out},W_{out},2)$ (4-D case) 
- **mode** (*str*) – interpolation mode to calculate output values `'bilinear'` | `'nearest'`. Default: `'bilinear'`
- **padding_mode** (*str*) – padding mode for outside grid values `'zeros'` | `'border'` | `'reflection'`. Default: `'zeros'`
- **align_corners** (*bool,* *optional*) – Geometrically, we consider the pixels of the input as squares rather than points. If set to `True`, the extrema (`-1` and `1`) are considered as referring to the center points of the input’s corner pixels. If set to `False`, they are instead considered as referring to the corner points of the input’s corner pixels, making the sampling more resolution agnostic. This option parallels the `align_corners` option in [`interpolate()`](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate), and so whichever option is used here should also be used there to resize the input image before grid sampling. Default: `False`



### affine_grid

> torch.nn.functional.**affine_grid**(theta, size, align_corners=None)

生成二维或三维流场(采样网格)，给定一批仿射矩阵θ。此函数通常与`grid_sample()`一起使用来构建空间转换网络。

- **theta** (*Tensor*） – input batch of affine matrices with shape (N x 2 x 3 ) for 2D or (N x3 x 4) for 3D
- **size** (*torch.Size*) – the target output image size. (N*×*C*×*H*×*W for 2D or N*×*C*×*D*×*H*×*W for 3D) Example: torch.Size((32, 3, 24, 24))
- **align_corners** (*bool**,* *optional*) – if `True`, consider `-1` and `1` to refer to the centers of the corner pixels rather than the image corners. Refer to [`grid_sample()`](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample) for a more complete description. A grid generated by [`affine_grid()`](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.affine_grid) should be passed to [`grid_sample()`](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample) with the same setting for this option. Default: `False`



pytorch 中提供了對Tensor進行Crop的方法，可以使用GPU實現。具體函式是

`torch.nn.functional.affine_grid`和`torch.nn.functional.grid_sample`。前者用於生成二維網格，後者對輸入Tensor按照網格進行雙線性取樣。

下面進行簡單的實驗：

1. 首先生成一個1x1x5x5的Tensor變數
2. 裁剪視窗為x1 = 2.5, x2 = 4.5, y1 = 0.5, y2 = 3.5，size為1x1x3x2，根據座標設定theta矩陣
3. 進行裁剪，並與numpy計算結果相比較。

```python
a = torch.rand((1, 1, 5, 5))
print(a)

# x1 = 2.5, x2 = 4.5, y1 = 0.5, y2 = 3.5
# out_w = 2, out_h = 3
size = torch.Size((1, 1, 3, 2))
print(size)

# theta
theta_np = np.array([[0.5, 0, 0.75], [0, 0.75, 0]]).reshape(1, 2, 3)
theta = torch.from_numpy(theta_np)
print('theta:')
print(theta)
print()

flowfield = torch.nn.functional.affine_grid(theta, size)
sampled_a = torch.nn.functional.grid_sample(a, flowfield.to(torch.float32))
sampled_a = sampled_a.numpy().squeeze()
print('sampled_a:')
print(sampled_a)

# compute bilinear at (0.5, 2.5), using (0, 3), (0, 4), (1, 3), (1, 4)
# quickly compute(https://blog.csdn.net/lxlclzy1130/article/details/50922867)
print()
coeff = np.array([[0.5, 0.5]])
A = a[0, 0, 0:2, 2:2+2]
print('torch sampled at (0.5, 3.5): %.4f' % sampled_a[0,0])
print('numpy compute: %.4f' % np.dot(np.dot(coeff, A), coeff.T).squeeze())
```















## torch.utils.Data

***

PyTorch数据加载实用程序的核心是`torch.utils.data.DataLoader`类。

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```



### 数据集类型

DataLoader构造函数最重要的参数是DataSet，它指示要从DataSet加载数据的DataSet对象。PyTorch支持两种不同类型的数据集：

-  **Map-style datasets** 映射样式数据集是实现__getitem__()和__len__()协议的数据集，并表示来自(可能是非整数)索引/键到数据样本的映射。例如，当使用DataSet[idx]访问这样的数据集时，可以从磁盘上的文件夹读取idx-th图像及其相应的标签。
-  **Iterable-style datasets** 可迭代样式数据集是`IterableDataset`子类的一个实例，该子类实现`__iter__()`协议，并表示数据样本上的可迭代性。这种类型的数据集特别适用于随机读取昂贵甚至不可能的情况，以及批处理大小取决于获取的数据的情况。例如，这样的数据集(称为ITER(DataSet))可以返回从数据库、远程服务器甚至实时生成的日志中读取的数据流。



### 数据装载顺序和采样器

对于可迭代样式的数据集，数据加载顺序完全由用户定义的迭代控制.这允许更容易地实现块读取和动态批处理大小(例如，每次生成一个批处理样本)。

本节的其余部分涉及到`map-style datasets`的情况。`torch.utils.data.Sampler`用于指定在数据加载中使用的索引/键的顺序。它们表示数据集索引上的可迭代对象。例如，在使用随机梯度下降(SGD)的常见情况下，一个采样器可以随机排列一个指数列表并一次生成每个指标，或者为小型批量SGD生成少量的指数。

将根据DataLoader的洗牌参数自动构造一个顺序的或洗牌的采样器。或者，用户可以使用SAMPLER参数来指定自定义的SAMPLER对象，该对象每次都会生成下一个索引/键来获取。



### 加载分批和非分批数据

`DataLoader`支持通过参数` batch_size`、`drop_last`和`batch_sampler`自动整理单个获取的数据样本。



#### Automatic batching (default)

这是最常见的情况，并对应于获取一小批数据并将其排序为批处理样本，即包含以一维为批处理维度(通常为第一个)的张量。

当`batch_size` (默认`1`) 不是 `None`, `DataLoader`产生分批样本，而不是单个样本。`batch_size` and`drop_last` arguments are used to specify how the data loader obtains batches of dataset keys. For map-style datasets, users can alternatively specify `batch_sampler`, which yields a list of keys at a time.



#### Disable automatic batching

在某些情况下，用户可能希望在DataSet代码中手动处理分批，或者只是加载单个示例。例如，直接加载分批数据(例如，从数据库批量读取数据或读取连续内存块)成本更低，或者分批大小依赖于数据，或者程序设计用于处理单个样本。在这些情况下，最好不要使用自动批处理(其中`collate_fn`用于整理示例)，而是让数据加载器直接返回DataSet对象的每个成员。

当`batch_size`和`batch_sampler`都为`None`(batch_sampler的默认值已经为None)时，将禁用自动批处理。从DataSet中获取的每个示例都使用传递为`collate_fn`参数的函数进行处理。

当禁用自动批处理时，默认的`collate_fn`只需将NumPy数组转换为PyTorch张量，并将其他所有内容保持不变。



#### Working with collate_fn

启用或禁用自动批处理时，`collate_fn`的使用略有不同。

当禁用自动批处理时，对每个单独的数据样本调用`collate_fn`，并从数据加载器迭代器获得输出。在这种情况下，默认的`collate_fn`只转换PyTorch张量中的NumPy数组。

启用自动批处理时，每次使用**数据样本列表**调用`collate_fn`。它将把输入样本整理成一批，以便从数据加载器迭代器获得结果。

本节的其余部分描述了本例中默认`collate_fn`的行为。

例如，如果每个数据样本包含一个3通道图像和一个完整的类标签，即数据集的每个元素都返回一个元组`(Image，class_index)`，则默认的`collate_fn`将这样的元组列表整理成一个分批图像张量和批类标签张量的单个元组。特别是，默认的`collate_fn`具有以下属性：

- 它总是将一个新维度作为批维度。
- 它自动将NumPy数组和Python数值转换为PyTorch张量。
- 它保留了数据结构，例如，如果每个示例都是字典，则它输出具有相同密钥集的字典，但作为值批处理张量(如果不能将值转换为张量，则列出列表)。lists、tuples、namedtuples等也是如此。

用户可以使用定制的`collate_fn`实现自定义批处理，例如沿着第一个维度以外的维度排序，填充不同长度的序列，或者添加对自定义数据类型的支持。



### 单进程和多进程数据加载

默认情况下，`DataLoader`使用单进程数据加载。

在Python进程中，全局解释器锁(Global解释器锁，GIL)防止在线程之间真正地完全并行化Python代码。为了避免在加载数据时阻塞计算代码，PyTorch提供了一个简单的开关，通过简单地将参数`num_workers`设置为正整数来执行多进程数据加载。



#### Single-process data loading (default)

在这种模式下，数据获取是在初始化DataLoader的同一进程中完成的。因此，数据加载可能会阻碍计算。然而，当用于在进程之间共享数据的资源(例如共享内存、文件描述符)受到限制时，或者当整个数据集很小并且可以完全加载在内存中时，这种模式可能是首选的。此外，单进程加载通常显示更易读的错误跟踪，因此对调试很有用.



#### Multi-process data loading

将参数`num_workers`设置为正整数将打开多进程数据加载，加载指定数量的加载器工作进程。

在这种模式下，每次创建DataLoader的迭代器(例如，当您调用枚举(Dataloader)时)，`num_worker进程`就会被创建。此时，`DataSet`、`collate_fn`和`work_init_fn`被传递给每个工作人员，在那里它们被用于初始化和获取数据。这意味着DataSet访问与其内部IO、Transform(包括`collate_fn`)一起在工作进程中运行。

`torch.utils.data.get_worker_info()`返回工作进程中的各种有用信息(包括worker id、dataset副本、初始种子等)，并在主进程中不返回任何信息。用户可以在DataSet代码和/或`work_init_fn`中使用此函数单独配置每个DataSet副本，并确定代码是否在工作进程中运行。例如，这对于切分数据集特别有用。

对于map样式的数据集，主进程使用采样器生成索引并将它们发送给工作人员。因此，任何随机洗牌都是在引导加载的主要过程中进行的。

对于可迭代样式的数据集，由于每个工作进程都获得DataSet对象的副本，简单的多进程加载通常会导致数据重复。使用`torch.utils.data.get_worker_info()`和/或`work_init_fn`，用户可以独立地配置每个副本。(关于如何实现这一点，请参阅IterableDataset文档。)出于类似的原因，在多进程加载中，DROP_LEST参数将删除每个员工的可迭代样式数据集副本的最后一批非完整批。

一旦到达迭代结束，或者迭代器变成垃圾收集，工人就会被关闭。

> 通常不建议在多进程加载中返回CUDA张量，因为在多处理过程中使用CUDA和共享CUDA张量有许多微妙之处(见多重处理中的CUDA)。相反，我们建议使用自动内存钉扎(即设置PIN_Memory=True)，使数据能够快速传输到启用CUDA的GPU。



**Platform-specific behaviors**:

由于工作人员依赖于Python `multiprocessing`，因此在Windows上的工作启动行为与Unix不同。

- On Unix, `fork()` is the default `multiprocessing` start method. Using `fork()`, child workers typically can access the `dataset` and Python argument functions directly through the cloned address space.
- On Windows, `spawn()` is the default `multiprocessing` start method. Using `spawn()`, another interpreter is launched which runs your main script, followed by the internal worker function that receives the `dataset`, `collate_fn` and other arguments through `pickle` serialization.

这种单独的序列化意味着您应该采取两个步骤来确保您在使用多进程数据加载时与Windows兼容：

- Wrap most of you main script’s code within `if __name__ == '__main__':` block, to make sure it doesn’t run again (most likely generating error) when each worker process is launched. You can place your dataset and [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) instance creation logic here, as it doesn’t need to be re-executed in workers.
- Make sure that any custom `collate_fn`, `worker_init_fn` or `dataset` code is declared as top level definitions, outside of the`__main__` check. This ensures that they are available in worker processes. (this is needed since functions are pickled as references only, not `bytecode`.)



**Randomness in multi-process data loading(多进程加载的随机性)**:

默认情况下，每个工作程序都将其PyTorch种子设置为base_seed + worker_id，其中base_seed 是由使用其RNG的主进程长时间生成的(因此，强制使用RNG状态)。但是，在初始化工作人员(W.G.，NumPy)时，可能会复制其他库的种子，从而导致每个工作人员返回相同的随机数。(见常见问题中的本节)。

在Worker_init_fn中，您可以使用`torch.utils.data.get_worker_info().seed`或`torch.Initialed()`访问每个工作者的PyTorch种子集，并在加载数据之前使用它为其他库添加种子。



### Memory Pinning









### API

#### DataLoader

> torch.utils.data.**DataLoader**(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None)

数据加载器。组合数据集和采样器，并提供对给定数据集的可迭代性。

DataLoader支持具有单进程或多进程加载、自定义加载顺序和可选自动批处理(排序)和内存钉扎的映射样式和可迭代样式数据集。









#### Dataset

> torch.utils.data.**Dataset**

表示数据集的抽象类。

表示从键到数据示例的映射的所有数据集都应该将其子类化。所有子类都应该覆盖`__getitem__()`，支持获取给定键的数据样本。子类还可以选择性地覆盖`__len__()`，这将由许多`Sampler`实现返回数据集的大小和`DataLoader`的默认选项。

> 默认情况下，DataLoader构造一个生成积分索引的索引采样器。要使它使用具有非整数索引/键的地图样式数据集，必须提供自定义采样器。





> torch.utils.data.**IterableDataset**

> torch.utils.data.**TensorDataset**(*tensors)

> torch.utils.data.**ConcatDataset**(datasets)

> torch.utils.data.**ChainDataset**(datasets)

> torch.utils.data.**Subset**(dataset, indices)

> torch.utils.data.**get_worker_info**()

> torch.utils.data.**random_split**(dataset, lengths)

> torch.utils.data.**Sampler**(data_source)

> torch.utils.data.**SequentialSampler**(data_source)

> torch.utils.data.**RandomSampler**(data_source, replacement=False, num_samples=None)

> torch.utils.data.**SubsetRandomSampler**(indices)

> torch.utils.data.**WeightedRandomSampler**(weights, num_samples, replacement=True)

> torch.utils.data.**BatchSampler**(sampler, batch_size, drop_last)

> torch.utils.data.distributed.**DistributedSampler**(dataset, num_replicas=None, rank=None, shuffle=True)



## torch.optim

optim是一个实现各种优化算法的包。大多数常用的方法已经得到了支持，并且接口足够通用，因此将来也可以很容易地集成更复杂的方法。



### How to use an optimizer

要使用`torch.optim`，您必须构造一个优化器对象，该对象将保存当前状态并根据计算的梯度更新参数。

要构造一个优化器，必须给它一个可迭代的参数(都应该是`Variable` s)来进行优化。然后，可以指定特定于优化器的选项，如学习速率、权重衰减等。

如果需要通过.cuda()将模型移动到GPU，在为其构造优化器之前，请先这样做。.cuda()之后的模型参数将是调用前的不同对象。通常，在构造和使用优化器时，应确保优化参数处于一致的位置。

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```



优化器还支持指定每个参数选项.要做到这一点，不要传递变量s的可迭代性，而是传递一个可迭代的`dict`。每个参数组将定义一个单独的参数组，并且应该包含一个`params`键，其中包含属于它的参数列表。其他键应与优化器接受的关键字参数匹配，并将用作此组的优化选项。

```python
optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
```

这意味着`Model.base`的参数将使用默认的学习速率1e-2，`model.classifier`的参数将使用1e-3的学习速率，对于所有的参数使用动量为0.9。



### Taking an optimization step

所有优化器都实现一个`step()`方法，该方法更新参数。它可以通过两种方式使用：

- **optimizer.step()**

  这是大多数优化器支持的简化版本。一旦使用`backward()`计算了梯度，就可以调用该函数。

  ```python
  for input, target in dataset:
      optimizer.zero_grad()
      output = model(input)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
  ```

  

- **optimizer.step(closure)**

  一些优化算法，如共轭梯度和LBFGS需要多次重新评估函数，因此您必须传递一个闭包，允许它们重新计算您的模型。闭包应该清除渐变，计算损失，并返回它。

  ```python
  for input, target in dataset:
      def closure():
          optimizer.zero_grad()
          output = model(input)
          loss = loss_fn(output, target)
          loss.backward()
          return loss
      optimizer.step(closure)
  ```



### Algorithms

> CLASS torch.optim.**Optimizer**(params, defaults)

> CLASS torch.optim.**Adadelta**(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

> CLASS torch.optim.**Adagrad**(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

> CLASS torch.optim.**Adam**(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

> CLASS torch.optim.**AdamW**(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

> CLASS torch.optim.**SparseAdam**(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

> CLASS torch.optim.**Adamax**(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

> CLASS torch.optim.**ASGD**(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

> CLASS torch.optim.**LBFGS**(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)

> CLASS torch.optim.**RMSprop**(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

> CLASS torch.optim.**Rprop**(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))

> CLASS torch.optim.**SGD**(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)



#### Optimizer

> CLASS torch.optim.**Optimizer**(params, defaults)

所有优化器的基类。

| Parameters              |                                                              |
| ----------------------- | ------------------------------------------------------------ |
| **params** (*iterable*) | an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized. |
| **defaults** – (dict)   | a dict containing default values of optimization options (used when a parameter group doesn’t specify them). |



| Methods                          |                                                              |
| -------------------------------- | ------------------------------------------------------------ |
| `add_param_group`(*param_group*) |                                                              |
| `load_state_dict`(*state_dict*)  |                                                              |
| `state_dict`()                   | Returns the state of the optimizer as a `dict`.<br />It contains two entries:<br />- `state` - a dict holding current optimization state. Its content<br />- `param_groups` - a dict containing all parameter groups |
| `step`(*closure*)                | 执行单个优化步骤(参数更新)。                                 |
| `zero_grad`()                    |                                                              |



### How to adjust Learning Rate

`torch.optim.lr_scheduler` 提供了几种基于`epochs`数调整学习速度的方法。

`torch.optim.lr_scheduler.ReduceLROnPlateau` 允许基于某些验证度量的动态学习速率降低。

学习速率调度应该在优化器更新之后应用；例如，您应该以这样的方式编写代码：

```python
>>> scheduler = ...
>>> for epoch in range(100):
>>>     train(...)
>>>     validate(...)
>>>     scheduler.step()
```



> CLASS torch.optim.lr_scheduler.**LambdaLR**(optimizer, lr_lambda, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**MultiplicativeLR**(optimizer, lr_lambda, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**StepLR**(optimizer, step_size, gamma=0.1, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**MultiStepLR**(optimizer, milestones, gamma=0.1, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**ExponentialLR**(optimizer, gamma, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**CosineAnnealingLR**(optimizer, T_max, eta_min=0, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**ReduceLROnPlateau**(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

> CLASS torch.optim.lr_scheduler.**CyclicLR**(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**OneCycleLR**(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)

> CLASS torch.optim.lr_scheduler.**CosineAnnealingWarmRestarts**(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)







## torch.utils.tensorboard

`SummaryWriter`类是记录TensorBoard使用和可视化数据的主入口

```python
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

...

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()

# tensorboard --logdir=runs
```

我们可以通过分层命名来对图进行分组。例如，“Loss/train”和“Loss/test”将被分组在一起而“Accuracy/train”和“Accuracy/test”将分别在TensorBoard接口分组。

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```





> CLASS torch.utils.tensorboard.writer.**SummaryWriter**(log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')

| Methods                                                      |                                                       |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| `__init__`(*log_dir=None*, *comment=''*, *purge_step=None*, *max_queue=10*, *flush_secs=120*, *filename_suffix=''*) | 创建一个SummaryWriter，它将为事件文件写出事件和摘要。 |
| `add_scalar`(*tag*, *scalar_value*, *global_step=None*, *walltime=None*) |                                                       |
| `add_scalars`(*main_tag*, *tag_scalar_dict*, *global_step=None*, *walltime=None*) |                                                       |
| `add_histogram`(*tag*, *values*, *global_step=None*, *bins='tensorflow'*, *walltime=None*, *max_bins=None*) |                                                       |
| `add_image`(*tag*, *img_tensor*, *global_step=None*, *walltime=None*, *dataformats='CHW'*) |                                                       |
| `add_images`(*tag*, *img_tensor*, *global_step=None*, *walltime=None*, *dataformats='NCHW'*) |                                                       |
| `add_figure`(*tag*, *figure*, *global_step=None*, *close=True*, *walltime=None*) |                                                       |
| `add_video`(*tag*, *vid_tensor*, *global_step=None*, *fps=4*, *walltime=None*) |                                                       |
| `add_audio`(*tag*, *snd_tensor*, *global_step=None*, *sample_rate=44100*, *walltime=None*) |                                                       |
| `add_text`(*tag*, *text_string*, *global_step=None*, *walltime=None*) |                                                       |
| `add_graph`(*model*, *input_to_model=None*, *verbose=False*) |                                                       |
| `add_embedding`(*mat*, *metadata=None*, *label_img=None*, *global_step=None*, *tag='default'*, *metadata_header=None*) |                                                       |
| `add_pr_curve`(*tag*, *labels*, *predictions*, *global_step=None*, *num_thresholds=127*, *weights=None*, *walltime=None*) |                                                       |
| `add_custom_scalars`(*layout*)                               |                                                       |
| `add_mesh`(*tag*, *vertices*, *colors=None*, *faces=None*, *config_dict=None*, *global_step=None*, *walltime=None*) |                                                       |
| `add_hparams`(*hparam_dict=None*, *metric_dict=None*)        |                                                       |
| `flush`()                                                    |                                                       |
| `close`()                                                    |                                                       |



## torch.onnx

### Example: End-to-end AlexNet from PyTorch to ONNX

这是一个简单的脚本，可以将Torchvision中定义的经过预训练的AlexNet导出到ONNX中。它运行一轮推断，然后将生成的跟踪模型保存到`alexnet.onnx`：

```python
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
```

生成的`alexnet.onnx`是一个二进制protobuf文件，其中包含您导出的模型的网络结构和参数（在本例中为AlexNet）。 关键字参数verbose = True使导出程序打印出人类可读的网络表示形式：

```
# These are the inputs and parameters to the network, which have taken on
# the names we specified earlier.
graph(%actual_input_1 : Float(10, 3, 224, 224)
      %learned_0 : Float(64, 3, 11, 11)
      %learned_1 : Float(64)
      %learned_2 : Float(192, 64, 5, 5)
      %learned_3 : Float(192)
      # ---- omitted for brevity ----
      %learned_14 : Float(1000, 4096)
      %learned_15 : Float(1000)) {
  # Every statement consists of some output tensors (and their types),
  # the operator to be run (with its attributes, e.g., kernels, strides,
  # etc.), its input tensors (%actual_input_1, %learned_0, %learned_1)
  %17 : Float(10, 64, 55, 55) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1), scope: AlexNet/Sequential[features]/Conv2d[0]
  %18 : Float(10, 64, 55, 55) = onnx::Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
  %19 : Float(10, 64, 27, 27) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2]
  # ---- omitted for brevity ----
  %29 : Float(10, 256, 6, 6) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
  # Dynamic means that the shape is not known. This may be because of a
  # limitation of our implementation (which we would like to fix in a
  # future release) or shapes which are truly dynamic.
  %30 : Dynamic = onnx::Shape(%29), scope: AlexNet
  %31 : Dynamic = onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet
  %32 : Long() = onnx::Squeeze[axes=[0]](%31), scope: AlexNet
  %33 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
  # ---- omitted for brevity ----
  %output1 : Float(10, 1000) = onnx::Gemm[alpha=1, beta=1, broadcast=1, transB=1](%45, %learned_14, %learned_15), scope: AlexNet/Sequential[classifier]/Linear[6]
  return (%output1);
}
```

您还可以使用ONNX库验证protobuf。您可以使用conda安装ONNX：

```
conda install -c conda-forge onnx
```

```python
import onnx

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)
```

您也可以使用ONNX Runtime运行导出的模型，您将需要安装ONNX Runtime：请按照以下说明进行操作。 一旦安装了这些，就可以将后端用于ONNX Runtime：

```python
# ...continuing from above
import onnxruntime as ort

ort_session = ort.InferenceSession('alexnet.onnx')

outputs = ort_session.run(None, {'actual_input_1': np.random.randn(10, 3, 224, 224).astype(np.float32)})

print(outputs[0])
```





### Functions

#### export

> torch.onnx.**export**(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, opset_version=None, _retain_param_name=True, do_constant_folding=False, example_outputs=None, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None)



Parameters:

- **model** ([*torch.nn.Module*](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)) – the model to be exported.

- **args** (*tuple of arguments*) – the inputs to the model, e.g., such that `model(*args)` is a valid invocation of the model. Any non-Tensor arguments will be hard-coded into the exported model; any Tensor arguments will become inputs of the exported model, in the order they occur in args. If args is a Tensor, this is equivalent to having called it with a 1-ary tuple of that Tensor. (Note: passing keyword arguments to the model is not currently supported. Give us a shout if you need it.)

  模型的输入，例如，使`model（* args）`是模型的有效调用。 任何非Tensor参数将被硬编码到导出的模型中； 任何Tensor参数将按照在args中出现的顺序成为导出模型的输入。 如果args是一个Tensor，则相当于用该Tensor的1元元组调用了它。 （注意：当前不支持将关键字参数传递给模型。如果需要，请给我们喊叫。）

- **f** – a file-like object (has to implement fileno that returns a file descriptor) or a string containing a file name. A binary Protobuf will be written to this file.

  类似于文件的对象（必须实现返回文件描述符的fileno）或包含文件名的字符串。 二进制Protobuf将被写入此文件。

- **export_params** (*bool**,* *default True*) – if specified, all parameters will be exported. Set this to False if you want to export an untrained model. In this case, the exported model will first take all of its parameters as arguments, the ordering as specified by `model.state_dict().values()`

  如果指定，将导出所有参数。 如果要导出未经训练的模型，请将其设置为False。 在这种情况下，导出的模型将首先以其所有参数作为参数，顺序由`model.state_dict().values()`指定。

- **training** (*bool**,* *default False*) – export the model in training mode. At the moment, ONNX is oriented towards exporting models for inference only, so you will generally not need to set this to True.

  以训练模式导出模型。 目前，ONNX仅面向导出模型以进行推理，因此通常不需要将其设置为True。

- 





```python
dummy_input = torch.randn(10, 3, 224, 224)
torch.onnx.export(model, dummy_input, "alexnet.onnx")
```

















