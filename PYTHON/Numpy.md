# Numpy









# API

## 索引

**Indexing routines**



### 生成索引数组

|                                                   |      |
| ------------------------------------------------- | ---- |
| `nonzero(a)`                                      |      |
| `where(condition, [x, y])`                        |      |
| `indices(dimensions[, dtype, sparse])`            |      |
| `ix_(\*args)`                                     |      |
| `ogrid`                                           |      |
| `ravel_multi_index(multi_index, dims[, mode, …])` |      |
| `unravel_index(indices, shape[, order])`          |      |
| `diag_indices(n[, ndim])`                         |      |
| `diag_indices_from(arr)`                          |      |
| `mask_indices(n, mask_func[, k])`                 |      |
| `tril_indices(n[, k, m])`                         |      |
| `tril_indices_from(arr[, k])`                     |      |
| `triu_indices(n[, k, m])`                         |      |
| `triu_indices_from(arr[, k])`                     |      |



#### where

> numpy.**where**(condition[, x, y])

返回从x或y中选择的元素(取决于条件)。

如果只提供条件，则此函数是`np.asarray(condition).nonzero()`的缩写。直接使用非零应该是首选的，因为它对子类的行为是正确的。本文档的其余部分仅涉及提供所有三个参数的情况。

**Args:**

- **condition** :array_like, bool

  真，返回x，否则返回y

- **x, y** ：array_like

  可从中选择的值。X，y和条件需要广播到某种形状。

  

**Return：**

- **out** ：ndarray

  一个包含来自x的元素的数组，其中条件为True，其他地方的元素来自y。

  

```python
a = np.arange(10)
a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.where(a < 5, a, 10*a)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
```



#### nonzero

> numpy.**nonzero**(a)

返回非零元素的索引。

返回数组的元组，每个数组为a的每个维度，包含该维度中非零元素的索引。a中的值总是按照行主C样式的顺序进行测试和返回。

若要按元素(而不是维度)对索引进行分组，请使用`argwhere`，它为每个非零元素返回一行。

当在零d数组或标量上调用时，`nonzero(a)`被视为`nonzero(atleast1d(a))`。从版本1.17.0开始不推荐使用atleast1d

**Args:**

- **a** : array_like

  Input array.



**Return:**

- **tuple_of_arrays** : tuple

  非零元素的索引。



```python
x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
x
array([[3, 0, 0],
       [0, 4, 0],
       [5, 6, 0]])

np.nonzero(x)
(array([0, 1, 2, 2]), array([0, 1, 0, 1]))
```















### 索引式操作

### 将数据插入数组

### 迭代数组







## 函数式编程

**Functional programming**

|                    |                                       |
| ------------------ | ------------------------------------- |
| `apply_along_axis` | 沿给定轴对一维切片应用函数.           |
| `apply_over_axes`  | 在多个轴上反复应用函数                |
| `vectorize`        | 广义函数类                            |
| `frompyfunc`       | 接受任意Python函数并返回NumPy ufunc。 |
| `piecewise`        | 评估一个分段定义的函数。              |



###  frompyfunc

> numpy.**frompyfunc**(func, nin, nout, *[, identity])

接受任意Python函数并返回NumPy ufunc。例如，可以将广播添加到内置的Python函数(参见示例部分)。

**Args:**

- **func** Python function object 

  任意的Python函数。

- **nin** int

  输入参数的数量。

- **nout** int

  func返回的对象数。

- **identity** object, optional

  要用于结果对象的Identity属性的值。如果指定，这相当于将基础C标识字段设置为`PyUFunc_IdentityValue`。如果省略，则标识设置为`PyUFunc_None`。注意，这相当于将标识设置为None，这意味着操作是可重排序的。



**Return:**

- **out** ufunc

  返回NumPy通用函数(Ufunc)对象。



```python
oct_array = np.frompyfunc(oct, 1, 1)
oct_array(np.array((10, 30, 100)))
array(['0o12', '0o36', '0o144'], dtype=object)

np.array((oct(10), oct(30), oct(100))) # for comparison
array(['0o12', '0o36', '0o144'], dtype='<U5')
```



## 数组创建

### Ones and zeros

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`empty`](https://numpy.org/devdocs/reference/generated/numpy.empty.html#numpy.empty)(shape[, dtype, order]) | Return a new array of given shape and type, without initializing entries. |
| [`empty_like`](https://numpy.org/devdocs/reference/generated/numpy.empty_like.html#numpy.empty_like)(prototype[, dtype, order, subok, …]) | Return a new array with the same shape and type as a given array. |
| [`eye`](https://numpy.org/devdocs/reference/generated/numpy.eye.html#numpy.eye)(N[, M, k, dtype, order]) | Return a 2-D array with ones on the diagonal and zeros elsewhere. |
| [`identity`](https://numpy.org/devdocs/reference/generated/numpy.identity.html#numpy.identity)(n[, dtype]) | Return the identity array.                                   |
| [`ones`](https://numpy.org/devdocs/reference/generated/numpy.ones.html#numpy.ones)(shape[, dtype, order]) | Return a new array of given shape and type, filled with ones. |
| [`ones_like`](https://numpy.org/devdocs/reference/generated/numpy.ones_like.html#numpy.ones_like)(a[, dtype, order, subok, shape]) | Return an array of ones with the same shape and type as a given array. |
| [`zeros`](https://numpy.org/devdocs/reference/generated/numpy.zeros.html#numpy.zeros)(shape[, dtype, order]) | Return a new array of given shape and type, filled with zeros. |
| [`zeros_like`](https://numpy.org/devdocs/reference/generated/numpy.zeros_like.html#numpy.zeros_like)(a[, dtype, order, subok, shape]) | Return an array of zeros with the same shape and type as a given array. |
| [`full`](https://numpy.org/devdocs/reference/generated/numpy.full.html#numpy.full)(shape, fill_value[, dtype, order]) | Return a new array of given shape and type, filled with *fill_value*. |
| [`full_like`](https://numpy.org/devdocs/reference/generated/numpy.full_like.html#numpy.full_like)(a, fill_value[, dtype, order, …]) | Return a full array with the same shape and type as a given array. |



### From existing data

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`array`](https://numpy.org/devdocs/reference/generated/numpy.array.html#numpy.array)(object[, dtype, copy, order, subok, ndmin]) | Create an array.                                             |
| [`asarray`](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray)(a[, dtype, order]) | Convert the input to an array.                               |
| [`asanyarray`](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray)(a[, dtype, order]) | Convert the input to an ndarray, but pass ndarray subclasses through. |
| [`ascontiguousarray`](https://numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray)(a[, dtype]) | Return a contiguous array (ndim >= 1) in memory (C order).   |
| [`asmatrix`](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.                             |
| [`copy`](https://numpy.org/devdocs/reference/generated/numpy.copy.html#numpy.copy)(a[, order]) | Return an array copy of the given object.                    |
| [`frombuffer`](https://numpy.org/devdocs/reference/generated/numpy.frombuffer.html#numpy.frombuffer)(buffer[, dtype, count, offset]) | Interpret a buffer as a 1-dimensional array.                 |
| [`fromfile`](https://numpy.org/devdocs/reference/generated/numpy.fromfile.html#numpy.fromfile)(file[, dtype, count, sep, offset]) | Construct an array from data in a text or binary file.       |
| [`fromfunction`](https://numpy.org/devdocs/reference/generated/numpy.fromfunction.html#numpy.fromfunction)(function, shape, \*[, dtype]) | Construct an array by executing a function over each coordinate. |
| [`fromiter`](https://numpy.org/devdocs/reference/generated/numpy.fromiter.html#numpy.fromiter)(iterable, dtype[, count]) | Create a new 1-dimensional array from an iterable object.    |
| [`fromstring`](https://numpy.org/devdocs/reference/generated/numpy.fromstring.html#numpy.fromstring)(string[, dtype, count, sep]) | A new 1-D array initialized from text data in a string.      |
| [`loadtxt`](https://numpy.org/devdocs/reference/generated/numpy.loadtxt.html#numpy.loadtxt)(fname[, dtype, comments, delimiter, …]) | Load data from a text file.                                  |



### Numerical ranges

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`arange`](https://numpy.org/devdocs/reference/generated/numpy.arange.html#numpy.arange)([start,] stop[, step,][, dtype]) | Return evenly spaced values within a given interval.         |
| [`linspace`](https://numpy.org/devdocs/reference/generated/numpy.linspace.html#numpy.linspace)(start, stop[, num, endpoint, …]) | Return evenly spaced numbers over a specified interval.      |
| [`logspace`](https://numpy.org/devdocs/reference/generated/numpy.logspace.html#numpy.logspace)(start, stop[, num, endpoint, base, …]) | Return numbers spaced evenly on a log scale.                 |
| [`geomspace`](https://numpy.org/devdocs/reference/generated/numpy.geomspace.html#numpy.geomspace)(start, stop[, num, endpoint, …]) | Return numbers spaced evenly on a log scale (a geometric progression). |
| [`meshgrid`](https://numpy.org/devdocs/reference/generated/numpy.meshgrid.html#numpy.meshgrid)(\*xi[, copy, sparse, indexing]) | Return coordinate matrices from coordinate vectors.          |
| [`mgrid`](https://numpy.org/devdocs/reference/generated/numpy.mgrid.html#numpy.mgrid) | *nd_grid* instance which returns a dense multi-dimensional “meshgrid”. |
| [`ogrid`](https://numpy.org/devdocs/reference/generated/numpy.ogrid.html#numpy.ogrid) | *nd_grid* instance which returns an open multi-dimensional “meshgrid”. |



#### meshgrid

> numpy.**meshgrid**(*xi, copy=True, sparse=False, indexing='xy')

从坐标向量返回坐标矩阵。在N-D网格上，为N-D标量场/矢量场的矢量化计算制作N-D坐标阵列，给定一维坐标阵x1，x2，…、Xn.



**Args：**

|                  |                        |                                                              |
| ---------------- | ---------------------- | ------------------------------------------------------------ |
| **x1, x2,…, xn** | array_like             | 表示网格坐标的一维数组                                       |
| **indexing**     | {‘xy’, ‘ij’}, optional | 笛卡尔(‘XY’，默认值)或矩阵(‘ij’)索引输出。                   |
| **sparse**       | bool, optional         | 如果是True，则返回稀疏网格以节省内存。默认是假的             |
| **copy**         | bool, optional         | 如果为false，则返回原始数组中的视图，以节省内存。默认值为True。<br />请注意sparse=false，Copy=false很可能返回非连续数组。此外，广播阵列的多个元素可以引用单个存储器位置。如果需要写入数组，请先复制。 |
|                  |                        |                                                              |

**Return:**

- **X1, X2,…, XN** ：ndarray

  对于向量x1，x2，…，“xn”的长度为Ni=len(Xi)，返回(N1，N2，N3，.NN)形状数组，如果索引=‘ij’或(N2，N1，N3，…NN)形状数组，则索引=‘XY’时，XI的元素重复重复，沿第1维填充矩阵，第二个数组填充x2，等等。



```python
nx, ny = (3, 2)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
xv
array([[0. , 0.5, 1. ],
       [0. , 0.5, 1. ]])
yv
array([[0.,  0.,  0.],
       [1.,  1.,  1.]])

xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays
xv
array([[0. ,  0.5,  1. ]])
yv
array([[0.],
       [1.]])
```







### Building matrices

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`diag`](https://numpy.org/devdocs/reference/generated/numpy.diag.html#numpy.diag)(v[, k]) | Extract a diagonal or construct a diagonal array.            |
| [`diagflat`](https://numpy.org/devdocs/reference/generated/numpy.diagflat.html#numpy.diagflat)(v[, k]) | Create a two-dimensional array with the flattened input as a diagonal. |
| [`tri`](https://numpy.org/devdocs/reference/generated/numpy.tri.html#numpy.tri)(N[, M, k, dtype]) | An array with ones at and below the given diagonal and zeros elsewhere. |
| [`tril`](https://numpy.org/devdocs/reference/generated/numpy.tril.html#numpy.tril)(m[, k]) | Lower triangle of an array.                                  |
| [`triu`](https://numpy.org/devdocs/reference/generated/numpy.triu.html#numpy.triu)(m[, k]) | Upper triangle of an array.                                  |
| [`vander`](https://numpy.org/devdocs/reference/generated/numpy.vander.html#numpy.vander)(x[, N, increasing]) | Generate a Vandermonde matrix.                               |



### The Matrix class

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`mat`](https://numpy.org/devdocs/reference/generated/numpy.mat.html#numpy.mat)(data[, dtype]) | Interpret the input as a matrix.                             |
| [`bmat`](https://numpy.org/devdocs/reference/generated/numpy.bmat.html#numpy.bmat)(obj[, ldict, gdict]) | Build a matrix object from a string, nested sequence, or array. |





## 数组操作

### 基础操作

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`copyto`](https://numpy.org/devdocs/reference/generated/numpy.copyto.html#numpy.copyto)(dst, src[, casting, where]) | Copies values from one array to another, broadcasting as necessary. |
| [`shape`](https://numpy.org/devdocs/reference/generated/numpy.shape.html#numpy.shape)(a) | Return the shape of an array.                                |



### 改变数组形状

|                                                              |                                                          |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| [`reshape`](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape)(a, newshape[, order]) | Gives a new shape to an array without changing its data. |
| [`ravel`](https://numpy.org/devdocs/reference/generated/numpy.ravel.html#numpy.ravel)(a[, order]) | Return a contiguous flattened array.                     |
| [`ndarray.flat`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flat.html#numpy.ndarray.flat) | A 1-D iterator over the array.                           |
| [`ndarray.flatten`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.flatten.html#numpy.ndarray.flatten)([order]) | Return a copy of the array collapsed into one dimension. |



### Transpose-like操作

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`moveaxis`](https://numpy.org/devdocs/reference/generated/numpy.moveaxis.html#numpy.moveaxis)(a, source, destination) | Move axes of an array to new positions.                      |
| [`rollaxis`](https://numpy.org/devdocs/reference/generated/numpy.rollaxis.html#numpy.rollaxis)(a, axis[, start]) | Roll the specified axis backwards, until it lies in a given position. |
| [`swapaxes`](https://numpy.org/devdocs/reference/generated/numpy.swapaxes.html#numpy.swapaxes)(a, axis1, axis2) | Interchange two axes of an array.                            |
| [`ndarray.T`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.T.html#numpy.ndarray.T) | The transposed array.                                        |
| [`transpose`](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose)(a[, axes]) | Reverse or permute the axes of an array; returns the modified array. |



### 改变维度数量

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`atleast_1d`](https://numpy.org/devdocs/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d)(\*arys) | Convert inputs to arrays with at least one dimension.        |
| [`atleast_2d`](https://numpy.org/devdocs/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d)(\*arys) | View inputs as arrays with at least two dimensions.          |
| [`atleast_3d`](https://numpy.org/devdocs/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d)(\*arys) | View inputs as arrays with at least three dimensions.        |
| [`broadcast`](https://numpy.org/devdocs/reference/generated/numpy.broadcast.html#numpy.broadcast) | Produce an object that mimics broadcasting.                  |
| [`broadcast_to`](https://numpy.org/devdocs/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to)(array, shape[, subok]) | Broadcast an array to a new shape.                           |
| [`broadcast_arrays`](https://numpy.org/devdocs/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays)(\*args[, subok]) | Broadcast any number of arrays against each other.           |
| [`expand_dims`](https://numpy.org/devdocs/reference/generated/numpy.expand_dims.html#numpy.expand_dims)(a, axis) | Expand the shape of an array.                                |
| [`squeeze`](https://numpy.org/devdocs/reference/generated/numpy.squeeze.html#numpy.squeeze)(a[, axis]) | Remove single-dimensional entries from the shape of an array. |



#### broadcast_to

> numpy.**broadcast_to**(array, shape, subok=False)[source]

将数组广播到新形状。

```python
x = np.array([1, 2, 3])
np.broadcast_to(x, (3, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
```









### 改变数组类型

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`asarray`](https://numpy.org/devdocs/reference/generated/numpy.asarray.html#numpy.asarray)(a[, dtype, order]) | Convert the input to an array.                               |
| [`asanyarray`](https://numpy.org/devdocs/reference/generated/numpy.asanyarray.html#numpy.asanyarray)(a[, dtype, order]) | Convert the input to an ndarray, but pass ndarray subclasses through. |
| [`asmatrix`](https://numpy.org/devdocs/reference/generated/numpy.asmatrix.html#numpy.asmatrix)(data[, dtype]) | Interpret the input as a matrix.                             |
| [`asfarray`](https://numpy.org/devdocs/reference/generated/numpy.asfarray.html#numpy.asfarray)(a[, dtype]) | Return an array converted to a float type.                   |
| [`asfortranarray`](https://numpy.org/devdocs/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray)(a[, dtype]) | Return an array (ndim >= 1) laid out in Fortran order in memory. |
| [`ascontiguousarray`](https://numpy.org/devdocs/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray)(a[, dtype]) | Return a contiguous array (ndim >= 1) in memory (C order).   |
| [`asarray_chkfinite`](https://numpy.org/devdocs/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite)(a[, dtype, order]) | Convert the input to an array, checking for NaNs or Infs.    |
| [`asscalar`](https://numpy.org/devdocs/reference/generated/numpy.asscalar.html#numpy.asscalar)(a) | Convert an array of size 1 to its scalar equivalent.         |
| [`require`](https://numpy.org/devdocs/reference/generated/numpy.require.html#numpy.require)(a[, dtype, requirements]) | Return an ndarray of the provided type that satisfies requirements. |



### 连接数组

|                                                              |                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| [`concatenate`](https://numpy.org/devdocs/reference/generated/numpy.concatenate.html#numpy.concatenate)([axis, out]) | Join a sequence of arrays along an existing axis.       |
| [`stack`](https://numpy.org/devdocs/reference/generated/numpy.stack.html#numpy.stack)(arrays[, axis, out]) | Join a sequence of arrays along a new axis.             |
| [`column_stack`](https://numpy.org/devdocs/reference/generated/numpy.column_stack.html#numpy.column_stack)(tup) | Stack 1-D arrays as columns into a 2-D array.           |
| [`dstack`](https://numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack)(tup) | Stack arrays in sequence depth wise (along third axis). |
| [`hstack`](https://numpy.org/devdocs/reference/generated/numpy.hstack.html#numpy.hstack)(tup) | Stack arrays in sequence horizontally (column wise).    |
| [`vstack`](https://numpy.org/devdocs/reference/generated/numpy.vstack.html#numpy.vstack)(tup) | Stack arrays in sequence vertically (row wise).         |
| [`block`](https://numpy.org/devdocs/reference/generated/numpy.block.html#numpy.block)(arrays) | Assemble an nd-array from nested lists of blocks.       |



### 分割数组

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`split`](https://numpy.org/devdocs/reference/generated/numpy.split.html#numpy.split)(ary, indices_or_sections[, axis]) | Split an array into multiple sub-arrays as views into *ary*. |
| [`array_split`](https://numpy.org/devdocs/reference/generated/numpy.array_split.html#numpy.array_split)(ary, indices_or_sections[, axis]) | Split an array into multiple sub-arrays.                     |
| [`dsplit`](https://numpy.org/devdocs/reference/generated/numpy.dsplit.html#numpy.dsplit)(ary, indices_or_sections) | Split array into multiple sub-arrays along the 3rd axis (depth). |
| [`hsplit`](https://numpy.org/devdocs/reference/generated/numpy.hsplit.html#numpy.hsplit)(ary, indices_or_sections) | Split an array into multiple sub-arrays horizontally (column-wise). |
| [`vsplit`](https://numpy.org/devdocs/reference/generated/numpy.vsplit.html#numpy.vsplit)(ary, indices_or_sections) | Split an array into multiple sub-arrays vertically (row-wise). |



### 重复数组

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`tile`](https://numpy.org/devdocs/reference/generated/numpy.tile.html#numpy.tile)(A, reps) | Construct an array by repeating A the number of times given by reps. |
| [`repeat`](https://numpy.org/devdocs/reference/generated/numpy.repeat.html#numpy.repeat)(a, repeats[, axis]) | Repeat elements of an array.                                 |



### 添加删除元素

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`delete`](https://numpy.org/devdocs/reference/generated/numpy.delete.html#numpy.delete)(arr, obj[, axis]) | Return a new array with sub-arrays along an axis deleted.    |
| [`insert`](https://numpy.org/devdocs/reference/generated/numpy.insert.html#numpy.insert)(arr, obj, values[, axis]) | Insert values along the given axis before the given indices. |
| [`append`](https://numpy.org/devdocs/reference/generated/numpy.append.html#numpy.append)(arr, values[, axis]) | Append values to the end of an array.                        |
| [`resize`](https://numpy.org/devdocs/reference/generated/numpy.resize.html#numpy.resize)(a, new_shape) | Return a new array with the specified shape.                 |
| [`trim_zeros`](https://numpy.org/devdocs/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros)(filt[, trim]) | Trim the leading and/or trailing zeros from a 1-D array or sequence. |
| [`unique`](https://numpy.org/devdocs/reference/generated/numpy.unique.html#numpy.unique)(ar[, return_index, return_inverse, …]) | Find the unique elements of an array.                        |



#### unique

> numpy.**unique**(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)

查找数组的唯一元素。



```python
np.unique([1, 1, 2, 2, 3, 3])
array([1, 2, 3])
a = np.array([[1, 1], [2, 3]])
np.unique(a)
array([1, 2, 3])
```







### 重新排列元素

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`flip`](https://numpy.org/devdocs/reference/generated/numpy.flip.html#numpy.flip)(m[, axis]) | Reverse the order of elements in an array along the given axis. |
| [`fliplr`](https://numpy.org/devdocs/reference/generated/numpy.fliplr.html#numpy.fliplr)(m) | Flip array in the left/right direction.                      |
| [`flipud`](https://numpy.org/devdocs/reference/generated/numpy.flipud.html#numpy.flipud)(m) | Flip array in the up/down direction.                         |
| [`reshape`](https://numpy.org/devdocs/reference/generated/numpy.reshape.html#numpy.reshape)(a, newshape[, order]) | Gives a new shape to an array without changing its data.     |
| [`roll`](https://numpy.org/devdocs/reference/generated/numpy.roll.html#numpy.roll)(a, shift[, axis]) | Roll array elements along a given axis.                      |
| [`rot90`](https://numpy.org/devdocs/reference/generated/numpy.rot90.html#numpy.rot90)(m[, k, axes]) | Rotate an array by 90 degrees in the plane specified by axes. |



### padding array

> numpy.**pad**(array, pad_width, mode='constant', **kwargs)







## 排序, 搜索, 和计数

### 排序

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`sort`](https://numpy.org/devdocs/reference/generated/numpy.sort.html#numpy.sort)(a[, axis, kind, order]) | Return a sorted copy of an array.                            |
| [`lexsort`](https://numpy.org/devdocs/reference/generated/numpy.lexsort.html#numpy.lexsort)(keys[, axis]) | Perform an indirect stable sort using a sequence of keys.    |
| [`argsort`](https://numpy.org/devdocs/reference/generated/numpy.argsort.html#numpy.argsort)(a[, axis, kind, order]) | Returns the indices that would sort an array.                |
| [`ndarray.sort`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.sort.html#numpy.ndarray.sort)([axis, kind, order]) | Sort an array in-place.                                      |
| [`msort`](https://numpy.org/devdocs/reference/generated/numpy.msort.html#numpy.msort)(a) | Return a copy of an array sorted along the first axis.       |
| [`sort_complex`](https://numpy.org/devdocs/reference/generated/numpy.sort_complex.html#numpy.sort_complex)(a) | Sort a complex array using the real part first, then the imaginary part. |
| [`partition`](https://numpy.org/devdocs/reference/generated/numpy.partition.html#numpy.partition)(a, kth[, axis, kind, order]) | Return a partitioned copy of an array.                       |
| [`argpartition`](https://numpy.org/devdocs/reference/generated/numpy.argpartition.html#numpy.argpartition)(a, kth[, axis, kind, order]) | Perform an indirect partition along the given axis using the algorithm specified by the *kind*keyword. |



####  argsort

> numpy.**argsort**(a, axis=-1, kind=None, order=None)

返回排序数组的索引。

使用kind关键字指定的算法沿给定的轴执行间接排序。它返回的索引数组的形状与该索引数据沿给定轴的排序顺序相同。



```python
x = np.array([3, 1, 2])
np.argsort(x)
array([1, 2, 0])
```

```python
x = np.array([[0, 3], [2, 2]])
x
array([[0, 3],
       [2, 2]])

ind = np.argsort(x, axis=0)  # sorts along first axis (down)
ind
array([[0, 1],
       [1, 0]])
np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)列
array([[0, 2],
       [2, 3]])

ind = np.argsort(x, axis=1)  # sorts along last axis (across)行
ind
array([[0, 1],
       [0, 1]])
np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
array([[0, 3],
       [2, 2]])
```







### 搜索



|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`argmax`](https://numpy.org/devdocs/reference/generated/numpy.argmax.html#numpy.argmax)(a[, axis, out]) | Returns the indices of the maximum values along an axis.     |
| [`nanargmax`](https://numpy.org/devdocs/reference/generated/numpy.nanargmax.html#numpy.nanargmax)(a[, axis]) | Return the indices of the maximum values in the specified axis ignoring NaNs. |
| [`argmin`](https://numpy.org/devdocs/reference/generated/numpy.argmin.html#numpy.argmin)(a[, axis, out]) | Returns the indices of the minimum values along an axis.     |
| [`nanargmin`](https://numpy.org/devdocs/reference/generated/numpy.nanargmin.html#numpy.nanargmin)(a[, axis]) | Return the indices of the minimum values in the specified axis ignoring NaNs. |
| [`argwhere`](https://numpy.org/devdocs/reference/generated/numpy.argwhere.html#numpy.argwhere)(a) | Find the indices of array elements that are non-zero, grouped by element. |
| [`nonzero`](https://numpy.org/devdocs/reference/generated/numpy.nonzero.html#numpy.nonzero)(a) | Return the indices of the elements that are non-zero.        |
| [`flatnonzero`](https://numpy.org/devdocs/reference/generated/numpy.flatnonzero.html#numpy.flatnonzero)(a) | Return indices that are non-zero in the flattened version of a. |
| [`where`](https://numpy.org/devdocs/reference/generated/numpy.where.html#numpy.where)(condition, [x, y]) | Return elements chosen from *x* or *y* depending on *condition*. |
| [`searchsorted`](https://numpy.org/devdocs/reference/generated/numpy.searchsorted.html#numpy.searchsorted)(a, v[, side, sorter]) | Find indices where elements should be inserted to maintain order. |
| [`extract`](https://numpy.org/devdocs/reference/generated/numpy.extract.html#numpy.extract)(condition, arr) | Return the elements of an array that satisfy some condition. |



### 计数

|                                                              |                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [`count_nonzero`](https://numpy.org/devdocs/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero)(a[, axis]) | Counts the number of non-zero values in the array `a`. |
|                                                              |                                                        |



## 数学函数

### maximum

> numpy.**maximum**(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'maximum'>

数组元素的逐元素最大值。

比较两个数组，并返回一个包含按元素的最大值的新数组。**如果要比较的元素之一是NaN，则返回该元素**。如果两个元素均为NaN，则返回第一个。后一种区分对于复杂的NaN至关重要，后者被定义为至少一个实部或虚部为NaN。最终结果是**NaN会传播**。

```python
np.maximum([2, 3, 4], [1, 5, 2])
array([2, 5, 4])
```

```python
np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
array([nan, nan, nan])
np.maximum(np.Inf, 1)
inf
```





### fmax

> numpy.**fmax**(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'fmax'>

数组元素的逐元素最大值。

比较两个数组，并返回一个包含元素级极大值的新数组。**如果要比较的元素之一是NaN，则返回非NaN元素**。如果这两个元素都是NAN，则返回第一个元素。后一种区分对于复杂的NAN很重要，它被定义为至少一个实部或虚部为NaN。净效果是，在可能的情况下，**NAN被忽略**。

```python
np.fmax([np.nan, 0, np.nan],[0, np.nan, np.nan])
array([ 0.,  0., nan])
```

