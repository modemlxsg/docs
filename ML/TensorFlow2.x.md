#  TensorFlow2.x

## 01、分类

### Import

```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import time
from tensorflow import keras
print(tf.__version__)
```

### 加载数据

```python
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]

print(x_valid.shape,x_train.shape,x_test.shape)
```

### 归一化

```python
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28) #-1表示默认值，自动计算
x_valid_scaled = scalar.transform(x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scalar.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
```

### 构建模型

```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation="selu"))
model.add(keras.layers.AlphaDropout(rate=0.5))
# AlphaDropout: 1. 均值和方差不变 2. 归一化性质也不变
# model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
```

### 训练

```python
# callbacks
logdir = os.path.join('cnn-callbacks')
print(logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]
history = model.fit(x_train_scaled,y_train,epochs=10,
                    validation_data=(x_valid_scaled,y_valid),
                    callbacks=callbacks)
```

### 绘图

```python
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)
```



## 02、回归

### Model

```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])
model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd")
```

### wide & deep

![image-20191231045955865](./images\TensorFlow2.x.assets\image-20191231045955865.png)

### 函数式API

```python
# 函数式API 功能API
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
# 复合函数: f(x) = h(g(x))

concat = keras.layers.concatenate([input, hidden2]) # shape(None,38) input.shape(None,8) hidden2.shape(None,30)
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs = [input],
                           outputs = [output])

model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd")

callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]
history = model.fit(x_train_scaled, y_train,
                    validation_data = (x_valid_scaled, y_valid),
                    epochs = 100,
                    callbacks = callbacks)
```

### 子类化API

```python
# 子类API
class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型的层次"""
        self.hidden1_layer = keras.layers.Dense(30, activation='relu')
        self.hidden2_layer = keras.layers.Dense(30, activation='relu')
        self.output_layer = keras.layers.Dense(1)
    
    def call(self, input):
        """完成模型的正向计算"""
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output
# model = WideDeepModel()
model = keras.models.Sequential([
    WideDeepModel(),
])

model.build(input_shape=(None, 8))
        
model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]
history = model.fit(x_train_scaled, y_train,
                    validation_data = (x_valid_scaled, y_valid),
                    epochs = 1,
                    callbacks = callbacks)
```

### 多输入

```python
# 多输入
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_wide, input_deep],
                           outputs = [output])
        

model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]
model.summary()
```

### 多输出

```python
# 多输出
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs = [input_wide, input_deep],
                           outputs = [output, output2])
        

model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]
model.summary()
```

### 超参数搜索

```python
# RandomizedSearchCV
# 1. 转化为sklearn的model
# 2. 定义参数集合
# 3. 搜索参数

def build_model(hidden_layers = 1,
                layer_size = 30,
                learning_rate = 3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size, activation='relu',input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size,activation = 'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss = 'mse', optimizer = optimizer)
    return model

sklearn_model = KerasRegressor(build_fn = build_model)
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = sklearn_model.fit(x_train_scaled, y_train,
                            epochs = 10,
                            validation_data = (x_valid_scaled, y_valid),
                            callbacks = callbacks)
```

```python
from scipy.stats import reciprocal
# f(x) = 1/(x*log(b/a)) a <= x <= b

param_distribution = {
    "hidden_layers":[1, 2, 3, 4],
    "layer_size": np.arange(1, 100),
    "learning_rate": reciprocal(1e-4, 1e-2),
}

from sklearn.model_selection import RandomizedSearchCV

random_search_cv = RandomizedSearchCV(sklearn_model,
                                      param_distribution,
                                      n_iter = 10,
                                      cv = 3,
                                      n_jobs = 1)
random_search_cv.fit(x_train_scaled, y_train, epochs = 100,
                     validation_data = (x_valid_scaled, y_valid),
                     callbacks = callbacks)

# cross_validation: 训练集分成n份，n-1训练，最后一份验证.

print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_)

model = random_search_cv.best_estimator_.model
model.evaluate(x_test_scaled, y_test)
```

![image-20191231052116047](J:\03_NOTES\ML\images\TensorFlow2.x.assets\image-20191231052116047.png)

## 03、基础API

### 常量

```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])

# index
print(t)
print(t[:, 1:])
print(t[..., 1])

tf.Tensor(
[[1. 2. 3.]
 [4. 5. 6.]], shape=(2, 3), dtype=float32)
tf.Tensor(
[[2. 3.]
 [5. 6.]], shape=(2, 2), dtype=float32)
tf.Tensor([2. 5.], shape=(2,), dtype=float32)

# ops
print(t+10)
print(tf.square(t))
print(t @ tf.transpose(t))

tf.Tensor(
[[11. 12. 13.]
 [14. 15. 16.]], shape=(2, 3), dtype=float32)
tf.Tensor(
[[ 1.  4.  9.]
 [16. 25. 36.]], shape=(2, 3), dtype=float32)
tf.Tensor(
[[14. 32.]
 [32. 77.]], shape=(2, 2), dtype=float32)

# numpy conversion
print(t.numpy())
print(np.square(t))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(np_t))

[[1. 2. 3.]
 [4. 5. 6.]]
[[ 1.  4.  9.]
 [16. 25. 36.]]
tf.Tensor(
[[1. 2. 3.]
 [4. 5. 6.]], shape=(2, 3), dtype=float64)

# Scalars
t = tf.constant(2.718)
print(t.numpy())
print(t.shape)

2.718
()
```

### Strings

```python
# strings
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t, "UTF8"))

tf.Tensor(b'cafe', shape=(), dtype=string)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor([ 99  97 102 101], shape=(4,), dtype=int32)

# string array
t = tf.constant(["cafe", "coffee", "咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t, "UTF8")
print(r)

tf.Tensor([4 6 2], shape=(3,), dtype=int32)
<tf.RaggedTensor [[99, 97, 102, 101], [99, 111, 102, 102, 101, 101], [21654, 21857]]>
```

### ragged tensor

```python
# ragged tensor
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
# index op
print(r)
print(r[1])
print(r[1:2])

<tf.RaggedTensor [[11, 12], [21, 22, 23], [], [41]]>
tf.Tensor([21 22 23], shape=(3,), dtype=int32)
<tf.RaggedTensor [[21, 22, 23]]>

# ops on ragged tensor
r2 = tf.ragged.constant([[51, 52], [], [71]])
print(tf.concat([r, r2], axis = 0))

<tf.RaggedTensor [[11, 12], [21, 22, 23], [], [41], [51, 52], [], [71]]>

r3 = tf.ragged.constant([[13, 14], [15], [], [42, 43]])
print(tf.concat([r, r3], axis = 1))

<tf.RaggedTensor [[11, 12, 13, 14], [21, 22, 23, 15], [], [41, 42, 43]]>

print(r.to_tensor())

tf.Tensor(
[[11 12  0]
 [21 22 23]
 [ 0  0  0]
 [41  0  0]], shape=(4, 3), dtype=int32)
```

### sparse tensor

```python
# sparse tensor
s = tf.SparseTensor(indices = [[0, 1], [1, 0], [2, 3]],
                    values = [1., 2., 3.],
                    dense_shape = [3, 4])
print(s)
print(tf.sparse.to_dense(s))

SparseTensor(indices=tf.Tensor(
[[0 1]
 [1 0]
 [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
tf.Tensor(
[[0. 1. 0. 0.]
 [2. 0. 0. 0.]
 [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)

# ops on sparse tensors
s2 = s * 2.0
print(s2)

try:
    s3 = s + 1
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])
print(tf.sparse.sparse_dense_matmul(s, s4))

SparseTensor(indices=tf.Tensor(
[[0 1]
 [1 0]
 [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
unsupported operand type(s) for +: 'SparseTensor' and 'int'
tf.Tensor(
[[ 30.  40.]
 [ 20.  40.]
 [210. 240.]], shape=(3, 2), dtype=float32)

# sparse tensor
s5 = tf.SparseTensor(indices = [[0, 2], [0, 1], [2, 3]],
                    values = [1., 2., 3.],
                    dense_shape = [3, 4])
print(s5)
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))

SparseTensor(indices=tf.Tensor(
[[0 2]
 [0 1]
 [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
tf.Tensor(
[[0. 2. 1. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)
```

### 变量

```python
# Variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
print(v.value())
print(v.numpy())


<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>
tf.Tensor(
[[1. 2. 3.]
 [4. 5. 6.]], shape=(2, 3), dtype=float32)
[[1. 2. 3.]
 [4. 5. 6.]]

# assign value
v.assign(2*v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())

[[ 2.  4.  6.]
 [ 8. 10. 12.]]
[[ 2. 42.  6.]
 [ 8. 10. 12.]]
[[ 2. 42.  6.]
 [ 7.  8.  9.]]
```

### 自定义loss

```python
def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])
model.summary()
model.compile(loss=customized_mse, optimizer="sgd",
              metrics=["mean_squared_error"])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(x_train_scaled, y_train,
                    validation_data = (x_valid_scaled, y_valid),
                    epochs = 100,
                    callbacks = callbacks)
```

### 自定义layer

```python
# lambda
# tf.nn.softplus : log(1+e^x)
customized_softplus = keras.layers.Lambda(lambda x : tf.nn.softplus(x))
print(customized_softplus([-10., -5., 0., 5., 10.]))
```

```python
# customized dense layer.
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """构建所需要的参数"""
        # x * w + b. input_shape:[None, a] w:[a,b]output_shape: [None, b]
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (input_shape[1], self.units),
                                      initializer = 'uniform',
                                      trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                    shape = (self.units, ),
                                    initializer = 'zeros',
                                    trainable = True)
        super(CustomizedDenseLayer, self).build(input_shape)
    
    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)

model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu',
                         input_shape=x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # keras.layers.Dense(1, activation="softplus"),
    # keras.layers.Dense(1), keras.layers.Activation('softplus'),
])
model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

for i in range(5):
    exec('var{}={}'.format(i,i))
print(var0,var1,var2,var3,var4)
```

### tf.function & autograph

在TensorFlow 2.0中，默认情况下启用了急切执行。 对于用户而言直观且灵活（运行一次性操作更容易，更快），但这可能会牺牲性能和可部署性。

要获得最佳性能并使模型可在任何地方部署，可以优先使用`tf.function`从程序中构建图。 因为有AutoGraph，可以使用tf.function构建高效性能的Python代码，但仍有一些陷阱需要警惕。

> `%time` 	  可以测量一行代码执行的时间
> `%timeit` 	可以测量一行代码多次执行的时间

```python
# tf.function and auto-graph.
def scaled_elu(z, scale=1.0, alpha=1.0):
    # z >= 0 ? scale * z : scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))

print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2.5])))

scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))

print(scaled_elu_tf.python_function is scaled_elu)

%timeit scaled_elu(tf.random.normal((1000, 1000)))
%timeit scaled_elu_tf(tf.random.normal((1000, 1000)))

@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total

print(converge_to_2(20))
```

```python
def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    display(Markdown('```python\n{}\n```'.format(code)))
    
display_tf_code(scaled_elu)
```

```python
var = tf.Variable(0.)

@tf.function
def add_21():
    return var.assign_add(21) # += 

print(add_21())

@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)

try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)
    
print(cube(tf.constant([1, 2, 3])))


```

```python
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')]) #类型限制(可以不加)
def cube(z):
    return tf.pow(z, 3)

try:
    print(cube(tf.constant([1., 2., 3.])))
except ValueError as ex:
    print(ex)
    
print(cube(tf.constant([1, 2, 3])))
```

```python
# @tf.function py func -> tf graph
# get_concrete_function -> add input signature -> SavedModel

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32))
print(cube_func_int32)
print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1, 2, 3])))

cube_func_int32.graph
cube_func_int32.graph.get_operations()

[<tf.Operation 'x' type=Placeholder>,
 <tf.Operation 'Pow/y' type=Const>,
 <tf.Operation 'Pow' type=Pow>,
 <tf.Operation 'Identity' type=Identity>]

pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)

name: "Pow"
op: "Pow"
input: "x"
input: "Pow/y"
attr {
  key: "T"
  value {
    type: DT_INT32
  }
}

print(list(pow_op.inputs))
print(list(pow_op.outputs))
cube_func_int32.graph.get_operation_by_name("x")
cube_func_int32.graph.get_tensor_by_name("x:0")
cube_func_int32.graph.as_graph_def()
```

### 自定义求导

```python
def f(x):
    return 3. * x ** 2 + 2. * x - 1

def approximate_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)

print(approximate_derivative(f, 1.))

7.999999999999119

def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)

def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2

print(approximate_gradient(g, 2., 3.))

(8.999999999993236, 41.999999999994486)
```

```python
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
print(dz_x1)

try:
    dz_x2 = tape.gradient(z, x2)
except RuntimeError as ex:
    print(ex)
    
tf.Tensor(9.0, shape=(), dtype=float32)
GradientTape.gradient can only be called once on non-persistent tapes.
```

```python
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent = True) as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1, dz_x2)

del tape

tf.Tensor(9.0, shape=(), dtype=float32) tf.Tensor(42.0, shape=(), dtype=float32)
```

```python
# 变量求导
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])

print(dz_x1x2)

[<tf.Tensor: id=95, shape=(), dtype=float32, numpy=9.0>, <tf.Tensor: id=101, shape=(), dtype=float32, numpy=42.0>]

#常量求导
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])

print(dz_x1x2)

[None, None]
```

```python
# 常量求导
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])

print(dz_x1x2)
[<tf.Tensor: id=192, shape=(), dtype=float32, numpy=9.0>, <tf.Tensor: id=204, shape=(), dtype=float32, numpy=42.0>]
```

```python
# 多函数求导
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
tape.gradient([z1, z2], x)

<tf.Tensor: id=261, shape=(), dtype=float32, numpy=13.0>
```

```python
# 二阶导数
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2]) for inner_grad in inner_grads]
print(outer_grads)
del inner_tape
del outer_tape

[[None, <tf.Tensor: id=324, shape=(), dtype=float32, numpy=6.0>], [<tf.Tensor: id=378, shape=(), dtype=float32, numpy=6.0>, <tf.Tensor: id=361, shape=(), dtype=float32, numpy=14.0>]]
```

```python
# 梯度下降
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)
print(x)

<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.3333333>
```

```python
# optimizer
learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr = learning_rate)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])
print(x)

<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.3333333>
```

### 自定义求导与tf.keras实例

#### Import

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras
```

#### 数据集

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state = 11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)
```

#### 归一化

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)
```

#### metric使用

```python
# metric使用
metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))
print(metric([0.], [1.])) # 自动累加
print(metric.result())

metric.reset_states() # 重置
metric([1.], [3.])
print(metric.result())
```

#### 手动训练求导

```python
# 1. batch 遍历训练集 metric
#    1.1 自动求导
# 2. epoch结束 验证集 metric

epochs = 100
batch_size = 32
steps_per_epoch = len(x_train_scaled) // batch_size
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()

def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=x_train.shape[1:]),
    keras.layers.Dense(1),
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_per_epoch):
        x_batch, y_batch = random_batch(x_train_scaled, y_train,
                                        batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_batch, y_pred)
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradients(grads_and_vars)
        print("\rEpoch", epoch, " train mse:",
              metric.result().numpy(), end="")
    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(
        keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\t", "valid mse: ", valid_loss.numpy())
```



## 04、tf.data

### 基础API

```python
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

# 遍历
for item in dataset:
    print(item)

# 1. repeat epoch
# 2. get batch
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
    
tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
tf.Tensor([8 9], shape=(2,), dtype=int64)

# interleave: 
# case: 文件dataset -> 具体数据集
dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v), # map_fn
    cycle_length = 5, # cycle_length 并行
    block_length = 5, # block_length 块大小
)
for item in dataset2:
    print(item)
```

```python
x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

<TensorSliceDataset shapes: ((2,), ()), types: (tf.int64, tf.string)>
[1 2] b'cat'
[3 4] b'dog'
[5 6] b'fox'
```

```python
dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x,
                                               "label": y})
for item in dataset4:
    print(item["feature"].numpy(), item["label"].numpy())

[1 2] b'cat'
[3 4] b'dog'
[5 6] b'fox'
```

### CSV

```python
def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, # 为了示例更容易展示，手动设置较小的值
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
```

### TFRecord

```python
# tfrecord 文件格式
# -> tf.train.Example
#    -> tf.train.Features -> {"key": tf.train.Feature}
#       -> tf.train.Feature -> tf.train.ByteList/FloatList/Int64List

favorite_books = [name.encode('utf-8')
                  for name in ["machine learning", "cc150"]]
favorite_books_bytelist = tf.train.BytesList(value = favorite_books)
print(favorite_books_bytelist)

hours_floatlist = tf.train.FloatList(value = [15.5, 9.5, 7.0, 8.0])
print(hours_floatlist)

age_int64list = tf.train.Int64List(value = [42])
print(age_int64list)

features = tf.train.Features(
    feature = {
        "favorite_books": tf.train.Feature(bytes_list = favorite_books_bytelist),
        "hours": tf.train.Feature(float_list = hours_floatlist),
        "age": tf.train.Feature(int64_list = age_int64list),
    }
)
print(features)
```

### 构建图片数据集

https://github.com/modemlxsg/docs/blob/master/ML/notebooks/tf_data_image.ipynb







## 05、tf.estimator

#### feature_columns

```python
categorical_columns = ['sex', 'n_siblings_spouses','parch', 'class', 'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']

feature_columns = []
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    print(categorical_column, vocab)
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column,vocab)))
    
for categorical_column in numeric_columns:
    print(categorical_column, vocab)
    feature_columns.append(tf.feature_column.numeric_column(categorical_column,dtype=tf.float32))
```

```python
def make_dataset(data_df, label_df, epochs=10, shuffle=True, batch_size = 32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset

train_dataset = make_dataset(train_df, y_train, batch_size=5)
```

```python
# keras.layers.DenseFeature
# 转换测试
for x,y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_column = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column,dtype='floate32')(x).numpy())
    print(keras.layers.DenseFeatures(gender_column,dtype='float32')(x).numpy())

# 转换feature_columns
for x,y in train_dataset.take(1):
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())
```

#### keras to estimator

```python
model = keras.Sequential()
model.add(keras.layers.DenseFeatures(feature_columns))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
```

```python
# 1. model.fit
# 2. model -> estimator -> train

train_dataset = make_dataset(train_df, y_train, epochs=100)
eval_dataset = make_dataset(eval_df, y_eval, epochs=1, shuffle=False)
history = model.fit(train_dataset,
                    validation_data=eval_dataset,
                    steps_per_epoch=10,
                    validation_steps=8,
                    epochs=100)
```

```python
estimator = keras.estimator.model_to_estimator(model)
# input_fn格式
# 1. function
# 2. 必须返回return a.(features, labels) b.(dataset) -> (feature, label)
estimator.train(input_fn = lambda:make_dataset(train_df, y_train, epochs=100)) # lambda封装成没有参数的函数
```

#### 使用预定义estimator

```python

```



#### 交叉特征

```python
# cross feature:对离散特征做笛卡尔积 age:[1,2,3,4,5] gender:['male','fmale']
# age_x_gender: [(1,male),(2,male),...,(5,male),(1.fmale),...,(5,fmale)]
# 100000:100 -> hash(100000 values) % 100 把可能的10万个稀疏特征hash映射到100个桶中

feature_columns.append(
    tf.feature_column.indicator_column(
        tf.feature_column.crossed_column(['age','sex'],
                                         hash_bucket_size=100)))
```



## 06、卷积网络

### 基本结构

```python
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='relu',
                              input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, # 经过池化卷积核翻倍
                              padding='same',
                              activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
```



### 深度可分离卷积

深度可分离卷积分为两步：

- 第一步用三个卷积对三个通道分别做卷积，这样在一次卷积后，输出3个数。
- 这输出的三个数，再通过一个1x1x3的卷积核（pointwise核），得到一个数。

所以深度可分离卷积其实是通过两次卷积实现的。

第一步，对三个通道分别做卷积，输出三个通道的属性：

![image-20200102003959848](images\TensorFlow2.x.assets\image-20200102003959848.png)

第二步，用卷积核1x1x3对三个通道再次做卷积，这个时候的输出就和正常卷积一样，是8x8x1：

![image-20200102004013504](J:\03_NOTES\ML\images\TensorFlow2.x.assets\image-20200102004013504.png)

如果要提取更多的属性，则需要设计更多的1x1x3卷积核心就可以

![image-20200102004116073](J:\03_NOTES\ML\images\TensorFlow2.x.assets\image-20200102004116073.png)

```python
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation='relu',
                              input_shape=(28,28,1)))
model.add(keras.layers.SeparableConv2D(filters=32, kernel_size=3, # 输入层以外用深度可分离卷积
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.SeparableConv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.SeparableConv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
```

### 数据增强



### 迁移学习

```python
resnet50_fine_tune = keras.models.Sequential()
resnet50_fine_tune.add(keras.applications.ResNet50(include_top = False,
                                                   pooling = 'avg',
                                                   weights = 'imagenet'))
resnet50_fine_tune.add(keras.layers.Dense(num_classes, activation = 'softmax'))
resnet50_fine_tune.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
```

```python
# 后5层可训练
resnet50 = keras.applications.ResNet50(include_top = False,
                                       pooling = 'avg',
                                       weights = 'imagenet')
for layer in resnet50.layers[0:-5]:
    layer.trainable = False
    
resnet50_new = keras.models.Sequential([
    resnet50,
    keras.layers.Dense(num_classes, activation = 'softmax')
])
```





## 07、循环网络

## 08、分布式训练

## 09、模型保存与部署

![image-20200201162847217](images/TensorFlow2.x.assets/image-20200201162847217.png)



### 模型保存



#### 全模型保存

This file includes:

- The model's architecture
- The model's weight values (which were learned during training)
- The model's training config (what you passed to `compile`), if any
- The optimizer and its state, if any (this enables you to restart training where you left off)

```python
# Save the model
model.save('path_to_my_model.h5')

# Recreate the exact same model purely from the file
new_model = keras.models.load_model('path_to_my_model.h5')
```



#### 导出为SavedModel格式

还可以将整个模型导出为`TensorFlow SavedModel`格式。SavedModel是TensorFlow对象的独立序列化格式，由TensorFlow服务以及Python以外的TensorFlow实现支持

The `SavedModel` files that were created contain:

- A TensorFlow `checkpoint` containing the model weights.
- A `SavedModel` proto containing the underlying TensorFlow graph.

```python
# Export the model to a SavedModel
model.save('path_to_saved_model', save_format='tf')

# Recreate the exact same model
new_model = keras.models.load_model('path_to_saved_model')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.
```



#### 只保存模型结构

有时，您只对模型的体系结构感兴趣，而不需要保存权重值或优化器。在本例中，您可以通过`get_config()方法`检索模型的“config”。配置是Python dict，它允许您重新创建相同的模型--从头开始初始化，而不需要以前在培训期间学到的任何信息。

```python
config = model.get_config()
reinitialized_model = keras.Model.from_config(config)

# Note that the model state is not preserved! We only saved the architecture.
new_predictions = reinitialized_model.predict(x_test)
assert abs(np.sum(predictions - new_predictions)) > 0.
```

您也可以使用`to_json() 和 from_json()`，它使用JSON字符串来存储配置，而不是Python dict。这对于将配置保存到磁盘非常有用。

```python
json_config = model.to_json()
reinitialized_model = keras.models.model_from_json(json_config)
```





#### 只保存模型参数

有时，您只对模型的状态感兴趣--它的权重值--而不是对体系结构感兴趣。在本例中，您可以通过`get_weights()`检索权重值作为Numpy数组的列表，并通过`set_weights()`设置模型的状态：

```python
weights = model.get_weights()  # Retrieves the state of the model.
model.set_weights(weights)  # Sets the state of the model.
```

您可以将`get_config()/from_config()`和`get_weights()/set_weights()`组合起来，以便在相同的状态下重新创建模型。但是与`model.Save()`不同，这将不包括培训配置和优化器。在使用该模型进行培训之前，您必须再次调用`compile()`。

```python
config = model.get_config()
weights = model.get_weights()

new_model = keras.Model.from_config(config)
new_model.set_weights(weights)

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer was not preserved,
# so the model should be compiled anew before training
# (and the optimizer will start from a blank state).
```

```python
# Save JSON config to disk
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)
# Save weights to disk
model.save_weights('path_to_my_weights.h5')

# Reload the model from the 2 files we saved
with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer was not preserved.
```



#### 使用checkpoints保存参数

注意，`save_weights()`可以创建Keras `HDF5格式`的文件，也可以创建TensorFlow检查点格式的文件。该格式是从您提供的文件扩展名推断出来的：如果是`“.h5”或“.keras`”，则框架使用Keras HDF 5格式。其他任何默认设置为检查点。

```python
# 对于完全显式性，格式可以通过Save_Format参数显式传递，该参数可以取值“tf”或“h5”
model.save_weights('path_to_my_tf_checkpoint', save_format='tf')
```





#### 保存子类化模型

 **Sequential** 模型和**Functional** 模型是表示层的DAG的数据结构。因此，它们可以安全地序列化和反序列化。

子类模型的不同之处在于它不是数据结构，而是一段代码。模型的体系结构是通过调用方法的主体来定义的。这意味着模型的体系结构不能安全地序列化。要加载模型，您需要访问创建它的代码(模型子类的代码)。或者，您可以将此代码序列化为字节码，但这是不安全的，而且通常不可移植。

首先，无法保存从未使用过的子类模型。这是因为需要对某些数据调用子类模型，以创建其权重。

保存子类模型的推荐方法是使用`save_weights()`创建一个TensorFlow SavedModel Checkpoint，该检查点将包含与模型关联的所有变量的值：

- The layers' weights
- The optimizer's state
- Any variables associated with stateful model metrics (if any)

```python
model.save_weights('path_to_my_weights', save_format='tf')
```

若要还原模型，需要访问创建模型对象的代码。请注意，为了恢复优化器状态和任何有状态度量的状态，您应该编译模型(参数与前面完全相同)，并在调用`load_weights()`之前对某些数据进行调用：

```python
# Recreate the model
new_model = get_model()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())

# This initializes the variables used by the optimizers,
# as well as any stateful metric variables
new_model.train_on_batch(x_train[:1], y_train[:1])

# Load the state of the old model
new_model.load_weights('path_to_my_weights')

# Check that the model state has been preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# The optimizer state is preserved as well,
# so you can resume training where you left off
new_first_batch_loss = new_model.train_on_batch(x_train[:64], y_train[:64])
assert first_batch_loss == new_first_batch_loss
```



### 模型部署

TFLite - FlatBuffer



## 10、机器翻译实战



# TensorFlow2.x API



## tf

***

### tf.fill

```python
tf.fill(
    dims,
    value,
    name=None
)
```

此操作创建形状`dims`的张量，并将其填充为`value`。

```python
tf.fill([2, 3], 9) 

<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[9, 9, 9],
       [9, 9, 9]])>
```



## tf.nn

***

### ctc_loss

```python
tf.nn.ctc_loss(
    labels,
    logits,
    label_length,
    logit_length,
    logits_time_major=True,
    unique=None,
    blank_index=None,
    name=None
)
```

**Notes:**

1、标签可以是密集的、零填充的张量，带有标签序列长度的矢量，也可以作为SparseTensor。

2、在TPU和GPU上：只支持密集的填充标签。

3、在CPU上：调用者可以使用SparseTensor或稠密的填充标签，但是使用SparseTenser调用将大大加快速度。



**Args:**

- **`labels`**: tensor of shape **[batch_size, max_label_seq_length]** or SparseTensor
- **`logits`**: tensor of shape **[frames, batch_size, num_labels]**, if logits_time_major == False, shape is [batch_size, frames, num_labels].
- **`label_length`**: tensor of shape **[batch_size]** `None if labels is SparseTensor` Length of reference label sequence in labels.
- **`logit_length`**: tensor of shape **[batch_size]** Length of input sequence in logits.
- **`logits_time_major`**: (optional) If True (default), logits is shaped [time, batch, logits]. If False, shape is [batch, time, logits]
- **`unique`**: (optional) Unique label indices as computed by ctc_unique_labels(labels). If supplied, enable a faster, memory efficient implementation on TPU.
- **`blank_index`**: (optional) Set the class index to use for the blank label. Negative values will start from num_classes, ie, -1 will reproduce the ctc_loss behavior of using **num_classes - 1** for the blank symbol. There is some memory/performance overhead to switching from the default of 0 as an additional shifted copy of the logits may be created.
- **`name`**: A name for this `Op`. Defaults to "ctc_loss_dense".



**Returns:**

- **`loss`**: tensor of shape **[batch_size]**, negative log probabilities.



###  ctc_greedy_decoder

```python
tf.nn.ctc_greedy_decoder(
    inputs,
    sequence_length,
    merge_repeated=True
)
```

如果`merge_repeated`为真，则`ABB_B_B`合并为`ABBB`。如果为假则为`ABBBB`

**Args:**

- **`inputs`**: 3-D `float` `Tensor` sized `[max_time, batch_size, num_classes]`. The logits.
- **`sequence_length`**: 1-D `int32` vector containing sequence lengths, having size `[batch_size]`.
- **`merge_repeated`**: Boolean. Default: True.



**Returns:**

A tuple `(decoded, neg_sum_logits)` where

- **`decoded`**: A single-element list. `decoded[0]` is an `SparseTensor` containing the decoded outputs.

  `decoded.indices`: Indices matrix `(total_decoded_outputs, 2)`. The rows store: `[batch, time]`.

  `decoded.values`: Values vector, size `(total_decoded_outputs)`. The vector stores the decoded classes.

  `decoded.dense_shape`: Shape vector, size `(2)`. The shape values are: `[batch_size, max_decoded_length]`

- **`neg_sum_logits`**: A `float` matrix `(batch_size x 1)` containing, for the sequence found, the negative of the sum of the greatest logit at each timeframe.















## tf.keras

***

### metrics



#### Metric

|                  |                                                              |
| ---------------- | ------------------------------------------------------------ |
| **add_weight**   | 添加状态变量。仅供子类使用。                                 |
| **reset_states** | 重置所有度量状态变量。当在训练期间对度量进行评估时，将在各epochs/steps之间调用此函数。 |
| **result**       | 计算并返回度量值张量。                                       |
| **update_state** | 为度量积累统计信息。                                         |
| **init**         |                                                              |

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.fit(dataset, epochs=10)
```



#### Mean

计算给定值的(加权)平均值。

这个度量创建了两个变量，总数`total`和计数`count`，用于计算值的平均值。这个平均值最终作为平均值返回，这是一个幂等运算，它简单地将总数除以计数。

```python
m = tf.keras.metrics.Mean() 
_ = m.update_state([1, 3, 5, 7]) 
m.result().numpy() 

m.reset_states() 
_ = m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0]) 
m.result().numpy() 
```



### optimizers



#### Optimizer

这个类定义了添加Ops来训练模型的API。您从不直接使用这个类，而是实例化它的一个子类

```python
# Create an optimizer with the desired parameters.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# `loss` is a callable that takes no argument and returns the value
# to minimize.
loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
# In graph mode, returns op that minimizes the loss by updating the listed
# variables.
opt_op = opt.minimize(loss, var_list=[var1, var2])
opt_op.run()
# In eager mode, simply call minimize to update the list of variables.
opt.minimize(loss, var_list=[var1, var2])
```



**手动训练：**

```python
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

loss_fn = lambda: tf.keras.losses.mse(model(input), output)
var_list_fn = lambda: model.trainable_weights

for input, output in data:
  opt.minimize(loss_fn, var_list_fn)
```



| 属性           |                                    |
| -------------- | ---------------------------------- |
| **iterations** | 变量。此优化器运行的培训步骤数     |
| **weights**    | 根据创建的顺序返回此优化器的变量。 |



|                     |                                                              |
| ------------------- | ------------------------------------------------------------ |
| **add_slot**        |                                                              |
| **add_weigth**      |                                                              |
| **apply_gradients** | 将梯度应用于变量。这是`minimize()`的第二部分。它返回一个应用梯度的`Operation`。参数：grads_and_vars : List of (gradient, variable) pairs |
| **from_config**     |                                                              |
| **get_config**      |                                                              |
| **get_gradients**   |                                                              |
| **get_slot**        |                                                              |
| **get_slot_names**  |                                                              |
| **get_updates**     |                                                              |
| **get_weights**     |                                                              |
| **minimize**        | 通过更新`var_list`将`loss`降到最低。此方法使用`tf.GradientTape`计算梯度，并调用`apply_gradients()`。如果您想在应用之前处理梯度，那么就显式地调用，而不是使用这个函数。 |
| **set_weights**     |                                                              |
| **variables**       |                                                              |
|                     |                                                              |



#### schedules

|                            |                                      |
| -------------------------- | ------------------------------------ |
| **ExponentialDecay**       | 使用**指数衰减**的学习率计划表       |
| **InverseTimeDecay**       | 使用**逆时间衰减**的学习率计划表     |
| **LearningRateSchedule**   | 使用**可串行化**的学习率衰减时间表。 |
| **PiecewiseConstantDecay** | 使用**分段常数衰减**的学习率计划表。 |
| **PolynomialDecay**        | 使用**多项式衰减**的学习速率计划表。 |

```python
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, epochs=5)
```



**ExponentialDecay**

在训练模型时，往往建议随着训练的进行而降低学习率。该计划在给定初始学习速率的情况下，将指数衰减函数应用于优化器步骤。

```python
__init__(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)
```

计算如下：

如果参数`staircase`为真，那么`step / decay_steps`是整数除法，衰减学习率遵循阶梯函数。学习速率在离散间隔。

```python
def decayed_learning_rate(step):
  return initial_learning_rate * decay_rate ^ (step / decay_steps)
```













## tf.data

***

### Dataset



#### map方法

```python
map(
    map_func,
    num_parallel_calls=None
)
```

此转换将**map_func**应用于此数据集的每个元素，并返回一个包含转换后的元素的新数据集，其顺序与它们在输入中出现的顺序相同。map_func可用于更改数据集元素的值和结构。例如，向每个元素添加1，或投影元素组件的子集。

**map_func的输入参数取决于此数据集中每个元素的结构**。

```python
# Each element is a tuple containing two `tf.Tensor` objects. 
elements = [(1, "foo"), (2, "bar"), (3, "baz)")] 
dataset = tf.data.Dataset.from_generator( 
    lambda: elements, (tf.int32, tf.string)) 
# `map_func` takes two arguments of type `tf.Tensor`. This function 
# projects out just the first component. 
result = dataset.map(lambda x_int, y_str: x_int) 
list(result.as_numpy_iterator()) 
```

**map_func返回的值确定返回数据集中每个元素的结构。**

```python
dataset = tf.data.Dataset.range(3) 
# `map_func` returns two `tf.Tensor` objects. 
def g(x): 
  return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"]) 
result = dataset.map(g) 
result.element_spec 

# Python primitives, lists, and NumPy arrays are implicitly converted to 
# `tf.Tensor`. 
def h(x): 
  return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64) 
result = dataset.map(h) 
result.element_spec 

# `map_func` can return nested structures. 
def i(x): 
  return (37.0, [42, 16]), "foo" 
result = dataset.map(i) 
result.element_spec 
```

**map_func可以接受作为参数并返回任何类型的DataSet元素。**

要在函数中使用Python代码，有两个选项：

1)依赖签名将Python代码转换为等效的图计算。这种方法的缺点是签名可以转换一些但不是全部Python代码。

2)使用**tf.py_function**，它允许您编写任意Python代码，但通常会导致性能比1差。例如：

```python
d = tf.data.Dataset.from_tensor_slices(['hello', 'world']) 
# transform a string tensor to upper case string using a Python function 
def upper_case_fn(t: tf.Tensor): 
  return t.numpy().decode('utf-8').upper() 

d = d.map(lambda x: tf.py_function(func=upper_case_fn, inp=[x], Tout=tf.string)) 
list(d.as_numpy_iterator()) 
```



#### interleave方法

```python
interleave(
    map_func,
    cycle_length=AUTOTUNE,
    block_length=1,
    num_parallel_calls=None
)
```

将map_func映射到此数据集，并将结果交织在一起。

**首先该方法会从该Dataset中取出cycle_length个element，然后对这些element apply map_func, 得到cycle_length个新的Dataset对象。然后从这些新生成的Dataset对象中取数据，每个Dataset对象一次取block_length个数据。当新生成的某个Dataset的对象取尽时，从原Dataset中再取一个element，然后apply map_func，以此类推。**



```python
dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ] 
# NOTE: New lines indicate "block" boundaries. 
dataset = dataset.interleave( 
    lambda x: Dataset.from_tensors(x).repeat(6), 
    cycle_length=2, block_length=4) 
list(dataset.as_numpy_iterator()) 
[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5]
```



#### batch方法

将此数据集的连续元素组合成批

结果元素的组件将有一个额外的外部维度，即Batch_Size

```python
batch(
    batch_size,
    drop_remainder=False
)
```

```python
dataset = tf.data.Dataset.range(8) 
dataset = dataset.batch(3) 
list(dataset.as_numpy_iterator()) 

[array([0, 1, 2], dtype=int64),
 array([3, 4, 5], dtype=int64),
 array([6, 7], dtype=int64)]
```

```python
dataset = tf.data.Dataset.range(8) 
dataset = dataset.batch(3, drop_remainder=True) 
list(dataset.as_numpy_iterator()) 

[array([0, 1, 2], dtype=int64), array([3, 4, 5], dtype=int64)]
```



####  padded_batch方法

```python
padded_batch(
    batch_size,
    padded_shapes=None,
    padding_values=None,
    drop_remainder=False
)
```

将此数据集的连续元素组合到填充的批中,此转换将输入数据集的多个连续元素组合为单个元素。

与`tf.data.Dataset.batch`不同，要组成批的输入元素可能具有不同的形状，此转换将将每个组件以`padded_shapes`的形式填充到相应的形状。`padded_shapes`参数确定输出元素中每个组件的每个维度的最终形状：

- 如果维度是常量，则该组件将在该维度中填充到该长度。

- 如果维度未知，则该组件将被填充到该维度中所有元素的最大长度。

```python
A = tf.data.Dataset.range(1, 5).map(lambda x: tf.fill([x], x))
#[1]
#[2 2]
#[3 3 3]
#[4 4 4 4]
B = A.padded_batch(2, padded_shapes=[None])

for element in B.as_numpy_iterator():
    print(element) 
    
[[1 0]
 [2 2]]
[[3 3 3 0]
 [4 4 4 4]]
```







### TFRecordWriter

将数据集写入TFRecord文件

数据集的元素必须是标量字符串。要将DataSet元素序列化为字符串，可以使用`tf.io.serialize_tensor`函数。

```python
#存储
dataset = tf.data.Dataset.range(3)
dataset = dataset.map(tf.io.serialize_tensor)
writer = tf.data.experimental.TFRecordWriter("/path/to/file.tfrecord")
writer.write(dataset)
```

```python
# 读取
dataset = tf.data.TFRecordDataset("/path/to/file.tfrecord")
dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.int64))
```

若要在多个TFRecord文件中分解数据集，请执行以下操作

```python
dataset = ... # dataset to be written

def reduce_func(key, dataset):
  filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(dataset.map(lambda _, x: x))
  return tf.data.Dataset.from_tensors(filename)

dataset = dataset.enumerate()
dataset = dataset.apply(tf.data.experimental.group_by_window(
  lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
))
```











## tf.lookup

***

class **KeyValueTensorInitializer**: 由给出的键和值张量初始化表。

class **StaticHashTable**: 初始化后不可变的泛型哈希表。

class **StaticVocabularyTable**: 将词汇表外键分配给桶的字符串到ID表包装器。

class **TextFileIndex**: 从每一行获取的键和值内容。

class **TextFileInitializer**: 文本文件中的表初始化器。

Class **DenseHashTable**: 使用张量作为后备存储的通用可变哈希表实现



### StaticHashTable

```python
__init__(
    initializer,
    default_value,
    name=None
)
```



```python
keys_tensor = tf.constant([1, 2])
vals_tensor = tf.constant([3, 4])
input_tensor = tf.constant([1, 5])
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
print(table.lookup(input_tensor))
```

**属性:**

|                     |                              |
| ------------------- | ---------------------------- |
| **default_value**   |                              |
| **key_dtype**       |                              |
| **value_dtype**     |                              |
| **name**            |                              |
| **resource_handle** | 返回与此资源关联的资源句柄。 |



**方法：**

|            |                                                              |
| ---------- | ------------------------------------------------------------ |
| **export** | 返回表中所有键和值的张量。                                   |
| **lookup** | 在表中查找键，输出相应的值。lookup( keys, name=None ) 参数keys是tensor或sparsetensor |
| **size**   | 计算此表中的元素数。                                         |



### KeyValueTensorInitializer

```python
__init__(
    keys,
    values,
    key_dtype=None,
    value_dtype=None,
    name=None
)
```



### TextFileInitializer

```python
__init__(
    filename,
    key_dtype,
    key_index,
    value_dtype,
    value_index,
    vocab_size=None,
    delimiter='\t',
    name=None
)
```



**TextFileIndex.LINE_NUMBER** :

**TextFileIndex.WHOLE_LINE** : 



## tf.strings

***

|                                 |      |
| ------------------------------- | ---- |
| **as_string**                   |      |
| **bytes_split**                 |      |
| **format**                      |      |
| **join**                        |      |
| **length**                      |      |
| **lower**                       |      |
| **ngrams**                      |      |
| **reduce_join**                 |      |
| **regex_full_match**            |      |
| **regex_replace**               |      |
| **split**                       |      |
| **strip**                       |      |
| **substr**                      |      |
| **to_hash_bucket**              |      |
| **to_hash_bucket_fast**         |      |
| **to_hash_bucket_strong**       |      |
| **to_number**                   |      |
| **unicode_decode**              |      |
| **unicode_decode_with_offsets** |      |
| **unicode_encode**              |      |
| **unicode_script**              |      |
| **unicode_split**               |      |
| **unicode_split_with_offsets**  |      |
| **unicode_transcode**           |      |
| **unsorted_segment_join**       |      |
| **upper**                       |      |











## tf.sparse

***





### to_dense方法

```python
tf.sparse.to_dense(
    sp_input,
    default_value=None,
    validate_indices=True,
    name=None
)
```



## tf.image

***

|                                 |                                                        |
| ------------------------------- | ------------------------------------------------------ |
| adjust_brightness               | 调整RGB或灰度图像的亮度                                |
| adjust_contrast                 | 调整RGB或灰度图像的对比度。                            |
| adjust_gamma                    | 对输入图像执行伽玛校正                                 |
| adjust_hue                      | 调整RGB图像的色调                                      |
| adjust_jpeg_quality             | 调整图像的jpeg编码质量                                 |
| adjust_saturation               | 调整RGB图像的饱和度。                                  |
| central_crop                    | 裁剪图像的中心区域                                     |
| combined_non_max_suppression    | 贪婪地按分数的降序选择包围框的子集                     |
| convert_image_dtype             | 将图像转换为dtype，并在需要时缩放其值                  |
| crop_and_resize                 | 从输入图像张量中提取作物并调整它们的大小               |
| crop_to_bounding_box            | 将图像裁剪到指定的边框中                               |
| decode_and_crop_jpeg            | 解码并裁剪JPEG编码的图像到uint 8张量                   |
| decode_bmp                      |                                                        |
| decode_gif                      |                                                        |
| decode_image                    |                                                        |
| decode_jpeg                     |                                                        |
| decode_png                      |                                                        |
| draw_bounding_boxes             | 在一批图像上绘制边框                                   |
| encode_jpeg                     |                                                        |
| encode_png                      |                                                        |
| extract_glimpse                 | 从输入张量中提取一瞥                                   |
| extract_jpeg_shape              |                                                        |
| extract_patches                 |                                                        |
| flip_left_right                 |                                                        |
| flip_up_down                    |                                                        |
| generate_bounding_box_proposals |                                                        |
| grayscale_to_rgb                |                                                        |
| hsv_to_rgb                      |                                                        |
| image_gradients                 |                                                        |
| is_jpeg                         |                                                        |
| non_max_suppression             | 贪婪地按分数的降序选择包围框的子集。                   |
| non_max_suppression_overlaps    |                                                        |
| non_max_suppression_padded      |                                                        |
| non_max_suppression_with_scores |                                                        |
| pad_to_bounding_box             | 衬垫图像与零到指定的高度和宽度。                       |
| per_image_standardization       | 对图像中的每幅图像进行线性缩放，使其均值为0，方差为1。 |
| psnr                            | 返回a和b之间的峰值信噪比。                             |
| random_brightness               |                                                        |
| random_contrast                 |                                                        |
| random_crop                     |                                                        |
| random_flip_left_right          |                                                        |
| random_flip_up_down             |                                                        |
| random_hue                      |                                                        |
| random_jpeg_quality             |                                                        |
| random_saturation               |                                                        |
| resize                          |                                                        |
| resize_with_crop_or_pad         |                                                        |
| resize_with_pad                 | 调整图像大小并将其设置为目标宽度和高度                 |
| rgb_to_grayscale                |                                                        |
| rgb_to_hsv                      |                                                        |
| rgb_to_yiq                      |                                                        |
| rgb_to_yuv                      | 将一个或多个图像从RGB转换为YUV。                       |
| rot90                           | 逆时针旋转图像90度.                                    |
| sample_distorted_bounding_box   | 为图像生成一个随机扭曲的包围框。                       |
| sobel_edges                     | 返回一个保持Sobel边映射的张量。                        |
| ssim                            | 计算img 1和img 2之间的ssim索引。                       |
| ssim_multiscale                 | 计算img1和img2之间的MS-ssim。                          |
| total_variation                 | 计算并返回一个或多个图像的总变化。                     |
| transpose                       | 通过交换高度和宽度尺寸来转换图像。                     |
| yiq_to_rgb                      | 将一个或多个图像从YIQ转换为RGB。                       |
| yuv_to_rgb                      | 将一个或多个图像从YUV转换为RGB                         |



### draw_bounding_boxes

```python
tf.image.draw_bounding_boxes(
    images,
    boxes,
    colors,
    name=None
)
```

输出图像副本，但在框中位置指定的像素零或多个边界框的顶部绘制。框中每个边框的坐标编码为[y_min，x_min，y_max，x_max]。包围框坐标在[0.0，1.0]中相对于基础图像的宽度和高度浮动。

例如，如果图像为100 x 200像素(高度x宽度)，而边界框为[0.1，0.2，0.5，0.9]，则边界框的左上角和右下角坐标将为(40，10)至(180，50)(在(x，y)坐标中)。



Args:

- **`images`**: A `Tensor`. Must be one of the following types: `float32`, `half`. 4-D with shape `[batch, height, width, depth]`. A batch of images.
- **`boxes`**: A `Tensor` of type `float32`. 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding boxes.
- **`colors`**: A `Tensor` of type `float32`. 2-D. A list of RGBA colors to cycle through for the boxes.
- **`name`**: A name for the operation (optional).

Returns:

A `Tensor`. Has the same type as `images`.

```python
# create an empty image
img = tf.zeros([1, 28, 28, 3])
box = np.array([0, 0, 0.5, 0.5])
boxes = box.reshape([1, 1, 4])

# alternate between red and blue
colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
img_bbox = tf.image.draw_bounding_boxes(img, boxes, colors)
```

