## 1、Getting Start

src/lib.rs

```rust
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
/// Formats the sum of two numbers as string.
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
/// A Python module implemented in Rust.
fn string_sum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;

    Ok(())
}
```

cargo.toml

```rust
[package]
name = "string-sum"
authors = ["modemlxsg@gmail.com"]
version = "0.1.0"
edition = "2018"

[lib]
name = "string_sum"
crate-type = ["cdylib"]

[dependencies.pyo3]
version = "0.9.2"
features = ["extension-module"]
```

`cargo build --release`

`maturin.exe build`

`pip install ./target/wheels/string_sum-0.1.0-cp37-none-win_amd64.whl`

```python
import string_sum
c = string_sum.sum_as_string(1, 2)
print(c)

3
```











## 2、Python Module

如“入门”一章中所示，您可以按以下方式创建一个模块

```rust
use pyo3::prelude::*;

// add bindings to the generated Python module
// N.B: "rust2py" must be the name of the `.so` or `.pyd` file.

/// This module is implemented in Rust.
#[pymodule]
fn rust2py(py: Python, m: &PyModule) -> PyResult<()> {
    // PyO3 aware function. All of our Python interfaces could be declared in a separate module.
    // Note that the `#[pyfn()]` annotation automatically converts the arguments from
    // Python objects to Rust values, and the Rust return value back into a Python object.
    // The `_py` argument represents that we're holding the GIL.
    #[pyfn(m, "sum_as_string")]
    fn sum_as_string_py(_py: Python, a: i64, b: i64) -> PyResult<String> {
        let out = sum_as_string(a, b);
        Ok(out)
    }

    Ok(())
}

// logic implemented as a normal Rust function
fn sum_as_string(a: i64, b: i64) -> String {
    format!("{}", a + b)
}
```

`＃[pymodule]`过程宏属性负责将模块的初始化函数导出到Python。 它可以将模块的名称作为参数，该名称必须是.so或.pyd文件的名称； 默认值为Rust函数的名称。

要导入模块，请按照入门中的说明复制共享库，或使用工具（例如使用maturin或`python setup.py develop`使用setuptools-rust开发。

在Python中，模块是一流的对象。这意味着您可以将它们存储为值或将它们添加到字典或其他模块中：

```rust
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule};
use pyo3::types::IntoPyDict;

#[pyfunction]
fn subfunction() -> String {
    "Subfunction".to_string()
}

#[pymodule]
fn submodule(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pyfunction!(subfunction))?;
    Ok(())
}

#[pymodule]
fn supermodule(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(submodule))?;
    Ok(())
}

fn nested_call() {
    let gil = GILGuard::acquire();
    let py = gil.python();
    let supermodule = wrap_pymodule!(supermodule)(py);
    let ctx = [("supermodule", supermodule)].into_py_dict(py);

    py.run("assert supermodule.submodule.subfunction() == 'Subfunction'", None, Some(&ctx)).unwrap();
}
```

这样，您可以在单个扩展模块中创建模块层次结构。



## 3、Python Functions

PyO3支持两种在Python中定义自由函数的方式。两者都需要将功能注册到模块。

一种方法是在模块定义中定义函数，并用`＃[pyfn]`注释。

```rust
use pyo3::prelude::*;

#[pymodule]
fn rust2py(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "sum_as_string")]
    fn sum_as_string_py(_py: Python, a:i64, b:i64) -> PyResult<String> {
        Ok(format!("{}", a + b))
    }

    Ok(())
}
```

另一个是使用`＃[pyfunction]`注释一个函数，然后使用`wrap_pyfunction!`宏将其添加到模块中

```rust
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn double(x: usize) -> usize {
    x * 2
}

#[pymodule]
fn module_with_functions(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(double)).unwrap();

    Ok(())
}

```



### 参数解析

\#[pyfunction]和#[pyfn]属性都支持指定参数解析的细节。详细信息在“方法参数”一节中给出。下面是一个函数的例子，它接受任意的关键字参数(Python语法中的**kwargs)并返回传递的数字:

```rust
extern crate pyo3;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyDict;

#[pyfunction(kwds="**")]
fn num_kwds(kwds: Option<&PyDict>) -> usize {
    kwds.map_or(0, |dict| dict.len())
}

#[pymodule]
fn module_with_functions(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(num_kwds)).unwrap();
    Ok(())
}

fn main() {}

```













