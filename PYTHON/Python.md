## 1、Virtualenv

安装命令 `pip install virtualenv`

文档 https://virtualenv.pypa.io/en/stable/userguide/



**创建虚拟环境**

命令：virtualenv 环境名称  `virtualenv env_tf2`

可选参数：**-p** 指定python版本。

可选参数：**--system-site-packages** 是否继承系统第三方库



**激活/退出虚拟环境**

激活命令：`source bin/activate`

退出命令：`deactivate`



**删除虚拟环境**

直接删除文件夹



## 2、Pathlib

该模块提供表示文件系统路径的类，其语义适用于不同的操作系统。

路径类被分为提供纯计算操作而没有 I/O 的**纯路径purepath**，以及从纯路径继承而来但提供 I/O 操作的**具体路径**。

![image-20200127012443650](images/Python.assets/image-20200127012443650.png)



**具体路径**



|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **pathlib.Path(*pathsegments)**                              | 一个 `PurePath` 的子类，此类以当前系统的路径风格表示路径（实例化为 `PosixPath` 或 `WindowsPath`） |
| **cwd()**                                                    | 返回一个新的表示**当前目录**的路径对象                       |
| **home()**                                                   | 返回一个表示**当前用户家目录**的新路径对象                   |
| **stat()**                                                   | 返回此路径的信息                                             |
| **chmod(*mode*)**                                            | 改变文件的模式和权限                                         |
| **exists()**                                                 | 是否指向一个已存在的文件或目录                               |
| **expanduser()**                                             | 返回展开了包含 `~` 和 `~user` 的构造                         |
| **glob(pattern)**                                            | 解析相对于此路径的通配符 *pattern*，产生所有匹配的文件       |
| **group()**                                                  | 返回拥有此文件的用户组                                       |
| **is_dir()**                                                 |                                                              |
| **is_file()**                                                |                                                              |
| **iterdir()**                                                | 当路径指向一个目录时，产生该路径下的对象的路径               |
| **mkdir(mode=0o777, parents=False, exist_ok=False)**         | 新建给定路径的目录                                           |
| **open(mode='r', buffering=-1, encoding=None, errors=None, newline=None)** | 打开路径指向的文件，就像内置的 **open()**函数所做的一样      |
| **read_bytes()**                                             |                                                              |
| **read_text(encoding=None, errors=None)**                    |                                                              |
| **rename(target)**                                           |                                                              |
| **replace(target)**                                          | 将文件名目录重命名为给定的 *target*，并返回一个新的指向 *target* 的 Path 实例 |
| **resolve(strict=False)**                                    | 将路径绝对化，解析任何符号链接。返回新的路径对象             |
| **rmdir()**                                                  |                                                              |
| **touch(mode=0o666, exist_ok=True)**                         |                                                              |
| **write_bytes(data)**                                        |                                                              |
| **write_text(data, encoding=None, errors=None)**             |                                                              |
|                                                              |                                                              |



## 3、argparse

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ta", "--train_annotation_paths", type=str, 
                    required=True, nargs="+", 
                    help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, nargs="+", 
                    help="The path of val data annotation file.")
parser.add_argument("-tf", "--train_parse_funcs", type=str, required=True,
                    nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-vf", "--val_parse_funcs", type=str, nargs="+", 
                    help="The parse functions of annotaion files.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-e", "--epochs", type=int, default=20, 
                    help="Num of epochs to train.")
args = parser.parse_args()
```

`default`：没有设置值情况下的默认参数

`required`: 表示这个参数是否一定需要设置

`type`：参数类型

`choices`：参数值只能从几个选项里面选择 choices=['alexnet', 'vgg']

`help`：指定参数的说明信息

`dest`：设置参数在代码中的变量名

`nargs`： 设置参数在使用可以提供的个数

```
N   参数的绝对个数（例如：3）
'?'   0或1个参数
'*'   0或所有参数
'+'   所有，并且至少一个参数
```









## 4、pytest

https://docs.pytest.org/en/latest/

### start

`pip install pytest`

编写pytest测试样例非常简单，只需要按照下面的规则：

- 测试文件以test_开头（以\_test结尾也可以）
- 测试类以Test开头，并且不能带有 **init** 方法
- 测试函数以test_开头
- 断言使用基本的assert即可

```python
# content of test_sample.py
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5
```

```python
# content of test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")
```



### fixture的scope参数

它用`@pytest.fixture`标识,在你编写测试函数的时候，你可以将此函数名称做为传入参数，pytest将会以依赖注入方式，将该函数的返回值作为测试函数的传入参数。

scope参数有四种，分别是'function','module','class','session'，默认为function。

- function：每个test都运行，默认是function的scope
- class：每个class的所有test只运行一次
- module：每个module的所有test只运行一次
- session：每个session只运行一次



### setup和teardown操作

- setup，在测试函数或类之前执行，完成准备工作，例如数据库链接、测试数据、打开文件等
- teardown，在测试函数或类之后执行，完成收尾工作，例如断开数据库链接、回收内存资源等
- 备注：也可以通过在fixture函数中通过yield实现setup和teardown功能



### 通过pytest.mark对test方法分类执行

通过@pytest.mark控制需要执行哪些feature的test，例如在执行test前增加修饰`@pytest.mark.website`



### @pytest.mark.parametrize

```python
@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
    
@pytest.mark.parametrize(
    "test_input,expected",
    [("3+5", 8), ("2+4", 6), pytest.param("6*9", 42, marks=pytest.mark.xfail)],
)
def test_eval(test_input, expected):
    assert eval(test_input) == expected
    
@pytest.mark.parametrize("x", [0, 1])
@pytest.mark.parametrize("y", [2, 3])
def test_foo(x, y):
    pass
```



## 5、yaml

yaml是一个专门用来写配置文件的语言。



### 1. yaml文件规则

- 区分大小写；
- 使用缩进表示层级关系；
- 使用空格键缩进，而非Tab键缩进
- 缩进的空格数目不固定，只需要相同层级的元素左侧对齐；
- 文件中的字符串不需要使用引号标注，但若字符串包含有特殊字符则需用引号标注；
- 注释标识为`#`



### 2. yaml文件数据结构

- 对象：键值对的集合（简称 "映射或字典"）
   键值对用冒号 `:` 结构表示，冒号与值之间需用空格分隔
- 数组：一组按序排列的值（简称 "序列或列表"）
   数组前加有 “`-`” 符号，符号与值之间需用空格分隔
- 纯量(scalars)：单个的、不可再分的值（如：字符串、bool值、整数、浮点数、时间、日期、null等）
   None值可用null可 ~ 表示



### 3. python读取yaml

`pip install pyyaml`

`import yaml`



python通过open方式读取文件数据，再通过`yaml.load`函数将数据转化为列表或字典；

```python
import yaml
import os

def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    print(file_data)
    print("类型：", type(file_data))

    # 将字符串转化为字典或列表
    print("***转化yaml数据为字典或列表***")
    data = yaml.load(file_data)
    print(data)
    print("类型：", type(data))
    return data
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "config.yaml")
get_yaml_data(yaml_path)

"""
***获取yaml文件数据***
# yaml键值对：即python中字典
usr: my
psw: 123455
类型：<class 'str'>
***转化yaml数据为字典或列表***
{'usr': 'my', 'psw': 123455}
类型：<class 'dict'>
"""
```



yaml文件中内容为键值对：

```bash
# yaml键值对嵌套：即python中字典嵌套字典
usr1:
  name: a
  psw: 123
usr2:
  name: b
  psw: 456
```

python解析yaml文件后获取的数据：

```bash
{'usr1': {'name': 'a', 'psw': 123}, 'usr2': {'name': 'b', 'psw': 456}}
```

yaml文件中“键值对”中嵌套“数组”

```bash
# yaml键值对中嵌套数组
usr3:
  - a
  - b
  - c
usr4:
  - b
```

python解析yaml文件后获取的数据：

```bash
{'usr3': ['a', 'b', 'c'], 'usr4': ['b']}
```

yaml文件“数组”中嵌套“键值对”

```cpp
# yaml"数组"中嵌套"键值对"
- usr1: aaa
- psw1: 111
  usr2: bbb
  psw2: 222
```

python解析yaml文件后获取的数据：

```bash
[{'usr1': 'aaa'}, {'psw1': 111, 'usr2': 'bbb', 'psw2': 222}]
```

yaml文件中基本数据类型：

```dart
# 纯量
s_val: name              # 字符串：{'s_val': 'name'}
spec_s_val: "name\n"    # 特殊字符串：{'spec_s_val': 'name\n'
num_val: 31.14          # 数字：{'num_val': 31.14}
bol_val: true           # 布尔值：{'bol_val': True}
nul_val: null           # null值：{'nul_val': None}
nul_val1: ~             # null值：{'nul_val1': None}
time_val: 2018-03-01t11:33:22.55-06:00     # 时间值：{'time_val': datetime.datetime(2018, 3, 1, 17, 33, 22, 550000)}
date_val: 2019-01-10    # 日期值：{'date_val': datetime.date(2019, 1, 10)}
```



### 4. python读取多个yaml文档

```python
# 分段yaml文件中多个文档
---
animal1: dog
age: 2
---
animal2: cat
age: 3
```

python获取yaml数据时需使用`load_all`函数来解析全部的文档，再从中读取对象中的数据

```python
# yaml文件中含有多个文档时，分别获取文档中数据
def get_yaml_load_all(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    all_data = yaml.load_all(file_data)
    for data in all_data:
        print(data)
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "config.yaml")
get_yaml_load_all(yaml_path)
"""结果
{'animal1': 'dog', 'age': 2}
{'animal2': 'cat', 'age': 3}
"""
```



## 6、trdg

| 参数                                   | 说明                                                         |
| -------------------------------------- | :----------------------------------------------------------- |
| --output_dir                           | The output directory                                         |
| "-i","--input_file"                    | When set, this argument uses a specified text file as source for the text |
| "-l","--language",                     | The language to use, should be fr (French), en (English), es (Spanish), <br />de (German), or cn (Chinese). |
| "-c","--count"                         | The number of images to be created.                          |
| "-rs",<br />"--random_sequences",      | Use random sequences as the source text for the generation. <br />Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three. |
| "-let",<br />"--include_letters"       | Define if random sequences should contain letters. Only works with -rs |
| "-num",<br />"--include_numbers"       | Define if random sequences should contain numbers. Only works with -rs |
| "-sym",<br />"--include_symbols"       | Define if random sequences should contain symbols. Only works with -rs |
| "-w","--length"                        | Define how many words should be included in each generated sample. <br />If the text source is Wikipedia, this is the MINIMUM length" |
| "-r","--random"                        | Define if the produced string will have variable word count (with --length being the maximum) |
| "-f","--format"                        | Define the height of the produced images if horizontal, else the width |
| "-t","--thread_count"                  | Define the number of thread to use for image generation      |
| "-e","--extension"                     | Define the extension to save the image with                  |
| "-k","--skew_angle"                    | Define skewing angle of the generated text. In positive degrees |
| "-rk","--random_skew"                  | When set, the skew angle will be randomized between the value set with -k and it's opposite |
| "-wk","--use_wikipedia"                | Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s" |
| "-bl","--blur"                         | Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius |
| "-rbl","--random_blur"                 | When set, the blur radius will be randomized between 0 and -bl. |
| "-b","--background"                    | Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures |
| "-hw","--handwritten"                  | Define if the data will be "handwritten" by an RNN           |
| "-na","--name_format"                  | Define how the produced files will be named. <br />0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] <br />2: [ID].[EXT] + one file labels.txt containing id-to-label mappings |
| "-om","--output_mask"                  | Define if the generator will return masks for the text       |
| "-d","--distorsion"                    | Define a distorsion applied to the resulting image. <br />0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random |
| "-do",<br />"--distorsion_orientation" | Define the distorsion's orientation. <br />Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both |
| "-wd","--width"                        | Define the width of the resulting image. <br />If not set it will be the width of the text + 10. <br />If the width of the generated text is bigger that number will be used |
| "-al","--alignment"                    | Define the alignment of the text in the image. <br />Only used if the width parameter is set. 0: left, 1: center, 2: right |
| "-or","--orientation"                  | Define the orientation of the text. 0: Horizontal, 1: Vertical |
| "-tc","--text_color"                   | Define the text's color, should be either a single hex color or a range in the ?,? format. |
| "-sw","--space_width"                  | Define the width of the spaces between words. 2.0 means twice the normal space width |
| "-cs","--character_spacing"            | Define the width of the spaces between characters. 2 means two pixels |
| "-m","--margins"                       | Define the margins around the text when rendered. In pixels  |
| "-fi","--fit"                          | Apply a tight crop around the rendered text                  |
| "-ft", "--font"                        | Define font to be used                                       |
| "-fd","--font_dir"                     | Define a font directory to be used                           |
| "-ca","--case"                         | Generate upper or lowercase only. arguments: upper or lower. Example: --case upper |
| "-dt", "--dict"                        | Define the dictionary to be used                             |
| "-ws", "--word_split",                 | Split on words instead of on characters (preserves ligatures, no character spacing) |

