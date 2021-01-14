

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









## 7、Queue

在Python文档中搜索队列（queue）会发现，Python标准库中包含了四种队列，分别是

`queue.Queue` / `asyncio.Queue` / `multiprocessing.Queue` / `collections.deque`。



### collections.deque

deque是双端队列（double-ended queue）的缩写，由于两端都能编辑，既可以用来实现栈（stack）也可以用来实现队列（queue）。

collections.deque是一个可以方便实现队列的数据结构，具有线程安全的特性，并且有很高的性能。



### queue.Queue & asyncio.Queue

`queue.Queue`和`asyncio.Queue`都是支持多生产者、多消费者的队列，基于collections.deque，他们都提供了Queue（FIFO队列）、PriorityQueue（优先级队列）、LifoQueue（LIFO队列），接口方面也相同。

区别在于queue.Queue适用于多线程的场景，asyncio.Queue适用于协程场景下的通信，由于asyncio的加成，queue.Queue下的阻塞接口在asyncio.Queue中则是以返回协程对象的方式执行



### multiprocessing.Queue

multiprocessing提供了三种队列，分别是`Queue`、`SimpleQueue`、`JoinableQueue`。

multiprocessing.Queue既是线程安全也是进程安全的，相当于queue.Queue的多进程克隆版。和threading.Queue很像，multiprocessing.Queue支持put和get操作，底层结构是multiprocessing.Pipe



## 8、comtypes



## 9、lxml

常见的 XML 编程接口有 DOM 和 SAX，这两种接口处理 XML 文件的方式不同，当然使用场合也不同。

Python 有三种方法解析 XML，SAX，DOM，以及 ElementTree:

1.**SAX** (simple API for XML )
Python 标准库包含 SAX 解析器，SAX 用事件驱动模型，通过在解析XML的过程中触发一个个的事件并调用用户定义的回调函数来处理XML文件。

2.**DOM**(Document Object Model)
将 XML 数据在内存中解析成一个树，通过对树的操作来操作XML

3.**ElementTree**(元素树)
ElementTree就像一个轻量级的DOM，具有方便友好的API。代码可用性好，速度快，消耗内存少。



**Element对象**
tag:string      元素代表的数据种类。
text:string     元素的内容。
tail:string      元素的尾形。
attrib:dictionary     元素的属性字典。

**针对属性的操作**
clear()          清空元素的后代、属性、text和tail也设置为None。
get(key, default=None)     获取key对应的属性值，如该属性不存在则返回default值。
items()         根据属性字典返回一个列表，列表元素为(key, value）。
keys()           返回包含所有元素属性键的列表。
set(key, value)     设置新的属性键与值。



**针对后代的操作**
append(subelement)     添加直系子元素。

extend(subelements)    增加一串元素对象作为子元素。

find(match)             寻找第一个匹配子元素，匹配对象可以为tag或path。

findall(match)          寻找所有匹配子元素，匹配对象可以为tag或path。

findtext(match)         寻找第一个匹配子元素，返回其text值。匹配对象可以为tag或path。

insert(index, element)  在指定位置插入子元素。

iter(tag=None)           生成遍历当前元素所有后代或者给定tag的后代的迭代器。

iterfind(match)          根据tag或path查找所有的后代。

itertext()               遍历所有后代并返回text值。

remove(subelement)      删除子元素。



## 10、xpath

**XPath 是一门在 XML 文档中查找信息的语言。XPath 用于在 XML 文档中通过元素和属性进行导航。**

- XPath 使用路径表达式在 XML 文档中进行导航
- XPath 包含一个标准函数库
- XPath 是 XSLT 中的主要元素
- XPath 是一个 W3C 标准

XPath 使用路径表达式来选取 XML 文档中的节点或者节点集。这些路径表达式和我们在常规的电脑文件系统中看到的表达式非常相似。

XPath 含有超过 100 个内建的函数。这些函数用于字符串值、数值、日期和时间比较、节点和 QName 处理、序列处理、逻辑值等等。



### 节点

**在 XPath 中，有七种类型的节点：元素、属性、文本、命名空间、处理指令、注释以及文档节点（或称为根节点）。**

```xml
<?xml version="1.0" encoding="ISO-8859-1"?>

<bookstore>

<book>
  <title lang="en">Harry Potter</title>
  <author>J K. Rowling</author> 
  <year>2005</year>
  <price>29.99</price>
</book>

</bookstore>
```

```
<bookstore> （文档节点）
<author>J K. Rowling</author> （元素节点）
lang="en" （属性节点）
```



**节点关系**：**父**（Parent）、**子**（Children）、**同胞**（Sibling）、**先辈**（Ancestor）、**后代**（Descendant）



### 语法

**XPath 使用路径表达式来选取 XML 文档中的节点或节点集。节点是通过沿着路径 (path) 或者步 (steps) 来选取的。**

| nodename | 选取此节点的所有子节点。                                   |
| -------- | ---------------------------------------------------------- |
| /        | 从根节点选取。                                             |
| //       | 从匹配选择的当前节点选择文档中的节点，而不考虑它们的位置。 |
| .        | 选取当前节点。                                             |
| ..       | 选取当前节点的父节点。                                     |
| @        | 选取属性。                                                 |

| 通配符 | 描述                 |
| :----- | :------------------- |
| *      | 匹配任何元素节点。   |
| @*     | 匹配任何属性节点。   |
| node() | 匹配任何类型的节点。 |

通过在路径表达式中使用“|”运算符，您可以选取**若干个路径**。

| 路径表达式                       | 结果                                                         |
| :------------------------------- | :----------------------------------------------------------- |
| //book/title \| //book/price     | 选取 book 元素的所有 title 和 price 元素。                   |
| //title \| //price               | 选取文档中的所有 title 和 price 元素。                       |
| /bookstore/book/title \| //price | 选取属于 bookstore 元素的 book 元素的所有 title 元素，以及文档中所有的 price 元素。 |

**谓语**用来查找某个特定的节点或者包含某个指定的值的节点。

| 路径表达式                         | 结果                                                         |
| :--------------------------------- | :----------------------------------------------------------- |
| /bookstore/book[1]                 | 选取属于 bookstore 子元素的第一个 book 元素。                |
| /bookstore/book[last()]            | 选取属于 bookstore 子元素的最后一个 book 元素。              |
| /bookstore/book[last()-1]          | 选取属于 bookstore 子元素的倒数第二个 book 元素。            |
| /bookstore/book[position()<3]      | 选取最前面的两个属于 bookstore 元素的子元素的 book 元素。    |
| //title[@lang]                     | 选取所有拥有名为 lang 的属性的 title 元素。                  |
| //title[@lang='eng']               | 选取所有 title 元素，且这些元素拥有值为 eng 的 lang 属性。   |
| /bookstore/book[price>35.00]       | 选取 bookstore 元素的所有 book 元素，且其中的 price 元素的值须大于 35.00。 |
| /bookstore/book[price>35.00]/title | 选取 bookstore 元素中的 book 元素的所有 title 元素，且其中的 price 元素的值须大于 35.00。 |



## 11、print倒计时

> **print**(value,sep=' ',end='\n',file=sys.stdout,flush=False)

- **value** : 要打印的字符串

- **sep** : 则是value之间的间隔

- **end** : 是打印完成之后要打印的东西

- **file** ：打印到哪里

- **flush** ：是否开启缓冲区

```python
import time
for x in range(5, -1, -1):
    mystr = "倒计时" + str(x) + "秒"
    print(mystr, end="")
    print("\b" * (len(mystr)*2), end="", flush=True)
    time.sleep(1)
```



## 12、格式化输出

### %用法

#### 1、整数的输出

%o —— oct 八进制
%d —— dec 十进制
%x —— hex 十六进制

```python
1 >>> print('%o' % 20)
2 24
3 >>> print('%d' % 20)
4 20
5 >>> print('%x' % 20)
6 14
```

#### 2、浮点数输出

%f ——保留小数点后面六位有效数字
　　%.3f，保留3位小数位
%e ——保留小数点后面六位有效数字，指数形式输出
　　%.3e，保留3位小数位，使用科学计数法
%g ——在保证六位有效数字的前提下，使用小数方式，否则使用科学计数法
　　%.3g，保留3位有效数字，使用小数或科学计数法

```python
 1 >>> print('%f' % 1.11)  # 默认保留6位小数
 2 1.110000
 3 >>> print('%.1f' % 1.11)  # 取1位小数
 4 1.1
 5 >>> print('%e' % 1.11)  # 默认6位小数，用科学计数法
 6 1.110000e+00
 7 >>> print('%.3e' % 1.11)  # 取3位小数，用科学计数法
 8 1.110e+00
 9 >>> print('%g' % 1111.1111)  # 默认6位有效数字
10 1111.11
11 >>> print('%.7g' % 1111.1111)  # 取7位有效数字
12 1111.111
13 >>> print('%.2g' % 1111.1111)  # 取2位有效数字，自动转换为科学计数法
14 1.1e+03
```

#### 3、字符串输出

%s
%10s——右对齐，占位符10位
%-10s——左对齐，占位符10位
%.2s——截取2位字符串
%10.2s——10位占位符，截取两位字符串

```python
 1 >>> print('%s' % 'hello world')  # 字符串输出
 2 hello world
 3 >>> print('%20s' % 'hello world')  # 右对齐，取20位，不够则补位
 4          hello world
 5 >>> print('%-20s' % 'hello world')  # 左对齐，取20位，不够则补位
 6 hello world         
 7 >>> print('%.2s' % 'hello world')  # 取2位
 8 he
 9 >>> print('%10.2s' % 'hello world')  # 右对齐，取2位
10         he
11 >>> print('%-10.2s' % 'hello world')  # 左对齐，取2位
12 he    
```



### format的用法

```python
salary = 9999.99
print(f'My salary is {salary:10.3f}')
My salary is   9999.990

s = 3
print(f"{s:05d}")
00003

s = 3
print(f"{s:06.3f}")
03.000
```



## 13、.gitignore

常用的规则：

```python
1）/mtk/             过滤整个文件夹
2）*.zip             过滤所有.zip文件
3）/mtk/do.c         过滤某个具体文件
```

gitignore还可以指定要将哪些文件添加到版本管理中:

```python
1）!*.zip
2）!/mtk/one.txt
```

唯一的区别就是规则开头多了一个感叹号



## 14、setuptools

### basic use

```python
from setuptools import setup, find_packages
setup(
    name="HelloWorld",
    version="0.1",
    packages=find_packages(),
)
```



```python
from setuptools import setup, find_packages
setup(
    name="HelloWorld",
    version="0.1",
    packages=find_packages(),
    
    install_requires=["docutils>=0.3"],
    scripts=["say_hello.py"],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # And include any *.msg files found in the "hello" package, too:
        "hello": ["*.msg"],
    },
)
```



### key words

|                               |      |
| ----------------------------- | ---- |
| name                          |      |
| version                       |      |
| description                   |      |
| long_description              |      |
| long_description_content_type |      |
| author                        |      |
| author_email                  |      |
| maintainer                    |      |
| maintainer_email              |      |
| url                           |      |
| download_url                  |      |
| packages                      |      |
| py_modules                    |      |
| scripts                       |      |
| ext_package                   |      |
| ext_modules                   |      |
| classifiers                   |      |
| distclass                     |      |
| script_name                   |      |
| script_args                   |      |
| options                       |      |
| license                       |      |
| keywords                      |      |
| platforms                     |      |
| cmdclass                      |      |
| package_dir                   |      |
| obsoletes                     |      |
| provides                      |      |
| include_package_data          |      |
| exclude_package_data          |      |
| package_data                  |      |
| zip_safe                      |      |
| install_requires              |      |
| entry_points                  |      |
| extras_require                |      |
| python_requires               |      |
| setup_requires                |      |
| namespace_packages            |      |
| test_suite                    |      |
| tests_require                 |      |
| test_loader                   |      |
| eager_resources               |      |
| use_2to3                      |      |
| convert_2to3_doctests         |      |
| use_2to3_fixers               |      |
| use_2to3_exclude_fixers       |      |
| project_urls                  |      |
|                               |      |
|                               |      |



### find_packages

find_packages()接受一个源目录和两个要排除和包含的包名称模式列表。如果省略，源目录默认与设置脚本所在的目录相同。有些项目使用src或lib目录作为源树的根目录，这些项目当然会使用“src”或“lib”作为find_packages()的第一个参数。(这些项目的setup()参数中还需要像package_dir={"": "src"}这样的东西，但这只是一个普通的distutils。)

无论如何，find_packages()遍历目标目录，根据包含模式进行过滤，并找到Python包(任何目录)。包只有在包含剩余的.py文件时才被识别。最后，应用排除模式删除匹配包。

包含和排除模式是包名，可以选择包含通配符。例如，find_packages(exclude=["*.tests"])将排除姓氏部分是tests的所有包。或者,find_packages(排除= [" *。test "， "*.tests.*"])也会排除名为tests的包的任何子包，但仍然不会排除顶级测试包或其子包。事实上，如果你真的不想要任何测试包，你需要这样的东西:

```python
find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
```

不管参数是什么，find_packages()函数都会返回一个适合用作setup()的包参数的包名列表，因此这通常是在设置脚本中设置该参数的最简单方法。特别是因为它使您不必在项目增长额外的顶级包或子包时记得修改设置脚本。



### find_namespace_packages

```python
├── namespace
│   └── mypackage
│       ├── __init__.py
│       └── mod1.py
├── setup.py
└── tests
    └── test_mod1.py
```

```python
from setuptools import setup, find_namespace_packages

setup(
    name="namespace.mypackage",
    version="0.1",
    packages=find_namespace_packages(include=["namespace.*"])
)
```

```python
├── setup.py
├── src
│   └── namespace
│       └── mypackage
│           ├── __init__.py
│           └── mod1.py
└── tests
    └── test_mod1.py
```

```python
setup(name="namespace.mypackage",
      version="0.1",
      package_dir={"": "src"},
      packages=find_namespace_packages(where="src"))
```





### Automatic Script Creation

在使用distutils时，打包和安装脚本可能会有些笨拙。首先，在Windows和POSIX平台上，没有一种简单的方法可以使脚本的文件名匹配本地约定。另一方面，当您实际的“main”是某个模块中的函数时，您常常必须为“main”脚本创建一个单独的文件。甚至在Python 2.4中，使用-m选项也仅适用于没有安装在包中的实际.py文件。

setuptools通过使用正确的扩展为您自动生成脚本来修复所有这些问题，在Windows上它甚至会创建一个.exe文件，这样用户就不必更改他们的PATHEXT设置。使用此特性的方法是在安装脚本中定义“入口点”，指示生成的脚本应该导入和运行什么函数。例如，要创建两个控制台脚本foo和bar，以及一个GUI脚本baz，你可以这样做:

```python
setup(
    # other arguments here...
    entry_points={
        "console_scripts": [
            "foo = my_package.some_module:main_func",
            "bar = other_module:some_func",
        ],
        "gui_scripts": [
            "baz = my_package_gui:start_func",
        ]
    }
)
```

当这个项目安装在非windows平台上(使用“setup.py install”、“setup.py develop”或pip)时，将安装一组foo、bar和baz脚本，从指定的模块导入main_func和some_func。调用您指定的函数时不带参数，它们的返回值被传递给sys.exit()，因此您可以返回一个errorlevel或消息来打印到stderr。

在Windows上，将创建一组foo.exe，bar.exe和baz.exe启动器，以及一组foo.py，bar.py和baz.pyw文件。 .exe包装器找到并执行正确版本的Python，以运行.py或.pyw文件。



### Declaring Dependencies

包含需求说明符的最简单方法是对setup()使用`install_requires`参数。它接受一个字符串或包含需求说明符的字符串列表。



如果您的项目依赖于PyPI上不存在的软件包，则只要它们可以通过以下方式下载，您仍然可以依赖它们：

- an egg, in the standard distutils `sdist` format,
- a single `.py` file, or
- a VCS repository (Subversion, Mercurial, or Git).

您只需要向setup()的`dependency_links`参数添加一些url

1. direct download URLs,
2. the URLs of web pages that contain direct download links, or
3. the repository’s URL

通常，链接到web页面更好，因为更新web页面通常比发布项目的新版本更简单。如果你所依赖的包是通过SourceForge发布的，你也可以使用SourceForge的showfiles.php链接。



有时，项目具有“推荐”依赖关系，而对于项目的所有使用而言，并不是必需的。 例如，如果安装了ReportLab，则一个项目可能提供可选的PDF输出，如果安装了docutils，则一个项目可能提供reStructuredText支持。 这些可选功能称为“ extras”，setuptools也允许您定义它们的要求。 这样，需要这些可选功能的其他项目可以通过在install_requires中命名所需的附加项来强制安装附加要求。

例如，假设项目A提供了可选的PDF和reST支持：

```python
setup(
    name="Project-A",
    ...
    extras_require={
        "PDF":  ["ReportLab>=1.2", "RXP"],
        "reST": ["docutils>=0.3"],
    }
)
```



### Including Data Files

Setuptools提供了三种方法来指定要包含在软件包中的数据文件。

首先，您可以简单地使用`include_package_data`

```python
from setuptools import setup, find_packages
setup(
    ...
    include_package_data=True
)
```

这告诉setuptools安装它在包中找到的任何数据文件。数据文件必须通过distutils的`MANIFEST.in`来指定。

|                 Command                 |                         Description                          |
| :-------------------------------------: | :----------------------------------------------------------: |
|        **include pat1 pat2 ...**        |    include all files matching any of the listed patterns     |
|        **exclude pat1 pat2 ...**        |    exclude all files matching any of the listed patterns     |
| **recursive-include dir pat1 pat2 ...** | include all files under *dir* matching any of the listed patterns |
| **recursive-exclude dir pat1 pat2 ...** | exclude all files under *dir* matching any of the listed patterns |
|    **global-include pat1 pat2 ...**     | include all files anywhere in the source tree matching — & any of the listed patterns |
|    **global-exclude pat1 pat2 ...**     | exclude all files anywhere in the source tree matching — & any of the listed patterns |
|              **prune dir**              |                exclude all files under *dir*                 |
|              **graft dir**              |                include all files under *dir*                 |



如果您希望对包含的文件进行更细粒度的控制（例如，如果您的软件包目录中有文档文件，并希望将其排除在安装范围之外），则也可以使用`package_data`关键字，例如：

```python
from setuptools import setup, find_packages
setup(
    ...
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # And include any *.msg files found in the "hello" package, too:
        "hello": ["*.msg"],
    }
)
```

package_data参数是一个从包名映射到全局遍历模式列表的字典。如果数据文件包含在包的子目录中，则globs可以包含子目录名。例如，如果包树是这样的:

```python
setup.py
src/
    mypkg/
        __init__.py
        mypkg.txt
        data/
            somefile.dat
            otherdata.dat
```

```python
from setuptools import setup, find_packages
setup(
    ...
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},   # tell distutils packages are under src

    package_data={
        # If any package contains *.txt files, include them:
        "": ["*.txt"],
        # And include any *.dat files found in the "data" subdirectory
        # of the "mypkg" package, also:
        "mypkg": ["data/*.dat"],
    }
)
```

如果数据文件包含在不是包本身的包的子目录中（没有\_\_init__.py），则package_data参数中需要子目录名称（或*）（如上所示，带有“ data / *.dat” ）。



有时候，单独的include_package_data或package_data选项不足以精确地定义您想要包含的文件。例如，您可能希望在您的版本控制系统和源发行版中包含软件包自述文件，但不希望安装它们。所以，setuptools也提供了`exclude_package_data`选项，它允许你做这样的事情:

```python
from setuptools import setup, find_packages
setup(
    ...
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},   # tell distutils packages are under src

    include_package_data=True,    # include everything in source control

    # ...but exclude README.txt from all packages
    exclude_package_data={"": ["README.txt"]},
)
```

总之，这三个选项使您可以：

**include_package_data**接受MANIFEST.in匹配的所有数据文件和目录。 

**package_data**指定其他模式以匹配MANIFEST.in可能匹配或可能不匹配或在源代码控制中找到的文件。 

**exclude_package_data**指定安装软件包时不应包含的数据文件和目录的模式，即使由于使用上述选项而原本会包含它们也是如此。













## 15、subprocess

推荐的调用子进程的方式是在任何它支持的用例中使用 [`run()`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.run) 函数。对于更进阶的用例，也可以使用底层的 [`Popen`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen) 接口。

> subprocess.**run**(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None, **other_popen_kwargs)

运行被 *arg* 描述的指令. 等待指令完成, 然后返回一个 [`CompletedProcess`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.CompletedProcess) 实例.

```python
subprocess.run(["ls", "-l", "/dev/null"])
```



**capture_output** 如果设为 true，stdout 和 stderr 将会被捕获。在使用时，内置的 [`Popen`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen) 对象将自动用 `stdout=PIPE` 和 `stderr=PIPE` 创建。*stdout* 和 *stderr* 参数不应当与 *capture_output* 同时提供。如果你希望捕获并将两个流合并在一起，使用 `stdout=PIPE` 和 `stderr=STDOUT` 来代替 *capture_output*。

***timeout*** 参数将被传递给 [`Popen.communicate()`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen.communicate)。如果发生超时，子进程将被杀死并等待。 [`TimeoutExpired`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.TimeoutExpired) 异常将在子进程中断后被抛出。

***input*** 参数将被传递给 [`Popen.communicate()`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen.communicate) 以及子进程的 stdin。 如果使用此参数，它必须是一个字节序列。 如果指定了 *encoding* 或 *errors* 或者将 *text* 设置为 `True`，那么也可以是一个字符串。 当使用此参数时，在创建内部 [`Popen`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen) 对象时将自动带上 `stdin=PIPE`，并且不能再手动指定 *stdin* 参数。

如果 ***check*** 设为 True, 并且进程以非零状态码退出, 一个 [`CalledProcessError`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.CalledProcessError) 异常将被抛出. 这个异常的属性将设置为参数, 退出码, 以及标准输出和标准错误, 如果被捕获到.

如果 ***encoding*** 或者 ***error*** 被指定, 或者 ***text*** 被设为 True, 标准输入, 标准输出和标准错误的文件对象将通过指定的 *encoding* 和 *errors* 以文本模式打开, 否则以默认的 [`io.TextIOWrapper`](https://docs.python.org/zh-cn/3/library/io.html#io.TextIOWrapper) 打开. ***universal_newline*** 参数等同于 *text* 并且提供了向后兼容性. 默认情况下, 文件对象是以二进制模式打开的.

如果 ***env*** 不是 `None`, 它必须是一个字典, 为新的进程设置环境变量; 它用于替换继承的当前进程的环境的默认行为. 它将直接被传递给 [`Popen`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen).

如果 ***shell*** 设为 `True`,，则使用 shell 执行指定的指令。如果您主要使用 Python 增强的控制流（它比大多数系统 shell 提供的强大），并且仍然希望方便地使用其他 shell 功能，如 shell 管道、文件通配符、环境变量展开以及 `~` 展开到用户家目录，这将非常有用。





此模块的底层的进程创建与管理由 [`Popen`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen) 类处理。它提供了很大的灵活性，因此开发者能够处理未被便利函数覆盖的不常见用例。

> class subprocess.**Popen**(args, bufsize=-1, executable=None, stdin=None, stdout=None, stderr=None, preexec_fn=None, close_fds=True, shell=False, cwd=None, env=None, universal_newlines=None, startupinfo=None, creationflags=0, restore_signals=True, start_new_session=False, pass_fds=(), *, encoding=None, errors=None, text=None)

在一个新的进程中执行子程序。在 POSIX，此类使用类似于 [`os.execvp()`](https://docs.python.org/zh-cn/3/library/os.html#os.execvp) 的行为来执行子程序。在 Windows，此类使用了 Windows `CreateProcess()` 函数。



## 16、vscode设置src文件夹

**settings.json**

```json
{
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${workspaceFolder}/src",
    },
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}/src",
    },
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}/src",
    },
    "python.envFile": "${workspaceFolder}/.env"
}
```

必须为编辑器的Python环境和集成终端都设置`PYTHONPATH`。 扩展程序使用了编辑器的Python环境，并提供了整理和测试功能。 调试时使用集成终端以激活新的python环境。

以上配置将覆盖现有的PYTHONPATH。要扩展，请使用以下设置：

```json
# Use path separator ';' on Windows.
{
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/src",
    },
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}/src",
    },
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}/src",
    }
}
```



## 17、concurrent

`concurrent.futures`是一个非常简单易用的库，主要用来实现多线程和多进程的异步并发

 python 3.x中自带了concurrent.futures模块



### Executor对象

`class concurrent.futures.Executor`

Executor是一个抽象类，它提供了异步执行调用的方法。它不能直接使用，但可以通过它的两个子类`ThreadPoolExecutor`或者`ProcessPoolExecutor`进行调用



#### Executor.submit(fn, \*args, \**kwargs)

`fn`：需要异步执行的函数

```python
from concurrent import futures
 
def test(num):
    import time
    return time.ctime(),num

with futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(test,1)
    print future.result()
```



#### Executor.map(func, \*iterables, timeout=None)

相当于`map(func, *iterables)`，但是func是异步执行。timeout的值可以是int或float，如果操作超时，会返回`raisesTimeoutError`；如果不指定timeout参数，则不设置超时间。

`func`：需要异步执行的函数

`*iterables`：可迭代对象，如列表等。每一次func执行，都会从iterables中取参数。

```python
from concurrent import futures
 
def test(num):
    import time
    return time.ctime(),num
 
data=[1,2,3]
with futures.ThreadPoolExecutor(max_workers=1) as executor:
    for future in executor.map(test,data):
        print future
```



#### Executor.shutdown(wait=True)

释放系统资源,在Executor.submit()或 Executor.map()等异步操作后调用。**使用with语句可以避免显式调用此方法**。



### ThreadPoolExecutor对象

ThreadPoolExecutor类是Executor子类，使用线程池执行异步调用.**使用max_workers数目的线程池执行异步调用**



### ProcessPoolExecutor对象

使用max_workers数目的进程池执行异步调用，如果max_workers为None则使用机器的处理器数目

```python
from concurrent import futures
 
def test(num):
    import time
    return time.ctime(),num
 
def muti_exec(m,n):
    #m 并发次数
    #n 运行次数
 
    with futures.ProcessPoolExecutor(max_workers=m) as executor: #多进程
    #with futures.ThreadPoolExecutor(max_workers=m) as executor: #多线程
        executor_dict=dict((executor.submit(test,times), times) for times in range(m*n))
 
    for future in futures.as_completed(executor_dict):
        times = executor_dict[future]
        if future.exception() is not None:
            print('%r generated an exception: %s' % (times,future.exception()))
        else:
            print('RunTimes:%d,Res:%s'% (times, future.result()))
 
if __name__ == '__main__':
    muti_exec(5,1)
```



















