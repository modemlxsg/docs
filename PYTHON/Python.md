# Python

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

















