## GCP环境搭建


```shell
# python 安装
sudo apt-get install python3

# pip安装
sudo apt-get install software-properties-common
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get install python3-pip

# virtualenv安装
sudo pip3 install -U virtualenv

#创建python环境
mkdir ./envs
cd envs
virtualenv --system-site-packages -p python3 env_tf2
source ./env_tf2/bin/activate #激活环境

# 安装TensorFlow
pip install tensorflow

# 安装其他库
pip install numpy pandas matplotlib sklearn jupyterlab

## 安装npm
sudo apt-get install npm

```

配置VPC IP地址 --改静态ip

配置防火墙规则

```shell
# jupyter 配置
jupyter lab --generate-config

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 6006
c.NotebookApp.allow_remote_access = True
```



## GPU配置

https://tensorflow.google.cn/install/gpu

```powershell
mkdir cuda_install
cd cuda_install/
vim install.sh
# 添加安装脚本
sh +x install.sh #运行脚本,最好一条条运行
# 重启 
nvidia-smi #查看驱动是否安装成功

# 测试GPU是否可用
tf.test.is_gpu_available()

#解决依赖问题
#aptitude 与 apt-get 一样，是 Debian 及其衍生系统中功能极其强大的包管理工具。与 apt-get 不同的是，aptitude #在处理依赖问题上更佳一些。
sudo apt-get install aptitude
sudo aptitude install `缺少的包`
```





## MobaXterm 连接

**SSH秘钥对生成**

tools -> ssh key generate (RSA 2048 移动鼠标加速生成) -> 修改key comment为GCP用户 -> save public and private key

![image-20200122151329080](images/GCP安装.assets/image-20200122151329080.png)

**GCP 元数据中添加公钥**

![image-20200122151515705](images/GCP安装.assets/image-20200122151515705.png)

**moba连接使用私钥**

![image-20200122151638176](images/GCP安装.assets/image-20200122151638176.png)



![image-20200122151734811](images/GCP安装.assets/image-20200122151734811.png)

