## 一、简介

**Docker**是一个开源的应用容器引擎；是一个轻量级容器技术；

Docker支持将软件编译成一个镜像；然后在镜像中各种软件做好配置，将镜像发布出去，其他使用者可以直接使用这个镜像；

运行中的这个镜像称为容器，容器启动是非常快速的。

**核心概念**

docker主机(Host)：安装了Docker程序的机器（Docker直接安装在操作系统之上）；

docker客户端(Client)：连接docker主机进行操作；

docker仓库(Registry)：用来保存各种打包好的软件镜像；

docker镜像(Images)：软件打包好的镜像；放在docker仓库中；

docker容器(Container)：镜像启动后的实例称为一个容器；容器是独立运行的一个或一组应用

**常用命令**

| 操作 | 命令                                            | 说明                                                     |
| ---- | ----------------------------------------------- | -------------------------------------------------------- |
| 检索 | docker  search 关键字  eg：docker  search redis | 我们经常去docker  hub上检索镜像的详细信息，如镜像的TAG。 |
| 拉取 | docker pull 镜像名:tag                          | :tag是可选的，tag表示标签，多为软件的版本，默认是latest  |
| 列表 | docker images                                   | 查看所有本地镜像                                         |
| 删除 | docker rmi image-id                             | 删除指定的本地镜像                                       |



镜像仓库：https://hub.docker.com/



## 二、安装Docker-ce

如果你过去安装过 docker，先删掉:

```bash
sudo apt-get remove docker docker-engine docker.io
```

首先安装依赖:

```bash
sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common
```

信任 Docker 的 GPG 公钥:

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

对于 amd64 架构的计算机，添加软件仓库:

```bash
sudo add-apt-repository \
   "deb [arch=amd64] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

最后安装

```bash
sudo apt-get update
sudo apt-get install docker-ce
```

启动docker

```bash
systemctl start docker
docker -v
```

开机启动docker

```
systemctl enable docker
```

停止docker

```
systemctl stop docker
```







## 三、命令大全

### 容器生命周期管理

#### run

```bash
docker run ：创建一个新的容器并运行一个命令
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```

- **a stdin:** 指定标准输入输出内容类型，可选 STDIN/STDOUT/STDERR 三项；
- **-d:** 后台运行容器，并返回容器ID；
- **-i:** 以交互模式运行容器，通常与 -t 同时使用；
- **-p:** 端口映射，格式为：主机(宿主)端口:容器端口
- **-t:** 为容器重新分配一个伪输入终端，通常与 -i 同时使用；
- **--name="nginx-lb":** 为容器指定一个名称；
- **--dns 8.8.8.8:** 指定容器使用的DNS服务器，默认和宿主一致；
- **--dns-search example.com:** 指定容器DNS搜索域名，默认和宿主一致；
- **-h "mars":** 指定容器的hostname；
- **-e username="ritchie":** 设置环境变量；
- **--env-file=[]:** 从指定文件读入环境变量；
- **--cpuset="0-2" or --cpuset="0,1,2":** 绑定容器到指定CPU运行；
- **-m :**设置容器使用内存最大值；
- **--net="bridge":** 指定容器的网络连接类型，支持 bridge/host/none/container: 四种类型；
- **--link=[]:** 添加链接到另一个容器；
- **--expose=[]:** 开放一个端口或一组端口；



####　start/stop/restart

```
docker start :启动一个或多个已经被停止的容器
docker stop :停止一个运行中的容器
docker restart :重启容器
docker start [OPTIONS] CONTAINER [CONTAINER...]
```



#### kill

```
docker kill :杀掉一个运行中的容器。
docker kill [OPTIONS] CONTAINER [CONTAINER...]
```

**-s :**向容器发送一个信号



#### rm

```
docker rm ：删除一个或多少容器
```

- **-f :**通过SIGKILL信号强制删除一个运行中的容器
- **-l :**移除容器间的网络连接，而非容器本身
- **-v :**-v 删除与容器关联的卷



#### pause/unpause

```
docker pause :暂停容器中所有的进程。
docker unpause :恢复容器中所有的进程。
```



#### create

```
docker create ：创建一个新的容器但不启动它
```

```
runoob@runoob:~$ docker create  --name myrunoob  nginx:latest  
```



#### exec

```
docker exec ：在运行的容器中执行命令
```

- **-d :**分离模式: 在后台运行
- **-i :**即使没有附加也保持STDIN 打开
- **-t :**分配一个伪终端

在容器mynginx中开启一个交互模式的终端

````
runoob@runoob:~$ docker exec -it  mynginx /bin/bash
root@b1a0703e41e7:/#
````



### 容器操作

#### ps

```
docker ps : 列出容器
```

- **-a :**显示所有的容器，包括未运行的。
- **-f :**根据条件过滤显示的内容。
- **--format :**指定返回值的模板文件。
- **-l :**显示最近创建的容器。
- **-n :**列出最近创建的n个容器。
- **--no-trunc :**不截断输出。
- **-q :**静默模式，只显示容器编号。
- **-s :**显示总的文件大小。



#### inspect

```
docker inspect : 获取容器/镜像的元数据
```

- **-f :**指定返回值的模板文件。
- **-s :**显示总的文件大小。
- **--type :**为指定类型返回JSON。



#### top

```
docker top :查看容器中运行的进程信息，支持 ps 命令参数。
```



#### attach

```
docker attach :连接到正在运行中的容器
```



#### events

```
docker events : 从服务器获取实时事件
```

- **-f ：**根据条件过滤事件；
- **--since ：**从指定的时间戳后显示所有事件;
- **--until ：**流水时间显示到指定的时间为止；



#### logs

```
docker logs : 获取容器的日志
```

- **-f :** 跟踪日志输出
- **--since :**显示某个开始时间的所有日志
- **-t :** 显示时间戳
- **--tail :**仅列出最新N条容器日志



#### wait

```
docker wait : 阻塞运行直到容器停止，然后打印出它的退出代码。
```



####  export

```
docker export :将文件系统作为一个tar归档文件导出到STDOUT
```



#### port 

```
docker port :列出指定的容器的端口映射，或者查找将PRIVATE_PORT NAT到面向公众的端口
```



#### commit 

```
docker commit :从容器创建一个新的镜像
```

- **-a :**提交的镜像作者；
- **-c :**使用Dockerfile指令来创建镜像；
- **-m :**提交时的说明文字；
- **-p :**在commit时，将容器暂停



### 镜像仓库

#### login/logout

```
docker login : 登陆到一个Docker镜像仓库，如果未指定镜像仓库地址，默认为官方仓库 Docker Hub
docker logout : 登出一个Docker镜像仓库，如果未指定镜像仓库地址，默认为官方仓库 Docker Hub
```

- **-u :**登陆的用户名
- **-p :**登陆的密码



#### pull 

````
docker pull : 从镜像仓库中拉取或者更新指定镜像
````

- **-a :**拉取所有 tagged 镜像
- **--disable-content-trust :**忽略镜像的校验,默认开启



#### push 

```
docker push : 将本地的镜像上传到镜像仓库,要先登陆到镜像仓库
```



#### search

```
docker search : 从Docker Hub查找镜像
```



### 本地镜像管理

#### images 

```
docker images : 列出本地镜像。
```

- **-a :**列出本地所有的镜像（含中间映像层，默认情况下，过滤掉中间映像层）；
- **--digests :**显示镜像的摘要信息
- **-f :**显示满足条件的镜像；
- **--format :**指定返回值的模板文件；
- **--no-trunc :**显示完整的镜像信息；
- **-q :**只显示镜像ID。



#### rmi 

```
docker rmi : 删除本地一个或多少镜像。
```

- **-f :**强制删除；
- **--no-prune :**不移除该镜像的过程镜像，默认移除；



#### tag 

```
docker tag : 标记本地镜像，将其归入某一仓库。
```



#### build

```
docker build 命令用于使用 Dockerfile 创建镜像。
```

- **--build-arg=[] :**设置镜像创建时的变量；
- **--cpu-shares :**设置 cpu 使用权重；
- **--cpu-period :**限制 CPU CFS周期；
- **--cpu-quota :**限制 CPU CFS配额；
- **--cpuset-cpus :**指定使用的CPU id；
- **--cpuset-mems :**指定使用的内存 id；
- **--disable-content-trust :**忽略校验，默认开启；
- **-f :**指定要使用的Dockerfile路径；
- **--force-rm :**设置镜像过程中删除中间容器；
- **--isolation :**使用容器隔离技术；
- **--label=[] :**设置镜像使用的元数据；
- **-m :**设置内存最大值；
- **--memory-swap :**设置Swap的最大值为内存+swap，"-1"表示不限swap；
- **--no-cache :**创建镜像的过程不使用缓存；
- **--pull :**尝试去更新镜像的新版本；
- **--quiet, -q :**安静模式，成功后只输出镜像 ID；
- **--rm :**设置镜像成功后删除中间容器；
- **--shm-size :**设置/dev/shm的大小，默认值是64M；
- **--ulimit :**Ulimit配置。
- **--tag, -t:** 镜像的名字及标签，通常 name:tag 或者 name 格式；可以在一次构建中为一个镜像设置多个标签。
- **--network:** 默认 default。在构建期间设置RUN指令的网络模式



#### history

```
docker history : 查看指定镜像的创建历史。
```

- **-H :**以可读的格式打印镜像大小和日期，默认为true；
- **--no-trunc :**显示完整的提交记录；
- **-q :**仅列出提交记录ID。



#### save

```
docker save : 将指定镜像保存成 tar 归档文件
```

**-o :**输出到的文件。



#### import

```
docker import : 从归档文件中创建镜像
```

- **-c :**应用docker 指令创建镜像
- **-m :**提交时的说明文字；



### docker信息

#### info

```
docker info : 显示 Docker 系统信息，包括镜像和容器数
```



#### version

```
docker version :显示 Docker 版本信息。
```



