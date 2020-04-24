# 官方教程

## 安装

`pip install Django`



## 创建项目

`django-admin startproject mysite`

```
mysite/
    manage.py
    mysite/
        __init__.py
        settings.py
        urls.py
        asgi.py
        wsgi.py
```

- `manage.py`: 一个命令行实用程序，可让您以各种方式与此Django项目进行交互。

- `mysite/settings.py`: 此Django项目的设置/配置。 Django设置将告诉您所有设置的工作方式。

- `mysite/urls.py`: 该Django项目的URL声明； Django支持的网站的“目录”。
- `mysite/asgi.py`: 兼容ASGI的Web服务器为您的项目提供服务的入口点
- `mysite/wsgi.py`: 兼容WSGI的Web服务器为您的项目提供服务的入口点



## 开发服务器

`python manage.py runserver`



**Changing the port** ： `python manage.py runserver 8080`



