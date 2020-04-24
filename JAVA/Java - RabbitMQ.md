# RabbitMQ

## 安装

### windows

先安装**Erlang**和对应版本**rabbitmq**，全部默认安装即可。

> **安装路径不能包含中文和空格**

进入**C:\RabbitMQServer\rabbitmq_server-3.7.10\sbin**安装目录，输入 **rabbitmq-plugins enable rabbitmq_management** 启动管理插件。

运行 **rabbitmq-server.bat**，进入**http://localhost:15672/#/**控制台，默认用户**guest**、**guest**。

`以上启动方式，关闭cmd窗口服务即关闭。`



rabbitmq启动方式有2种

1、以应用方式启动

**rabbitmq-server -detached** 后台启动

**Rabbitmq-server** 直接启动，如果你关闭窗口或者需要在改窗口使用其他命令时应用就会停止

 关闭:rabbitmqctl stop

2、以服务方式启动（安装完之后在任务管理器中服务一栏能看到RabbtiMq）

**rabbitmq-service install** 安装服务

**rabbitmq-service start** 开始服务

**Rabbitmq-service stop**  停止服务

**Rabbitmq-service enable** 使服务有效

**Rabbitmq-service disable** 使服务无效

**rabbitmq-service help** 帮助

当rabbitmq-service install之后默认服务是enable的，如果这时设置服务为disable的话，rabbitmq-service start就会报错。

当rabbitmq-service start正常启动服务之后，使用disable是没有效果的

关闭:**rabbitmqctl stop**

3、Rabbitmq 管理插件启动，可视化界面

**rabbitmq-plugins enable rabbitmq_management** 启动

**rabbitmq-plugins disable rabbitmq_management** 关闭

 

4、Rabbitmq节点管理方式

**Rabbitmqctl**



**1：无法启动**

CMD返回错误:---乱码

解决方法:

该问题一般是由于系统环境变量没有配置好引起的.

检查以下两个环境变量配置:

- ERLANG_HOME 
- RABBITMQ_BASE

重新执行 安装命令:

rabbitmq-service.bat remove

rabbitmq-service.bat install

rabbitmq-service.bat start.

ok,成功.



# Spring AMQP

## 快速入门

```java
import org.springframework.amqp.core.AmqpAdmin;
import org.springframework.amqp.core.AmqpTemplate;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.rabbit.connection.CachingConnectionFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitAdmin;
import org.springframework.amqp.rabbit.core.RabbitTemplate;

ConnectionFactory connectionFactory = new CachingConnectionFactory();
AmqpAdmin admin = new RabbitAdmin(connectionFactory);
admin.declareQueue(new Queue("myqueue"));
AmqpTemplate template = new RabbitTemplate(connectionFactory);
template.convertAndSend("myqueue", "foo");
String foo = (String) template.receiveAndConvert("myqueue");
```

**SpringBoot**

```java
@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

    @Bean
    public ApplicationRunner runner(AmqpTemplate template) {
        return args -> template.convertAndSend("myqueue", "foo");
    }

    @Bean
    public Queue myQueue() {
        return new Queue("myqueue");
    }

    @RabbitListener(queues = "myqueue")
    public void listen(String in) {
        System.out.println(in);
    }

}
```



## AMQP抽象

Spring AMQP由几个模块组成，每个模块在发行版中由一个JAR表示。这些模块是:spring-amqp和spring-rabbit。spring-amqp模块包含**org.springframe.amqp**核心包。

在这个包中，您将找到表示核心AMQP“模型”的类。我们的目的是提供不依赖于任何特定AMQP代理实现或客户端库的通用抽象。最终用户代码将更易于跨供应商实现移植，因为它只能针对抽象层进行开发。然后由特定于代理的模块(如spring-rabbit)实现这些抽象。目前只有一个RabbitMQ实现;然而，这些抽象已经在. net中使用Apache Qpid和RabbitMQ进行了验证。由于AMQP原则上是在协议级别上操作的，所以RabbitMQ客户机可以与支持相同协议版本的任何代理一起使用，但是我们目前不测试任何其他代理。

### Message

```java
public class Message {

    private final MessageProperties messageProperties;

    private final byte[] body;

    public Message(byte[] body, MessageProperties messageProperties) {
        this.body = body;
        this.messageProperties = messageProperties;
    }

    public byte[] getBody() {
        return this.body;
    }

    public MessageProperties getMessageProperties() {
        return this.messageProperties;
    }
}
```

**MessageProperties**接口定义了几个常见属性，如messageId、时间戳、内容类型等。通过调用**setHeader(String key, Object value)**方法，还可以使用用户定义的头文件扩展这些属性

### Exchange

```java
public interface Exchange {

    String getName();

    String getExchangeType();

    boolean isDurable();

    boolean isAutoDelete();

    Map<String, Object> getArguments();

}
```

**Exchange**还具有一个类型，该类型由**ExchangeTypes**中定义的常量表示。基本类型有:**Direct**、**Topic**、**Fanout**和**Headers**。

### Queue

```java
public class Queue  {

    private final String name;

    private volatile boolean durable;

    private volatile boolean exclusive;

    private volatile boolean autoDelete;

    private volatile Map<String, Object> arguments;

    /**
     * The queue is durable, non-exclusive and non auto-delete.
     *
     * @param name the name of the queue.
     */
    public Queue(String name) {
        this(name, true, false, false);
    }

    // Getters and Setters omitted for brevity

}
```

自动生成队列的**exclusive**和**autoDelete**属性都将设置为**true**

### Binding

考虑到生产者向交换器发送消息，而消费者从队列接收消息，将队列连接到交换器的绑定对于通过消息传递连接这些生产者和消费者至关重要。在Spring AMQP中，我们定义了一个绑定类来表示这些连接。让我们回顾一下将队列绑定到exchange的基本选项

```java
new Binding(someQueue, someDirectExchange, "foo.bar")
new Binding(someQueue, someTopicExchange, "foo.*")
new Binding(someQueue, someFanoutExchange)
Binding b = BindingBuilder.bind(someQueue).to(someTopicExchange).with("foo.*");
```

可以在**@Configuration**类中使用Spring的**@bean**样式定义绑定实例

## 连接和资源管理

尽管我们在上一节中描述的AMQP模型是通用的，并且适用于所有实现，但是当我们进入资源管理时，细节是特定于代理实现的。因此，在本节中，我们将重点介绍只存在于“spring-rabbit”模块中的代码，因为此时RabbitMQ是惟一受支持的实现。



## AmqpTemplate

## Sending messages

## Receiving messages

## Containers and Broker-Named queues

## Message Converters

## Modifying Messages - Compression and More

## Request/Reply Messaging

## Configuring the broker

## Broker Event Listener

## Delayed Message Exchange

## RabbitMQ REST API

## Exception Handling

## Transactions

## Message Listener Container Configuration

## Listener Concurrency

