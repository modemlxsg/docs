## Websockt



Spring 底层Api

Spring Boot为嵌入式Tomcat、Jetty和Undertow提供WebSockets自动配置。如果将war文件部署到独立容器中，Spring Boot假定容器负责其WebSocket支持的配置。
```java
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry webSocketHandlerRegistry) {
        webSocketHandlerRegistry.addHandler(wsHandler(),"/websockt");
    }

    @Bean
    public WSHandler wsHandler(){
        return new WSHandler();
    }
}
```


```java
public class WSHandler extends AbstractWebSocketHandler {

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        System.out.println("建立连接:" + session.getId());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        System.out.println("关闭连接" + session.getId());
    }

    @Override
    public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) throws Exception {
        System.out.println(message.getPayloadLength());
    }
}
```

## STOMP

### 概述
WebSocket协议定义了两种类型的消息(文本和二进制)，但是它们的内容没有定义。该协议为客户机和服务器定义了一种机制，用于协商在WebSocket上使用的子协议(即更高级别的消息传递协议)，以定义每个消息可以发送什么类型的消息、格式是什么、每个消息的内容，等等。子协议的使用是可选的，但是无论如何，客户机和服务器都需要就定义消息内容的某些协议达成一致。

STOMP(简单的面向文本的消息传递协议)最初是为脚本语言(如Ruby、Python和Perl)创建的，用于连接到企业消息代理。它旨在处理常用消息传递模式的最小子集。STOMP可以用于任何可靠的双向流网络协议，如TCP和WebSocket。尽管STOMP是一个面向文本的协议，但是消息有效负载可以是文本或二进制的。

STOMP是一种基于帧的协议，其帧是基于HTTP建模的。下面的清单显示了STOMP帧的结构:
```
COMMAND
header1:value1
header2:value2

Body^@
```
客户端可以使用**SEND**或**SUBSCRIBE**命令来发送或订阅消息，以及一个目标标头，该标头描述消息的内容和接收消息的对象。这支持一个简单的发布-订阅机制，您可以使用该机制通过代理向其他连接的客户机发送消息，或者向服务器发送消息以请求执行某些工作。

在使用Spring的STOMP支持时，Spring WebSocket应用程序充当客户机的STOMP代理。消息被路由到@Controller消息处理方法或简单的内存代理，该代理跟踪订阅并向订阅的用户广播消息。您还可以将Spring配置为使用专用的STOMP代理(如RabbitMQ、ActiveMQ等)来实际广播消息。在这种情况下，Spring维护到代理的TCP连接，将消息转发给代理，并将消息从代理向下传递到已连接的WebSocket客户机。因此，Spring web应用程序可以依赖统一的基于http的安全性、通用验证和熟悉的消息处理编程模型。

下面的示例显示了一个订阅股票报价的客户端，服务器可以定期发出股票报价(例如，通过一个计划好的任务，该任务通过SimpMessagingTemplate向代理发送消息):
```
SUBSCRIBE
id:sub-1
destination:/topic/price.stock.*

^@
```
下面的例子展示了一个发送交易请求的客户端，服务器可以通过@MessageMapping方法来处理:
```
SEND
destination:/queue/trade
content-type:application/json
content-length:44

{"action":"BUY","ticker":"MMM","shares",44}^@
```
执行之后，服务器可以向客户机广播交易确认消息和详细信息。

目标的含义在STOMP规范中故意保持不透明，它可以是任何字符串，完全由STOMP服务器来定义它们支持的目标的语义和语法。然而，目的地通常是/topic/..的类路径字符串。意味着发布-订阅(一对多)和/queue/意味着点对点(一对一)消息交换。

STOMP服务器可以使用MESSAGE命令向所有订阅者广播消息。下面的示例显示服务器向订阅的客户端发送股票报价:
```
MESSAGE
message-id:nxahklf6-1
subscription:sub-1
destination:/topic/price.stock.MMM

{"ticker":"MMM","price":129.45}^@
```
### 优势
服务器不能发送未经请求的消息。来自服务器的所有消息都必须响应特定的客户机订阅，并且服务器消息的订阅id头必须与客户机订阅的id头匹配。

与使用原始WebSockets相比，使用STOMP作为子协议可以让Spring框架和Spring Security提供更丰富的编程模型。关于HTTP与原始TCP以及它如何让Spring MVC和其他web框架提供丰富的功能，也可以提出同样的观点。以下是一些好处:
- 不需要发明自定义消息传递协议和消息格式。
- 可以使用STOMP客户机，包括Spring框架中的Java客户机。
- 您可以(可选地)使用消息代理(如RabbitMQ、ActiveMQ等)来管理订阅和广播消息。
- 应用程序逻辑可以组织在任意数量的@Controller实例中，并且可以根据STOMP目标头将消息路由到它们，而不是针对给定连接使用单个WebSocketHandler处理原始WebSocket消息。
- 您可以使用Spring Security基于STOMP目的地和消息类型来保护消息。

### 使用STOMP

**spring-messaging**和**spring-websocket**模块中提供了对WebSocket的STOMP支持。一旦你有了这些依赖，你可以公开一个STOMP端点，通过WebSocket与SockJS回退，如下面的例子所示:

```
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/portfolio").withSockJS();  
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.setApplicationDestinationPrefixes("/app"); 
        config.enableSimpleBroker("/topic", "/queue"); 
    }
}
```
- **/portfolio**是WebSocket(或SockJS)客户端在WebSocket握手时需要连接到的端点的HTTP URL。
- 目的地标头以/app开头的STOMP消息被路由到@Controller类中的@MessageMapping方法。
- 使用内置的message broker订阅和广播消息，并将目的地标头以/topic '或' /queue开头的消息路由到代理。

> 对于内置的简单代理，/topic和/queue前缀没有任何特殊含义。它们只是区分发布-订阅和点到点消息传递(即多个订阅者和一个使用者)的惯例。当您使用外部代理时，请检查代理的STOMP页面，以了解它支持哪种类型的STOMP目的地和前缀。 

要从浏览器连接，对于SockJS，可以使用SockJS -client。对于STOMP，许多应用程序都使用了jmesnil/ STOMP -websocket库(也称为STOMP .js)，该库具有完整的特性，已经在生产中使用多年，但不再维护。目前，JSteunou/webstomp-client是该库最积极维护和发展的继承者。下面的示例代码是基于它的:
```js
var socket = new SockJS("/spring-websocket-portfolio/portfolio");
var stompClient = webstomp.over(socket);

stompClient.connect({}, function(frame) {
}
```
或者，如果您通过WebSocket连接(没有SockJS)，您可以使用以下代码:
```js
var socket = new WebSocket("/spring-websocket-portfolio/portfolio");
var stompClient = Stomp.over(socket);

stompClient.connect({}, function(frame) {
}
```

### 消息流
一旦公开了STOMP端点，Spring应用程序就成为连接客户机的STOMP代理。本节描述服务器端的消息流。

Spring -messaging模块包含对源自Spring Integration的消息传递应用程序的基本支持，这些应用程序后来被提取并集成到Spring框架中，以便在许多Spring项目和应用程序场景中广泛使用。下面的列表简要描述了一些可用的消息传递抽象:
- Message：消息的简单表示，包括头和负载
- MessageHandler: 处理消息
- MessageChannel: 发送消息的契约，该消息支持生产者和消费者之间的松散耦合
- SubscribableChannel: 带有MessageHandler订阅者的MessageChannel
- ExecutorSubscribableChannel: 使用执行程序传递消息的SubscribableChannel。


![image](https://docs.spring.io/spring/docs/5.1.3.RELEASE/spring-framework-reference/images/message-flow-simple-broker.png)

- clientInboundChannel: 用于传递从WebSocket客户端接收的消息。
- clientOutboundChannel: 用于向WebSocket客户端发送服务器消息。
- brokerChannel: 用于从服务器端应用程序代码中向message broker发送消息。


当从WebSocket连接接收到消息时，它们被解码为STOMP帧，转换为Spring消息表示，并发送到clientInboundChannel进行进一步处理。例如，目的地标头以/app开头的STOMP消息可以路由到带注释控制器中的@MessageMapping方法，而/topic和/queue消息可以直接路由到message broker。

处理来自客户机的STOMP消息的带注释的@Controller可以通过brokerChannel向message broker发送消息，代理通过clientOutboundChannel将消息广播给匹配的订阅者。相同的控制器也可以对HTTP请求做出相同的响应，因此客户端可以执行HTTP POST，然后@PostMapping方法可以向message broker发送消息以广播到订阅的客户端。

### 注解
应用程序可以使用带注释的@Controller类来处理来自客户机的消息。这些类可以声明@MessageMapping、@SubscribeMapping和@ExceptionHandler方法，如下面的主题所述:
- **@MessageMapping**
- **@SubscribeMapping**
- **@MessageExceptionHandler**

#### @MessageMapping

您可以使用@MessageMapping来注释基于消息的目的地路由消息的方法。它在方法级别和类型级别都受到支持。在类型级别，@MessageMapping用于表示控制器中所有方法之间的共享映射。

默认情况下，映射值是ant风格的路径模式(例如/thing*、/thing/**)，包括对模板变量的支持(例如/thing/{id})。可以通过@DestinationVariable方法参数引用这些值。应用程序还可以切换到用于映射的点分隔目标约定，如点将其解释为分隔符。

支持方法参数

| Method argument                                              | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `Message`                                                    | For access to the complete message.                          |
| `MessageHeaders`                                             | For access to the headers within the `Message`.              |
| `MessageHeaderAccessor`, `SimpMessageHeaderAccessor, andStompHeaderAccessor` | For access to the headers through typed accessor methods.    |
| `@Payload`                                                   | For access to the payload of the message, converted (for example, from JSON) by a configured `MessageConverter`.The presence of this annotation is not required since it is, by default, assumed if no other argument is matched.You can annotate payload arguments with `@javax.validation.Valid` or Spring’s `@Validated`, to have the payload arguments be automatically validated. |
| `@Header`                                                    | For access to a specific header value — along with type conversion using an`org.springframework.core.convert.converter.Converter`, if necessary. |
| `@Headers`                                                   | For access to all headers in the message. This argument must be assignable to`java.util.Map`. |
| `@DestinationVariable`                                       | For access to template variables extracted from the message destination. Values are converted to the declared method argument type as necessary. |
| `java.security.Principal`                                    | Reflects the user logged in at the time of the WebSocket HTTP handshake. |

**返回值**

默认情况下，`@MessageMapping`方法的返回值通过匹配的MessageConverter序列化到有效负载，并作为消息发送到brokerChannel，然后从brokerChannel广播给订阅者。出站消息的目的地与入站消息的目的地相同，但是以/topic作为前缀

可以使用@SendTo和@SendToUser注释自定义输出消息的目的地。@SendTo用于自定义目标目的地或指定多个目的地。@SendToUser用于将输出消息仅指向与输入消息关联的用户。看到用户目的地。

您可以在同一方法上同时使用@SendTo和@SendToUser，这两种方法在类级别上都受到支持，在这种情况下，它们作为类中方法的默认值。但是，请记住，任何方法级别的@SendTo或@SendToUser注释都会在类级别覆盖此类注释。

消息可以异步处理，@MessageMapping方法可以返回ListenableFuture、CompletableFuture或CompletionStage。

注意，@SendTo和@SendToUser只是一种方便，相当于使用SimpMessagingTemplate发送消息。如果需要，对于更高级的场景，@MessageMapping方法可以直接使用SimpMessagingTemplate。可以这样做，而不是返回一个值，或者可能是另外返回一个值。看到发送消息

#### @SubscribeMapping

@SubscribeMapping类似于@MessageMapping，但是只将映射缩小到订阅消息。它支持与@MessageMapping相同的方法参数。但是对于返回值，默认情况下，消息直接发送到客户机(通过clientOutboundChannel响应订阅)，而不是发送到代理(通过brokerChannel，作为对匹配订阅的广播)。添加@SendTo或@SendToUser将重写此行为并将其发送到代理。

什么时候有用?假设代理映射到/topic和/queue，而应用程序控制器映射到/app。在此设置中，代理存储用于重复广播的/topic和/queue的所有订阅，应用程序不需要参与其中。客户机还可以订阅某个/app目的地，控制器可以返回响应该订阅的值，而不需要代理参与，不需要存储或再次使用订阅(实际上是一次性的请求-应答交换)。

什么时候没用呢?不要尝试将代理和控制器映射到相同的目标前缀，除非出于某种原因，您希望两者独立地处理消息，包括订阅。入站消息是并行处理的。无法保证代理或控制器是否首先处理给定的消息。如果目标是在存储订阅并为广播做好准备时得到通知，那么客户机应该在服务器支持的情况下索要收据(simple broker不支持)。例如，使用Java STOMP客户端，您可以执行以下操作来添加收据:

```java
@Autowired
private TaskScheduler messageBrokerTaskScheduler;

// During initialization..
stompClient.setTaskScheduler(this.messageBrokerTaskScheduler);

// When subscribing..
StompHeaders headers = new StompHeaders();
headers.setDestination("/topic/...");
headers.setReceipt("r1");
FrameHandler handler = ...;
stompSession.subscribe(headers, handler).addReceiptTask(() -> {
    // Subscription ready...
});
```

服务器端选项是在brokerChannel上注册ExecutorChannelInterceptor，并实现在处理消息(包括订阅)之后调用的aftermessagehandling方法。



#### @MessageExceptionHandler

应用程序可以使用@MessageExceptionHandler方法来处理来自@MessageMapping方法的异常。如果希望访问异常实例，可以在注释本身或通过方法参数声明异常。下面的例子通过方法参数声明了一个异常:

```java
@Controller
public class MyController {

    // ...

    @MessageExceptionHandler
    public ApplicationError handleException(MyException exception) {
        // ...
        return appError;
    }
}
```

@MessageExceptionHandler方法支持灵活的方法签名，并支持与@MessageMapping方法相同的方法参数类型和返回值。

通常，@MessageExceptionHandler方法应用于声明它们的@Controller类(或类层次结构)中。如果您想要这些方法应用于更全局的(跨控制器的)，您可以在一个标有@ControllerAdvice的类中声明它们。这类似于Spring MVC中提供的类似支持



### 发送消息

如果您想从应用程序的任何部分向连接的客户机发送消息，该怎么办?任何应用程序组件都可以向**brokerChannel**发送消息。最简单的方法是注入**SimpMessagingTemplate**并使用它发送消息。通常，您会按类型注入它，如下面的示例所示:

```java
@Controller
public class GreetingController {

    private SimpMessagingTemplate template;

    @ 
    public GreetingController(SimpMessagingTemplate template) {
        this.template = template;
    }

    @RequestMapping(path="/greetings", method=POST)
    public void greet(String greeting) {
        String text = "[" + getTimestamp() + "]:" + greeting;
        this.template.convertAndSend("/topic/greetings", text);
    }

}
```



### 简单消息代理（simple broker）

内置的**simple message broker**处理来自客户机的订阅请求，将它们存储在内存中，并将消息广播给具有匹配目的地的连接客户机。代理支持类似路径的目的地，包括订阅**ant**样式的目的地模式。

```java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    private TaskScheduler messageBrokerTaskScheduler;

    @Autowired
    public void setMessageBrokerTaskScheduler(TaskScheduler taskScheduler) {
        this.messageBrokerTaskScheduler = taskScheduler;
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {

        registry.enableSimpleBroker("/queue/", "/topic/")
                .setHeartbeatValue(new long[] {10000, 20000})
                .setTaskScheduler(this.messageBrokerTaskScheduler);

        // ...
    }
}
```



### 外部消息代理(External Broker)

简单代理非常适合入门，但只支持STOMP命令的一个子集(它不支持**ack**、**receipt**和其他一些特性)，依赖于一个简单的消息发送循环，不适合集群。

另一种选择是，您可以升级您的应用程序以使用功能齐全的message broker。请参阅选择的消息代理的STOMP文档(如RabbitMQ、ActiveMQ等)，安装代理，并在启用STOMP支持的情况下运行它。然后可以在Spring配置中启用STOMP代理中继(而不是简单的代理)。

```java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/portfolio").withSockJS();
    }

    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        registry.enableStompBrokerRelay("/topic", "/queue");
        registry.setApplicationDestinationPrefixes("/app");
    }

}
```

前面配置中的STOMP代理中继是Spring MessageHandler，它通过将消息转发到外部消息代理来处理消息。为此，它建立到代理的TCP连接，将所有消息转发给代理，然后通过代理的WebSocket会话将从代理接收到的所有消息转发给客户机。本质上，它充当一个“中继”，在两个方向上转发消息。

