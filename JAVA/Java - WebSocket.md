## 一、简介

**WebSocket **协议RFC 6455提供了一种标准化的方法，可以通过单个TCP连接在客户机和服务器之间建立全双工、双向通信通道。它是与HTTP不同的TCP协议，但设计用于在HTTP上工作，使用端口80和443，并允许重用现有防火墙规则。



## 二、WebSocket API

Spring框架提供了一个WebSocket API，您可以使用它来编写处理WebSocket消息的客户机和服务器端应用程序。

### `1、WebSocketHandler`

创建WebSocket服务器就像实现WebSocketHandler一样简单，或者更有可能扩展**TextWebSocketHandler**或**BinaryWebSocketHandler**。

写一个**Handler**类，继承**TextWebSocketHandler** 或 **BinaryWebSocketHandler**

```java
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.TextMessage;

public class MyHandler extends TextWebSocketHandler {

    @Override
    public void handleTextMessage(WebSocketSession session, TextMessage message) {
        // ...
    }

}
```

添加配置**@EnableWebSocket** 实现 **WebSocketConfigurer**


```java
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(myHandler(), "/myHandler");
    }

    @Bean
    public WebSocketHandler myHandler() {
        return new MyHandler();
    }

}
```



### 2、WebSocket Handshake

**addInterceptors** -> **HttpSessionHandshakeInterceptor**

```java
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new MyHandler(), "/myHandler")
            .addInterceptors(new HttpSessionHandshakeInterceptor());
    }

}
```



### 3、Server Configuration

每个底层WebSocket引擎都公开控制运行时特征的配置属性，例如消息缓冲区大小、空闲超时等等

对于Tomcat、WildFly和GlassFish，可以将ServletServerContainerFactoryBean添加到WebSocket Java配置中，如下面的示例所示

```java
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Bean
    public ServletServerContainerFactoryBean createWebSocketContainer() {
        ServletServerContainerFactoryBean container = new ServletServerContainerFactoryBean();
        container.setMaxTextMessageBufferSize(8192);
        container.setMaxBinaryMessageBufferSize(8192);
        return container;
    }

}
```



### 4、Allowed Origins & SockJS

```java
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(myHandler(), "/myHandler")
        .setAllowedOrigins("*").withSockJS();
    }

    @Bean
    public WebSocketHandler myHandler() {
        return new MyHandler();
    }

}
```



## 三、STOMP

STOMP(简单的面向文本的消息传递协议)最初是为脚本语言(如Ruby、Python和Perl)创建的，用于连接到企业消息代理。它被设计用于处理常用消息传递模式的最小子集。STOMP可以用于任何可靠的双向流网络协议，如TCP和WebSocket。虽然STOMP是一个面向文本的协议，但是消息有效负载可以是文本或二进制



### 1、Enable STOMP

```java
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
        config.enableSimpleBroker("/topic"); 
    }
}
```

> `/portfolio` is the HTTP URL for the endpoint to which a WebSocket (or SockJS) client needs to connect for the WebSocket handshake.
>
> STOMP messages whose destination header begins with `/app` are routed to `@MessageMapping` methods in `@Controller` classes.
>
> Use the built-in message broker for subscriptions and broadcasting and route messages whose destination header begins with `/topic ` to the broker.



### 2、Flow of Messages消息流

```java
@Controller
public class GreetingController {

    @MessageMapping("/greeting") {
    public String handle(String greeting) {
        return "[" + getTimestamp() + ": " + greeting;
    }

}
```



以上代码运行流程：

1、客户端连接到http://localhost:8080/portfolio，一旦建立了WebSocket连接，STOMP帧就开始在其上流动。

2、客户端发送带有`/topic/greeting`的目标标题的订阅框架。接收并解码后，消息被发送到`clientInboundChannel`，然后路由到存储客户端订阅的`message broker`。

3、客户端发送一个aSEND帧到/app/greeting。/app前缀有助于将其路由到带注释的控制器。去掉/app前缀后，目标的剩余/greeting部分映射到GreetingController中的@MessageMapping方法。

4、从GreetingController返回的值被转换为一个Spring消息，其负载基于返回值和/topic/greeting的缺省目的地标头(从输入目的地派生，/app替换为/topic)。结果消息被发送到broker通道，并由message broker处理。

5、`message broker`找到所有匹配的订阅者，并通过`clientOutboundChannel`向每个订阅者发送消息框架，消息从该通道编码为STOMP帧并在WebSocket连接上发送。



