# Protobuf 语言指南（proto3）

`Protocol Buffer`是Google的跨语言，跨平台，可扩展的，用于序列化结构化数据 - 对比XML，但更小，更快，更简单。您可以定义数据的结构化，然后可以使用特殊生成的源代码轻松地在各种数据流中使用各种语言编写和读取结构化数据。

## 定义消息类型

先来看一个非常简单的例子。假设你想定义一个“搜索请求”的消息格式，每一个请求含有一个**查询字符串**、你感兴趣的**查询结果所在的页数**，以及每一页**多少条查询结果**。可以采用如下的方式来定义消息类型的`.proto`文件了：

```protobuf
syntax = "proto3";

message SearchRequest {
  string query = 1;
  int32 page_number = 2;
  int32 result_per_page = 3;
}
```

- 该文件的第一行指定您正在使用`proto3`语法：如果您不这样做，protobuf 编译器将假定您正在使用[proto2](https://developers.google.com/protocol-buffers/docs/proto)。这必须是文件的第一个非空的非注释行。

- 所述`SearchRequest`消息定义了三个字段（名称/值对），对应着我需要的消息内容。每个字段都有一个名称和类型。

- 在上面的示例中，所有字段都是**标量类型**：两个整数（`page_number`和`result_per_page`）和一个字符串（`query`）。但是，您还可以为字段指定合成类型，包括**枚举**和其他消息类型。



### 分配标识号

正如上述文件格式，在消息定义中，每个字段都有唯一的一个**数字标识符**。这些标识符是用来在消息的二进制格式中识别各个字段的，一旦开始使用就不能够再改变。

> 注：[1,15]之内的标识号在编码的时候会占用一个字节。[16,2047]之内的标识号则占用2个字节。所以应该为那些频繁出现的消息元素保留 [1,15]之内的标识号。切记：要为将来有可能添加的、频繁出现的标识号预留一些标识号。

最小的标识号可以从1开始，最大到2^29 - 1, or 536,870,911。**不可以使用其中的[19000－19999]的标识号**， Protobuf协议实现中对这些进行了预留。如果非要在.proto文件中使用这些预留标识号，编译时就会报错。



### 添加注释

要为`.proto`文件添加注释，请使用C / C ++ - 样式`//`和`/* ... */`语法。



### .proto文件最终生成什么

当你使用`protoc` 来编译一个`.proto`文件的时候，编译器将利用你在文件中定义的类型生成你打算使用的语言的代码文件。生成的代码包括getting setting 接口和序列化，反序列化接口。

对于**C ++**，编译器会从每个`.proto`文件生成一个`.h`和一个`.cc`文件，并为您文件中描述的每种消息类型提供一个类。

**Python**有点不同 - Python编译器生成一个模块，其中包含每个消息类型的静态描述符，然后，用一个元类在运行时创建必要的Python数据访问类。



## 标量值类型

| .proto type | notes                                                        | C ++ type | Java type   | Python type      | Type    |
| ----------- | ------------------------------------------------------------ | --------- | ----------- | ---------------- | ------- |
| double      |                                                              | double    | double      | float            | float64 |
| float       |                                                              | float     | float       | float            | FLOAT32 |
| INT32       | 使用可变长度编码。编码负数的效率低 - 如果您的字段可能有负值，请改用sint32。 | INT32     | INT         | INT              | INT32   |
| Int64       | 使用可变长度编码。编码负数的效率低 - 如果您的字段可能有负值，请改用sint64。 | Int64     | long        | int / long [3]   | Int64   |
| UINT32      | 使用可变长度编码。                                           | UINT32    | int [1]     | int / long [3]   | UINT32  |
| UINT64      | 使用可变长度编码。                                           | UINT64    | Long [1]    | int / long [3]   | UINT64  |
| SINT32      | 使用可变长度编码。签名的int值。这些比常规int32更有效地编码负数。 | INT32     | INT         | INT              | INT32   |
| sint64      | 使用可变长度编码。签名的int值。这些比常规int64更有效地编码负数。 | Int64     | long        | int / long [3]   | Int64   |
| fixed32     | 总是四个字节。如果值通常大于2 28，则比uint32更有效。         | UINT32    | int [1]     | int / long [3]   | UINT32  |
| fixed64     | 总是八个字节。如果值通常大于2 56，则比uint64更有效。         | UINT64    | Long [1]    | int / long [3]   | UINT64  |
| sfixed32    | 总是四个字节。                                               | INT32     | INT         | INT              | INT32   |
| sfixed64    | 总是八个字节。                                               | Int64     | long        | int / long [3]   | Int64   |
| Boolean     |                                                              | Boolean   | Boolean     | Boolean          | Boolean |
| string      | 字符串必须始终包含UTF-8编码或7位ASCII文本。                  | string    | string      | str / unicode[4] | string  |
| byte        | 可以包含任意字节序列。                                       | string    | Byte string | Strait           | []byte  |



## 默认值

- 对于字符串，默认值为空字符串。
- 对于字节，默认值为空字节。
- 对于bools，默认值为false。
- 对于数字类型，默认值为零。
- 对于[枚举](https://developers.google.com/protocol-buffers/docs/proto3#enum)，默认值是第**一个定义的枚举值**，该**值**必须为0。
- 对于消息字段，未设置该字段。它的确切值取决于语言。



## 枚举

当你定义一个消息的时候，你可能希望它其中的某个字段一定是预先定义好的一组值中的一个。你如说我要在SearchRequest中添加corpus字段。它只能是 UNIVERSAL, WEB , IMAGES , LOCAL, NEWS ,PRODUCTS, 或者 VIDEO 。你可以很简单的在你的消息中定义一个枚举并且定义corpus字段为枚举类型，如果这个字段给出了一个不再枚举中的值，那么解析器就会把它当作一个未知的字段。

```protobuf
message SearchRequest {
    string query = 1;
    int32 page_number = 2;
    int32 result_per_page = 3;
    enum Corpus {
      UNIVERSAL = 0;
      WEB = 1;
      IMAGES = 2;
      LOCAL = 3;
      NEWS = 4;
      PRODUCTS = 5;
      VIDEO = 6;
    }
    Corpus corpus = 4;
  }
```

如您所见，`Corpus`枚举的第一个常量映射为零：每个枚举定义**必须**包含一个映射到零的常量作为其第一个元素。这是因为：

- 必须有一个零值，以便我们可以使用0作为数字[默认值](https://developers.google.com/protocol-buffers/docs/proto3#default)。
- 零值必须是第一个元素，以便与[proto2](https://developers.google.com/protocol-buffers/docs/proto)语义兼容，其中第一个枚举值始终是默认值。



## 使用其他消息类型

您可以使用其他消息类型作为字段类型。例如，假设你想包括`Result`每个消息的`SearchResponse`消息-要做到这一点，你可以定义一个`Result`在同一个消息类型`.proto`，然后指定类型的字段`Result`中`SearchResponse`：

```protobuf
message SearchResponse {
    repeated Result results = 1;
}
 
message Result {
    string url = 1;
    string title = 2;
    repeated string snippets = 3;
}
```



## 导入定义

您可以`.proto`通过*导入*来使用其他文件中的定义。要导入其他`.proto`的定义，请在文件顶部添加import语句：

```protobuf
import“myproject/other_protos.proto”;
```



## 嵌套类型

您可以在其他消息类型中定义和使用消息类型，如下例所示 - 此处`Result`消息在消息中定义`SearchResponse`：

```protobuf
message SearchResponse {
    message Result {
      string url = 1;
      string title = 2;
      repeated string snippets = 3;
    }
    repeated Result results = 1;
  }
```

如果要在其父消息类型之外重用此消息类型，请将其称为： `*Parent*.*Type*`

```proto
 message SomeOtherMessage {
    SearchResponse.Result result = 1;
  }
```



## 更新消息类型

如果现有的消息类型不再满足您的所有需求 - 例如，您希望消息格式具有额外的字段 - 但您仍然希望使用使用旧格式创建的代码，请不要担心！在不破坏任何现有代码的情况下更新消息类型非常简单。请记住以下规则：



....





# grpc

## HelloWorld

### proto

```protobuf
syntax="proto3";

service HelloWorld{
    rpc sayHello (HelloRequest) returns (HelloResponse);
}

message HelloRequest{
    string name=1;
}

message HelloResponse{
    string msg=1;
}
```



#### generate code:

```shell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./filename.proto
```



### server

```python
import proto.hello_pb2 as hello_pb2
import proto.hello_pb2_grpc as hello_pb2_grpc
import grpc
from concurrent import futures
import logging


class HelloWorld(hello_pb2_grpc.HelloWorldServicer):

    def sayHello(self, request, context):
        return hello_pb2.HelloResponse(msg=f"Hello {request.name}")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hello_pb2_grpc.add_HelloWorldServicer_to_server(HelloWorld(), server)
    server.add_insecure_port('[::]:50505')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()

```



### client

```python
import proto.hello_pb2 as hello_pb2
import proto.hello_pb2_grpc as hello_pb2_grpc
import grpc


def run():
    with grpc.insecure_channel("localhost:50505") as channel:
        stub = hello_pb2_grpc.HelloWorldStub(channel)
        response = stub.sayHello(hello_pb2.HelloRequest(name="linx"))
    print("HelloWorld client received: " + response.msg)


if __name__ == "__main__":
    run()

```

































