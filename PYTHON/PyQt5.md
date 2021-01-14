## 安装

`pip install PyQt5`

`pip install PyQt5-tools`



## vscode 环境配置

安装扩展 `PYQT Integration` 并配置路径

```c++
import sys
from PyQt5 import QtWidgets
  
app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QWidget()
widget.resize(360, 360)
widget.setWindowTitle("hello, pyqt5")
widget.show()
sys.exit(app.exec())
```







## 对信号和槽的支持

Qt的主要特征之一是它使用信号和插槽在对象之间进行通信。 它们的使用鼓励了可重用组件的开发。

当潜在的事情发生时，发出信号。 插槽是Python可调用的。 如果将信号连接到插槽，则在发出信号时将调用该插槽。 如果没有连接信号，则什么也不会发生。 发出信号的代码（或组件）不知道或不在乎是否正在使用该信号。

信号/插槽机制具有以下功能。

- 信号可能连接到许多插槽。

- 一个信号也可以连接到另一个信号。

- 信号参数可以是任何Python类型。

- 插槽可以连接到许多信号。

- 连接可以是直接的（即同步）或排队的（即异步）。

- 可以跨线程进行连接。

- 信号可能断开。



### 绑定和解绑信号

信号（特别是未绑定的信号）是类属性。当信号被引用为该类实例的属性时，PyQt5会自动将该实例绑定到该信号，以创建绑定信号。这与Python本身用于从类函数创建绑定方法的机制相同。

绑定信号具有实现关联功能的`connect（）`，`disconnect（）`和`generate（）`方法。它还具有一个`signal`属性，它是Qt的`SIGNAL（）`宏返回的信号的签名。

信号可能过载，即。具有特定名称的信号可能支持多个签名。可以用签名对信号进行索引，以选择所需的信号。签名是一系列类型。类型可以是Python类型对象，也可以是C ++类型名称的字符串。 C ++类型的名称会自动进行规范化，例如，可以使用QVariant代替非规范化的const QVariant＆。

如果信号过载，那么它将有一个默认值，如果没有给出索引，它将被使用。

发出信号时，如有可能，所有参数都将转换为C ++类型。如果参数没有相应的C ++类型，则将其包装在特殊的C ++类型中，该参数允许在Qt的元类型系统中传递该参数，同时确保正确维护其引用计数。



### 使用pyqtSignal定义新的信号

PyQt5自动为所有Qt的内置信号定义信号。可以使用pyqtSignal工厂将新信号定义为类属性。

> PyQt5.QtCore.**pyqtSignal**（types [，name [，revision = 0 [，arguments = []]]]）
> 创建一个或多个重载的未绑定信号作为类属性。

参数：
**types**–定义信号的C ++签名的类型。每个类型都可以是Python类型对象，也可以是C ++类型名称的字符串。或者，每个参数可以是一系列类型参数。在这种情况下，每个序列定义了不同信号过载的特征。默认为第一次过载。

**name**–信号名称。如果省略，则使用class属性的名称。这只能作为keyword 参数给出。

**revision**–导出到QML的信号的修订版。这只能作为keyword 参数给出。

**arguments** –导出到QML的信号参数名称的顺序。这只能作为keyword 参数给出。

返回类型:

未绑定的信号



以下示例显示了许多新信号的定义：

```python
from PyQt5.QtCore import QObject, pyqtSignal

class Foo(QObject):

    # This defines a signal called 'closed' that takes no arguments.
    closed = pyqtSignal()

    # This defines a signal called 'rangeChanged' that takes two
    # integer arguments.
    range_changed = pyqtSignal(int, int, name='rangeChanged')

    # This defines a signal called 'valueChanged' that has two overloads,
    # one that takes an integer argument and one that takes a QString
    # argument.  Note that because we use a string to specify the type of
    # the QString argument then this code will run under Python v2 and v3.
    valueChanged = pyqtSignal([int], ['QString'])
```

新信号只能在QObject的子类中定义。它们必须是类定义的一部分，并且在定义了类之后不能动态添加为类属性

以这种方式定义的新信号将自动添加到类的QMetaObject中。这意味着它们将出现在Qt Designer中，并且可以使用QMetaObject API进行内省。

当参数的Python类型没有对应的C ++类型时，应谨慎使用重载信号。 PyQt5使用相同的内部C ++类来表示此类对象，因此可能会产生带有不同Python签名的重载信号，这些信号用相同的C ++签名实现，会产生意外的结果。 以下是一个示例：

```python
class Foo(QObject):

    # This will cause problems because each has the same C++ signature.
    valueChanged = pyqtSignal([dict], [list])
```



### Connecting, Disconnecting and Emitting

> `connect`(*slot*[, *type=PyQt5.QtCore.Qt.AutoConnection*[, *no_receiver_check=False*]]) → PyQt5.QtCore.QMetaObject.Connection

Connect a signal to a slot. An exception will be raised if the connection failed.

- Parameters

  **slot** – the slot to connect to, either a Python callable or another bound signal.**type** – the type of the connection to make.**no_receiver_check** – suppress the check that the underlying C++ receiver instance still exists and deliver the signal anyway.

- Returns

  a [Connection](https://www.riverbankcomputing.com/static/Docs/PyQt5/api/qtcore/qmetaobject-connection.html) object which can be passed to [`disconnect()`](https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#disconnect). This is the only way to disconnect a connection to a lambda function.



> `disconnect`([*slot*])

Disconnect one or more slots from a signal. An exception will be raised if the slot is not connected to the signal or if the signal has no connections at all.

- Parameters

  **slot** – the optional slot to disconnect from, either a [Connection](https://www.riverbankcomputing.com/static/Docs/PyQt5/api/qtcore/qmetaobject-connection.html) object returned by [`connect()`](https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#connect), a Python callable or another bound signal. If it is omitted then all slots connected to the signal are disconnected.



`emit(**args*)`

Emit a signal.

- Parameters

  **args** – the optional sequence of arguments to pass to any connected slots.



```python
from PyQt5.QtCore import QObject, pyqtSignal

class Foo(QObject):

    # Define a new signal called 'trigger' that has no arguments.
    trigger = pyqtSignal()

    def connect_and_emit_trigger(self):
        # Connect the trigger signal to a slot.
        self.trigger.connect(self.handle_trigger)

        # Emit the signal.
        self.trigger.emit()

    def handle_trigger(self):
        # Show that the slot has been called.

        print "trigger signal received"
```

```python
from PyQt5.QtWidgets import QComboBox

class Bar(QComboBox):

    def connect_activated(self):
        # The PyQt5 documentation will define what the default overload is.
        # In this case it is the overload with the single integer argument.
        self.activated.connect(self.handle_int)

        # For non-default overloads we have to specify which we want to
        # connect.  In this case the one with the single string argument.
        # (Note that we could also explicitly specify the default if we
        # wanted to.)
        self.activated[str].connect(self.handle_string)

    def handle_int(self, index):
        print "activated signal passed integer", index

    def handle_string(self, text):
        print "activated signal passed QString", text
```

```python
act = QAction("Action", self)
act.triggered.connect(self.on_triggered)

act = QAction("Action", self, triggered=self.on_triggered)

act = QAction("Action", self)
act.pyqtConfigure(triggered=self.on_triggered)
```



### pyqtSlot() 装饰器

> PyQt5.QtCore.**pyqtSlot**(*types*[, *name*[, *result*[, *revision=0*]]])

Decorate a Python method to create a Qt slot.

- Parameters

  **types** – the types that define the C++ signature of the slot. Each type may be a Python type object or a string that is the name of a C++ type.**name** – the name of the slot that will be seen by C++. If omitted the name of the Python method being decorated will be used. This may only be given as a keyword argument.**revision** – the revision of the slot that is exported to QML. This may only be given as a keyword argument.**result** – the type of the result and may be a Python type object or a string that specifies a C++ type. This may only be given as a keyword argument.

``` python
from PyQt5.QtCore import QObject, pyqtSlot

class Foo(QObject):

    @pyqtSlot()
    def foo(self):
        """ C++: void foo() """

    @pyqtSlot(int, str)
    def foo(self, arg1, arg2):
        """ C++: void foo(int, QString) """

    @pyqtSlot(int, name='bar')
    def foo(self, arg1):
        """ C++: void bar(int) """

    @pyqtSlot(int, result=int)
    def foo(self, arg1):
        """ C++: int foo(int) """

    @pyqtSlot(int, QObject)
    def foo(self, arg1):
        """ C++: int foo(int, QObject *) """
```



### Connecting Slots By Name

PyQt5支持`connectSlotsByName（）`函数，pyuic5生成的Python代码最常使用该函数将信号自动连接到符合简单命名约定的插槽。 但是，在类重载Qt信号的地方（即，具有相同名称但具有不同参数的Qt信号），PyQt5需要附加信息才能自动连接正确的信号。

For example the [QSpinBox](https://www.riverbankcomputing.com/static/Docs/PyQt5/api/qtwidgets/qspinbox.html) class has the following signals:

```
void valueChanged(int i);
void valueChanged(const QString &text);
```

当旋转框的值更改时，将同时发出这两个信号。 如果您实现了一个名为`on_spinbox_valueChanged`的插槽（假设您为QSpinBox实例指定了名称spinbox），则它将连接到信号的两个变体。 因此，当用户更改值时，您的广告位将被调用两次-一次使用整数参数，一次使用字符串参数。

pyqtSlot（）装饰器可用于指定应将哪些信号连接到插槽。

```python
@pyqtSlot(int, name='on_spinbox_valueChanged')
def spinbox_int_value(self, i):
    # i will be an integer.
    pass

@pyqtSlot(str, name='on_spinbox_valueChanged')
def spinbox_qstring_value(self, s):
    # s will be a Python string object (or a QString if they are enabled).
    pass
```





