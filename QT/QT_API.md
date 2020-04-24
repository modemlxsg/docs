## QTimer

QTimer类提供了重复的和一次性的计时器。

QTimer类为计时器提供了一个高级编程接口。要使用它，请创建一个`QTimer`，将其`timeout()`信号连接到适当的插槽，并调用`start()`。从那时起，它将以固定的间隔发出timeout()信号。

```c++
QTimer *timer = new QTimer(this);
connect(timer, SIGNAL(timeout()), this, SLOT(update()));
timer->start(1000);
```

通过调用`setSingleShot(true)`，您可以设置一个计时器来超时一次。您还可以使用静态`QTimer::singleShot()`函数在指定的时间间隔之后调用槽:

```c++
QTimer::singleShot(200, this, SLOT(updateCaption()));
```

在多线程应用程序中，可以在任何具有事件循环的线程中使用QTimer。要从非gui线程启动事件循环，请使用`QThread::exec()`。Qt使用计时器的线程关联性来确定哪个线程将发出`timeout()`信号。因此，必须在其线程中启动和停止计时器;无法从另一个线程启动计时器。

作为一种特殊情况，超时为0的QTimer将在处理完窗口系统的事件队列中的所有事件后立即超时。这可以用来做繁重的工作，同时提供一个快速的用户界面:

```c++
QTimer *timer = new QTimer(this);
connect(timer, SIGNAL(timeout()), this, SLOT(processOneThing()));
timer->start();
```

从那时起，processOneThing()将被反复调用。它应该以这样一种方式编写，即它总是快速返回(通常在处理一个数据项之后)，以便Qt能够向用户界面交付事件，并在完成所有工作后立即停止计时器。这是在GUI应用程序中实现繁重工作的传统方式，但是随着多线程在越来越多的平台上变得可用，我们预计零毫秒的QTimer对象将逐渐被QThreads所取代。



## QApplication

QApplication类管理GUI应用程序的控制流和主要设置。

QApplication专为`QGuiApplication`专门提供一些基于QWidget的应用程序所需的功能。 它处理小部件特定的初始化，终止。

对于任何使用Qt的GUI应用程序，无论该应用程序在任何给定时间有0、1、2或更多窗口，都只有一个QApplication对象。 对于基于非QWidget的Qt应用程序，请改用QGuiApplication，因为它不依赖于QtWidgets库。

一些GUI应用程序提供了特殊的批处理模式，即。 提供用于执行任务的命令行参数，而无需人工干预。 在这种非GUI模式下，实例化一个普通的QCoreApplication通常足以避免不必要地初始化图形用户界面所需的资源。 以下示例显示如何动态创建适当类型的应用程序实例：

可通过`instance（）`函数访问QApplication对象，该函数返回与全局qApp指针等效的指针。

QApplication的主要职责是：

- 它使用用户的桌面设置（例如：palette（），font（）和doubleClickInterval（））初始化应用程序。万一用户全局更改桌面（例如通过某种控制面板），它会跟踪这些属性。

- 它执行事件处理，这意味着它从底层窗口系统接收事件并将其分派到相关的小部件。通过使用sendEvent（）和postEvent（），您可以将自己的事件发送到小部件。

- 它解析常见的命令行参数并相应地设置其内部状态。有关更多详细信息，请参见下面的构造函数文档。

- 它定义了应用程序的外观，并封装在QStyle对象中。可以在运行时使用setStyle（）进行更改。

- 它指定应用程序如何分配颜色。有关详细信息，请参见setColorSpec（）。

- 它提供了对本地用户可见的字符串的本地化功能，这些字符串可以通过translate（）看到。

- 它提供了一些神奇的对象，例如`desktop（）`和剪贴板（）。

- 它知道应用程序的窗口。您可以使用widgetAt（）询问哪个小部件在某个位置，获取topLevelWidgets（）和closeAllWindows（）的列表，等等。

- 它管理应用程序的鼠标光标处理，请参见setOverrideCursor（）


由于QApplication对象进行了大量初始化，因此必须在创建与用户界面相关的任何其他对象之前创建它。 QApplication还处理常见的命令行参数。因此，在应用程序本身对`argv`进行任何解释或修改之前，创建它通常是一个好主意。



## QPixmap



## QThreadPool

`QThreadPool`类管理QThreads的集合。

QThreadPool管理和回收单个QThread对象，以帮助减少使用线程的程序中的线程创建成本。 每个Qt应用程序都有一个全局QThreadPool对象，可以通过调用`globalInstance（）`来访问该对象。

要使用QThreadPool线程之一，请子类`QRunnable`并实现`run（）`虚拟函数。 然后创建该类的对象，并将其传递给`QThreadPool :: start（）`。

QThreadPool默认情况下会自动删除QRunnable。使用`QRunnable :: setAutoDelete（）`更改自动删除标志。

QThreadPool通过从QRunnable :: run（）中调用tryStart（this），支持多次执行同一QRunnable。如果启用了autoDelete，则当最后一个线程退出运行功能时，QRunnable将被删除。启用autoDelete时，使用相同的QRunnable多次调用start（）会创建竞争条件，因此不建议这样做。

在一定时间内未使用的线程将过期。默认的过期超时为30000毫秒（30秒）。可以使用`setExpiryTimeout（）`进行更改。设置负的到期超时将禁用到期机制。

调用`maxThreadCount（）`查询要使用的最大线程数。如果需要，可以使用setMaxThreadCount（）更改限制。默认的maxThreadCount（）是QThread :: idealThreadCount（）。 `activeThreadCount（）`函数返回当前正在工作的线程数。

`reserveThread（）`函数保留一个线程供外部使用。处理完线程后，请使用`releaseThread（）`，以便可以重用它。本质上，这些函数会暂时增加或减少活动线程数，并且在实现QThreadPool不可见的耗时操作时非常有用。

请注意，QThreadPool是用于管理线程的低级类，有关更高级的选择，请参阅Qt Concurrent模块。



## QMutex

QMutex类提供线程之间的访问序列化。

QMutex的目的是保护一个对象，数据结构或代码段，以便一次只能有一个线程可以访问它（这类似于Java sync关键字）。 通常最好将互斥锁与QMutexLocker一起使用，因为这样可以轻松确保一致地执行锁定和解锁。

例如，假设有一种方法可以在两行上向用户打印一条消息：

```c++
  int number = 6;
  void method1()
  {
      number *= 5;
      number /= 4;
  }
  void method2()
  {
      number *= 3;
      number /= 2;
  }
```

如果连续调用这两个方法，则会发生以下情况：

```c++
  // method1()
  number *= 5;        // number is now 30
  number /= 4;        // number is now 7

  // method2()
  number *= 3;        // number is now 21
  number /= 2;        // number is now 10
```

如果从两个线程同时调用这两个方法，则可能会导致以下顺序：

```c++
  // Thread 1 calls method1()
  number *= 5;        // number is now 30

  // Thread 2 calls method2().
  //
  // Most likely Thread 1 has been put to sleep by the operating
  // system to allow Thread 2 to run.
  number *= 3;        // number is now 90
  number /= 2;        // number is now 45

  // Thread 1 finishes executing.
  number /= 4;        // number is now 11, instead of 10
```

如果添加互斥锁，则应获得所需的结果：

```c++
  QMutex mutex;
  int number = 6;

  void method1()
  {
      mutex.lock();
      number *= 5;
      number /= 4;
      mutex.unlock();
  }

  void method2()
  {
      mutex.lock();
      number *= 3;
      number /= 2;
      mutex.unlock();
  }
```

那么在任何给定时间只有一个线程可以修改数字，并且结果是正确的。 当然，这是一个简单的示例，但适用于需要按特定顺序发生的任何其他情况。

当您在线程中调用lock（）时，其他尝试在同一位置调用lock（）的线程将阻塞，直到获得锁的线程调用unlock（）为止。 tryLock（）是lock（）的一种非阻塞替代方法。

QMutex经过优化，可以在非竞争情况下快速运行。 如果该互斥锁上没有争用，则非递归QMutex将不会分配内存。 它的构建和销毁几乎没有开销，这意味着可以将许多互斥锁作为其他类的一部分很好。



## QMutexLocker

`QMutexLocker`类是一个便捷类，它简化了互斥锁的锁定和解锁。

在复杂的函数和语句中或在异常处理代码中锁定和解锁QMutex容易出错，并且难以调试。 在此类情况下，可以使用QMutexLocker来确保始终正确定义互斥锁的状态。

应该在需要锁定QMutex的函数中创建QMutexLocker。 创建QMutexLocker时，互斥锁被锁定。 您可以使用unlock（）和relock（）来解锁和重新锁定互斥锁。 如果被锁定，则在销毁QMutexLocker时，互斥锁将被解锁。

例如，此复杂函数在进入函数时锁定QMutex并在所有出口点解锁互斥锁：

```c++
  int complexFunction(int flag)
  {
      mutex.lock();

      int retVal = 0;

      switch (flag) {
      case 0:
      case 1:
          retVal = moreComplexFunction(flag);
          break;
      case 2:
          {
              int status = anotherFunction();
              if (status < 0) {
                  mutex.unlock();
                  return -2;
              }
              retVal = status + flag;
          }
          break;
      default:
          if (flag > 10) {
              mutex.unlock();
              return -1;
          }
          break;
      }

      mutex.unlock();
      return retVal;
  }
```

该示例函数在开发过程中将变得更加复杂，从而增加了发生错误的可能性。

使用QMutexLocker大大简化了代码，并使代码更具可读性：

```c++
  int complexFunction(int flag)
  {
      QMutexLocker locker(&mutex);

      int retVal = 0;

      switch (flag) {
      case 0:
      case 1:
          return moreComplexFunction(flag);
      case 2:
          {
              int status = anotherFunction();
              if (status < 0)
                  return -2;
              retVal = status + flag;
          }
          break;
      default:
          if (flag > 10)
              return -1;
          break;
      }

      return retVal;
  }
```

现在，当QMutexLocker对象被销毁时（由于locker是自动变量，函数返回时），互斥锁将始终被解锁。

相同的原理适用于引发和捕获异常的代码。 在锁定互斥锁的函数中未捕获到的异常无法在将异常向上传递到调用函数之前将互斥锁解锁。

QMutexLocker还提供了一个Mutex（）成员函数，该函数返回QMutexLocker在其上运行的互斥量。 这对于需要访问互斥量的代码很有用，例如QWaitCondition :: wait（）。 例如：

```c++
  class SignalWaiter
  {
  private:
      QMutexLocker locker;

  public:
      SignalWaiter(QMutex *mutex)
          : locker(mutex)
      {
      }

      void waitForSignal()
      {
          ...
          while (!signalled)
              waitCondition.wait(locker.mutex());
          ...
      }
  };
```



## QWaitCondition

QWaitCondition类提供用于同步线程的条件变量。

QWaitCondition允许一个线程告诉其他线程已经满足某种条件。 一个或多个线程可以阻止等待QWaitCondition来使用`wakeOne（）`或`wakeAll（）`设置条件。 使用`wakeOne（）`唤醒一个随机选择的线程，或者使用`wakeAll（）`唤醒所有线程。

例如，假设我们有三个任务，每当用户按下一个键时就应执行。 每个任务都可以拆分为一个线程，每个线程都有一个run（）主体，如下所示：

```c++
  forever {
      mutex.lock();
      keyPressed.wait(&mutex);
      do_something();
      mutex.unlock();
  }
```

在这里，keyPressed变量是QWaitCondition类型的全局变量。

第四个线程将读取按键，并在每次收到一个按键时唤醒其他三个线程，如下所示：

```c
  forever {
      getchar();
      keyPressed.wakeAll();
  }
```

三个线程的唤醒顺序是不确定的。 另外，如果某些线程在按下键时仍处于do_something（）中，则不会被唤醒（因为它们没有等待条件变量），因此该键不会执行任务 。 可以使用计数器和QMutex来解决此问题。 例如，这是工作线程的新代码：

```c++
  forever {
      mutex.lock();
      keyPressed.wait(&mutex);
      ++count;
      mutex.unlock();

      do_something();

      mutex.lock();
      --count;
      mutex.unlock();
  }
```



这是第四个线程的代码：

```c++
  forever {
      getchar();

      mutex.lock();
      // Sleep until there are no busy worker threads
      while (count > 0) {
          mutex.unlock();
          sleep(1);
          mutex.lock();
      }
      keyPressed.wakeAll();
      mutex.unlock();
  }
```

互斥锁是必需的，因为试图同时更改同一变量的值的两个线程的结果是不可预测的。

等待条件是强大的线程同步原语。 “等待条件”示例示例说明如何使用QWaitCondition替代QSemaphore，以控制对生产者线程和使用者线程共享的循环缓冲区的访问。



## QSemaphore

QSemaphore类提供了常规计数信号量。

信号量是互斥量的泛化。 尽管互斥锁只能被锁定一次，但是有可能多次获取信号量。 信号量通常用于保护一定数量的相同资源。
信号量支持两个基本操作，`acquire（）`和`release（）`：

- acquire（n）尝试获取n个资源。 如果没有太多可用资源，则在这种情况下，调用将被阻塞。

- release（n）释放n个资源。

还有一个`tryAcquire（）`函数，如果它无法获取资源则立即返回，而`available（）`函数则随时返回可用资源的数量。
例：

```c++
  QSemaphore sem(5);      // sem.available() == 5

  sem.acquire(3);         // sem.available() == 2
  sem.acquire(2);         // sem.available() == 0
  sem.release(5);         // sem.available() == 5
  sem.release(5);         // sem.available() == 10

  sem.tryAcquire(1);      // sem.available() == 9, returns true
  sem.tryAcquire(250);    // sem.available() == 9, returns false
```





