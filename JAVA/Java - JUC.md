## 一、核心类

Java5.0添加了一个新的java.util.concurrent开发包

**核心类**
01）**Executor**：具有Runnable任务的执行者。
02）**ExecutorService**：一个线程池管理者，其实现类有多种，能把Runnable,Callable提交到池中让其调度。
03）**Semaphore**：一个计数信号量。
04）**ReentrantLock**：一个可重入的互斥锁定Lock，功能类似synchronized，但要强大的多。
05）**Future**：多线程交互的接口，比如一个线程执行结束后取返回的结果等，还提供了cancel终止线程。
06）**BlockingQueue**：阻塞队列。
07）**CompletionService**：ExecutorService的扩展，可以获得线程执行结果的。
08）**CountDownLatch**：在完成一组正在其他线程中执行的操作之前，它允许一个或多个线程一直等待。
09）**CyclicBarrier**：一个同步辅助类，它允许一组线程互相等待，直到到达某个公共屏障点。
10）**Future**：表示异步计算的结果。
11）**ScheduldExecutorService**：一个ExecutorService，可安排在给定的延迟后运行或定期执行的命令。



## 二、TimeUnit工具类

枚举类

| `DAYS`  时间单位代表二十四小时               |
| -------------------------------------------- |
| `HOURS`   时间单位代表六十分钟               |
| `MICROSECONDS`   时间单位代表千分之一毫秒    |
| `MILLISECONDS`   时间单位为千分之一秒        |
| `MINUTES`   时间单位代表60秒                 |
| `NANOSECONDS`   时间单位代表千分之一千分之一 |
| `SECONDS`   时间单位代表一秒                 |

线程休眠，相当于*Thread.Sleep*

``` java
TimeUnit.SECONDS.sleep(5);
```

时间转换

```java
//将三天时间转换为毫秒
long time = TimeUnit.MILLISECONDS.convert(3, TimeUnit.DAYS);
```



## 三、原子操作类

原子操作，是指操作过程不会被中断，保证数据操作是以原子方式进行的

• 基本类型：**AtomicInteger**, **AtomicLong**, **AtomicBoolean**.
• 数组类型：**AtomicIntegerArray**, **AtomicLongArray**.
• 引用类型：**AtomicReference**, **AtomicStampedRerence**.
• 对象的属性修改类型：**AtomicIntegerFieldUpdater**, **AtomicLongFieldUpdater**, **AtomicReferenceFieldUpdater**.

```java
//基本类型
AtomicInteger num = new AtomicInteger();
System.out.println(num.incrementAndGet());//自增获取

AtomicLong num2 = new AtomicLong(100) ;
System.out.println(num2.compareAndSet(100, 333));//true
System.out.println(num2);//333

//数组
AtomicLongArray array = new AtomicLongArray(new long[]{1, 2, 3});
array.set(0, 9); // 原子性的数组必须使用set修改内容
System.out.println(array);  // 输出结果：[9, 2, 3]

//对象
public class MLDNTestDemo {
    public static void main(String[] args) throws Exception {
        AtomicReference<Member> ref = new AtomicReference<Member>();
        Member memA = new Member("张三", 20);
        Member memB = new Member("李四", 30);
        ref.set(memA);// 对象引用变更只得依靠地址比较“==”
        ref.compareAndSet(memA, memB);
        System.out.println(ref);  // 结果：name = 李四、age = 30
    }
}
 
class Member {
    private String name;
    private int age;
    public Member(String name, int age) {
        this.name = name;   this.age = age;
    }
    @Override
    public String toString() {
        return "name = " + this.name + "、age = " + this.age;
    }
```



