# Rust程序设计

## 1、入门指南

### 安装

第一步是安装 Rust。我们通过 rustup 下载 Rust，这是一个管理 Rust 版本和相关工具的命令行工具。  

Linux and mac

```sh
$ curl https://sh.rustup.rs -sSf | sh
```

windows

在 Windows 上，前往 https://www.rust-lang.org/install.html 并按照说明安装 Rust  



#### 更新和卸载  

通过 rustup 安装了 Rust 之后，很容易更新到最新版本。  

```sh
$ rustup update
```

为了卸载 Rust 和 rustup ，在 shell 中运行如下卸载脚本:  

```sh
$ rustup self uninstall
```



#### 版本

```sh
$ rustc --version
```



#### 本地文档

```sh
$ rustup doc
```



### Hello,world

main.rs  

```rust
fn main() {
	println!("Hello, world!");
}
```

```sh
$ rustc main.rs
$ ./main
Hello, world!
```



### Hello,Cargo

Cargo 是 Rust 的构建系统和包管理器。  

```sh
$ cargo new hello_cargo
$ cd hello_cargo
```

Cargo.toml

```rust
[package]
name = "hello_cargo"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
edition = "2018"

[dependencies]
```

可以使用 `cargo build` 或 `cargo check` 构建项目。
可以使用 `cargo run` 一步构建并运行项目。
有别于将构建结果放在与源码相同的目录，Cargo 会将其放到 `target/debug` 目录。  

可以使用 `cargo build --release` 来优化编译项目 



## 2、猜猜看游戏

```rust
use std::io;
fn main() {
	println!("Guess the number!");
	println!("Please input your guess.");
	let mut guess = String::new();
	io::stdin().read_line(&mut guess)
		.expect("Failed to read line");
	println!("You guessed: {}", guess);
}
```

默认情况下，Rust 将 prelude 模块中少量的类型引入到每个程序的作用域中。  

`expect` 返回`io::Result` 枚举类型。如果是`Err`则程序停止输出信息

如果不调用 expect ，程序也能编译，不过会出现一个警告  ：Rust 警告我们没有使用 read_line 的返回值 Result ，说明有一个可能的错误没有处理。  



### 生成一个秘密数字

crate 是一个 Rust 代码包。我们正在构建的项目是一个 二进制 crate，它生成一个可执行文件。 rand crate 是一个 库 crate，库 crate 可以包含任意能被其他程序使用的代码。  

```rust
[dependencies]
rand = "0.5.5"
```



#### Cargo.lock 确保构建可重现

当第一次构建项目时，Cargo 计算出所有符合要求的依赖版本并写入Cargo.lock 文件。当将来构建项目时，Cargo 会发现 Cargo.lock 已存在并使用其中指定的版本，而不是再次计算所有的版本。这使得你拥有了一个自动化的可重现的构建。  



#### 更新 crate 到一个新版本  

```sh
$ cargo update
```



#### 生成一个随机数  

```rust
use std::io;
use rand::Rng;
fn main() {
	println!("Guess the number!");
	let secret_number = rand::thread_rng().gen_range(1, 101);
	println!("The secret number is: {}", secret_number);
	println!("Please input your guess.");
	let mut guess = String::new();
	io::stdin().read_line(&mut guess)
		.expect("Failed to read line");
	println!("You guessed: {}", guess);
}
```

首先，我们新增了一行 use ： use rand::Rng 。 Rng 是一个 trait，它定义了随机数生成器应实现的方法  



#### 比较猜测的数字和秘密数字  

```rust
use std::io;
use std::cmp::Ordering;
use rand::Rng;
fn main() {
// ---snip---
println!("You guessed: {}", guess);
match guess.cmp(&secret_number) {
	Ordering::Less => println!("Too small!"),
	Ordering::Greater => println!("Too big!"),
	Ordering::Equal => println!("You win!"),
	}
}
```

从标准库引入了一个叫做 `std::cmp::Ordering` 的类型。同Result 一样， Ordering 也是一个枚举，不过它的成员是 `Less 、 Greater 和 Equal` 。这是比较两个值时可能出现的三种结果。  



#### 完整代码

```rust
use std::io;
use std::cmp::Ordering;
use rand::Rng;
fn main() {
    println!("Guess the number!");
    let secret_number = rand::thread_rng().gen_range(1, 101);
    loop {
        println!("Please input your guess.");
        let mut guess = String::new();
        io::stdin().read_line(&mut guess)
            .expect("Failed to read line");
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };
        println!("You guessed: {}", guess);
        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
```



## 3、常见编程概念

### 变量和可变性

变量默认是不可改变的（immutable）   

```rust
let mut x = 5;
```



#### 变量和常量的区别  

声明常量使用 `const` 关键字而不是 `let` ，并且 必须 注明值的类型。  

常量只能被设置为常量表达式，而不能是函数调用的结果，或任何其他只能在运行时计算出的值  

```rust
const MAX_POINTS: u32 = 100_000;
```



#### 隐藏

```rust
fn main() {
	let x = 5;
	let x = x + 1;
	let x = x * 2;
	println!("The value of x is: {}", x);
}
```

这个程序首先将 x 绑定到值 5 上。接着通过 let x = 隐藏 x ，获取初始值并加 1 ，这样 x 的值就变成 6 了。第三个 let 语句也隐藏了 x ，将之前的值乘以 2 ， x 最终的值是 12 。

隐藏与将变量标记为 mut 是有区别的。当不小心尝试对变量重新赋值时，如果没有使用let 关键字，就会导致编译时错误。通过使用 let ，我们可以用这个值进行一些计算，不过计算完之后变量仍然是不变的。   

mut 与隐藏的另一个区别是，当再次使用 let 时，实际上创建了一个新变量，我们可以改变值的类型，但复用这个名字。

```rust
let spaces = " ";
let spaces = spaces.len();
```

这里允许第一个 spaces 变量是字符串类型，而第二个 spaces 变量，它是一个恰巧与第一个变量同名的崭新变量，是数字类型。  

  

### 数据类型

Rust 是 静态类型（statically typed） 语言，也就是说在编译时就必须知道所有变量的类型。  



#### 标量类型  

Rust 有四种基本的标量类型：**整型**、**浮点型**、**布尔类型**和**字符类型**  

表格 3-1: Rust 中的整型

| 长度    | 有符号 | 无符号 |
| ------- | ------ | ------ |
| 8-bit   | i8     | u8     |
| 16-bit  | i16    | u16    |
| 32-bit  | i32    | u32    |
| 64-bit  | i64    | u64    |
| 128-bit | i128   | u128   |
| arch    | isize  | usize  |



#### 复合类型

复合类型（Compound types） 可以将多个值组合成一个类型。Rust 有两个原生的复合类型：**元组**（tuple） 和**数组**（array） 。

##### 元组  

```rust
let tup: (i32, f64, u8) = (500, 6.4, 1);
let (x, y, z) = tup;
```

为了从元组中获取单个值，可以使用模式匹配（pattern matching） 来解构（destructure） 元组值  

除了使用模式匹配解构外，也可以使用点号（. ） 后跟值的索引来直接访问它们  

```rust
let x: (i32, f64, u8) = (500, 6.4, 1);
let five_hundred = x.0;
let six_point_four = x.1;
```



##### 数组

与元组不同，数组中的每个元素的类型必须相同，Rust 中的数组与一些其他语言中的数组不同，因为 Rust 中的数组是固定长度的：一旦
声明，它们的长度不能增长或缩小。  

可以像这样编写数组的类型：在方括号中包含每个元素的类型，后跟分号，再后跟数组元素的数量。  

```rust
let a = [1, 2, 3, 4, 5];
let a: [i32; 5] = [1, 2, 3, 4, 5];
let a = [3; 5];
```

访问数组元素  

```rust
let first = a[0];
let second = a[1];
```



### 函数

```rust
fn main() {
	let x = 5;
	let y = {
		let x = 3;
		x + 1
	};
	println!("The value of y is: {}", y);
}
```

语句和表达式



### 控制流

```rust
fn main() {
	let number = 6;
	if number % 4 == 0 {
		println!("number is divisible by 4");
	} else if number % 3 == 0 {
		println!("number is divisible by 3");
	} else if number % 2 == 0 {
		println!("number is divisible by 2");
	} else {
		println!("number is not divisible by 4, 3, or 2");
	}
}
```

let中使用`if`

```rust
let number = if condition {
		5
	} else {
		6
	};
```



Rust 有三种循环： `loop` 、 `while` 和 `for` 。  

```rust
fn main() {
	loop {
		println!("again!");
	}
}
```



```rust
fn main() {
	let mut number = 3;
	while number != 0 {
		println!("{}!", number);
		number = number - 1;
	} 
    println!("LIFTOFF!!!");
}
```



````rust
fn main() {
	let a = [10, 20, 30, 40, 50];
	for element in a.iter() {
		println!("the value is: {}", element);
	}
}
````







## 4、认识所有权

### 什么是所有权

Rust 的核心功能（之一） 是 所有权（ownership） 。  

1. Rust 中的每一个值都有一个被称为其 所有者（owner） 的变量。

2. 值有且只有一个所有者。

3. 当所有者（变量） 离开作用域，这个值将被丢弃。  



#### 变量作用域

```rust
{ 						// s 在这里无效, 它尚未声明
	let s = "hello"; 	// 从此处起，s 是有效的
	// 使用 s
} 						// 此作用域已结束，s 不再有效
```



#### 变量与数据交互的方式（一） ：移动  

```rust
let x = 5;
let y = x; //栈上分配，x,y都有效

let s1 = String::from("hello");
let s2 = s1; //堆上分配s1失效
```



#### 变量与数据交互的方式（二） ：克隆  

```rust
let s1 = String::from("hello");
let s2 = s1.clone();
println!("s1 = {}, s2 = {}", s1, s2);
```



#### 所有权与函数  

```rust
fn main() {
	let s = String::from("hello"); // s 进入作用域
	takes_ownership(s); // s 的值移动到函数里 ...
						// ... 所以到这里不再有效
	let x = 5; // x 进入作用域
	makes_copy(x); // x 应该移动函数里，
	// 但 i32 是 Copy 的，所以在后面可继续使用 x
} // 这里, x 先移出了作用域，然后是 s。但因为 s 的值已被移走，
// 所以不会有特殊操作
fn takes_ownership(some_string: String) { // some_string 进入作用域
	println!("{}", some_string);
} // 这里，some_string 移出作用域并调用 `drop` 方法。占用的内存被释放
fn makes_copy(some_integer: i32) { // some_integer 进入作用域
	println!("{}", some_integer);
} // 这里，some_integer 移出作用域。不会有特殊操作
```



### 引用与借用

```rust
fn main() {
    let s1 = String::from("hello");

    let len = calculate_length(&s1);

    println!("The length of '{}' is {}.", s1, len);
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
```

`&` 符号就是 引用，它们允许你使用值但不获取其所有权  

我们将获取引用作为函数参数称为 `借用`（borrowing）  

```rust
fn main() {
    let s = String::from("hello");

    change(&s);
}

fn change(some_string: &String) {
    some_string.push_str(", world");
} // 编译错误：尝试修改借用的值。正如变量默认是不可变的，引用也一样。（默认） 不允许修改引用的值。
```

```rust
fn main() {
    let mut s = String::from("hello");

    change(&mut s);
}

fn change(some_string: &mut String) {
    some_string.push_str(", world");
}
```

创建一个可变引用 `&mut s` 和接受一个可变引用`some_string: &mut String` 。  

可变引用有一个很大的限制：**在特定作用域中的特定数据有且只有一个可变引用**  

```rust
let mut s = String::from("hello");
let r1 = &s; // 没问题
let r2 = &s; // 没问题
let r3 = &mut s; // 大问题
println!("{}, {}, and {}", r1, r2, r3);
```

多个不可变引用是可以的  



### Slice类型

另一个没有所有权的数据类型是 slice。slice 允许你引用集合中一段连续的元素序列，而不用引用整个集合。  

```rust
let s = String::from("hello world");
let hello = &s[0..5];
let world = &s[6..11];
```

```rust
fn first_word(s: &String) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}
```



**字符串字面值就是 slice**  

```rust
let s = "Hello, world!";
```

这里 s 的类型是 `&str` ：它是一个指向二进制程序特定位置的 slice。这也就是为什么字符串字面值是不可变的； `&str` 是一个不可变引用。  



**字符串 slice 作为参数**  

```rust
fn first_word(s: &String) -> &str {
fn first_word(s: &str) -> &str {
```

如果有一个字符串 slice，可以直接传递它。如果有一个 String ，则可以传递整个 String的 slice。定义一个获取字符串 slice 而不是 String 引用的函数使得我们的 API 更加通用并且不会丢失任何功能  

```rust
let my_string = String::from("hello world");
// first_word 中传入 `String` 的 slice
let word = first_word(&my_string[..]);
let my_string_literal = "hello world";
// first_word 中传入字符串字面值的 slice
let word = first_word(&my_string_literal[..]);
// 因为字符串字面值 **就是** 字符串 slice，
// 这样写也可以，即不使用 slice 语法！
let word = first_word(my_string_literal);
```



**其他类型的 slice**  

```rust
let a = [1, 2, 3, 4, 5];
let slice = &a[1..3];
```

这个 slice 的类型是 `&[i32]`   



## 5、结构体





## 6、枚举与匹配模式

### 定义枚举

```rust
enum IpAddrKind {
	V4,
	V6,
}
let four = IpAddrKind::V4;
let six = IpAddrKind::V6;
fn route(ip_type: IpAddrKind) { }
route(IpAddrKind::V4);
route(IpAddrKind::V6);
```



```rust
enum IpAddrKind {
	V4,
	V6,
} 
struct IpAddr {
	kind: IpAddrKind,
	address: String,
} 
let home = IpAddr {
	kind: IpAddrKind::V4,
	address: String::from("127.0.0.1"),
};
let loopback = IpAddr {
	kind: IpAddrKind::V6,
	address: String::from("::1"),
};
```

更简洁方式,仅仅使用枚举并将数据直接放进每一个枚举成员而不是将枚举作为结构体的一部分.我们直接将数据附加到枚举的每个成员上，这样就不需要一个额外的结构体了。    

```rust
enum IpAddr {
	V4(String),
	V6(String),
} 
let home = IpAddr::V4(String::from("127.0.0.1"));
let loopback = IpAddr::V6(String::from("::1"));
```

用枚举替代结构体还有另一个优势：每个成员可以处理不同类型和数量的数据。  

```rust
enum Message {
	Quit,
	Move { x: i32, y: i32 },
	Write(String),
	ChangeColor(i32, i32, i32),
}
//rustlings enums3.rs
match message {
     Message::ChangeColor(a, b, c) => self.change_color((a, b, c)),
     Message::Echo(s) => self.echo(s),
     Message::Move { x, y } => self.move_position(Point { x: x, y: y }),
     Message::Quit => self.quit(),
}
```

结构体和枚举还有另一个相似点：就像可以使用 impl 来为结构体定义方法那样，也可以在枚举上定义方法。这是一个定义于我们 Message 枚举上的叫做 call 的方法：  

```rust
impl Message {
	fn call(&self) {
		// 在这里定义方法体
	}
} 
let m = Message::Write(String::from("hello"));
m.call();
```



### Option 枚举  

Option 类型应用广泛因为它编码了一个非常普遍的场景，即一个值要么有值要么没值  

Rust 并没有很多其他语言中有的空值功能。空值（Null ） 是一个值，它代表没有值  

```rust
enum Option<T> {
	Some(T),
	None,
}
```



### match 控制流运算符  

```rust
enum Coin {
	Penny,
	Nickel,
	Dime,
	Quarter,
} 
fn value_in_cents(coin: Coin) -> u8 {
	match coin {
		Coin::Penny => 1,
		Coin::Nickel => 5,
		Coin::Dime => 10,
		Coin::Quarter => 25,
	}
}
```



**绑定值的模式**  

匹配分支的另一个有用的功能是可以绑定匹配的模式的部分值。这也就是如何从枚举成员中提取值的  

```rust
# #[derive(Debug)]
# enum UsState {
# 	Alabama,
# 	Alaska,
# }
# 
# enum Coin {
# 	Penny,
# 	Nickel,
# 	Dime,
# 	Quarter(UsState),
# }
#
fn value_in_cents(coin: Coin) -> u8 {
	match coin {
		Coin::Penny => 1,
		Coin::Nickel => 5,
		Coin::Dime => 10,
		Coin::Quarter(state) => {
			println!("State quarter from {:?}!", state);
			25
		},
	}
}
```



**匹配 Option<T>**  

```rust
fn plus_one(x: Option<i32>) -> Option<i32> {
	match x {
		None => None,
		Some(i) => Some(i + 1),
	}
} 
let five = Some(5);
let six = plus_one(five);
let none = plus_one(None);
```

Rust 中的匹配是 穷尽的（exhaustive） ：必须穷举到最后的可能性来使代码有效  .



**_ 通配符**  

```rust
let some_u8_value = 0u8;
match some_u8_value {
	1 => println!("one"),
	3 => println!("three"),
	5 => println!("five"),
	7 => println!("seven"),
	_ => (),
}
```



### if let 简单控制流  

if let 语法让我们以一种不那么冗长的方式结合 if 和 let ，来处理**只匹配一个模式**的值而忽略其他模式的情况  

```rust
	let some_u8_value = Some(0u8);
	match some_u8_value {
		Some(3) => println!("three"),
		_ => (),
	}


	if let Some(3) = some_u8_value {
		println!("three");
	}
```





## 7、package、crate、mod



## 8、常见集合

### vector

新建一个vector  

```rust
let v: Vec<i32> = Vec::new();
let v = vec![1, 2, 3];
```

增加元素`push`

```rust
let mut v = Vec::new();
v.push(5);
v.push(6);
v.push(7);
v.push(8);
```

读取元素

访问 vector 中一个值的两种方式，索引语法或者 get 方法  

```rust
let v = vec![1, 2, 3, 4, 5];
let third: &i32 = &v[2];
v.get(2)
```

是必须用`usize`类型的值来索引

```rust
let j: i32 = 0;
v[j];

note: the type `collections::vec::Vec<_>` cannot be indexed by `i32`
```



遍历 vector  

```rust
let v = vec![100, 32, 57];
for i in &v {
	println!("{}", i);
}
```

使用枚举来存储多种类型

```rust
enum SpreadsheetCell {
	Int(i32),
	Float(f64),
	Text(String),
} 
let row = vec![
	SpreadsheetCell::Int(3),
	SpreadsheetCell::Text(String::from("blue")),
	SpreadsheetCell::Float(10.12),
];
```



### 字符串

Rust 的核心语言中只有一种字符串类型： `str` ，字符串 slice，它通常以被借用的形式出现， `&str` 。  

**新建字符串**

可以使用 `to_string` 方法，它能用于任何实现了 `Display trait` 的类型，字符串字面值也实现了它  

```rust
let mut s = String::new();
let data = "initial contents";
let s = data.to_string();
let s = "initial contents".to_string();
let s = String::from("initial contents");
```



**更新字符串**

使用 `push_str` 和 `push` 附加字符串  

```rust
let mut s = String::from("foo");
s.push_str("bar");
```

`push` 方法被定义为获取一个单独的字符作为参数，并附加到 String 中  

```rust
let mut s = String::from("lo");
s.push('l');
```

使用 `+` 运算符或 `format!` 宏拼接字符串  

```rust
let s1 = String::from("Hello, ");
let s2 = String::from("world!");
let s3 = s1 + &s2; // 注意 s1 被移动了，不能继续使用
```

```rust
let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = String::from("toe");
let s = format!("{}-{}-{}", s1, s2, s3);
```



**索引字符串**

Rust 的字符串不支持索引  

String 是一个 Vec<u8> 的封装  

```rust
let len = String::from("Hola").len(); // 4
let len = String::from("Здравствуйте").len(); // 不是12是24
```



**字符串 slice**  

索引字符串通常是一个坏点子，因为字符串索引应该返回的类型是不明确的：字节值、字符、字形簇或者字符串 slice。因此，如果你真的希望使用索引创建字符串 slice 时，Rust 会要求你更明确一些    

```rust
let hello = "Здравствуйте";
let s = &hello[0..4]; //“Зд”
```



**遍历字符串**  

如果你需要操作单独的 Unicode 标量值，最好的选择是使用 `chars` 方法  

```rust
for c in "नमİते".chars() {
	println!("{}", c);
}
```

`bytes` 方法返回每一个原始字节  

```rust
for b in "नमİते".bytes() {
	println!("{}", b);
}
```



### HashMap



**新建HashMap**

```rust
use std::collections::HashMap;
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);
```

类似于 vector，哈希 map 是同质的：所有的键必须是相同类型，值也必须都是相同类型。  

另一个构建哈希 map 的方法是使用一个元组的 vector 的 `collect` 方法  

```rust
use std::collections::HashMap;
let teams = vec![String::from("Blue"), String::from("Yellow")];
let initial_scores = vec![10, 50];
let scores: HashMap<_, _> = teams.iter().zip(initial_scores.iter()).collect();
```



**访问HashMap**   

可以通过 `get` 方法并提供对应的键  

```rust
use std::collections::HashMap;
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);
let team_name = String::from("Blue");
let score = scores.get(&team_name);
```

可以使用与 vector 类似的方式来遍历  

```rust
use std::collections::HashMap;
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);
for (key, value) in &scores {
	println!("{}: {}", key, value);
}
```



**更新HashMap**

```rust
use std::collections::HashMap;
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Blue"), 25);
println!("{:?}", scores);
```



**只在键没有对应值时插入**  

我们经常会检查某个特定的键是否有值，如果没有就插入一个值。为此哈希 map 有一个特有的 API，叫做 `entry` ，它获取我们想要检查的键作为参数  

```rust
use std::collections::HashMap;
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.entry(String::from("Yellow")).or_insert(50);
scores.entry(String::from("Blue")).or_insert(50);
println!("{:?}", scores);
```



**根据旧值更新一个值**  

```rust
use std::collections::HashMap;
let text = "hello world wonderful world";
let mut map = HashMap::new();
for word in text.split_whitespace() {
	let count = map.entry(word).or_insert(0);
	*count += 1;
} 
println!("{:?}", map);
```







## 9、错误处理

## 10、泛型、trait和生命周期



### trait

`trait` 告诉 Rust 编译器某个特定类型拥有可能与其他类型共享的功能。可以通过 `trait` 以一种抽象的方式定义共享的行为。可以使用 `trait bounds` 指定泛型是任何拥有特定行为的类型  



**定义trait**

```rust
pub trait Summary {
	fn summarize(&self) -> String;
}
```



**实现 trait**  

```rust

#![allow(unused_variables)]
fn main() {
pub trait Summary {
    fn summarize(&self) -> String;
}

pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
}

```

实现 trait 时需要注意的一个限制是，只有当 trait 或者要实现 trait 的类型位于 crate 的本地作用域时，才能为该类型实现 trait。

但是不能为外部类型实现外部 trait。  



**默认实现**  

```rust
pub trait Summary {
	fn summarize(&self) -> String {
		String::from("(Read more...)")
	}
}
```

如果想要对 NewsArticle 实例使用这个默认实现，而不是定义一个自己的实现，则可以通过`impl Summary for NewsArticle {}` 指定一个空的 impl 块。  

默认实现允许调用相同 trait 中的其他方法，哪怕这些方法没有默认实现。  

```rust
pub trait Summary {
	fn summarize_author(&self) -> String;
	fn summarize(&self) -> String {
		format!("(Read more from {}...)", self.summarize_author())
	}
}
```

为了使用这个版本的 Summary ，只需在实现 trait 时定义 summarize_author 即可  	



**trait 作为参数**  

```rust
pub fn notify(item: impl Summary) {
	println!("Breaking news! {}", item.summarize());
}
```

对于 item 参数，我们指定了 impl 关键字和 trait 名称，而不是具体的类型。该参数支持任何实现了指定 trait 的类型。  



**Trait Bound 语法**  

`impl Trait` 语法适用于直观的例子，它不过是一个较长形式的语法糖。这被称为 `trait bound`，这看起来像：  

```rust
pub fn notify<T: Summary>(item: T) {
	println!("Breaking news! {}", item.summarize());
}
```

impl Trait 很方便，适用于短小的例子。trait bound 则适用于更复杂的场景  

```rust
pub fn notify(item1: impl Summary, item2: impl Summary) {
pub fn notify<T: Summary>(item1: T, item2: T) { 
```



**通过 + 指定多个 trait bound**  

```rust
pub fn notify(item: impl Summary + Display) {
pub fn notify<T: Summary + Display>(item: T) {
```



**通过 where 简化 trait bound**  

```rust
fn some_function<T: Display + Clone, U: Clone + Debug>(t: T, u: U) -> i32 {
    
fn some_function<T, U>(t: T, u: U) -> i32
	where T: Display + Clone,
		U: Clone + Debug
{
```



**返回实现了 trait 的类型**  

```rust
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("horse_ebooks"),
        content: String::from(
            "of course, as you probably already know, people",
        ),
        reply: false,
        retweet: false,
    }
}
```







## 11、测试

### 编写测试

Rust 提供的专门用来编写测试的功能： `test` 属性、一些宏和 `should_panic` 属性。  

Rust 中的测试就是一个带有 `test` 属性注解的函数。`属性（attribute`） 是关于 Rust 代码片段的元数据.

为了将一个函数变成测试函数，需要在 fn 行之前加上 #[test] 。当使用 cargo test 命令运行测试时，Rust 会构建一个测试执行程序用来调用标记了 test 属性的函数，并报告每一个测试是通过还是失败。  

```rust
#[cfg(test)]
mod tests {
	#[test]
	fn it_works() {
		assert_eq!(2 + 2, 4);
	}
}
```

因为也可以在 tests 模块中拥有非测试的函数来帮助我们建立通用场景或进行常见操作，所以需要使用 #[test] 属性标明哪些函数是测试  

**panic!宏**

```rust
#[test]
fn another() {
	panic!("Make this test fail");
}
```

**自定义失败信息**  

可以向 `assert!` 、 `assert_eq!` 和 `assert_ne!` 宏传递一个可选的失败信息参数，可以在测试失败时将自定义失败信息一同打印出来  

```rust
#[test]
fn greeting_contains_name() {
	let result = greeting("Carol");
	assert!(
		result.contains("Carol"),
		"Greeting did not contain name, value was `{}`", result
	);
}
```



**should_panic** 

除了检查代码是否返回期望的正确的值之外，检查代码是否按照期望处理错误也是很重要的  

```rust
pub struct Guess {
	value: i32,
} 
impl Guess {
	pub fn new(value: i32) -> Guess {
		if value < 1 || value > 100 {
			panic!("Guess value must be between 1 and 100, got {}.", value);
		} 
        Guess {
			value
		}
	}
}
#[cfg(test)]
mod tests {
use super::*;
	#[test]
	#[should_panic]
	fn greater_than_100() {
		Guess::new(200);
	}
}
```

`#[should_panic]` 属性位于 `#[test]` 之后，对应的测试函数之前  

一些不是我们期望的原因而导致 panic 时也会通过。为了使should_panic 测试结果更精确，我们可以给 should_panic 属性增加一个可选的 `expected`参数  

```rust
if value < 1 {
	panic!("Guess value must be greater than or equal to 1, got {}.",
	value);
} else if value > 100 {
	panic!("Guess value must be less than or equal to 100, got {}.",
	value);
}

#[test]
#[should_panic(expected = "Guess value must be less than or equal to 100")]
fn greater_than_100() {
	Guess::new(200);
}
```



**将 Result<T, E> 用于测试**  

```rust
#[cfg(test)]
mod tests {
	#[test]
	fn it_works() -> Result<(), String> {
		if 2 + 2 == 4 {
			Ok(())
		} else {
			Err(String::from("two plus two does not equal four"))
		}
	}
}
```

不能对这些使用 Result<T, E> 的测试使用 #[should_panic] 注解。相反应该在测试失败时直接返回 Err 值。  



### 控制测试如何运行  

运行 `cargo test --help` 会提示 cargo test 的有关参数，而运行 `cargo test -- --help` 可以提示在分隔符 -- 之后使用的有关参数。  

**并行或连续的运行测试**  

当运行多个测试时， Rust 默认使用线程来并行运行 。你应该确保测试不能相互依赖，或依赖任何共享的状态  

如果你不希望测试并行运行，或者想要更加精确的控制线程的数量，可以传递 `--test-threads` 参数和希望使用线程的数量给测试二进制文件  

```sh
$ cargo test -- --test-threads=1
```



**显示函数输出**  

如果你希望也能看到通过的测试中打印的值，截获输出的行为可以通过 `--nocapture` 参数来禁用：  

```sh
$ cargo test -- --nocapture
```



**通过指定名字来运行部分测试**  

```rust
#[test]
fn add_two_and_two() {
	assert_eq!(4, add_two(2));
} 
#[test]
fn add_three_and_two() {
	assert_eq!(5, add_two(3));
} 
#[test]
fn one_hundred() {
	assert_eq!(102, add_two(100));
}
```

```sh
$ cargo test one_hundred 
```

我们可以指定部分测试的名称，任何名称匹配这个名称的测试会被运行  

```sh
$ cargo test add
...running 2 tests
```



**忽略某些测试**  

```rust
#[test]
#[ignore]
fn expensive_test() {
	// 需要运行一个小时的代码
}
```

如果我们只希望运行被忽略的测试，可以使用 `cargo test -- --ignored`   



### 测试的组织结构  

Rust 社区倾向于根据测试的两个主要分类来考虑问题：`单元测试（unit tests）` 与 `集成测试（integration tests）`  

**单元测试**

单元测试与他们要测试的代码共同存放在位于 src 目录下相同的文件中。规范是在每个文件中创建包含测试函数的 tests 模块，并使用 cfg(test)
标注模块。  

测试模块的 `#[cfg(test)]` 注解告诉 Rust 只在执行 cargo test 时才编译和运行测试代码，而在运行 cargo build 时不这么做  

Rust 的私有性规则确实允许你测试私有函数  

**集成测试**

集成测试对于你需要测试的库来说完全是外部的。同其他使用库的代码一样使用库文件，也就是说它们只能调用一部分库中的公有 API  

为了编写集成测试，需要在项目根目录创建一个 `tests` 目录，与 `src` 同级  。tests 文件夹在Cargo 中是一个特殊的文件夹， Cargo 只会在运行 cargo test 时编译这个目录中的文件  

```rust
// tests/integration_test.rs
use adder;

#[test]
fn it_adds_two() {
	assert_eq!(4, adder::add_two(2));
}
```





















## 12、一个I/O项目

## 13、迭代器与闭包  

### 闭包

Rust 的 闭包（closures） 是可以保存进变量或作为参数传递给其他函数的匿名函数。 不同于函数，闭包允许捕获调用者作用域中的值。  

```rust
    let expensive_closure = |num: u32| -> u32 {
        println!("calculating slowly...");
        thread::sleep(Duration::from_secs(2));
        num
    };
```



闭包的定义以一对竖线（| ） 开始，在竖线中指定闭包的参数；这个闭包有一个参数 num ；如果有多于一个参数，可以使用逗号分隔，比如
|param1, param2| 。  

参数之后是存放闭包体的大括号 —— 如果闭包体只有一行则大括号是可以省略的。大括号之后闭包的结尾，需要用于 let 语句的分号。  

调用闭包类似于调用函数；指定存放闭包定义的变量名并后跟包含期望使用的参数的括号  

```rust
expensive_closure(intensity)
```



闭包不要求像 fn 函数那样在参数和返回值上注明类型  

```rust
fn add_one_v1 (x: u32) -> u32 { x + 1 }
let add_one_v2 = |x: u32| -> u32 { x + 1 };
let add_one_v3 = |x| { x + 1 };
let add_one_v4 = |x| x + 1 ;
```



#### 带有泛型和 Fn trait 的闭包 



### 迭代器

迭代器模式允许你对一个项的序列进行某些处理。迭代器（iterator） 负责遍历序列中的每一项和决定序列何时结束的逻辑。当使用迭代器时，我们无需重新实现这些逻辑。  

迭代器是 惰性的（lazy） ，这意味着在调用方法使用迭代器之前它都不会有效果  

迭代器都实现了一个叫做 `Iterator` 的定义于标准库的 trait  

```rust
pub trait Iterator {
	type Item;
	fn next(&mut self) -> Option<Self::Item>;
	// 此处省略了方法的默认实现
}
```

type Item 和 Self::Item ，他们定义了 trait 的 `关联类型（associated type）`  

next 是 Iterator 实现者被要求定义的唯一方法  



**消费迭代器的方法**  

这些调用 next 方法的方法被称为 `消费适配器（consuming adaptors）` ，因为调用他们会消耗迭代器。一个消费适配器的例子是 `sum` 方法。这个方法获取迭代器的所有权并反复调用next 来遍历迭代器，因而会消费迭代器。  

```rust
#[test]
fn iterator_sum() {
	let v1 = vec![1, 2, 3];
	let v1_iter = v1.iter();
	let total: i32 = v1_iter.sum();
	assert_eq!(total, 6);
}
```

调用 sum 之后不再允许使用 `v1_iter` 因为调用 sum 时它会获取迭代器的所有权。  



**产生其他迭代器的方法**  

Iterator trait 中定义了另一类方法，被称为 `迭代器适配器（iterator adaptors）` ，他们允许我们将当前迭代器变为不同类型的迭代器。可以链式调用多个迭代器适配器  

调用迭代器适配器方法 map 的例子，该 map 方法使用闭包来调用每个元素以生成新的迭代器。  

```rust
let v1: Vec<i32> = vec![1, 2, 3];
let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();
assert_eq!(v2, vec![2, 3, 4]);
```



**使用闭包获取环境**  

```rust
#[derive(PartialEq, Debug)]
struct Shoe {
	size: u32,
	style: String,
} 
fn shoes_in_my_size(shoes: Vec<Shoe>, shoe_size: u32) -> Vec<Shoe> {
	shoes.into_iter()
		.filter(|s| s.size == shoe_size)
		.collect()
}
```



**自定义迭代器**  

...



## 14、Cargo 和 Crates.io  

## 15、智能指针  

指针 （pointer） 是一个包含内存地址的变量的通用概念。  Rust 中最常见的指针是第四章介绍的 引用（reference）  

智能指针（smart pointers） 是一类数据结构，他们的表现类似指针，但是也拥有额外的元数据和功能  

在 Rust 中，普通引用和智能指针的一个额外的区别是引用是一类只借用数据的指针；相反，在大部分情况下，智能指针 拥有 他们指向的数据。  

本章并不会覆盖所有现存的智能指针。这里将会讲到的是来自标准库中最常用的一些：  

- `Box<T>` ，用于在堆上分配值
- `Rc<T>` ，一个引用计数类型，其数据可以有多个所有者
- `Ref<T>` 和 `RefMut<T>` ，通过 RefCell<T> 访问，一个在运行时而不是在编译时执行借用规则的类型。



### 使用 Box <T> 指向堆上的数据  

最简单直接的智能指针是 box，其类型是 Box<T> 。 box 允许你将一个值放在堆上而不是栈上。留在栈上的则是指向堆数据的指针  

除了数据被储存在堆上而不是栈上之外，box 没有性能损失。不过也没有很多额外的功能。它们多用于如下场景：

- 当有一个在编译时未知大小的类型，而又想要在需要确切大小的上下文中使用这个类型值的时候
- 当有大量数据并希望在确保数据不被拷贝的情况下转移所有权的时候
- 当希望拥有一个值并只关心它的类型是否实现了特定 trait 而不是其具体类型的时候  



**使用 Box<T> 在堆上储存数据**  

```rust
fn main() {
	let b = Box::new(5);
	println!("b = {}", b);
}
```

这里定义了变量 b ，其值是一个指向被分配在堆上的值 5 的 Box 。这个程序会打印出 b= 5 ；在这个例子中，我们可以像数据是储存在栈上的那样访问 box 中的数据  



**Box 允许创建递归类型**  

Rust 需要在编译时知道类型占用多少空间。一种无法在编译时知道大小的类型是 递归类型**（recursive type）** ，其值的一部分可以是相同类型的另一个值。这种值的嵌套理论上可以无限的进行下去，所以 Rust 不知道递归类型需要多少空间。  

```rust
enum List {
	Cons(i32, List),
	Nil,
}

use crate::List::{Cons, Nil};
fn main() {
	let list = Cons(1, Cons(2, Cons(3, Nil)));
} //错误表明这个类型 “有无限的大小”。其原因是 List 的一个成员被定义为是递归的：它
//直接存放了另一个相同类型的值。
```

使用 List 枚举储存列表 1, 2, 3  

修改为

```rust
enum List {
	Cons(i32, Box<List>),
	Nil,
} 

use crate::List::{Cons, Nil};

fn main() {
	let list = Cons(1,Box::new(Cons(2,Box::new(Cons(3,Box::new(Nil))))));
}
```

Box<T> 类型是一个智能指针，因为它实现了 `Deref trai`t，它允许 Box<T> 值被当作引用对待。当 Box<T> 值离开作用域时，由于 Box<T> 类型 `Drop trait` 的实现，box 所指向的堆数据也会被清除。  



### 通过 Deref trait 将智能指针当作常规引用处理  

实现 Deref trait 允许我们重载 解引用运算符（dereference operator） `*` （与乘法运算符或通配符相区别）   

**像引用一样使用 Box<T>**  

可以使用 Box<T> 代替引用，解引用运算符也一样能工作  

```rust
fn main() {
	let x = 5;
	let y = Box::new(x);
	assert_eq!(5, x);
	assert_eq!(5, *y);
}
```



### 自定义智能指针  

从根本上说， Box<T> 被定义为包含一个元素的元组结构体  

```rust
struct MyBox<T>(T);

impl<T> MyBox<T> {
	fn new(x: T) -> MyBox<T> {
		MyBox(x)
	}
}
```



**MyBox<T> 上的 Deref 实现**  

`Deref trait`，由标准库提供，要求实现名为 `deref` 的方法  

```rust
use std::ops::Deref;

# struct MyBox<T>(T);

impl<T> Deref for MyBox<T> {
	type Target = T;
	fn deref(&self) -> &T {
		&self.0
	}
}
```

deref 方法体中写入了 `&self.0` ，这样 deref 返回了我希望通过 `*` 运算符访问的值的引用。  

当我们输入 `*y` 时，Rust 事实上在底层运行了如下代码：  `*(y.deref())`  Rust 将 * 运算符替换为先调用 deref 方法再进行普通解引用的操作  



**解引用强制多态（deref coercions）**  

解引用强制多态（deref coercions） 是 Rust 在函数或方法传参上的一种便利。其将实现了`Deref` 的类型的引用转换为原始类型通过 Deref 所能够转换的类型的引用  

```rust
# use std::ops::Deref;
# 
# struct MyBox<T>(T);
# 
# impl<T> MyBox<T> {
# 	fn new(x: T) -> MyBox<T> {
# 		MyBox(x)
# 	}
# }
# 
# impl<T> Deref for MyBox<T> {
# 	type Target = T;
# 
#	fn deref(&self) -> &T {
# 		&self.0
# 	}
# }
# 
# fn hello(name: &str) {
# 	println!("Hello, {}!", name);
# }
# 
fn main() {
	let m = MyBox::new(String::from("Rust"));
	hello(&m);
}
```

这里使用 `&m` 调用 hello 函数，其为 MyBox<String> 值的引用  

因为在`MyBox<T>` 上实现了 `Deref trait`，Rust 可以通过 `deref` 调用将 `&MyBox<String>` 变为 `&String` 。  



**解引用强制多态如何与可变性交互**  

类似于如何使用 Deref trait 重载不可变引用的 * 运算符，Rust 提供了 `DerefMut trait` 用于重载可变引用的 * 运算符。  

Rust 在发现类型和 trait 实现满足三种情况时会进行解引用强制多态：

- 当 T: Deref<Target=U> 时从 &T 到 &U 。
- 当 T: DerefMut<Target=U> 时从 &mut T 到 &mut U 。
- 当 T: Deref<Target=U> 时从 &mut T 到 &U 。  



### 使用 Drop Trait 运行清理代码  

对于智能指针模式来说第二个重要的 trait 是 Drop ，其允许我们在值要离开作用域时执行一些代码。  Box<T> 自定义了 Drop 用来释放 box 所指向的堆空间  

指定在值离开作用域时应该执行的代码的方式是实现 Drop trait。 Drop trait 要求实现一个叫做 drop 的方法，它获取一个 self 的可变引用。  

```rust
struct CustomSmartPointer {
	data: String,
} 

impl Drop for CustomSmartPointer {
	fn drop(&mut self) {
		println!("Dropping CustomSmartPointer with data `{}`!", self.data);
	}
} 
fn main() {
	let c = CustomSmartPointer { data: String::from("my stuff") };
	let d = CustomSmartPointer { data: String::from("other stuff") };
	println!("CustomSmartPointers created.");
}

CustomSmartPointers created.
Dropping CustomSmartPointer with data `other stuff`!
Dropping CustomSmartPointer with data `my stuff`!
```



**通过 std::mem::drop 提早丢弃值**  

然而，有时你可能需要提早清理某个值。一个例子是当使用智能指针管理锁时；你可能希望强制运行 drop 方法来释放锁以便作用域中的其
他代码可以获取锁。Rust 并不允许我们主动调用 Drop trait 的 drop 方法；当我们希望在作用域结束之前就强制释放变量的话，我们应该使用的是由标准库提供的 `std::mem::drop` 。  

```rust
fn main() {
let c = CustomSmartPointer { data: String::from("some data") };
	println!("CustomSmartPointer created.");
	drop(c); //std::mem::drop 位于 prelude
	println!("CustomSmartPointer dropped before the end of main.");
}
```



### Rc<T> 引用计数智能指针  

大部分情况下所有权是非常明确的：可以准确地知道哪个变量拥有某个值。然而，有些情况单个值可能会有多个所有者。例如，在图数据结构中，多个边可能指向相同的结点，而这个结点从概念上讲为所有指向它的边所拥有。结点直到没有任何边指向它之前都不应该被清理。  

为了启用多所有权，Rust 有一个叫做 Rc<T> 的类型。其名称为 引用计数（referencecounting） 的缩写。引用计数意味着记录一个值引用的数量来知晓这个值是否仍在被使用。如果某个值有零个引用，就代表没有任何有效引用并可以被清理。  

注意 `Rc<T>` 只能用于单线程场景；第十六章并发会涉及到如何在多线程程序中进行引用计数  



**使用 Rc<T> 共享数据**  

```rust
enum List {
	Cons(i32, Rc<List>),
	Nil,
} 
use crate::List::{Cons, Nil};
use std::rc::Rc;

fn main() {
	let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
	let b = Cons(3, Rc::clone(&a));
	let c = Cons(4, Rc::clone(&a));
}
```

需要使用 use 语句将 Rc<T> 引入作用域，因为它不在 prelude 中。在 main 中创建了存放5 和 10 的列表并将其存放在 a 的新的 Rc<List> 中。接着当创建 b 和 c 时，调用`Rc::clone` 函数并传递 a 中 Rc<List> 的引用作为参数。  **克隆 Rc<T> 会增加引用计数**



### RefCell<T> 和内部可变性模式  

**内部可变性（Interior mutability）** 是 Rust 中的一个设计模式，它允许你即使在有不可变引用时也可以改变数据，这通常是借用规则所不允许的。  



**通过 RefCell<T> 在运行时检查借用规则**  

对于引用和 Box<T> ，借用规则的不可变性作用于编译时。对于 RefCell<T> ，这些不可变性作用于 运行时。对于引用，如果违反这些规则，会得到一个编译错误。而对于RefCell<T> ，如果违反这些规则程序会 panic 并退出。  

RefCell<T> 正是用于当你确信代码遵守借用规则，而编译器不能理解和确定的时候。  

RefCell<T> 只能用于单线程场景  







## 16、并发  

## 17、面向对象特性  

## 18、模式用来匹配值的结构  

## 19、高级特征  

## 20、构建多线程 web server  







# RustRefence

## Rustc

### 什么是rustc

rustc是项目本身提供的Rust编程语言的编译器。编译器将您的源代码作为库或可执行文件获取并生成二进制代码。

大多数Rust程序员并不直接调用rustc，而是通过Cargo来执行。一切尽在rustc！如果您想了解货运如何称呼rustc，可以

```sh
$ cargo build --verbose
```

它将打印出每个rustc调用。 本书可以帮助您了解这些选项的作用。 此外，尽管大多数Rustaceans使用Cargo，但并非全部使用Cargo：有时，他们将rustc集成到其他构建系统中。 本书应提供您所需的所有选项的指南。

请注意，我们只将rustc传递给*crate root*,，而不传递我们希望编译的每个文件。例如，如果我们的main.rs如下所示：

```rust
mod foo;

fn main() {
    foo::hello();
}
```

要编译此代码，我们将运行以下命令：

```sh
$ rustc main.rs
```



### 命令行参数

|                     |      |
| ------------------- | ---- |
| -h / --help         |      |
| -cfg                |      |
| -L                  |      |
| -l                  |      |
| --crate-type        |      |
| --crate-name        |      |
| --edition           |      |
| --emit              |      |
| --print             |      |
| -g                  |      |
| -O                  |      |
| -o                  |      |
| --out-dir           |      |
| --test              |      |
| --target            |      |
| -W                  |      |
| -A                  |      |
| -D                  |      |
| -F                  |      |
| -Z                  |      |
| --cap-lint          |      |
| -C /--codegen       |      |
| -V/--version        |      |
| -v / --verbose      |      |
| --extern            |      |
| --sysroot           |      |
| --error-format      |      |
| --color             |      |
| --remap-path-prefix |      |

















## cargo.toml

`cargo.toml`和`cargo.lock`是cargo项目代码管理的核心两个文件，cargo工具的所有活动均基于这两个文件。

`cargo.toml`是cargo特有的项目数据描述文件，对于猿们而言，`cargo.toml`文件存储了项目的所有信息，它直接面向rust猿，猿们如果想让自己的rust项目能够按照期望的方式进行构建、测试和运行，那么，必须按照合理的方式构建’cargo.toml’。

而`cargo.lock`文件则不直接面向猿，猿们也不需要直接去修改这个文件。lock文件是cargo工具根据同一项目的toml文件生成的项目依赖详细清单文件，所以我们一般不用不管他，只需要对着`cargo.toml`文件撸就行了。

### package段落

[package]段落描述了软件开发者对本项目的各种元数据描述信息，例如[name]字段定义了项目的名称，[version]字段定义了项目的当前版本，[authors]定义了该项目的所有作者，当然，[package]段落不仅仅包含这些字段，[package]段落的其他可选字段详见cargo参数配置章节。

```rust
[package]
 # 软件包名称，如果需要在别的地方引用此软件包，请用此名称。
name = "hello_world"

# 当前版本号，这里遵循semver标准，也就是语义化版本控制标准。
version = "0.1.0"    # the current version, obeying semver

# 软件所有作者列表
authors = ["you@example.com"]

# 非常有用的一个字段，如果要自定义自己的构建工作流，
# 尤其是要调用外部工具来构建其他本地语言（C、C++、D等）开发的软件包时。
# 这时，自定义的构建流程可以使用rust语言，写在"build.rs"文件中。
build = "build.rs"

# 显式声明软件包文件夹内哪些文件被排除在项目的构建流程之外，
# 哪些文件包含在项目的构建流程中
exclude = ["build/**/*.o", "doc/**/*.html"]
include = ["src/**/*", "Cargo.toml"]

# 当软件包在向公共仓库发布时出现错误时，使能此字段可以阻止此错误。
publish = false

# 关于软件包的一个简短介绍。
description = "..."

# 下面这些字段标明了软件包仓库的更多信息
documentation = "..."
homepage = "..."
repository = "..."

# 顾名思义，此字段指向的文件就是传说中的ReadMe，
# 并且，此文件的内容最终会保存在注册表数据库中。
readme = "..."

# 用于分类和检索的关键词。
keywords = ["...", "..."]

# 软件包的许可证，必须是cargo仓库已列出的已知的标准许可证。
license = "..."

# 软件包的非标许可证书对应的文件路径。
license-file = "..."
```



### 定义项目依赖

在cargo的toml文件描述中，主要通过各种依赖段落来描述该项目的各种依赖项。toml中常用的依赖段落包括一下几种：

- 基于rust官方仓库crates.io，通过版本说明来描述：

- 基于项目源代码的git仓库地址，通过URL来描述：

- 基于本地项目的绝对路径或者相对路径，通过类Unix模式的路径来描述：

  这三种形式具体写法如下：

```rust
[dependencies]
typemap = "0.3"
plugin = "0.2*"
hammer = { version = "0.5.0"}
color = { git = "https://github.com/bjz/color-rs" }
geometry = { path = "crates/geometry" }
```

上述例子中，2-4行为方法一的写法，第5行为方法二的写法，第6行为方法三的写法。



### 定义集成测试用例

cargo另一个重要的功能，即将软件开发过程中必要且非常重要的测试环节进行集成，并通过代码属性声明或者toml文件描述来对测试进行管理。其中，单元测试主要通过在项目代码的测试代码部分前用`#[test]`属性来描述，而集成测试，则一般都会通过toml文件中的`[[test]]`段落进行描述。

例如，假设集成测试文件均位于tests文件夹下，则toml可以这样来写：

```rust
[[test]]
name = "testinit"
path = "tests/testinit.rs"
[[test]]
name = "testtime"
path = "tests/testtime.rs"
```

上述例子中，name字段定义了集成测试的名称，path字段定义了集成测试文件相对于本toml文件的路径。

需要注意的是:

- 如果没有在Cargo.toml里定义集成测试的入口，那么tests目录(不包括子目录)下的每个rs文件被当作集成测试入口.
- 如果在Cargo.toml里定义了集成测试入口，那么定义的那些rs就是入口，不再默认指定任何集成测试入口.



### 定义项目示例和可执行程序

上面我们介绍了cargo项目管理中常用的三个功能，还有两个经常使用的功能：example用例的描述以及bin用例的描述。其描述方法和test用例描述方法类似。不过，这时候段落名称’[[test]]’分别替换为：’[[example]]’或者’[[bin]]’。例如：

```rust
[[example]]
name = "timeout"
path = "examples/timeout.rs"
[[bin]]
name = "bin1"
path = "bin/bin1.rs"
```

对于’[[example]]’和’[[bin]]’段落中声明的examples和bins，需要通过’cargo run —example NAME’或者’cargo run —bin NAME’来运行，其中NAME对应于你在name字段中定义的名称。



## Attributes

属性是应用于某些模块、crate 或项的元数据（metadata）。

当属性作用于整个 crate 时，它们的语法为 `#![crate_attribute]`，当它们用于模块 或项时，语法为 `#[item_attribute]`（注意少了感叹号 `!`）。

属性可以接受参数，有不同的语法形式：

- `#[attribute = "value"]`
- `#[attribute(key = "value")]`
- `#[attribute(value)]`



编译器提供了 `dead_code`（死代码，无效代码）[*lint*](https://en.wikipedia.org/wiki/Lint_(software))，这会对未使用的函数 产生警告。可以用一个**属性**来禁用这个 lint。

```rust
fn used_function() {}

// `#[allow(dead_code)]` 属性可以禁用 `dead_code` lint
#[allow(dead_code)]
fn unused_function() {}

fn noisy_unused_function() {}
// 改正 ^ 增加一个属性来消除警告

fn main() {
    used_function();
}
```



`crate_type` 属性可以告知编译器 crate 是一个二进制的可执行文件还是一个 库（甚至是哪种类型的库），`crate_name` 属性可以设定 crate 的名称。不过，一定要注意在使用 cargo 时，这两种类型时都**没有**作用。由于大多数 Rust 工程都使用 cargo，这意味着 `crate_type` 和 `crate_name` 的作用事实上很有限。

```rust
// 这个 crate 是一个库文件
#![crate_type = "lib"]
// 库的名称为 “rary”
#![crate_name = "rary"]

pub fn public_function() {
    println!("called rary's `public_function()`");
}

fn private_function() {
    println!("called rary's `private_function()`");
}

pub fn indirect_access() {
    print!("called rary's `indirect_access()`, that\n> ");

    private_function();
}
```



条件编译可能通过两种不同的操作符实现：

- `cfg` 属性：在属性位置中使用 `#[cfg(...)]`
- `cfg!` 宏：在布尔表达式中使用 `cfg!(...)`

```rust
// 这个函数仅当目标系统是 Linux 的时候才会编译
#[cfg(target_os = "linux")]
fn are_you_on_linux() {
    println!("You are running linux!")
}

// 而这个函数仅当目标系统 **不是** Linux 时才会编译
#[cfg(not(target_os = "linux"))]
fn are_you_on_linux() {
    println!("You are *not* running linux!")
}

fn main() {
    are_you_on_linux();
    
    println!("Are you sure?");
    if cfg!(target_os = "linux") {
        println!("Yes. It's definitely linux!");
    } else {
        println!("Yes. It's definitely *not* linux!");
    }
}
```

有部分条件如 `target_os` 是由 `rustc` 隐式地提供的，但是自定义条件必须使用 `--cfg` 标记来传给 `rustc`。

```rust
#[cfg(some_condition)]
fn conditional_function() {
    println!("condition met!")
}

fn main() {
    conditional_function();
}

```





















