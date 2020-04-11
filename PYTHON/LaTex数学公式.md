# LaTex数学公式

###  公式排版

- 行内公式：用 `$`包裹公式
- 独立公式：用 `$$`包裹公式。

### 换行

**\\\\** 表示换行  

###  角标（上下标）

**上标命令**：`^{}`
**下标命令**：`_{}`
如果角标为单个字符，可以不使用花括号；否则必须使用花括号。

### 分式

**分式命令**：`$\frac{分子}{分母}$`

### 根式

**二次根式命令**：`$\sqrt{表达式}$`
**N次根式命令**：`$\sqrt[n]{表达式}$`

### 对数

`\log_ax` $\log_ax$

`\ln x` $\ln x$ 

`\lg x` $\lg x$ 

### 求和、积分、极限

求和命令：`\sum_{k=1}^nf(x)`：$\sum_{k=1}^nf(x)$

积分命令：`\int_a^b表达式`:$\int_a^b$

极限命令：`\lim_{x\to\infty}\frac{1}{x}` : $\lim_{x\to\infty}\frac{1}{x}$

**改变上下限位置**的命令：`\limits`(强制上下限在上下侧) 和 `\nolimits`(强制上下限在左右侧)

`\sum\limits_{k=1}^n`: $\sum\limits_{k=1}^n$ 和 `\sum\nolimits_{k=1}^n` : $\sum\nolimits_{k=1}^n$

### 加减乘除

|  加   |  减   |     乘     |    除    |      |      |      |      |
| :---: | :---: | :--------: | :------: | ---- | ---- | ---- | ---- |
| **+** | **-** | **\times** | **\div** |      |      |      |      |

### 逻辑运算

|   ⊕    |  ∨   |   ∧    |      |      |      |      |      |
| :----: | :--: | :----: | ---- | ---- | ---- | ---- | ---- |
| \oplus | \vee | \wedge |      |      |      |      |      |

### 绝对值

直接插入竖线|即可，使用`\left`和`\right`标签指定竖线的垂直长度。

`\left| \sum_i \vec{v}_i \left|\Delta t_i\right|  \right|`

$\left| \sum_i \vec{v}_i \left|\Delta t_i\right|  \right|$

### 其他运算

|  ∑   |  ∫   |   ∮   |   ∏   |      |      |      |      |
| :--: | :--: | :---: | :---: | ---- | ---- | ---- | ---- |
| \sum | \int | \oint | \prod |      |      |      |      |

### 关系

|   ≠   |    ≈    |   ≡    |  ≤   |  ≥   |  ≫   |  ≪   |      |
| :---: | :-----: | :----: | :--: | :--: | :--: | :--: | :--: |
| \not= | \approx | \equiv | \le  | \ge  | \gg  | \ll  |      |

### 集合


|  ∈   |  ∋   |    ⊂    |    ⊃    |     ⊆     |     ⊇     | $A\cap B$ | $A\cup B$ |
| :--: | :--: | :-----: | :-----: | :-------: | :-------: | :-------: | :-------: |
| \in  | \ni  | \subset | \supset | \subseteq | \supseteq |   \cap    |   \cup    |
### 存在

|    ∃    |    ∀    |      |      |      |      |      |      |
| :-----: | :-----: | ---- | ---- | ---- | ---- | ---- | ---- |
| \exists | \forall |      |      |      |      |      |      |

### 箭头

|   ←   |  →   |     ⇐      |      ⇒      |        ⇔        |      |      |      |
| :---: | :--: | :--------: | :---------: | :-------------: | ---- | ---- | ---- |
| \gets | \to  | \Leftarrow | \Rightarrow | \Leftrightarrow |      |      |      |

### 头顶符号

| $\hat{x}$ | $\dot{x}$ | $\bar{x}$ | $\ddot{x}$ | $\vec{x}$ |      |      |      |
| :-------: | :-------: | :-------: | :--------: | :-------: | ---- | ---- | ---- |
|  \hat{x}  |  \dot{x}  |  \bar{x}  |  \ddot{x}  |  \vec{x}  |      |      |      |

### 上划线和下划线

`\overline{a+bi}` : $\overline{a+bi}$

`\underline{431}` : $\underline{431}$



### 环境对齐

对于非简写的环境（begin和end包裹的）都有两种形式，一种直接写环境名，会参与**自动编号**；另一种是在环境名后面加一个星号“*”，不会参与编号。

```latex
\begin{align*}
 f(x) &= (x+a)(x+b) \\
 &= x^2 + (a+b)x + ab
\end{align*}
```

$$
\begin{align*}
 f(x) &= (x+a)(x+b) \\
 &= x^2 + (a+b)x + ab
\end{align*}
$$

```latex
\begin{array}{cc}
        (A)\quad 4 & \hspace{4cm}(B)\quad 3\\\\
        (B)\quad 2 & \hspace{4cm}(D)\quad 1
\end{array}
```

$$
\begin{array}{cc}
        (A)\quad 4 & \hspace{4cm}(B)\quad 3\\\\
        (B)\quad 2 & \hspace{4cm}(D)\quad 1
\end{array}
$$

- **tip：**`\quad`和`\hspace{}`都是表示空格，但是空的个数不同





### 矩阵

**普通写法：**

```latex
\begin{array}{ccc}
    1 & 0 & 0\\\\
    0 & 1 & 0\\\\
    0 & 0 & 1\\\\
\end{array}
```

$$
\begin{array}{ccc}
    1 & 0 & 0\\
    0 & 1 & 0\\
    0 & 0 & 1\\
\end{array}
$$

```latex
\left[
    \begin{array}{ccc}
        1 & 0 & 0\\
        0 & 1 & 0\\
        0 & 0 & 1\\
    \end{array}
\right]
```

$$
\left[
    \begin{array}{ccc}
        1 & 0 & 0\\
        0 & 1 & 0\\
        0 & 0 & 1\\
    \end{array}
\right]
$$

- {ccc}是指元素的对齐方法（居中），此外还有`l`和`r`的参数可选，分别表示左和右



**矩阵环境写法：**

直接用`matrix、pmatrix(带小括号)、bmatrix(带中括号)、Bmatrix(带大括号)、vmatrix(行列式)、Vmatrix(两条竖线)`环境:

```latex
\begin{gathered}
\begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix}
\quad
\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
\quad
\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
\quad
\begin{Bmatrix} 1 & 0 \\ 0 & -1 \end{Bmatrix}
\quad
\begin{vmatrix} a & b \\ c & d \end{vmatrix}
\quad
\begin{Vmatrix} i & 0 \\ 0 & -i \end{Vmatrix}
\end{gathered}
```

$$
\begin{gathered}
\begin{matrix} 0 & 1 \\ 1 & 0 \end{matrix}
\quad
\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}
\quad
\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
\quad
\begin{Bmatrix} 1 & 0 \\ 0 & -1 \end{Bmatrix}
\quad
\begin{vmatrix} a & b \\ c & d \end{vmatrix}
\quad
\begin{Vmatrix} i & 0 \\ 0 & -i \end{Vmatrix}
\end{gathered}
$$



**复杂矩阵：**

```latex
A = 
    \begin{pmatrix}
        a_{11} & a_{12} & \cdots & a_{1n}\\
        a_{21} & a_{22} & \cdots & a_{2n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n1} & a_{n2} & \cdots & a_{nn}\\
    \end{pmatrix}
```

$$
A = 
    \begin{pmatrix}
        a_{11} & a_{12} & \cdots & a_{1n}\\
        a_{21} & a_{22} & \cdots & a_{2n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n1} & a_{n2} & \cdots & a_{nn}\\
    \end{pmatrix}
$$

- **tip：**横排列的点 ⋯⋯ 用`\cdots`表示，列排列的点 ⋮⋮ 用`\vdots`表示，斜排列的点 ⋱⋱ 用`\ddots`表示



**表格:**

```latex
\begin{array}{|c|c|}
        \hline
        0 & 1 \\\\\hline
        1 & 0 \\\\\hline
\end{array}
```

$$
\begin{array}{|c|c|}
        \hline
        0 & 1 \\\hline
        1 & 0 \\\hline
\end{array}
$$

- **tip：**`\hline`表示水平线，而竖线可以使用`|`来表示



### 希腊字母

默认小写，大写首字母大写。例如 \Omega

小写	大写	latex
α	A	\alpha
β	B	\beta
γ	Γ	\gamma
δ	Δ	\delta
ϵ	E	\epsilon
ζ	Z	\zeta
ν	N	\nu
ξ	Ξ	\xi
ο	O	\omicron
π	Π	\pi
ρ	P	\rho
σ	Σ	\sigma
η	H	\eta
θ	Θ	\theta
ι	I	\iota
κ	K	\kappa
λ	Λ	\lambda
μ	M	\mu
τ	T	\tau
υ	Υ	\upsilon
ϕ	Φ	\phi，（φφ：\varphi）
χ	X	\chi
ψ	Ψ	\psi
ω	Ω	\omega







































