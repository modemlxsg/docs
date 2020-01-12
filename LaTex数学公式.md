# LaTex数学公式

###  公式排版

- 行内公式：用 `$`包裹公式
- 独立公式：用 `$$`包裹公式。

###  角标（上下标）

**上标命令**：`^{}`
**下标命令**：`_{}`
如果角标为单个字符，可以不使用花括号；否则必须使用花括号。

### 分式

**分式命令**：`$\frac{分子}{分母}$`

### 根式

**二次根式命令**：`$\sqrt{表达式}$`
**N次根式命令**：`$\sqrt[n]{表达式}$`

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

|  ∈   |  ∋   |    ⊂    |    ⊃    |     ⊆     |     ⊇     |      |      |
| :--: | :--: | :-----: | :-----: | :-------: | :-------: | ---- | ---- |
| \in  | \ni  | \subset | \supset | \subseteq | \supseteq |      |      |

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



### 希腊字母

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







































