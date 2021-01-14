# 一、扑克理论元素

## 01、扑克基础

### 1、基本概念



#### 牌桌位置

EP：前位。九人桌前三个位置，人少则没有前位

MP：lojack位置，第4个行动。Hijack位置，第5个行动

LP：co关口位置，bn按钮位置

盲注位置：SB、BB



#### 扑克术语

active player：活跃牌手，hero和villain

hero：第一视角牌手

villain：hero的对手

pre-flop：翻前

post-flop：翻后

ip：有利位置，后行动

oop：不利位置，先行动

relative position：相对位置

first in：第一个加注入池

stack size 和 stack depth：筹码深度总是相对大盲而言

effective stack：有效筹码，活跃牌手中的最小筹码量

bet-size：下注尺度以大盲活底池的一部分表示。

the-nuts：坚果牌

effective-nuts：强到可以当坚果游戏的牌

speculative hand：投机牌

air：空气牌

bluff catcher：抓诈牌

pure bluff：纯诈唬

semi bluff：半诈唬

showdown value：摊牌价值

nut advantage 或 nut threshold：坚果优势，如果一个牌手范围中的有效坚果牌相对对手的范围比重更大，那么他具有坚果优势。

aggressive player：激进牌手

pocket pair：口袋对

draw：听牌

gut shot 或 inside straight draw：卡顺听牌

broadway：坚果顺子

any two card (ATC)：包含所有可能底牌的范围

coin flip 或 race：跑马，两个牌手全压胜率差不多的情况

cooler：冤家牌

rake：抽水



#### 牌手行动

call

limp

raise

check

VPIP：主动投入资金到底池，投入盲注不算。

RFI：通过加注首先入池。open raise

2bet：盲注被视为第一个下注，如果无人limp，RFI是2bet的一种特殊情况。

steal：偷盲

isolate：隔离加注，在某人入池后加注，意图翻后和他单挑。

miniraise：最小加注。

minibet：最小下注。

overbet：超池加注。

3bet：再加注。

resteal：反偷盲。

push：all-in 

open shove或open jam：在你前面无人入池时全压。

3bet jam：在某人加注后全压。

cold 4bet：非初始加注者做出的4bet。

cold call：冷跟住。

squeeze：在某人加注其他人跟住后3bet。

cbet：持续下注。

donk bet或lead out：不利位置对之前回合主动者下注。

slow play：慢打。



#### 行动线

牌手采取的一系列下注行动。l/f ：limp/fold。x/f：check/fold



#### 底牌范围

当牌手达到一定阶段，他们不会简单的将对手推测到单单一手牌，而是给他们分配一个可能的底牌范围，并根据对手行动缩减这个范围。



#### 13x13 底牌矩阵

底牌矩阵（hand grid）是范围的一种图形表达方式。包含所有169种可能的翻前范围。

口袋对子13种，占5.88%。同花78种，占23.53%。非同花牌78种，占70.59%。



#### 组合数学

从n个对象中选出k个对象的组合公式：
$$
组合数 = \frac{n!}{k!(n-k)!}
$$

$$
手牌组合 = \frac{52!}{2!(52-2)!} = \frac{52\times51\times50!}{2\times50!} = \frac{52\times51}{2} = 1326
$$



**口袋对子：**
$$
口袋对子组合数 = \frac{x\times(x-1)}{2} = 4\times3\div2 = 6
$$
**非对子：**
$$
非对子牌组合数 = X\times Y （X,Y是每个rank有效牌数，例如AK，有4张A、4张K，所以组合数4\times 4 = 16）
$$


#### 底牌范围的组合数学



##### 翻前范围拆解

例：23%加注范围

<img src="images/Modern Poker Theory/image-20210106172352030.png" alt="image-20210106172352030" style="zoom:80%;" />

口袋对子：13x6 = 78种组合

同花牌：30x4 = 120种组合

非同花牌：9x12 = 108种组合

总范围：78 + 120 + 108 = 306 / 1326 = 23%



##### 翻后范围拆解

考虑在AcJh9h的特定翻牌面对手的范围



成牌157种：三条7种，两对10种，顶对38种，中对30种，低对15种，口袋对12种......



### 2、关键衡量指标



#### 底池赔率与补牌（pot odds and outs）

底池赔率就是**风险回报率**
$$
底池赔率 = \frac{风险额}{风险额 + 回报额}
$$







## 02、博弈论基础

### 核心概念

### 最大程度剥削策略



## 03、现代扑克软件

**PokerStrategy.com Equilab**

https://www.pokerstrategy.com/poker-software-tools/equilab-holdem/



**ICMIZER**

https://www.icmpoker.com/en/download/#Win



**SimpleGTO**

https://simplepoker.com/en/Solutions/Simple_GTO_Trainer?partner=mypokercoaching



**GTO+**

https://www.gtoplus.com/



**piosolver**

https://www.piosolver.com/

# 二、翻牌前理论与实践

## 04、翻牌前理论与实践

## 05、六人常规桌GTO策略

## 06、MTT打法理论

## 07、MTT均衡策略：首入的玩法

## 08、MTT均衡策略：防守

## 09、MTT均衡策略：对抗3bet



# 三、翻牌后理论与实践

## 10、翻后玩法理论

## 11、翻牌圈玩法理论

## 12、翻牌圈持续下注

## 13、GTO转牌圈策略

## 14、GTO河牌圈策略

