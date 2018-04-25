---
layout:     post
title:      "知识图谱入门 (七)" 
subtitle:   "知识推理"
date:       2018-04-24 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 本节对本体任务推理做一个简单的介绍，并介绍本体推理任务的分类。而后对本体推理的方法和工具做一个介绍。

* TOC
{:toc}

# 知识推理简介

## 知识推理任务分类

所谓推理就是通过各种方法**获取新的知识或者结论**，这些知识和结论满足语义。其具体任务可分为可满足性(satisfiability)、分类(classification)、实例化(materialization)。

可满足性可体现在本体上或概念上，在本体上即本体可满足性是检查一个本体是否可满足，即检查该本体是否有模型。如果本体不满足，说明存在不一致。概念可满足性即检查某一概念的可满足性，即检查是否具有模型，使得针对该概念的解释不是空集。

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_1.png)

上图是两个不可满足的例子，第一个本体那个是说，Man 和 Women 的交集是空集，那么就不存在同一个本体Allen 既是Man 又是Women。 第二个概念是说概念Eternity是一个空集，那么他不具有模型，即不可满足。

分类，针对Tbox的推理，计算新的概念包含关系。如:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_2.png)

即若Mother 是 Women的子集，Women是 Person的子集，那么我们就可以得出 Mother是 Person的子集这个新类别关系。

实例化即计算属于某个概念或关系的所有**实例的集合**。如:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_3.png)

第一个是计算新的类实例信息，首先已知Alice 是Mother，Mother 是 Women的子集，那么可知Alice 是一个Women。即为Women增加了一个新的实例。下面那个是计算新的二元关系，已知Alice 和Bob 有儿子，同时has_son 是has_child的子类，那么可知Alice 和Bob has_child。

## 知识推理简介

OWL本体语言是知识图谱中最规范(W3C制定)、最严谨(采用描述逻辑)。表达能力最强的语言(是一阶谓词逻辑的子集)，它基于RDF语法，使表示出来的文档具有语义理解的结构基础。促进了统一词汇表的使用，定义了丰富的语义词汇。同时允许逻辑推理。

关于OWL语言的规范性我们再之前讨论过，此处我们介绍一下它的逻辑基础：描述逻辑。

### 描述逻辑

**描述逻辑(Description Logic)是基于对象的知识表示的形式化，也叫概念表示语言或术语逻辑，是一阶谓词逻辑的一个可判定子集。**

一个**描述逻辑系统**由四个基本部分组成：

* 最基本的元素：概念、关系、个体    
* TBox术语集：概念术语的公理集合    
* Abox断言集：个体的断言集合    
* TBox 和 ABox上的推理机制

**不同的描述逻辑系统的表示能力与推理机制由于对这四个组分的不同选择而不同。**下面对四个组分中的概念做一个简单介绍。

最基本的元素有概念、关系、个体。    
* 概念即解释为一个领域的子集，如
$$ {x|student(x)} $$    
* 关系解释为该领域上的二元关系(笛卡尔积)，如 
$$ {<x, y> | friend(x, y)} $$    
* 个体解释为一个领域内的实例，如小明：{Ming}

TBox为术语集，它是泛化的知识，是描述概念和关系的知识，被称之为公理(Axiom)。由于概念之间存在包含关系，TBox 知识形成类似格(Lattice)的结构，这种结构是由包含关系决定的，与具体实现无关。TBox语言有定义和包含，其中定义为引入概念及关系的名称，如Mother、Person、has_child，包含指声明包含关系的公理，例如$$ Mother \sqsubseteq \exists has_child.Person $$

ABox是断言集，指具体个体的信息，ABox包含外延知识(又称为断言(Assertion)), 描述论域中的特定个体。**描述逻辑的知识库 K:= <T, A>， T即TBOx， A即ABOx。**ABox 语言包含概念断言和关系断言，概念断言即表示一个对象是否属于某个概念，例如Mother(Alice)、Person(Bob)。关系断言表示两个对象是否满足特定的关系，例如 has_child(Alice, Bob)。

描述逻辑语义：解释I是知识库K的模型,当且仅当I是K中每个断言的模型。若一个知识库K有一个模型,则称K是可满足的。若断言σ对于K的每个模型都是满足的,则称K逻辑蕴含σ,记为$$ K \models \sigma$$。对概念C,若K有一个模型I使得$$ C^I \neq \varnothing $$则称C是可满足的。

描述逻辑依据提供的构造算子,在简单的概念和关系上构造出复杂的概念和关系。描述逻辑至少包含以下构造算子:交 ($$ \cap $$),并($$ \cup $$),非 (¬),存在量词 ($$ \exists $$)和全称量词 ($$ \forall $$)。有了语义之后,我们可以进行推理。通过语义来保证推理的正确和完备性。

下图给出描述逻辑的语义表:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_4.png)

因为OWL采用描述逻辑，因此下图给出了描述逻辑与OWL词汇的对应表：

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_5.png)

# 本体推理方法与工具介绍

基于本体推理的方法常见的有基于Tableaux运算的方法、基于逻辑编程改写的方法、基于一阶查询重写的方法、基于产生式规则的方法等。

* 基于Tableaux运算适用于检查某一本体的可满足性，以及实例检测。    
* 基于逻辑编程改写的方法可以根据特定的场景定制规则，以实现用户自定义的推理过程。    
* 基于一节查询重写的方法可以高效低结合不同数据格式的数据源，重写方法关联起了不同的查询语言。以Datalog语言为中间语言,首先重写SPARQL语言为Datalog,再将Datalog重写为SQL查询；     
* 一种前向推理系统,可以按照一定机制执行规则从而达到某些目标,与一阶逻辑类似,也有区别；

下面对上面的几种方法做详细介绍。

## 基于Tableaux运算

基于Tableaux运算适用于检查某一本体的可满足性，以及实例检测。其基本思想是通过一系列规则构建Abox,以检测可满足性,或者检测某一实例是否存在于某概念。这种思想类似于一阶逻辑的归结反驳。

Tableaux运算规则(以主要DL算子举例)如下：

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_6.png)

这里对第一个解释一下，其他的类似。第一个是说如果C 和D(x) 的合取是$$ \varnothing $$，同时呢C(x) 和 D(x) 却不在$$ \varnothing $$里，那么也就是说$$ \varnothing $$有可能只包含了部分C，而C(x)不在里面，那么我们就把它们添加到$$ \varnothing $$里。下面我们举个实际的例子:

现在给定如下本体，检测实例Allen 是否在 Woman中? 即:

$$ Man \sqcap Woman \sqsubseteq \perp $$

$$ Man(Allen) $$ 

检测 Woman(Allen)？其解决流程为:

* 首先加入带反驳的结论:

$$ Man \sqcap Woman \in \perp $$

$$ Man(Allen)~~~~ Woman(Allen)$$ 

* 初始Abox，记为 $$ \varnothing $$，其内包含$$ Man(Allen)~~~~ Woman(Allen)$$。

* 运用 $$ \sqcap^{-} -$$规则，得到 $$ Man \sqcap Women(Allen) $$。将其加入到$$ \varnothing $$中，现在的 $$ \varnothing $$为 $$ Man(Allen)~~~~ Woman(Allen) ~~~ Man \sqcap Women(Allen) $$。

* 运用 $$ \sqsubseteq - $$ 规则到$$ Man \sqcap Women(Allen) $$ 与$$ Man \sqcap Woman \sqsubseteq \perp $$上，得到 $$ \perp(Aleen) $$。此时的$$ \varnothing $$包含 $$ Man(Allen)~~~~ Woman(Allen) ~~~ Man \sqcap Women(Allen)~~~ \perp(Aleen) $$。

* 运用$$ \perp - $$ 规则，拒绝现在的$$ \varnothing $$。

* 得出Allen 不在Woman的结论。如果Woman(Allen)在初始情况已存在于原始本体,那么推导出该本体不可满足!

Tableaux运算的基于Herbrand模型，Herbrand模型你可以把它简单的理解为所有可满足模型的最小模型，具体的可以去看逻辑方面的书。

### 相关工具简介

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_7.png)

## 基于逻辑编程改写的方法

本体推理具有一定的局限性，如仅支持预定义的本体公理上的推理，无法针对自定义的词汇支持灵活推理；用户无法定义自己的推理过程等。因此引入**规则推理**，它可以根据特定的场景定制规则,以实现用户自定义的推理过程。

基于以上描述，引入Datalog语言，它可以结合本体推理和规则推理。面向知识库和数据库设计的逻辑语言,表达能力与OWL相当,支持递归，便于撰写规则，实现推理。

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_8.png)

Datalog 的基本语法包含:

* 原子(Atom): $$ p(t_1, t_2, \ldots, t_n) $$， 其中p是谓词,n是目数,$$t_i$$ 是项 (变量或常量)，例如 has_child(X, Y)；    
* 规则(Rule)：$$ H :- B_1, B_2, \ldots, B_m $$，由原子构建，其中H 是头部原子，$$ B_1, B_2, \ldots, B_m $$是体部原子。例如:  has_child X, Y : −has_son X, Y    
* 事实(Fact)： $$ F(c_1, c_2, \ldots, c_n ):-$$ ，它是没有体部且没有变量的规则，例如  has_child Alice, Bob : −     
* Datalog程序是规则的集合

下图给出一个Datalog 推理的例子：

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_9.png)

### 相关工具简介

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_10.png)

## 基于一阶查询重写的方法

基于查询重写我们可以高效地结合不同数据格式的数据源；同时重写方法关联起了不同的查询语言。

一阶查询是具有一阶逻辑形式的语言，因为Datalog是数据库的一种查询语言，同时具有一阶逻辑形式，因此可以以Datalog 为中间语言，首先重写SPARQL 语言为Datalog ，再将Datalog 重写为SQL 查询。

$$ SPARQL \rightarrow Datalog \rightarrow SQL $$ 

下图给出查询重写的基本流程:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_11.png)

### 查询重写举例

查询所有研究人员及其所从事的项目? 用 SPARQL表述为:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_12.png)

给定Datalog 规则如下:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_13.png)

底层数据具体为某数据库中为下图中的两张表:

![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_14.png)

* 步骤一： 重写为Datalog 查询

    * 过滤不需要的公理 (通过语法层过滤)
    ![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_15.png)

    * 生成所有相关的Datalog 查询

    ![](/img/in-post/xiaoxiangkg_note7/xiaoxiangkg_note7_16.png)

* 步骤二： 将数据库关系表达式映射成Datalog原子

    * 
