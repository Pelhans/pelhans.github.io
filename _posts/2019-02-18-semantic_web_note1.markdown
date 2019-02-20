---
layout:     post
title:      "语义网基础教程笔记（一）"
subtitle:   "资源描述框架：RDF"
date:       2019-02-18 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - knowledge graph
---

> 看书不记笔记的下场就是还得看一遍。。。 RDF 是一种勇于表达有关对象(资源) 的生命的语言；它是一个标准的数据模型以提供机器可处理的语义。RDF模式提供了一组用于将RDF词汇表组织成带类型的层次结构的建模原语。

* TOC
{:toc}

# 简介

万维网的成功展现了使用标准化的信息交换和通信机制的力量。HTML是可编辑的网页的标准语言，它用于传递有关面向人类的文档的结构的信息。而对于语义网，我们的需求更加丰富。RDF(资源描述框架)恰好提供了这样一个灵活并且领域无关的数据模型。他的基础构件是一个实体-属性-取值三元组，称为声明。因为 RDF 不针对仁和领域及使用，对用户而言必须定义他们在这些声明中使用的术语。为此，需要利用 RDF 模式(RDFS)。RDFS 允许用户精确地定义他们的词汇表(vocabulary,即术语)应该如何解释。

综合起来，这些技术定义了在不同机器交换任意数据的一种标准化语言的组成部分：

* RDF -- 数据模型    
* RDFS -- 语义    
* Turtle / RDFa / RDF-XML -- 语法

尽管 RDF 主要是指数据模型，但它也经常被用来作为上述所有的总称。

# RDF：数据模型

RDF 中的基本概念包括资源、属性、声明和图。

## 资源

我们可以认为一个资源是一个对象。每个资源都有一个 URI。一个 URI 可以是一个URL((Uniform Resource Locator, 统一资源定位符)，或网址)或者，另一种唯一的标识符。URI 提供了一种机制来无歧义地标识我们想要谈论的一个事物。使用 URI 不必能访问到一个资源。但是使用可以解引用的 URL 作为资源标识符被认为是一种好的做法。

## 属性

属性是一类特殊的资源，它们描述了资源之间的关系。如 "located in"。和其他资源一样，属性也由 URI 标识。我们也可以解引用属性的 URL 来找到它们的描述。

## 声明

声明断言了资源的属性。一个声明是一个实体-属性-取值的三元组，由一个资源、一个属性和一个属性值组成。属性值要么是一个资源，要么是一个文字(literal)。文字是原子值，如数字、字符串或日期。我们经常使用主语一词来指称三元组里的实体，而使用宾语来指称其取值。

例如，对于声明 "BAron Way Building is located in Amsterdam"，可以这样写：

```
<http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayBuilding>
<http://dbpedia.org/ontology/location>
<http://dbpedia.org/resource/Amsterdam>.
```

## 图

我们也可以使用图形化的方式来书写相同的声明。如下图所示，其中我们为了提高可读性没用 URI。其中带标签的节点通过带标签的边连接。边是有向的，从声明的主语到声明的宾语，声明的属性被标记在边上。节点上的标签时主语和宾语的标识符，一个声明的宾语可以是另一个声明的主语。这种图形化的表示强调了 RDF 是一个以图为中心的数据模型这一概念。

![](/img/in-post/sematic_web_note/rdf_graph.png)

## 指向声明和图

有时能够指向特定的声明或图的某些部分是很有用的。RDF 为此提供了两种实现机制。一种称为具体化(reification)，具体化背后的关键思想是引入一个额外的对象，并将它和原来声明中的三个部分通过属性 subject、predicate、object 关联。但这种方式的代价比较高，因此在较新版本的 RDF标准中引入了命名图的概念。此时一个显式的标识符被赋予一个声明和声明集合。一个命名图允许圈出一个 RDF 声明的集合并为这些声明提供一个标识符。

# RDF 语法

前面已经介绍了一种 RDF 语法，即图形化的语法。但这种语法既不是机器可解释的，也不是标准化的。这里我们介绍一种标准的机器可解释的语法，称为 Turle。还有一些其他语法。

## Turtle

Turtle( Terse RDF Triple Language ) 是一种基于文本的 RDF 语法。Turtle 文本文件使用的后缀名是 ".ttl"。我们之前见到的那个三元组就是 Turtle 的一个声明。

```
<http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayBuilding>
<http://dbpedia.org/ontology/location>
<http://dbpedia.org/resource/Amsterdam>.
```

URL 包含在 尖括号中。一个声明的主语、属性和宾语依次出现，由句号结尾。

### 文字

除了像上面那样将资源链接在一起的声明。我们也能在RDF 中引入文字，即原子值。在 Turtle 中，我们简单地将值卸载引号中，并附上值的数据类型。数据类型包含字符串、日期、整数型或其他数据类型没数据类型也是用 URL 来表达。实践中建议使用 XML 模式定义的数据类型。如果一个文字之后没有指定数据类型，则假设数据类型是字符型。如：

```
<http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayAppartment>
<http://www.semanticwebprimer.org/ontology/apartments.ttl#hasNumberOfBedrooms>
"3"^^<http://www.w3.org/2001/XMLSchema#integer>.

<http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayAppartment>
<http://www.semanticwebprimer.org/ontology/apartments.ttl#isPartOf>
<http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayBuilding>.

<http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayBuilding>
<http://www.dbpedia.org/ontology/location>
<http://www.dbpedia.org/resource/Amsterdam>.
```

上面的例子相对而言不容易使用。为了更加清晰，Turtle 提供了一些构造子来使书写变得更加容易。

### 缩写

在上面的例子中，BaronWayAppartment 和 BaronWayBuilding 都定义在 http://www.semanticwebprimer.org/ontology/apartments.ttl 这个 URL 下。这个 URL 定义了这些资源的命名空间。Turtle 使用了这个惯例，允许 URL 被缩写。它引入 @prefix 语法来定义命名空间的替代形式。如 可以用 swp 作为 http://www.semanticwebprimer.org/ontology/apartments.ttl 的 替代的形式。这种替代称为限定名(qualified name)。使用限定名重写上述例子：

```
@prefix swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
@prefix dbpedia: <http://dbpedia.org/resource>.
@prefix dbpedia-owl: <http://dbpedia.org/ontology>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

swp:BaronWayAppartment swp:hasNumberOfBedrooms "3"^^<xsd:integer>.
swp:BaronWayAppartment swp:isPartOf swp:BaronWayBuilding.
swp:BaronWayBuilding dbpedia-owl:location dbpedia:Amsterdam.
```

Turtle 还允许我们重复使用某些主语的时候不需要再重复写。这可以通过在一个声明的结尾处使用一个分号来使得书写更加紧凑。

```
@prefix swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
@prefix dbpedia: <http://dbpedia.org/resource>.
@prefix dbpedia-owl: <http://dbpedia.org/ontology>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

swp:BaronWayAppartment swp:hasNumberOfBedrooms "3"^^<xsd:integer>；
                        swp:isPartOf swp:BaronWayBuilding.
swp:BaronWayBuilding dbpedia-owl:location dbpedia:Amsterdam.
```

如果主语和谓语都被重复使用，我们可以在声明的结尾处使用一个逗号。如想说明 BAron Way Building 不仅位于 Amsterdam 还位于 Netherlands，则：

```
@prefix swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
@prefix dbpedia: <http://dbpedia.org/resource>.
@prefix dbpedia-owl: <http://dbpedia.org/ontology>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

swp:BaronWayAppartment swp:hasNumberOfBedrooms "3"^^<xsd:integer>；
                        swp:isPartOf swp:BaronWayBuilding.

swp:BaronWayBuilding dbpedia-owl:location dbpedia:Amsterdam,
                                          dbpedia:Netherlands.
```

最后，Turtle 还允许我们简写常见的数据类型。例如数字不用引号来写，如果数字包含一个小数点，则被解释为小数。若不包含小数点则被解释为整型数。

### 命名图

我们之前讨论了一个指向一组声明的能力。Trig 是 Turtle 的一个扩展，它允许我们表达这个概念。为此我们将一组想要的声明用花括号括起来并赋予这组声明一个 URL。

```
@prefix swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
@prefix dbpedia: <http://dbpedia.org/resource>.
@prefix dbpedia-owl: <http://dbpedia.org/ontology>.
@prefix dc: <http://purl.org/dc/terms/>.

{
    <http://www.semanticwebprimer.org/ontology/apartments.ttl#>
    dc:creator <http://wwww.cs.vu.nl/frankh>
}

<http://www.semanticwebprimer.org/ontology/apartments.ttl#>
{
    swp:BaronWayAppartment swp: hasNumberOfBedrooms 3;
                           swp:isPartOf swp:BaronWayBuilding.
    swp:BaronWayBuilding dbpedia-owl:location dbpedia:Amsterdam,
                                            dbpedia:Netherlands.
}
```

在上面的例子中，位于花括号中但是之前没有 URL 的声明不是一个特定图的一部分。它成为默认图。

## 其他语法

除了 Turtle 之外，还存在其他一些可用于编写 RDF 的语法。其中有两个标准的语法： RDF/XML 和 RDFa。

### RDF/XML

RDF/XML 是 RDF 在 XML语言中的编码。它允许 RDF 被已有 XML 处理工具使用。起初，RDF/XML 是 RDF 的唯一语法。但是由于 Turtle 通常更容易阅读，所以作为一种额外标准被采纳。下例是 RDF/XML 的例子：

```
<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:dbpedia-owl="http://dbpedia.org/ontology/"
    xmlns:dbpedia="http://dbpedia.org/resource/"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:swp="http://www.semanticwebprimer.org/ontology/apartments.ttl#">
<rdf:Description
rdf:about="http://www.semanticwebprimer.org/ontology/apartments.ttl#BaronWayAppartment">
    <swp:hasNumberOfBedrooms
    rdf:datatype="http:www.w3.org/2001/XMLSchema#integer">
        3
    <swp:hasNumberOfBedrooms>
</rdf:Description>
```

### RDFa

RDF 的一个用例是描述或标注 HTML 网页的内容。为了使其更加简单，引入RDFa 语法来帮助实现这个用例。 RDFa 在 HTML 标签的属性(attribute) 中嵌入 RDF。

### 小结

上述每种 RDF 语法适用于不同的场景。但**尽管可能会用到不同的语法，但它们的底层数据模型和语义是相同的。**

# RDFS:添加语义

RDF 既不假设与任何特定应用领域有关，也不定义任何领域的语义。为了指明语义，一个开发者或者用户需要**通过 RDF 模式中定义的一组基本的领域无关的结构来定义其词汇表的含义。**

## 类和属性

为了描述特定的领域，除了需要个体对象外，还有类，它定义了对象的类型。一个；类可以被理解为一个元素集合。属于一个类的个体对象被称为该类的实例(instance)。RDF 给我们提供了一种通过使用一个特殊属性 rdf:type 来定义实例和类之间的联系方式。

## 类层次和继承

类之间是可以有关联的，一个类可以是另一个类的子类，也可以是另一个类的超类。一个类可以继承自另一个类。RDF 模式不要求所有的类形成一个严格的层次结构。通过创建这样的语义定义，RDFS 是一种(能力依然受限的)定义特定领域的语言。换句话说，RDF 模式是一种基本的本体语言。

## RDF 和 RDFS 的分层对比

下图展现了这个例子的 RDF 层次和 RDF 模式层次。其中，方块是属性，虚线以上的圆圈是类，而虚线以下的圆圈是实例。

![](/img/in-post/sematic_web_note/rdf_rdfs_compard.png)

# RDF 模式：语言

RDF 模式提供建模原语来表达上节的信息。现让我们定义 RDF 模式的建模原语。

## 核心类

核心类包括：

* rdfs:Resource，所有资源的类        
* rdfs:Class,所有类的类    
* rdfs:Literal，所有文字(字符串)的类。    
* rdf:Property，所有属性的类。    
* rdf:Statement，所有具体化声明的类。    

## 定义联系的核心属性

用来定义联系的核心属性包括：

* rdf:type，将一个资源关联到它的类。该资源被声明为该类的一个实例。    
* rdfs:subClassOf，将一个类关联到它的超类。一个类的所有实例都是它的超类的实例。注意，一个类可能是多个类的子类。    
* rdfs:subPropertyOf，将一个属性关联到它超属性中的一个。

需要注意的是，rdfs:subPropertyOf 和 rdfs:subClassOf 被定义为传递的。rdfs:Class 是 rdfs:Resource 的一个子类(所有的类都是资源)，同时 rdfs:Resource 是 rdfs:Class 的一个实例。出于同样的原因，每个类都是 rdfs:Class 的实例。

## 限制属性的核心属性

用来限制属性的核心属性包括：

* rdfs:domain，指定一个属性 P的定义域，声明任何拥有某个给定属性的资源是定义域类的一个实例。    
* rdfs:range，指定一个属性P的值域，声明一个属性的取值是值域类的实例。

## 对具体化有用的属性

下面是一些对具体化有用的属性：

* rdf:subject，讲一个具体化声明关联到它的主语。    
* rdf:predicate，将一个具体化声明关联到它的谓语。    
* rdf:object，讲一个具体化属性关联到它的宾语。

## 容器类

RDF 还允许用一个标准的方式表达容器。可以表达包、序列或选择。

* rdf:Bag，包的类    
* rdf:Seq，序列的类。    
* rdf:Alt，选择的类。    
* rdfs:Container，所有容器类的超类，包括前面提到的3种。

## 效用属性

一个资源可以在万维网上的许多地方被定义和描述。下列属性允许我们定义连接到这些地址：

* rdfs:seeAlso，将一个资源关联到另一个解释它的资源。    
* rdfs:isDefinedBy，它是rdfs:seeAlso 的一个子属性，将一个资源关联到它的定义之处，一般是一个RDF模式。    
* rdfs:comment，注释，一般是长的文本，可以与一个资源关联。    
* rdfs:label，讲一个人类友好的标签(名字)与一个资源相关联。

## 一个例子：汽车

这里给出一个简单的汽车本体，下图给出了它的类层次及表示。

![](/img/in-post/sematic_web_note/rdfs_car_0.png)

![](/img/in-post/sematic_web_note/rdfs_car_1.png)

![](/img/in-post/sematic_web_note/rdfs_car_2.png)

# 总结

* RDF 为表示和处理及其可理解的数据提供了基础。    
* RDF 使用基于图的数据模型。它的核心概念包括资源、属性、声明和图。一个声明是一个资源-属性-值的三元组。    
* RDF 拥有三种标准语法(Turtle、RDF/XML和RDFa)来支持语法层的互操作性。    
* RDF 使用分布式思想，允许递增式的构建知识，以及知识的共享和复用。    
* RDF 是领域无关的，RDF 模式提供了一种描述特定领域的机制。    
* RDF 模式是一种基本的本体语言。它提供一组具有固定含义的建模原语。RDF模式的核心概念有类、子类关系、属性、子属性关系，以及定义域和值域的限制。
