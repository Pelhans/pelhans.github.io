---
layout:     post
title:      "<语义网基础教程>笔记（二）"
subtitle:   "查询语义网：SPARQL"
date:       2019-02-20 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---

> 本章将介绍SPARQL 查询得以执行的基础设施，之后讨论SPARQL的基础知识并逐步介绍其更复杂的部分。

* TOC
{:toc}

# 简介

SPARQL 能够让我们通过选择、抽取等方式从被表示为 RDF 的知识中获取特定的部分。SPARQL 是专为 RDF 设计的，适合并依赖于万维网上的各种技术。如果你熟悉诸如SQL 等数据库查询语言，你会发现 SPARQL 和它们有很多相似之处。

# SPARQL 基础设施

想要执行一条 SPARQL 查询，就需要一个能执行查询的软件。能做到这一点的最常用的软件叫做三元组存储库(triple store)。本质上，一个三元组存储库就是一个RDF的数据库，在 SPARQL的相关规范中三元组存储库也称为图存储库。

当数据被加载进三元组存储库之后，就可以使用 SPARQL 协议来发送 SPARQL 查询去查询了。每个三元组存储库都提供一个端点(endpoint)，在此提交 SPARQL 查询。重要的一点是客户端使用 HTTP 协议向端点发送查询，因此你可以通过发送网络请求进行查询。

# 基础知识：匹配模式

假设我们有如下描述 Baron Way 公寓及其位置的 RDF 数据：

```
@prefix swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
@prefix dbpedia: <http://www.dbpedia.org/resource/>.
@prefix dbpedia-owl: <http://dbpedia.org/ontology/>.

swp:BaronWayApartment swp:hasNumberOfBedrooms 3;
                      swp:isPartOf swp:BaronWayBuilding.
swp:BaronWayBuilding dbpedia-owl:location dbpedia:Amsterdam,
                                          dbpedia:Netherlands.
```

在 SPARQL 中，我们可以将三元组中的任何一个元素替换为一个变量。变量的首字符是一个 ?。要引入一个变量表示位置，我们可以这样写：

```
swp:BaronWayBuilding dbpedia-owl:location ?location.
```

三元组存储库将接收这个图模式(graph pattern)并尝试去找到能够匹配这个模式的那些三元组集合。本质上，它找到了所有以 swp:BaronWayBuilding 作为主语， dbpedia-owl:location 作为谓语的三元组。

要构建一个完整的 SPARQL 查询，还需要增加一些内容。首先需要定义所有的前缀。还需要告诉三元组存储库我们对一个特定变量的结果感兴趣。因此，上述查询对应的完整的 SPARQL 查询如下：

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.

SELECT ?location
WHERE {
    swp:BaronWayBuilding dbpedia-owl:location ?location.
}
```

与 Turtle 类似，PREFIX 关键词指明各种 URL的缩写。 SELECT 关键词表明了哪些变量是感兴趣的。需要被匹配的图模式出现在 WHERE 关键词之后的括号中。返回的查询结果是一组称作绑定(binding)的映射，表示哪些元素对应到一个给定的变量。表格中的每一行是一条结果或一个绑定。因此，这条结果返回的查询结果如下：

```
?location    
http://dbpedia.org/resource/Amsterdam.    
http://dbpedia.org/resource/Netherlands.    
```

SPARQL 的全部基础就是这个简单的概念：尝试去找到能够匹配一个给定图模式的那些三元组集合。SPARQL 提供了更多的功能，用来指定更加复杂的模式并以不同的方式提供结果；但无论模式多么复杂，运用的过程都是一样的。下面给出一个更为复杂的查询，它包含多个三元组，且同时查询了多个变量。采用 LIMIT 语句限制返回结果数量。

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.

SELECT ?p ?o
WHERE {
    swp:BaronWayApartment swp:isPartOf ?p.
    ?p dbpedia-owl:location ?o.
}
LIMIT 10
```

不同于前面使用的由多个三元组模式构成的链。 SPARQL 提供了一种精确表述属性链的方式。这一机制成为属性路径(property path)。如：

```
?apartment swp:isPartOf ?building.
?building dbpedia-owl:location dbpedia:Amsterdam.

# 用属性路径表述为

?apartment swp:isPartOf/dbpedia-owl:location dbpedia:Amsterdam.
```

# 过滤器

当处理如大于或小于一个特定数量的查询时，可以使用 FILTER 关键字，该关键字支持数值型数据类型(即整型数、小数)和日期/时间都支持小于、大于和等于运算。

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.

SELECT ?apartment
WHERE {
    ?apartment swp:hasNumberOfBedrooms ?bedrooms.
    FILTER(?bedrooms >2).
}
```

SPARQL 也支持字符串的过滤。假设我们的数据集包含如下三元组：

```
swp:BaronWayApartment swp:address "4 Baron Way Circle"
```

我们想要找到所有地址中包含 "4 Baron Way"的资源。这可以使用 SPARQL 内置支持的正则表达式来实现--regex，这个函数的参数在随后的括号中给出。

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.

SELECT ?apartment
WHERE {
    ?apartment swp:address ?address.
    FILTER regex(?address, "^4 Baron Way")
}
```

SPARQL 还包含一些其他类型的过滤器，在一些特定的场合可能有用，但数值和字符串过滤器是最常用的。

最后介绍一个常用的函数 str。它将资源和文字转换为可以在正则表达式中使用的字符串表示。

# 处理一个开放世界的构造子

与传统数据库不同，不是每个语义网上的资源都会以相同的模式(schema)描述，或者都拥有相同的属性，这叫做开放世界假设。

如下面的例子中，一些公寓可能比其他公寓拥有更多的描述，更进一步的，它们可能以一种不同的词汇表来描述。为此 SPARQL 提供了两种构造子 UNION 和 OPTIONAL。

RDF 例子：

```
@prefix swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.    
@prefix dbpedia: <http://dbpedia.org/resource>.    
@prefix dbpedia-owl: <http://dbpedia.org/ontology>.    
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.    

swp:BaronWayApartment swp:hasNumberOfBedrooms 3.    
swp:BaronWayApartment dbpedia-pwl:location dbpedia:Amsterdam.    
swp:BaronWayApartment rdfs:label "Baron Way Apartment for Rent".    

swp:FloridaAveStudio swp:hasNumberOfBedrooms 1.    
swp:FloridaAveStudio dbpedia-owl:locationCity dbpedia:Amsterdam.    
```

SPARQL 查询：

```
PREFIX geo: <http://www.geonames.org/ontology#>.    
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.    
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.    
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.    
    
SELECT ?apartment ?label    
WHERE {    
    {?apartment dbpedia-owl:location dbpedia:Amsterdam.}    
    UNION    
    {?apartment dbpedia-owl:locationCity dbpedia:Amsterdam.}    
    OPTIONAL    
    {?apartment rdfs:label ?label.}    
}    
```

这个查询的结果是：

```
?apartment                  ?label
swp:BaronWayApartment      Baron Way Apartment for Rent    
swp:FloridaAveStudio    
```

UNION 关键词告诉三元组存储库返回那些仅匹配一个图模式或两个都匹配的结果。OPTIONAL 关键词告诉三元组存储库为特定的图模式返回结果--如果能找到，即对于待返回的查询而言，这个图模式未必要满足。

# 组织结果集

当我们想要查询结果以一种特定的方式返回：分组的、计数的或排序的。SPARQL 提供了一些函数来帮助我们，如 DISTINCT 来消除结果集中的重复结果，用法为： ```SELECT DISTINCT ?name WHERE {...}```。ORDER BY 关键词用来对返回结果进行排序，如 ```ORDER BY DESC(?bedrooms.)```。其他的还有聚集函数，如 COUNT、SUM、MIN、MAX、AVG等。聚集函数还可以与AS关键词组合使用来指明结果集中的变量。也可以用 GROUP BY 关键词来聚集特定的分组。

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.    
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.    
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.    

SELECT ?apartment ?bedrooms
WHERE {
    ?apartment swp:hasNumberOfBedrooms ?bedrooms.
}
OREDER BY DESC(?bedrooms)
```

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.    
PREFIX dbpedia: <http://www.dbpedia.org/resource/>.    
PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>.    

SELECT (AVG(?bedrooms) AS ?avgNumRooms)
WHERE {
    ?apartment swp:hasNumberOfBedrooms ?bedrooms
}
```

# 其他形式的 SPARQL 查询

除了 SELECT 以外， 其他的两种常用查询是 ASK 和 CONSTRUCT。ASK 形式的查询简单地检查一个数据集中是否存在一个图模式，而不是去返回结果。如果存在则返回真。CONSTRUCT 形式的查询用来从一个更大的 RDF 集中检索出一个 RDF 图。因此可以查询一个三元组存储库并检索一个 RDF 图而非一组变量绑定。CONSTRUCT 查询经常用来在模式(schema)之间转换--通过查询特定的模式，并用目标模式中的属性替换。

## 通过 SPARQL 更新来增加信息

当想插入三元组时，采用INSERT DATA 语句：

```
PREFIX swp: <http://www.semanticwebprimer.org/ontology/apartments.ttl#>.    
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>.

INSERT DATA{
    swp:LuxuryApartment rdfs:subClassOf swp:Apartment.
}
```

当想删除三元组时，一种是采用 DELETE DATA 关键词，它和插入的写法类似，注意这种形势下是不允许变量的，所有三元组都必须被完整指定。另一种更加灵活的方式是使用 DELETE WHERE 构造自，用来删除匹配指定图模式的那些三元组。最后，当要删除一个三元组存储库中所有内容时，可以用 CLEAR ALL 语句。

# 总结

* SPARQL 通过匹配图模式来选择信息，并提供基于数值和字符串比较的过滤机制。    
* SPARQL 查询采用类似 Turtle的语法。    
* 数据和模式(schema)都可以使用 SPARQL 来查询。    
* UNION 和 OPTIONAL 构造子允许 SPARQL 更容易地处理开放世界数据。    
* SPARQL 更新提供了从三元组存储库中更新和删除信息的机制。
