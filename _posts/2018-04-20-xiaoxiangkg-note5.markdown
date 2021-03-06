---
layout:     post
title:      "知识图谱入门 (五)" 
subtitle:   "知识存储"
date:       2018-04-20 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 知识存储，即获取到的三元组和schema如何存储在计算机中。本节从以Jena为例，对知识在数据库中的导入、存储、查询、更新做一个简要的介绍，而后对主流的图数据库进行介绍。

* TOC
{:toc}

# 图数据库简介

图数据库源起欧拉和图理论(graph theory),也称为面向/基于图的数据库，对应的英文是Graph Database。图数据库的基本含义是以“图”这种数据结构存储和查询数据。它的数据模型主要是以节点和关系(边)来体现，也可以处理键值对。它的优点是快速解决复杂的关系问题。

## Apache Jena 

Jena 是一个免费开源的支持构建语义网络和数据连接应用的Java框架。下图为Jena的框架：

![](/img/in-post/xiaoxiangkg_note5/xiaoxiangkg_note5_1.png)

其中，最底层的是数据库，包含SQL数据库和原生数据库，其中SDB用来导入SQL数据库， TDB导入RDF三元组。数据库之上的是内建的和外联的推理接口。在往上的就是SPARQL查询接口了。通过直接使用SPARQL语言或通过REfO等模块转换成SPARQL语言进行查询。

在上方我们看到有一个Fuseki模块，它相当于一个服务器端，我们的操作就是在它提供的端口上进行的。

### 数据的导入

数据导入分为两种方式，第一种是通过Fuseki的手动导入，第二种是通过TDB进行导入,对应的命令如下:

```
/jena-fuseki/tdbloader --loc=/jena-fuseki/data filename
```

数据导入后就可以启动Fuseki了，对应的命令如下:

```
/jena-fuseki/fuseki-server --loc=/jena-fuseki/data --update /music
```

### 查询

查询也有两种方式，第一种就是简单直接的通过Fuseki界面查询，另一种就是使用endpoint接口查询。

#### Endpoint接口查询

endpoint的SPARQL 查询网址为: http://localhost:3030/music/query;    
更新网址为：http://localhost:3030/music/update .

##### 查询举例

* 首先是最简单的单个语句查询,意在查询某一歌手所唱的所有歌曲：

```
SELECT DISTINCT ?trackID
WHERE {
    ?trackID track_artist artistID

}
```

可以看出查询语句整体和SQL很像的，下面多举几个例子。

* 查询某一位歌手所有歌曲的歌曲名:

```
SELECT ?name
WHERE {
    ?trackID track_artist artistID .
    ?trackID track_name ?name

}
```

* 使用CONCAT关键字进行连接，它的效果是在查询结果前增加一列叫专辑信息，它的结果以专辑名+ : + 查询结果组成：

```
SELECT ?歌曲id ?专辑id (CONCAT("专辑
                                   名",":",?专辑名) AS ?专辑信息)
WHERE {
    ?歌曲id track_name track_name .
    ?歌曲id track_album ?专辑id .
    ?专辑id album_name ?专辑名

}"))
```

* 其余还有LIMIT 关键字限制查询结果的条数

```
SELECT ?trackID
WHERE {
    ?albumID
    album_name album_name .
    ?trackID
    track_album ?albumID

}
LIMIT 2
```

* 使用COUNT进行计数；

```
SELECT (COUNT(?trackID) AS ?num)
WHERE {
    ?albumID album_name album_name .
    ?trackID track_album ?albumID

}
```

* 使用DISTINCT去重；

```
SELECT DISTINCT ?tag_name
WHERE {
    ?trackID track_artist artistID .
    ?trackID track_tag ?tag_name

}
```

* ORDER BY排序；

```
SELECT DISTINCT ?tag_name
WHERE {
    ?trackID track_artist artistID .
    ?trackID track_tag ?tag_name

}
ORDER BY DESC(?tag_name)
```

* UNION进行联合查询

```
SELECT (COUNT(?trackID ) AS ?num)
WHERE {
    {
        ?trackID track_tag tag_name .

    }
    UNION
    {
        ?trackID track_tag tag_name2 .
    }
}
```

* 使用FILTER对结果进行过滤

```
SELECT (count(?trackID ) as ?num)
WHERE {
    ?trackID track_tag ?trag_name
    FILTER (?tag_name = tag_name1 ||
           ?tag_name = tag_name2)

}
```

* ASK来询问是否存在,回答结果只有True或False

```
ASK
{
    ?trackID track_name ?track_name .
    FILTER regex(?track_name,‖xx‖)

}
```

##### 更新举例

在更新时要更换端口地址为: http://localhost:3030/music/update

* 使用INSERT DATA操作，对数据的属性和事例进行添加

```
INSERT DATA
{
    artistID artist_name artist_name .
}
```

* 使用WHERE定位，DELETE删除事例

```
DELETE
{
    artistID artist_name ?x .
}
WHERE
{
    artistID artist_name ?x .
}
```

对于更多的SPARQL用法请参见[官方文档](https://www.w3.org/TR/2013/REC-sparql11-query-20130321/)

#### 通过SPARQLWrapper 包查询和更新

首先通过pip安装SPARQLWrapper，而后就可以通过下图所示的方式进行查询了。具体的查询语句与端口的一样，此处不再赘述。

![](/img/in-post/xiaoxiangkg_note5/xiaoxiangkg_note5_2.png)

# 图数据库介绍

图数据库很多，其中开源的如RDF4j、gStore等。商业数据库如Virtuoso、AllegroGraph、Stardog等。原生图数据库如Neo4j、OrientDB、Titan等，涉及内容较广，我也是刚刚入门，不足以从大体上介绍，因此只对我打算用的几个图数据库进行简单介绍，其余的可以自己查阅文档了解。

图数据库的分类与发展如下图所示：

![](/img/in-post/xiaoxiangkg_note5/xiaoxiangkg_note5_3.png)

## 开源图数据库

### [RDF4j](http://docs.rdf4j.org/migration/)

它是处理RDF数据的Java框架，使用简单可用的API来实现RDF存储。支持SPARQL 查询和两种RDF存储机制，支持所有主流的RDF格式。

### [gStore](http://www.gstore-pku.com)

gStore从图数据库角度存储和检索RDF知识图谱数据， gStore支持W3C定义的SPARQL 1.1标准,包括含有Union,OPTIONAL,FILTER和聚集函数的查询;gStore支持有效的增删改操作。 gStore单机可以支持1Billion(十亿)三元组规模的RDF知识图谱的数据管理任务。

## 商业图数据库介绍

### [Virtuoso](http://virtuoso.openlinksw.com)

智能数据， 可视化与整合。可扩展和高性能数据管理，支持Web扩展和安全

### [Allgrograph](http://www.franz.com/agraph/allegrograph)

AllegroGraph是一个现代的高性能的，支持永久存储的图数据库。它基于Restful接入支持多语言编程。具有强大的加载速度、查询速度和高性能。

## 原生图数据库 

### Neo4j

Neo4j是一个高性能的,NOSQL图形数据库，它将结构化数据存储在网络上而不是表中。它是一个嵌入式的、基于磁盘的、具备完全的事务特性的Java持久化引擎，但是它将结构化数据存储在网络(从数学角度叫做图)上而不是表中。Neo4j也可以被看作是一个高性能的图引擎，该引擎具有成熟数据库的所有特性。内置Cypher 查询语言。

Neo4j具有以下特性：

* 图数据库 + Lucene索引    
* 支持图属性    
* 支持ACID    
* 高可用性    
* 支持320亿的结点,320亿的关系结点,640亿的属性

Neo4j的优点为：    
* 高连通数据    
* 推荐    
* 路径查找    
* A\*算法    
* 数据优先

# Ref
 
[王昊奋知识图谱教程](http://www.chinahadoop.cn/course/1048)
