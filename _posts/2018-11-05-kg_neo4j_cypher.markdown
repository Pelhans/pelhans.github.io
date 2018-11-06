---
layout:     post
title:      "从零开始构建知识图谱（六）"
subtitle:   "将数据存进Neo4j"
date:       2018-11-06 00:15:18
author:     "Pelhans"
header-img: "img/kg_bg.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - knowledge graph
---


> 图数据库是基于图论实现的一种新型NoSQL数据库。它的数据数据存储结构和数据的查询方式都是以图论为基础的。图论中图的节本元素为节点和边，对应于图数据库中的节点和关系。

* TOC
{:toc}

# 1. Cypher 简介

Neo4j 是由 Java 实现的开源 NoSQL 图数据库，它提供了完整的数据库特性，包括ADID事务的支持、集群的支持、备份与故障转移等。

Cypher 是一种声明式的图数据库查询语言，能高效地查询和更新图数据。Cypher 语句可分为三类，包括读语句、写语句和通用语句：

* 读语句： MATCH、OPTIONAL MATCH、WHERE、START、AGGREGATION、LOAD CSV。    
* 写语句： CREATE、MERGE、SET、DELETE、REMOVE、FOREACH、CREATE UNIQUE。
* 通用语句： RETURN、ORDER BY、LIMIT、SKIP、WITH、UNWIND、UNION、CALL。

熟悉SQL语句的应该能根据上面这些猜出大概用途了。

# 2. MYSQL 数据的导出

MYSQl 支持将查询结果直接导出为 CSV 文本格式，在使用 SELECT 语句查询数据时，在语句后面加上导出指令即可，格式如下：

* into outfile < 导出的目录和文件名>： 指定导出的目录和文件名。    
* fields terminated by <字段间分隔符>：定义字段间的分隔符。    
* optionally enclosed by <字段包围符>：定义包围字段的字符(数值型字段无效)。    
* lines terminated by <行间分隔符>：定义每行的分隔符。

下面我们基于[从零开始构建是知识图谱(一)](https://zhuanlan.zhihu.com/p/43447848)中建立的互动百科数据库，导出 actor 表中所有数据

其中 actor 表的属性为：

```
# 演员 : ID, 简介， 中文名，外文名，国籍，星座，出生地，出生日期，代表作品，主要成就，经纪公司；
# actor: actor_id, actor_bio, actor_chName, actor_foreName, actor_nationality,  actor_constellation, actor_birthPlace, actor_birthDay, actor_repWorks, actor_achiem, actor_brokerage;
```
mysql 中导出表的代码为：

```
SELECT * FROM actor into outfile '/var/lib/mysql-files/hudong_actor.csv' fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n';
```

hudong_actor.csv 的第一行数据为：

```
1,"周星驰（Stephen Chow），1962年6月22日生于香港，华语喜剧演员、导演、编剧、监制、制片人、出品人。1980年成为丽的电视特约演员，开始出道。1988年初涉影坛，后相继主演《唐伯虎点秋香》、《大话西游》等，自编自导自演《国产凌凌漆》、《食神》、《功夫》等多部影片，6度打破香港电影票房纪录，并获得8个香港电影年度票房冠军，创下打破票房纪录次数及获得年度票房冠军次数的纪录。2003年当选《时代周刊》“年度风云人物”，并成为“亚洲英雄”的封面人物。2004年《功夫》创下数十个国家和地区的华语电影票房纪录，并被《时代周刊》评为“2005年十大佳片”之一。2013年导演的《西游·降魔篇》破23项华语电影票房纪录，全球票房达2.18亿美元，刷新华语电影全球票房纪录。2014年执导的科幻电影《美人鱼》开拍，该片已于2016年2月8日上映，上映19天累计票房超过30亿，刷新了华语电影票房记录。2017年1月28日，担任监制、编剧的古装喜剧片《西游伏妖篇》上映。编辑摘要","周星驰","Stephen Chow","中国","巨蟹座","香港","None","唐伯虎点秋香、功夫","获得两届香港电影金像奖最佳影片 第21届香港电影金像奖最佳导演 第42届台湾电影金马奖最佳导演","创办星辉电影公司和上市比高集团"
```

类比于此，我们可以获得 movie 表、电影类型 genre 表、actor-movie表、 movie-genre 表中的数据：

```
SELECT * FROM movie into outfile '/var/lib/mysql-files/hudong_movie.csv' fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n';

SELECT * FROM genre into outfile '/var/lib/mysql-files/hudong_genre.csv' fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n';

SELECT * FROM actor_to_movie into outfile '/var/lib/mysql-files/hudong_actor_to_movie.csv' fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n';

SELECT * FROM movie_to_genre into outfile '/var/lib/mysql-files/hudong_movie_to_genre.csv' fields terminated by ',' optionally enclosed by '"' lines terminated by '\r\n';
```

# 3 将数据导入倒 Neo4j 中
## 3.1 导入实体

我们首先将 actor 实体导入到 Neo4j 中，对应文件为 hudong_actor.csv：

```
LOAD CSV FROM 'file:///hudong_actor.csv' AS line CREATE (:Actor { actor_id: line[0], actor_bio: line[1], actor_chName: line[2], actor_foreName: line[3],actor_nationality: line[4], actor_constellation: line[5], actor_birthPlace:  line[6], actor_birthDay: line[7], actor_repWorks: line[8], actor_achiem: line[9], actor_brokerage: line[10] })
```

其中 LOAD CSV FROM "file-url" AS line 是将指定路径下的 CSV 文件读取出来，"file-url" 就是文件的地址，它可以是本地文件路径也可以是网址。需要注意的是，对于本地文件，该路径默认为 import 文件夹，要想读取其他文件夹下的文件，需要修改配置文件才可以。

上面命令在读取 CSV 文件后，调用 CREATE 命令创建了对应的节点，也就是用圆括号 (:Actor {... line[10]}) 包起来的部分。节点的标签为 :Actor，节点的属性为 花括号{} 包起来的部分。包含 actor_id、actor_bio 等。属性的值从 line 中读取，line[0] 表示 CSV 文件中的第一列，其他列以此类推。

查看导入结果：

```
MATCH (n:Actor) RETURN n LIMIT 25;
```

![](/img/in-post/kg_neo4j_cypher/actor.png)


通过上面的命令我们就创建了 actor 实体的节点，下面我们按照类似的命令创建 movie 的节点和 genre 节点：

```
LOAD CSV FROM 'file:///hudong_movie.csv' AS line CREATE (:Movie { movie_id: line[0], movie_bio: line[1], movie_chName: line[2], movie_foreName: line[3],movie_prodTime: line[4], movie_prodCompany: line[5], movie_director: line[6], movie_screenwriter: line[7], movie_genre: line[8], movie_star: line[9], movie_length: line[10], movie_rekeaseTime: line[11], movie_language: line[12],  movie_achiem: line[13]  });

LOAD CSV FROM 'file:///hudong_genre.csv' AS line CREATE (:Genre { genre_id:  line[0], genre_name: line[1]  });
```

最终我们有 actor 节点 5392 个，属性 65252。movie 节点 13865 个，属性194110. genre 节点11，属性 11。


## 3.2 导入关系

前面我们只是建立的演员和电影的节点，并没有关系链接它们。关系在 MYSQL中 对应于 actor_to_movie 表 和 movie_to_genre 表，分别对应 演员 :ACTED_IN 电影 的关系 和 电影 :Belong_to 类别 关系。

导入演员 :ACTED_IN 电影关系对应的语句为：

```
LOAD CSV FROM 'file:///hudong_actor_to_movie.csv' AS line MATCH (a:Actor), (m:Movie) WHERE a.actor_id = line[1] AND m.movie_id = line[2] CREATE (a) - [r:ACTED_IN] -> (m) RETURN r;
```

其中采用 LOAD CSV FROM 语句读取 CSV 文件，而后采用 CREATE 语句创建关系。第一对括号 (a:Actor {actor_id: line[1]}) 表示对于 line[1] 中给出的 actor_id 对应的节点 :Actor 节点a。第二对括号 (m:Movie {movie_id: line[2]}) 表示 line[2] 给出的 movie_id 对应的 :Movie 节点 m。它们之间的 - [r:ACTED_IN] -> 表示一个有向的关系r,它的标签为:ACTED_IN。因此上述语句就是先找到节点a 和 m，而后创建它们之间的关系 r。

运行语句后，我们得到 actor :ACTED_IN movie 关系 800个。

查看导入结果：

```
MATCH p=()-[r:ACTED_IN]->() RETURN p LIMIT 25;
```

![](/img/in-post/kg_neo4j_cypher/actor_movie.png)


与此类似，我们可以建立 电影 :Belong_to 类别 关系。

```
LOAD CSV FROM 'file:///hudong_movie_to_genre.csv' AS line CREATE (a:Movie {movie_id: line[1]}) - [r:Belong_to] ->(m:Genre {genre_id: line[2]});
```

获得 movie :Belong_to genre 关系 14558 个。

查看导入结果：

```
MATCH p=()-[r:Belong_to]->() RETURN p LIMIT 25;
```

![](/img/in-post/kg_neo4j_cypher/movie_to_genre.png)


最终我们来一个总的结果：

```
MATCH p=() -[] -> () - [r:Belong_to]->() RETURN p LIMIT 25
```

![](/img/in-post/kg_neo4j_cypher/actor_movie_genre.png)


