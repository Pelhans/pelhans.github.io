---
layout:     post
title:      "知识图谱入门 (一)" 
subtitle:   "知识图谱与语义技术概览"
date:       2018-03-15 00:15:18
author:     "Pelhans"
header-img: "img/post-bg-universe.jpg"
header-mask: 0.3 
catalog:    true
tags:
    - Knowledge Graph
---


> 知识图谱与语义技术概览。主要介绍知识表示、知识抽取、知识存储、知识融合、知识推理、知识众包、语义搜索、知识问答等内容。同时还包含一些典型的应用案例。若理解有偏差还请指正。

* TOC
{:toc}

#  知识图谱与语义技术概览

## 知识图谱的概念演化

知识图谱(Knowledge Graph， KG)的概念演化可以用下面这幅图来概括:

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_1.png)

在1960年，语义网络(Semantic Networks)作为知识表示的一种方法被提出，主要用于自言语言理解领域。它是一种用图来表示知识的结构化方式。在一个语义网络中，信息被表达为一组结点，结点通过一组带标记的有向直线彼此相连，用于表示结点间的关系。如下图所示。简而言之，语义网络可以比较容易地让我们理解语义和语义关系。其表达形式简单直白，符合自然。然而，由于缺少标准，其比较难应用于实践。

![语义网络示意图](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_2.png)

1980s出现了本体论(Ontology)，该本体是由哲学概念引入到人工智能领域的，用来刻画知识。在1989年Time Berners-Lee发明了万维网，实现了文本间的链接。

1998年语义网(THe Semantic Web)被提出，它从超文本链接到语义链接。语义网是一个更官方的名称，也是该领域学者使用得最多的一个术语，同时，也用于指代其相关的技术标准。在万维网诞生之初，网络上的内容只是人类可读，而计算机无法理解和处理。比如，我们浏览一个网页，我们能够轻松理解网页上面的内容，而计算机只知道这是一个网页。网页里面有图片，有链接，但是计算机并不知道图片是关于什么的，也不清楚链接指向的页面和当前页面有何关系。语义网正是为了使得网络上的数据变得机器可读而提出的一个通用框架。“Semantic”就是用更丰富的方式来表达数据背后的含义，让机器能够理解数据。“Web”则是希望这些数据相互链接，组成一个庞大的信息网络，正如互联网中相互链接的网页，只不过基本单位变为粒度更小的数据，如下图。

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_3.png)

2006年Tim突出强调语义网的本质是要建立开放数据之间的链接，即链接数据(LInked Data)。2012年谷歌发布了其基于知识图谱的搜索引擎产品。可以看出，知识图谱的提出得益于Web的发展和数据层面的丰富，有着来源于知识表示(Knowledge Represention， KR)、自然语言处理(NLP)、Web、AI多个方面的基因。可用于搜索、问答、决策、AI推理等方面。

## 知识图谱的本质

知识图谱目前没有标准的定义，这里引用一下“Exploiting Linked Data and Knowledge Graphs in Large Organisations”这本书对于知识图谱的定义：
>A knowledge graph consists of a set of interconnected typed entities and their attributes.

即**知识图谱是由一些相互连接的实体和它们的属性构成的**。最简单情况下它长这样：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_4.png)

复杂一些是这样的：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_5.png)

前面说过，知识图谱综合了众多方面，其中从Web角度看KG，它像建立文本之间的超链接一样，建立数据之间的语义链接，并支持语义搜索。从NLP角度看，它主要在做怎么能够从文本中抽取语义和结构化的数据。从知识表示角度看是怎么利用计算机符号来表示和处理知识。从AI角度则是怎么利用知识库来辅助理解人类的语言。从数据库角度看就是用图的方式存储知识。因此要做好KG要综合利用好KR、NLP、Web、ML、DB等多方面的方法和技术。

# 知识图谱技术概览

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_6.png)

上图表示了知识图谱的技术体系，首先在最底层我们有大量的文本、结构化数据库、多媒体文件等数据来源。通过知识抽取、知识融合、知识众包等技术，获取我们需要的数据，而后通过知识表示和知识推理、知识链接等将知识规范有序的组织在一起并存储起来。最终用于知识问答、语义搜索、可视化等方面。

## 知识表示

知识表示研究怎么利用计算机符号来表示人脑中的知识，以及怎么通过符号之间的运算来模拟人脑的推理过程。

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_7.png)

上图给出了知识表示的演化过程，其中最主要根本的变化是从基于数理逻辑的知识表示过渡到基于向量空间学习的分布式知识表示。

下图给出官方推荐的语义网知识表示框架：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_8.png)

其中最底层的是URI/IRI是网络链接，其上是XML和RDF为资源表示框架。SPARQL是知识查询语言。被蓝色部分覆盖的是推理模块，它包含了如RDFS和OWL这样的支持推理的表示框架。在网上就是trust和interaction部分，暂时不需要了解(还不清楚是什么，只知道用不到。。。)。

### RDF
RDF(Resource Description Framework)即资源描述框架，是W3C制定的。用于描述实体/资源的标准数据模型。在知识图谱中，我们用RDF形式化地表示三元关系。(Subject, predicate, object)。例如:

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_9.png)

RDFS在RDF的基础上定义了一些固定的关键词如：Class，subClassOf，type， Property， subPropertyOf， Domain， Range以及多了Schema层。它的表示为：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_10.png)

### OWL

OWL(Web Ontology Language), 这个本体就是从哲学那面借鉴来的。OWL在RDF的基础上扩充了Schema层，使它支持推理等操作。如：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_11.png)

### SPARQL

SPARQL是RDF的查询语言，它基于RDF数据模型，可以对不同的数据集撰写复杂的连接，由所有主流的图数据库支持。其操作如：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_12.png)

### JSON-LD

JSON for Linking Data: 适用于作为程序之间做数据交换,在网页中嵌入语义数据和Restful Web Service。存储格式如:

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_13.png)

## 知识图谱的分布式表示--KG Embedding

其实看到 Embedding这个词我们就知道，它是一个向量嵌入。详细来说就是在保留语义的同时，将知识图谱中的实体和关系映射到连续的稠密的低维向量空间。

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_14.png) 

## 知识抽取

知识抽取是一个结合NLP和KR的工作，它的目标是抽取KR用的三元组、多元关系、模态知识等。具体流程如下：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_15.png)

文字表述为，首先从网络上获取大量的各种非结构化的文本数据，经过文本预处理后得到干净的文本数据。而后借助机器学习相关程序对文本进行分词、词性标注、词法解析、依存分析等工作，此时词法及句法层次的分析结束，接下来对该文本进行NER和实体链接工作，为关系抽取和时间抽取做准备，最终形成KR用的三元组、多元关系、模态知识等构成知识图谱。

## 知识问答

知识问答(Knowledge-Based Question Answering， KBQA)是基于知识库的问题回答，它以直接而准确的方式回答用户自然语言提问的自动问答系统，它将构成下一代搜索引擎的基本形态。如搜索姚明的身高，就可以给出226cm的回答。其实现流程为：

![](/img/in-post/xiaoxiangkg_note1/xiaoxiangkg_note1_16.png)

## 知识推理

简单而言，推理就是指基于已知事实推出未知的事实的计算过程，例如回答张三儿子的爸爸是谁？按照解决方法分类可分为：基于描述逻辑的推理、基于规则挖掘的推理、基于概率逻辑的推理、基于表示学习与神经网络的推理。按照推理类型分类可分为：缺省推理、连续变化推理、空间推理、因果关系推理等等。

## 知识融合

实体融合(Knowledge Fusion),也叫数据连接(Data Linking)等，目的是在不同的数据集中找出一个实体的描述记录，主要目的是对不同的数据源中的实体进行整合，形成更加全面的实体信息。典型的工具为Dedupe(一个基于python的工具包)和LIMES。

## 知识众包

允许各网站基于一定的方式如RDFa、JASON-LD等方式在网页和邮件等数据源中嵌入语义化数据，让个人和企业定制自己的知识图谱信息。
