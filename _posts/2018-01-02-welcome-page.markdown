---
layout:     post
title:      "庆祝小博客装修完成"
subtitle:   "Welcome to my simple blog"
date:       2018-01-02 17:17:00
author:     "Pelhans"
header-img: "img/post-bg-nextgen-web-pwa.jpg"
header-mask: 0.3
catalog:    true
tags:
    - NLP
    - ML
---


> 新年新气象~ Happy new year！！ <br><br>

* TOC
{:toc}

## 自然语言

到现在为止学习自然语言正好半年了,看了很多书籍、教程，也码了很多代码。但发现有很多东西会渐渐忘掉，以后复习的时候还要重新看，太麻烦。于是有了建立个自己小博客写下自己笔记感想的想法。程序员嘛，都想有个自己的网站，但奈何网络编程一窍不通。。。多亏原作者开源这个好用的框架，我也就沾光能有一个静态的小窝~开心。

在学习的过程中写了一个小的[中文语言处理工具包](https://github.com/Pelhans/ZNLP/)，大部分是基于神经网络的，本意时用来深入理解背后的理论和坑，但现在看着它的功能一点点增加，也就有了既然做就做好的想法，希望有一天它能像HanNLP、LTP那样具备良好的性能，广为流传~

好啦~做梦结束，新的一年，继续补充基础吧~


<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: 'location.href', // 可选。默认为 location.href
  owner: 'pelhans',
  repo: 'pelhans.github.io',
  oauth: {
    client_id: 'dbec37728f2282bb2d97',
    client_secret: 'b602c6c6c0f484eb0894d15d6a5898f5d1f13438',
  },
})
gitment.render('container')
</script>
