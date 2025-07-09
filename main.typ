#import "template.typ": *

#show: template.with(
  // 笔记标题
  title: [集合通信算法研究分析报告],
  // 在页眉展示的短标题（选填）
  short-title: "集合通信算法研究分析报告",
  // 笔记描述（选填）
  description: [
    /**/ \ summer 2025
  ],
  // 笔记创建日期（选填）
  date: datetime(year: 2025, month: 7, day: 6),
  // 作者信息（除 name 外，其他参数选填）
  authors: (
    (
      name: "WizHUA",
      github: "https://github.com/WizHUA",
      // homepage: "https://github.com/WizHUA",
      // affiliations: "1",
    ),
  ),

  // 所属组织列表，每一项包括一个 id 和 name。这些将显示在作者下方。
  affiliations: (
    (id: "", name: "NUDT 计算机学院"),
    // (id: "2", name: "Example Inc."),
  ),

  // 参考书目文件路径及引用样式
  bibliography-file: "refs.bib",
  bibstyle: "gb-7714-2015-numeric",

  // 页面尺寸，同时会影响页边距。
  paper-size: "a4",

  // 中英文文本和代码的字体
  fonts: (
    (
      en-font: "Linux Libertine",
      // zh-font: "Noto Sans CJK SC",
      zh-font: "",
      code-font: ("DejaVu Sans Mono", "LXGW WenKai Mono"),
    )
  ),
  
  // 主题色
  accent: orange,
  // 封面背景图片（选填图片路径或 none）
  cover-image: "./figures/cover-image.png",
  // 正文背景颜色（选填 HEX 颜色或 none）
  // background-color: "#FAF9DE"
  background-color: "#FAF9DE"
)

// 摘要
#include "content/00-abstract.typ"

// 第一部分：项目概述与研究背景
#include "content/01-overview.typ"

// 第二部分：Open MPI集合通信算法源码分析
#include "content/02-source-analysis.typ"

// 第三部分：集合通信参数配置分析  
#include "content/03-parameter-analysis.typ"

// 第四部分：数据集构建与应用场景设计
#include "content/04-dataset-construction.typ"

// 第五部分：机器学习建模与性能预测
#include "content/05-ml-modeling.typ"

// 第六部分：模型验证与检验
#include "content/06-validation.typ"

// 第七部分：结果分析与总结
#include "content/07-results-analysis.typ"

// 结论与展望
#include "content/08-conclusion.typ"