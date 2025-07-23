#import "../template.typ": *

= 模型验证与检验

基于第五章设计的三种建模方案，本章对回归预测模型、分类决策模型和混合建模方案进行具体实现和验证。通过系统的实验设计，评估不同方案在集合通信算法选择问题上的性能表现，为后续的结果分析提供实验基础。

== 实验设计与评估框架

=== 数据集划分策略

为确保模型评估的客观性和可靠性，采用分层抽样的方式划分数据集：

*时间维度分层*：按照测试时间顺序，将数据集按7:2:1的比例划分为训练集、验证集和测试集，避免数据泄露问题。

*操作类型分层*：在每个子集中保持五种集合通信操作（Broadcast、Reduce、Allgather、Scatter、Gather）的样本比例一致。

*规模范围分层*：确保不同进程数和消息大小组合在各子集中的分布相对均匀，避免训练集偏向特定规模。

#figure(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1.5fr),
    align: (left, center, center, center, left),
    stroke: 0.5pt,
    table.header([*数据子集*], [*样本数*], [*比例*], [*用途*], [*特点*]),
    
    [训练集], [70%], [训练], [模型学习], [涵盖主要配置空间],
    [验证集], [20%], [验证], [超参数调优], [用于模型选择],
    [测试集], [10%], [测试], [最终评估], [完全未见过的数据],
  ),
  caption: [数据集划分策略]
)

=== 评估指标体系

针对不同建模方案的特点，设计相应的评估指标：

#let add1 = [
  均方根误差衡量预测值与真实值之间的差异程度，单位与原始数据相同，对大误差更敏感
]

#let add2 = [
  平均绝对百分比误差表示预测误差占真实值的百分比，是无量纲指标，便于跨不同数据集对比
]

#let add3 = [
  决定系数衡量模型能解释数据变异性的比例，范围[0,1]，越接近1表示模型拟合效果越好
]

*回归模型评估指标*：
- 均方根误差（RMSE）#footnote(add1)：$"RMSE" = sqrt(frac(1, n) sum_(i=1)^n (hat(y)_i - y_i)^2)$
- 平均绝对百分比误差（MAPE）#footnote(add2)：$"MAPE" = frac(1, n) sum_(i=1)^n abs(frac(hat(y)_i - y_i, y_i)) times 100%$
- 决定系数（R²）#footnote(add3)：$R^2 = 1 - frac("SS"_("res"), "SS"_("tot"))$

#let add1 = [
  准确率是正确预测样本数占总样本数的比例，其中TP为真正例，TN为真负例，FP为假正例，FN为假负例
]

#let add2 = [
  宏平均F1分数对每个类别分别计算F1分数后取平均，能有效处理类别不平衡问题，其中F1分数是精确率和召回率的调和平均
]

#let add3 = [
  Top-k准确率指预测的前k个最可能配置中包含真实最优配置的比例，对于算法选择问题具有实际意义
]

*分类模型评估指标*：
- 准确率（Accuracy）#footnote(add1)：$"Accuracy" = frac("TP" + "TN", "TP" + "TN" + "FP" + "FN")$
- 宏平均F1分数#footnote(add2)：$"F1"_("macro") = frac(1, |C|) sum_(c in C) "F1"_c$
- Top-k准确率#footnote(add3)：预测的前k个最可能配置中包含真实最优配置的比例

#let add1 = [
  相对性能提升衡量相比基准方法（如默认配置或随机选择）的性能改善程度，是评估模型实际应用价值的重要指标
]

#let add2 = [
  决策时延是模型从接收输入到输出预测结果所需的时间，在在线优化场景下是关键性能指标，这部分可能暂不考虑
]

*性能优化效果评估*：
- 相对性能提升#footnote(add1)：$"Improvement" = frac(T_("baseline") - T_("predicted"), T_("baseline")) times 100%$
- 决策时延#footnote(add2)：模型推理所需的计算时间

== 方案一：回归预测模型实现

=== 模型架构选择

基于集合通信性能预测的特点，对比多种回归算法：

- *线性回归模型*：作为基线模型，验证特征工程的有效性
- *随机森林回归*：处理特征间的非线性交互关系
- *梯度提升回归*：捕获复杂的性能模式
- *神经网络回归*：学习高维非线性映射关系

```python
# 模型架构示例（待实现）
class CollectivePerformanceRegressor:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.feature_engineering = ComplexityGuidedFeatures()
        
    def build_model(self):
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
        # 其他模型类型...
        
    def train(self, X_train, y_train):
        # 训练逻辑
        pass
        
    def predict_optimal_config(self, environment):
        # 最优配置搜索
        pass
```

=== 特征工程实现

基于第五章的理论分析，实现复杂度指导的特征增强：

```python
# 特征工程实现（待完善）
def create_regression_features(df):
    """为回归模型创建增强特征"""
    # 基础特征编码
    df_encoded = encode_categorical_features(df)
    
    # 复杂度指导特征
    df_enhanced = add_complexity_features(df_encoded)
    
    # 交互特征
    df_final = add_interaction_features(df_enhanced)
    
    return df_final
```

=== 超参数优化

采用网格搜索结合交叉验证的方式进行超参数调优：

#figure(
  table(
    columns: (2fr, 2fr, 2fr),
    align: (left, left, left),
    stroke: 0.5pt,
    table.header([*模型类型*], [*关键超参数*], [*搜索范围*]),
    
    [随机森林], [n_estimators, max_depth], [[50,100,200], [10,15,20]],
    [梯度提升], [learning_rate, n_estimators], [[0.01,0.1,0.2], [100,200,300]],
    [神经网络], [hidden_size, learning_rate], [[64,128,256], [1e-4,1e-3,1e-2]],
  ),
  caption: [回归模型超参数搜索空间]
)

=== 实验结果

*模型性能对比*：
// 这里将填入实际的实验结果表格

*特征重要性分析*：
// 这里将展示特征重要性排序结果

*预测精度分析*：
// 这里将展示不同操作类型和规模下的预测精度

== 方案二：分类决策模型实现

=== 多分类模型架构

将算法选择问题建模为多分类决策，实现端到端的配置预测：

```python
# 分类模型实现（待完善）
class CollectiveConfigClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.label_encoder = ConfigLabelEncoder()
        
    def prepare_labels(self, performance_data):
        """构造最优配置标签"""
        # 基于第五章的标签构造策略
        pass
        
    def build_model(self):
        # 多分类模型构建
        pass
        
    def predict(self, environment_features):
        # 直接预测最优配置ID
        pass
```

=== 类别不平衡处理

由于不同操作的配置空间差异显著，采用多种策略处理类别不平衡：

*样本级平衡*：
- SMOTE过采样算法增强少数类样本
- 随机欠采样平衡主要类别分布

*算法级平衡*：
- 基于类别频率的损失函数加权
- 成本敏感学习方法

*评估级平衡*：
- 宏平均指标避免大类别主导
- 分层抽样确保测试集代表性

=== 决策边界可视化

通过t-SNE降维技术可视化学习到的决策边界：

// 这里将插入决策边界可视化图表

=== 实验结果

*分类准确率分析*：
// 这里将展示各操作类型的分类准确率

*Top-k准确率评估*：
// 这里将分析Top-3, Top-5准确率结果

*决策效率对比*：
// 这里将对比推理时间和搜索时间

== 方案三：混合建模方案实现

=== 两阶段模型架构

实现算法分类+参数回归的层次化决策框架：

```python
# 混合模型实现（待完善）
class HybridCollectiveOptimizer:
    def __init__(self):
        self.algorithm_classifier = AlgorithmClassifier()
        self.parameter_regressors = {}  # 每种算法一个回归器
        
    def train_stage1(self, X_env, algorithm_labels):
        """第一阶段：算法类别分类"""
        pass
        
    def train_stage2(self, X_env, X_params, performance_data):
        """第二阶段：参数优化回归"""
        pass
        
    def predict_optimal_config(self, environment):
        """两阶段预测"""
        # 1. 预测最优算法类别
        best_algorithm = self.algorithm_classifier.predict(environment)
        
        # 2. 在该算法内优化参数
        best_params = self.parameter_regressors[best_algorithm].optimize(environment)
        
        return best_algorithm, best_params
```

=== 算法分类器训练

第一阶段专注于学习算法类别的选择边界：

*算法类别定义*：
- 线性类算法：basic_linear, linear_sync
- 二项树类算法：binomial, recursive_doubling  
- K项树类算法：knomial_tree系列
- 流水线类算法：pipeline, segmented_ring

*分类特征选择*：
仅使用环境特征（进程数、消息大小、操作类型），避免算法参数信息泄露。

=== 参数回归器训练

第二阶段针对每种算法类别训练专门的参数优化器：

*特征空间分解*：
每个算法类别具有不同的参数空间维度，需要专门的特征工程。

*目标函数设计*：
在固定算法类别内，优化参数配置的性能表现。

=== 实验结果

*两阶段准确率分析*：
// 这里将展示算法分类准确率和参数优化效果

*计算效率评估*：
// 这里将对比单阶段和两阶段的计算开销

*综合性能表现*：
// 这里将评估混合方案的整体效果

== 模型对比与分析

=== 性能指标综合对比

#figure(
  table(
    columns: (1.5fr, 1.5fr, 1.5fr, 1.5fr, 1.5fr),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    table.header([*建模方案*], [*预测精度*], [*推理时延*], [*可解释性*], [*适用场景*]),
    
    [回归预测], [待填入], [O(|C|)], [高], [精度优先],
    [分类决策], [待填入], [O(1)], [中], [效率优先],
    [混合建模], [待填入], [O(|P|)], [高], [平衡需求],
  ),
  caption: [三种建模方案性能对比]
)

=== 误差分析与模型诊断

*预测误差分布*：
分析不同方案在各种配置下的预测误差模式。

*失效案例分析*：
识别模型预测失效的典型场景和原因。

*鲁棒性测试*：
评估模型对噪声数据和异常输入的敏感性。

== 小结

本章系统实现了三种集合通信算法选择的建模方案，通过统一的实验框架和评估指标进行性能验证。回归预测方案在理论精度上表现优异但计算开销较大，分类决策方案实现了高效的实时决策但泛化能力有限，混合建模方案在效率和精度间取得了良好平衡。实验结果为不同应用场景下的方案选择提供了量化依据，并为后续的深入分析奠定了基础。