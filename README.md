<div align="center">

# CAM-15/35: 可解释中文文本分类的共现关联矩阵框架

**基于局部共现模式的白盒特征提取与轻量级决策系统**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

</div>

## 📋 摘要 (Abstract)

CAM-15/35是一个面向中文短文本分类的**白盒可解释特征提取框架**，通过构建字符级共现关联矩阵（Co-occurrence Association Matrix）捕捉局部语义结构。区别于黑盒深度学习模型，本框架提供：
- **结构化稀疏特征**：15维（CAM-15）或35维（CAM-35）可解释特征向量
- **极端模型压缩**：通过LZMA+X双重压缩，完整模型体积可控制在**800KB级别**（对比TF-IDF方案的18MB）
- **高速推理**：292.87 samples/sec（约为TF-IDF+XGBoost的**2.05倍**）
- **白盒决策逻辑**：基于决策树的IF-THEN规则提取，支持置信度校准与OOD检测

本项目实现了从原始语料到可解释分类器的完整流水线，包括语料统计、词汇构建、共现矩阵估计、关联矩阵生成、特征压缩与白盒建模六大核心模块。

## 🚀 核心创新 (Key Innovations)

### 1. 层次化共现特征架构 (Hierarchical Co-occurrence Architecture)
- **CAM-15模式**：单一层级15维特征（中心自相关+前后向关联+统计矩+位置编码）
- **CAM-35模式**：层次化融合
  - Word层（15D）：字符级共现，窗口大小5
  - Phrase层（15D）：Bigram短语共现，窗口大小3  
  - Sentence层（5D）：长距离依赖，窗口大小10

### 2. 极端压缩存储方案 (Extreme Compression Strategy)
采用**双重压缩机制**：
- 第一层：Numpy结构化数组（较Pickle减少70%体积）
- 第二层：LZMA/XZ算法（压缩级别3-9可调）
- 内存映射加载：支持大模型RAM友好型推理

### 3. 白盒可解释性引擎 (Whitebox Explainability Engine)
- **决策规则提取**：从CCP剪枝决策树提取人类可读的IF-THEN规则
- **置信度校准**：基于Laplace平滑、样本支持度与规则复杂度修正原始置信度，解决过拟合导致的虚假高置信问题
- **特征重要性分析**：字符级特征贡献度追踪与可视化

### 4. 多尺度局部邻域构建 (Multi-scale Local Neighborhoods)
支持Unigram/Bigram/Trigram三种粒度：
- 正弦位置编码（Sinusoidal Position Encoding）
- 方向掩码（Direction Mask）：区分前向/后向关联
- 距离衰减权重（Distance Decay）：$$
w = \exp(-\lambda \cdot d)
$$

### 5. 语义增强共现计数 (Semantic-weighted Co-occurrence)
可选集成Word2Vec语义相似度：
- 基于余弦相似度的动态权重：$weight = \frac{1 + \cos(v_i, v_j)}{2}$
- 支持预训练向量或语料内训练轻量级模型

## 📊 性能基准 (Benchmarks)

在20,425条中文短文本（41分类）数据集上的对比实验：

| 指标 | CAM-15 (本框架)       | TF-IDF+PCA+XGBoost | 相对优势         |
|------|--------------------|--------------------|--------------|
| **模型体积** | ~800 KB            | ~18 MB             | **压缩率23×**   |
| **推理速度** | 292.87 samples/sec | 142.51 samples/sec | **+105%**    |
| **交叉验证准确率** | 40.08%*            | 60.63% ± 0.45%     | **TF-IDF占优** |
| **特征维度** | 15 / 35            | 15 (PCA后)          | 信息保留更优       |
| **可解释性** | 完整规则链              | 黑盒                 | 白盒透明         |

*注：CAM-15的准确率有待提升，但是不能排除是算法结构设计的根本问题，后续会持续更新考究*

## 🛠️ 快速开始 (Quick Start)

### 环境配置
```bash
# 克隆仓库
git clone https://github.com/ChaoJiht666/CAM-15.git
cd CAM-15

# 安装依赖
pip install -r requirements.txt
# 核心依赖：numpy, scipy, pandas, scikit-learn, joblib, jieba, tqdm
# 可选依赖：xgboost, gensim（用于语义增强）
```

### 数据格式
CSV文件需包含两列：
- `text`: 原始文本内容
- `title`: 类别标签

示例：
```csv
text,title
"故宫位于北京中轴线上，是中国明清两代的皇家宫殿",文化科普
"这家餐厅的烤鸭非常正宗，推荐尝试",美食推荐
```

### 一键训练 (CAM-15模式)
```bash
python train.py \
    --data Data/train \
    --config Config/System_Config.json \
    --output-base Output/train \
    --model-compress-level 3
```

### 层次化训练 (CAM-35模式)
```bash
python train.py \
    --data Data/train \
    --layered \
    --word-weight 0.4 \
    --phrase-weight 0.5 \
    --sentence-weight 0.1
```

### 批量测试与规则解释
```bash
python test_whitebox.py \
    --test-file Data/test/test.csv \
    --run run1 \
    --show 10
```

### 交互式预测
```bash
python predict.py --run run1

# 或直接预测单条文本
python predict.py --run run1 --text "自然语言处理最新进展"
```

## 📖 详细文档 (Documentation)

### 配置系统 (Configuration)
`Config/System_Config.json` 关键参数：
```json
{
    "feature_mode": "enhanced",      // "lite"(4D), "enhanced"(15D), "layered"(35D)
    "window_size": 5,                // 共现窗口大小
    "distance_decay": 0.5,           // 距离衰减系数
    "use_semantic_matrix": false,    // 是否启用语义加权
    "use_compression": true,         // 启用XZ压缩
    "matrix_cache_size": 1024,       // LRU缓存大小
    "laplace_alpha": 2.0             // 拉普拉斯平滑系数
}
```

### 核心模块架构
```
src/
├── Corpus_Statistics.py           # 语料统计与拉普拉斯系数计算
├── Vocabulary_Construction.py     # 词汇表构建（支持XZ压缩）
├── Cooccurrence_Matrix_Estimation.py  # 共现矩阵估计（float32优化）
├── Association_Matrix_Generation.py   # 关联矩阵生成（距离加权）
├── Local_Neighborhood_Construction.py # 局部邻域构建（多尺度）
├── Matrix_Statistical_Compression.py  # 矩阵压缩（15D特征提取）
├── Feature_Sequence_Output.py     # 特征序列输出（内存优化）
└── main.py                         # 统一操作控制器
```

### 白盒决策流程
1. **特征提取**：文本 → 局部邻域 → 关联矩阵 → 15D特征向量
2. **标准化**：StandardScaler缩放
3. **决策推理**：CCP剪枝决策树 → 叶节点规则匹配
4. **置信校准**：原始置信度 → 样本支持度修正 → 复杂度惩罚 → 熵不确定性量化
5. **规则解释**：输出IF-THEN规则链与特征贡献度

## 🔬 算法细节 (Technical Details)

### 共现矩阵构建算法
给定语料$C$和词汇表$V$，构建共现矩阵$M \in \mathbb{R}^{|V| \times |V|}$：

$$M_{ij} = \sum_{c \in C} \sum_{p \in \text{Positions}(c)} \mathbb{I}(c_p = v_i) \cdot \sum_{k=-w}^{w} \mathbb{I}(c_{p+k} = v_j) \cdot \sigma(v_i, v_j)$$

其中$\sigma(v_i, v_j)$为语义相似度权重（可选），$w$为窗口大小。

### 15D特征结构化定义
从3×3关联矩阵$A$提取：
- **结构特征**（6D）：$a_{00}$（中心自相关）, $a_{01}$（前向链接）, $a_{02}$（后向链接）, $a_{11}$（右自相关）, $a_{12}$（交叉链接）, $a_{22}$（左自相关）
- **统计特征**（4D）：迹（Trace）、最大值、均值、标准差
- **上下文特征**（2D）：不对称性$Asym = \frac{fwd - bwd}{|fwd| + |bwd| + \epsilon}$、集中度$Conc = \frac{max}{mean + \epsilon}$
- **位置编码**（3D）：正弦位置编码截取

### 置信度校准公式
$$Conf_{calibrated} = \frac{n \cdot p_{raw} + \alpha}{n + K\alpha} \times \phi(n) \times \psi(d) \times (1 - H_{norm})$$

其中：
- $n$：叶节点样本支持度
- $\phi(n)$：样本数惩罚（小样本惩罚）
- $\psi(d)$：规则深度惩罚（过拟合控制）
- $H_{norm}$：归一化熵（不确定性量化）

## 📂 项目结构 (Project Structure)

```
CAM-15/
├── Config/
│   └── System_Config.json         # 系统配置
├── Data/
│   ├── train/                     # 训练数据
│   └── test/                      # 测试数据
├── Output/
│   ├── train/run*/                # 训练输出（词汇表、矩阵、模型）
│   │   ├── vocab.xz               # 压缩词汇表
│   │   ├── cooccur_matrix_matrix.joblib.xz  # 压缩共现矩阵
│   │   ├── whitebox_tree.pkl      # 白盒决策树
│   │   ├── decision_rules.json    # 可解释规则集
│   │   └── classifier_meta.json   # 元数据
│   └── test/                      # 测试结果
├── src/                           # 源代码（见上文）
├── Demo/
│   ├── train.py                   # 训练脚本（全数据+交叉验证）
│   ├── test_whitebox.py           # 白盒测试脚本
│   ├── predict.py                 # 交互式预测
│   └── tfidf_pca_31d_xgb.py      # 基线对比实验
└── README.md
```

## 📝 实验复现 (Reproducibility)

### 复现CAM-15训练
```bash
cd Demo
python train.py --data ../Data/train --config ../Config/System_Config.json
```

### 复现TF-IDF+PCA基线
```bash
python tfidf_pca_31d_xgb.py train --data ../Data/train --dim 15
```

## 📚 引用 (Citation)

若在研究中使用了本项目，请引用：

```bibtex
@software{cam15_2024,
  author = {ChaoJiht666},
  title = {CAM-15/35: Co-occurrence Association Matrix for Interpretable Chinese Text Classification},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ChaoJiht666/CAM-15}}
}
```

## 🤝 贡献与联系 (Contribution)

**当前版本**: v1.0.0  
**作者**: ChaoJiht666  
**维护**: 欢迎通过Issue提交问题或PR贡献代码。

### 后续研究方向
- [ ] 考究CAM算法结构问题，提升准确率
- [ ] 集成LLM进行规则后验证（Rule Verification）


---

<div align="center">
<b>基于共现局部性的白盒AI，让每一次分类决策都可追溯、可解释、可信赖。</b>
</div>



