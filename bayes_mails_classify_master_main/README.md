# 运行代码
## 多项式朴素贝叶斯
代码使用多项式朴素贝叶斯分类器进行邮件分类，是朴素贝叶斯算法的一种变体。 多项式朴素贝叶斯基于"特征条件独立性假设"，
即假设文本中的每个词(特征)在给定类别条件下是相互独立的。 贝叶斯定理应用：在邮件分类中，算法计算给定邮件内容条件下属于某类别的后验概率。

## 代码中的数据处理流程包括以下步骤：
1. 文本读取：从txt文件中读取原始邮件内容
2. 无效字符过滤：使用正则表达式re.sub(r'[.【】0-9、——。，！~\*]', '', line)去除标点符号、数字等非文本字符
3. 分词处理：使用jieba分词库进行中文分词
4. 停用词过滤：通过filter(lambda word: len(word) > 1, line)过滤掉单字词(简单形式的停用词处理)
5. 词频统计：使用Counter统计词频，为特征选择做准备

## 方法差异
### 高频词特征选择
    选择语料库中出现频率最高的N个词作为特征，每个邮件的特征向量是这些词在该邮件中的出现次数
### TF-IDF特征加权
    TF-IDF = 词频(TF) × 逆文档频率(IDF)
### 主要差异
1. 高频词方法只考虑词频，TF-IDF同时考虑词频和文档频率
2. 高频词方法需要手动选择特征词，TF-IDF自动处理
3. TF-IDF通常能获得更好的分类效果，但计算复杂度稍高

<img src="images/zy.png" width="800" alt="zy">

# 灵活切换方式
在调用train_and_predict()函数时传入不同的参数即可
### 局部切换 对应代码classify_local.py
<img src="images/local.png" width="800" alt="local">

### 局部切换 对应代码classify_global.py
<img src="images/global.png" width="800" alt="global">

# 样本平衡处理
代码使用了SMOTE方法来处理邮件分类任务中的样本不平衡问题。通过对每个少数类样本，
从其k近邻中随机选择样本在特征空间中这两个样本之间的连线上随机生成新样本。

<img src="images/sample_balancing.png" width="800" alt="sample_balancing">

# 最终版_添加全局方法选择/样本平衡处理/模型评估指标
<img src="images/classify_all.png" width="800" alt="classify_all">