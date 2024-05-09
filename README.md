# Mindspore-CapsuleIDRR

### 介绍
*Interactive Capsule Networks with a Novel Dynamic Routing Mechanism for Implicit Discourse Relation Recognition（IJCNN 2023）*（https://ieeexplore.ieee.org/document/10191545/）

Mindspore代码实现

#### 模型架构
整个模型按照论文中叙述分为四个部分：
1）**Bi-LSTM Layer**，由于mindspore没有elmo相关库，这里省略了LSTM的操作
2）**Argument Feature Capsule Layer**，采用胶囊网络的方法融合论元特征
3）**Discourse Relation-aware Interaction Layer**，论元之间采用注意力机制
4）**Relation Classifier** 对论文的篇章关系进行预测

整体模型架构图如下：
![image](https://github.com/sunhaowen123456/CapsuleIDRR-mindspore/assets/156690615/62bf6696-72d0-4641-a607-42e6a5f5e5c5)

### 目录结构
CapsuleIDRR\data\ -----语料相关
CapsuleIDRR\data\ raw-----处理语料
builder.py----建立模型训练过程
CapsuleModels_v8.py----提出的胶囊网络模型
CatCapsuleModels.py----拼接的胶囊网络模型（Baseline）
CapsuleIDRR\config.py----参数配置文件
CapsuleIDRR\data.py----加载数据
CapsuleIDRR\main.py----主程序
CapsuleIDRR\model.py----构建神经网络
CapsuleIDRR\NeuralTensorModels.py----张量神经网络模型
CapsuleIDRR\optim_schedule.py----优化学习率
prepare_data.sh---进行语料处理

#### 运行教程

1. 安装依赖包
    mindnlp @ file://model_data/wheel/mindnlp-0.2.0.20231227-py3-none-any.whl#sha256=06abc7070bc84be365f158747f85aea600cb1f8c3cd97cfa56100908cc65684b
    mindspore @ https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.10/MindSpore/unified/x86_64/mindspore-2.2.10-cp38-cp38-linux_x86_64.whl#sha256=f190fa0f74632407ca20f3bd162fafc47333f73b6ccaa0b3b6853af63824f963
    pandas==2.0.3
    tqdm==4.66.1

3. 将PDTB 2.0.cvs经过预处理文件data\preprocess，生成char_table.pkl，sub_table.pkl，test.pkl，train.pkl，we.pkl文件。

4. 将GoogleNews-vectors-negative300.bin.bz文件放到`data/`下。

5. 执行下面命令：

   ```python
   python3 main.py
   ```
