[知乎Mamba介绍](https://zhuanlan.zhihu.com/p/684231320)

# Efficiently Modeling Long Sequences with Structured State Spaces
### Introduction
提出`S4`模型，改进`SSM`的计算瓶颈问题。

S4优势：
+ 能够快速建模长序列。
+ 提供decode序列的通用框架，减少专门的、任务相关的数据预处理环节。
+ 可以接受分辨率的变化，例如0.5倍速下的音频。

### Background: State Spaces
$x\prime(t)=Ax(t)+Bu(t)$

$y(t)=Cx(t)+Du(t)$

其中$u(t)$是一维输入序列，$x(t)$是N维中间状态，$y(t)$是一维输出序列。使用梯度下降更新A、B、C矩阵，其中D可以视为一个类似残差链接的作用，为了简便则忽略之。
+ A：状态转移矩阵
+ B：从输入到状态的矩阵
+ C：从状态到输出的矩阵
+ D：从输入直接到输出的矩阵

上面是连续的情况，在深度学习里都是离散的序列输入，所以需要把$u(t)$修改为$u_k=u(k\delta)$，其中$\delta$是最小的时间步长。

用双线性变换将连续的矩阵离散化$\bar{A}$、$\bar{B}$、$\bar{C}$。

通过推导，可以将SSM的操作变为卷积形式$y=\bar{K}*u$，其中$\bar{K}=(\bar{C}\bar{A}^i\bar{B})_{i\in L}$。$L$是输入序列长度。这里出现了计算瓶颈：$\bar{K}$！

### Method: Structured State Spaces (S4)
引理1：SSMs $(A,B,C)\sim(V^{-1}AV,V^{-1}B,CV)$

引理2：HiPPO矩阵可以被$V_{ij}=C_{i+j}^{i-j}$对角化。（证明过程复杂）

理想情况是如果矩阵$A$是一个正规矩阵，就可以被一个矩阵$V$对角化。但是正规矩阵的限制很强，且HiPPO矩阵不是正规矩阵。

可以通过变换（论文`Algorithm 1`）使得计算大大减少。

最后的深度模型的每一个SSM层以 (batch_size, seq_len, hidden_size) 为输入，输出与输入形状一致。可以看作是一个大型的CNN卷积核。


# Mamba: Linear-Time Sequence Modeling with Selective State Spaces
[知乎 -- 基于mlx实现的mamba](https://zhuanlan.zhihu.com/p/679885879)

[腾讯云开发者社区 -- 挑战Transformer的新架构Mamba解析以及Pytorch复现](https://cloud.tencent.com/developer/article/2377967)

解决S4模型在离散和数据密集类型序列（文本）上较差的建模能力。（S4模型是确定的参数）

处理序列需要有隐藏状态，高效意味着有限的状态（RNN），而有效意味着较低的效率（Transformer）。本文提出选择性机制，选择机制控制信息的传播与交互。

核心：
+ 参数不再是固定的，而是参与梯度下降；状态转换矩阵A仍然是固定的；输入到状态、状态到输出、delta都是输入依赖的了。
+ 针对 GPU cache 做了专门优化：不使用卷积，而使用循环扫描的方法，在GPU上非常高效。