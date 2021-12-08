> 本篇精读笔记，对重要部分做了严格翻译，如摘要和总结。对正文部分做了提炼，对重点部分突出标注。对参考文献做了分类。本文内容较长，如果时间有限可以直接跳到感兴趣的小节阅读。
>
> 注意arXiv上面的是老版论文，和新版略有差异。

本文提出了一种以编解码(encoder-decoder) transformer为核心的新的跟踪架构，称为STARK。编码器对目标对象和搜索区域之间的全局时空特征依赖关系进行建模，而解码器通过学习一个query embedding来预测目标对象的空间位置。目标跟踪作为一个直接的包围盒预测问题，不使用任何建议或预定义的锚点。通过编解码transformer，目标的预测只需使用一个简单的全卷积网络，直接估计目标的角点。整个方法是端到端的，不需要余弦窗口、包围盒平滑等后处理步骤，大大简化了现有的跟踪管线。在5个具有挑战性的短期和长期基准上达到了SOTA性能，并且可以实时运行，比Siam R-CNN快6倍。代码和模型在这里：https://github.com/researchmm/Stark

## 1. 介绍

在过去的几年里，基于卷积神经网络的目标跟踪取得了显著的进展。但是，卷积核不善于对图像内容和特征的长期相关性进行建模，因为它们只处理时间或空间的局部邻域。目前流行的跟踪器，包括离线跟踪器和在线学习的跟踪器，几乎都是建立在卷积运算上的。通常，这些方法只对图像内容的局部关系建模效果较好。这样的缺陷可能会降低模型处理全局上下文信息的能力，例如有大尺度变化或频繁进出视线的对象。

以前的Siamese跟踪器只利用空间信息进行跟踪，而在线方法使用历史预测进行模型更新。虽然这些方法取得了成功，但并没有直接建立空间和时间之间的关系。本文工作，借助于transformer对全局依赖性建模的优越能力，来整合用于跟踪的空间和时间信息，生成用于定位目标的具有辨识力的时空特征。

### 1.1 新架构的三个组件

- Encoder。接受输入：起始目标、当前帧、一个动态更新的模板。包含一个自注意模块用来学习输入序列之间的关系。
- Decoder。学习一个query embedding来预测目标的空间位置。
- Prediction head。一个基于角点的预测头用于估计当前帧中目标对象的包围盒；一个学习分数头来控制动态模板图像的更新。

大量实验表明，本文方法在多个数据集，以及短时跟踪和长时跟踪赛道都达到了SOTA。不像之前的跟踪模型，有多个模块组成，比如：跟踪器、目标验证模块、全局检测器等。本文方法只需要训练一个端到端的网络。模型更简单，速度也很快。见图一。

### 1.2 重要参考论文

- **Attention is all you need**。开创性的提出了Transformer。

  > Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. **Attention is all you need**. In NIPS, 2017.
- End-to end object detection with transformers。本文受这篇启发。

  > [DETR] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to end object detection with transformers. In ECCV, 2020

### 1.3 贡献总结

- 提出了一种新的基于transformer的视觉跟踪架构。它能够捕获视频序列中空间和时间信息的全局特征依赖关系。
- 整个方法是端到端的，不需要余弦窗、包围盒平滑等后处理步骤，大大简化了现有的跟踪管线。
- 提出的跟踪器在五个具有挑战性的短时和长时基准数据集上实现了最先进的性能（state-of-the-art, SOTA），并且可以实时速度运行。
- 建立了一个新的大规模跟踪基准数据集，以缓解小尺度数据集的过拟合问题。  

## 2. 相关工作

### 2.1 Transformer在语言和视觉领域的应用

Transformer最初是由Vaswani等人提出用来解决机器翻译任务，现在已成为语言建模的主流架构。Transformer将序列作为输入，扫描序列中的每个元素并了解它们的依赖关系。这一特点使得Transformer在捕获序列数据中的全局信息方面具有内在的优势。近年来，Transformer在视觉任务中显示出巨大的潜力，如：图像分类，目标检测，语义分割，多目标跟踪等。

#### 2.1.1 与DETR做比较

本文工作灵感受DETR启发，但也有所不同：

1. 研究问题不同。DETR是为目标检测设计的，而本文工作是研究目标跟踪。
2. 网络输入不同。DETR将整张图作为输入，而STARK的输入是三元组，包括一个搜索区域和两个模板。
3. 查询设计和训练策略不同。DETR使用100个对象查询，并在训练期间使用匈牙利算法将预测与GT进行匹配。相反，STARK只使用一个查询，并且总是将其与GT进行匹配，而不使用匈牙利算法。

#### 2.1.2 其他使用Transformer做目标跟踪的工作

1. TransTrack. 它的特点有：

   - 编码器将当前帧和前一帧的图像特征作为输入
   - 它有两个解码器，分别以学习到的对象查询和最后一帧中的查询作为输入。针对不同的查询，将编码器的输出序列分别转换为检测盒和跟踪盒。
   - 使用匈牙利算法基于IoU对预测的两组包围盒做匹配

   > Peize Sun, Yi Jiang, Rufeng Zhang, Enze Xie, Jinkun Cao, Xinting Hu, Tao Kong, Zehuan Yuan, Changhu Wang, and Ping Luo. TransTrack: Multiple-object tracking with transformer. arXiv preprint arXiv:2012.15460, 2020.

2. TrackFormer. 它的特点有：

   - 只将当前的帧特征作为编码器输入
   - 只有一个解码器，学习到的对象查询和上一帧跟踪结果相互作用
   - 它仅通过注意力操作来关联跟踪结果，不依赖于任何额外的匹配

   > Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. TrackFormer: Multi-object tracking with transformers. arXiv preprint arXiv:2101.02702, 2021

### 2.2 空间-时间信息挖掘

根据对时间和空间信息的利于，可以将现有的跟踪器分为两类。

#### 2.2.1 空间跟踪器

大多数离线Siamese跟踪器都是纯空间跟踪器，将目标跟踪视为初始模板和当前搜索区域之间的模板匹配问题。为了提取模板与搜索区域在空间维度上的关系，大多数跟踪器采用了各种不同的相关性，包括朴素相关性、深度相关和点相关。虽然近年来取得了显著的进展，但这些方法只捕捉了局部相似性，而忽略了全局信息。相比之下，Transformer中的自注意机制可以捕捉长期关系，适合于配对任务。

#### 2.2.2 空间-时间跟踪器

这一类跟踪器又可以分为两类：基于梯度和无梯度。

- 基于梯度。在推理时需要计算梯度。其中的经典工作之一是MD-Net，它用梯度下降来更新特定的层。为了提高优化效率，后面的工作采用了更先进的优化方法，如高斯-牛顿法或基于元学习的更新策略。然而，现实世界中许多部署深度学习的设备都不支持反向传播，这限制了基于梯度方法的应用。

  > [MD-Net] Hyeonseob Nam and Bohyung Han. Learning multi–domain convolutional neural networks for visual tracking. In CVPR, 2016

- 无梯度。利用一个额外的网络来更新Siamese跟踪器的模板。另一个代表作LTMU学习元更新器来预测当前状态是否足够可靠，以用于长期跟踪中的更新。虽然这些方法是有效的，但它们造成了空间和时间的分离。

  > [LTMU] Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, and Xiaoyun Yang. High-performance longterm tracking with meta-updater. In CVPR, 2020.

### 2.3 跟踪管线和后处理

以前的跟踪器的跟踪管线比较复杂。首先生成大量具有置信度的包围盒提案，然后使用各种后处理来选择最优的包围盒作为跟踪结果。常用的后处理方法有余弦窗、尺度或宽高比惩罚、包围盒平滑、基于tracklet的动态规划等。虽然效果较好，但后处理对超参数敏感。虽然有一些跟踪器试图简化跟踪管线，但它们的性能仍然远远落后于最先进的跟踪器。本文工作试图缩小这一差距，通过预测每帧中的单个包围盒来实现最佳性能。

## 3. 方法

### 3.1 A Simple Baseline Based on Transformer

如图二，这个简单的基准框架包含三个组件：一个卷积主干，一个编解码transformer，和一个包围盒预测头。

#### 3.1.1 Backbone

可以使用任意的卷积网络作为特征提取的主干。在不失通用性的情况下，采用ResNet作为主干。具体的说，除了去掉最后一阶和全连接层外，其他保持不变。主干的输入是一对图像：起始目标的模板图像z和当前帧的搜索区域x。输出是对应的两个特征图。

> Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. （引用数马上破10w了）

#### 3.1.2 Encoder

堆叠了N个编码器层。主干输出的特征图在送到编码器之前需要预处理，经过一个瓶颈层将通道数从D降到d，然后展平和拼接。根据原始Transformer的置换不变性（permutation-invariance），给输入序列增加了正弦位置嵌入（sinusoidal positional embeddings）。编码器捕获序列中所有元素之间的特征依赖关系并用全局上下文信息增强原始特征，使模型可以学到具有辨识力的特征用于定位目标。

#### 3.1.3 Decoder

堆叠了M个解码器层。将一个查询和编码器的输出作为输入。移除了DETR中的匈牙利算法。在编解码器注意力模块，目标查询可以注意到模板上的所有位置和搜索区域特征，因此最后预测的包围盒也比较稳定。

#### 3.1.4 Head

DETR使用一个三层感知机来预测包围盒坐标，但是，GFLoss指出直接对坐标回归相当于拟合一个狄拉克δ分布函数，缺乏对数据集二义性和不确定性的考虑。为了改进，设计了一个新的预测头，来估算包围盒四个角的概率分布。如图三。步骤：

1. 从编码器输出序列中取出搜索区域特征。

2. 计算搜索区域特征和解码器输出内嵌（embedding）之间的相似度。
3. 相似度分数逐像素乘以搜索区域特征，来增强重要区域。
4. 新的特征序列改造成特征图f，然后喂给一个简单的全卷积网络（FCN）。
5. FCN由L个堆叠的Conv-BN-ReLU层组成，输出两个概率图$P_{tl}(x, y)$和$P_{br}(x,y)$，分别代表包围盒左上角和右下角坐标。
6. 通过公式二计算角点概率分布的期望，得到包围盒坐标。

$$
(\widehat{x_{tl}},\widehat{y_{tl}})=(\sum_{y=0}^H \sum_{x=0}^W x\cdot P_{tl}(x,y),\sum_{y=0}^H \sum_{x=0}^W y\cdot P_{tl}(x,y)) , \\
(\widehat{x_{br}},\widehat{y_{br}})=(\sum_{y=0}^H \sum_{x=0}^W x\cdot P_{br}(x,y),\sum_{y=0}^H \sum_{x=0}^W y\cdot P_{br}(x,y)) ,\quad(2)
$$



#### 3.1.5 Training and Inference

- 训练。L1损失+IoU损失。损失函数公式：

$$
L=\lambda_{iou}L_{iou}(b_i,\hat{b_i})+\lambda_{L_1}L_1(b_i,\hat{b_i}). \quad (1)
$$

​		$b_i$和$\hat{b_i}$分别代表GT和预测的包围盒，$\lambda_{iou}$和\lambda_{L_1}是超参数。

- 推理。根据第一帧初始化模板图像和它的主干特征并保持固定。跟踪时对于每一帧，用当前帧的一个搜索区域作为网络输入，然后返回预测的包围盒作为最终输出，不使用余弦窗口或包围盒平滑等任何后处理。

### 3.2 Spatio-Temporal Transformer Tracking

跟踪目标可能随时间产生巨大变化，比如镜头拉近拉远等。所以捕捉目标最新的状态是很重要的。这就需要同时利用空间信息和时间信息。需要对Baseline做三个主要变化。

#### 3.2.1 Input

增加动态更新的模板作为第三个输入。如图四。动态模板可以提供目标随时间变化的信息。

#### 3.2.2 Head

跟踪时，有些情况动态模板不应该更新。比如当目标被遮挡，消失，或发生漂移时，模板并不可靠。简单起见，只有搜索区域中有目标时才可以更新。为了自动判定当前状态是否可靠，加入一个简单的分数预测头，一个三层感知机加上sigmoid激活。如果当前状态的分数大于一个阈值则认为是可靠的。

####  3.2.3 Training and Inference

最近的工作指出，同时训练定位和分类会导致次优的结果。因此，将训练过程分为两个阶段，将定位作为主要任务，分类作为次要任务。具体过程：

1. 第一阶段，使用公式一的定位相关损失，训练除了分数头外的整个网络。

2. 第二阶段，使用公式三的交叉熵损失，训练分数头。其中，$y_i$是GT标签，$P_i$是预测的置信度。训练期间其他参数保持不变。
   $$
   L_{ce}=y_ilog(P_i)+(1-y_i)log(1-P_i),\quad (3)
   $$

推理过程：

1. 用第一帧初始化两个模板和对应的特征。
2. 裁剪一个搜索区域喂给网络，生成一个包围盒和置信度。
3. 仅当到达更新间隔并且置信度大于阈值时，更新模板。

## 4. 实验

### 4.1 实现细节

- Python 3.6
- PyTorch 1.5.1
- 8个Tesla V100 GPU(16GB)

#### 4.1.1 模型

三个STARK变种：

- STARK-S50，只利用空间信息，使用ResNet-50做主干
- STARK-ST50，利用时间和空间信息，使用ResNet-50做主干
- STARK-ST101，利用时间和空间信息，使用ResNet-101做主干

使用ImageNet上预训练出来的参数初始化主干，训练时BatchNorm层固定。

> Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015

网络结构参数：

- 编解码层都为6个，包括多头注意力层(MHA)和前馈网络(FFN)
- MHA有8个头，宽度256
- FFN有2048个隐藏单元
- Dropout率0.1
- 包围盒预测头（FCN），5个堆叠的Conv-BN-ReLU层
- 分类头：三层感知机，每层256个单元

#### 4.1.2 训练

训练数据集：

- LaSOT
- GOT-10K，根据VOT2019的要求，去掉了1k被禁的序列。
- COCO2017
- TrackingNet

超参数：

- 搜索图尺寸：320x320
- 模板尺寸：128x128

数据增广：

- 水平翻转
- 亮度抖动

STARK-ST训练过程：

- 定位500ephch，分类50epoch
- 每个epoch三元组数：60000
- 优化器AdamW, 权重衰减10^-4
- $\lambda_{L1}=5,\lambda_{iou}=2$
- Batch size = 128, 每个GPU16
- 初始学习率，主干=10^-5, 其他=10^-4；衰减倍率10，第一阶段400epoch后衰减，第二阶段40epoch后衰减。

#### 4.1.3 推理

- 动态模板更新区间：每200帧更新一次
- 置信度阈值0.5

### 4.2 实验结果比较

比较了三个ST短时基准和两个LT长时基准。

#### 4.2.1 GOT-10K

> Lianghua Huang, Xin Zhao, and Kaiqi Huang. GOT-10k: A large high-diversity benchmark for generic object tracking in the wild. TPAMI, 2019

要求只使用GOT-10K的训练集训练。结果见表一。STARK-ST101的AO分数达到68.8%，超过Siam R-CNN 3.9%.

#### 4.2.2 TrackingNet

TrackingNet是一个包含511个视频的短时跟踪测试集。见表二。STARK-ST101的AUC达到82.0%，超过Siam R-CNN 0.8%.

#### 4.2.3 VOT2020

受AlphaRef启发，增加了改进模块。STARK50+AR的EAO指标达到了最好，STARK101+AR的accurracy指标达到最好，Robusstness指标没有达到最好，最好的是OceanPlus.

#### 4.2.4 LaSOT

LaSOT是一个长时跟踪测试集，包含280个平均长度2448帧的视频。见图五，STARK-ST101获得了最好67.1%，超过Siam R-CNN 2.3%。

#### 4.2.5 VOT2020-LT

包含50个长视频，跟踪目标会频繁进出视野。见表四。STARK-ST50和STARK-ST101的F-score分别为70.2%和70.1%，获得了最优。而且比冠军方案LTMU_B更简单。

#### 4.2.6 Speed, FLOPs andd Params

见表六。STARK-S50可以到达40fps。与SiamRPN++相比，FLOPS是它的1/4，参数个数是它的1/2。

### 4.3 新的测试基准

建立了一个新的数据集叫NOTU。收集了来自NFS, OTB100, TC-128, UAV123的401个视频。表五结果显示，STARK的泛化能力很强。

### 4.4 组件分析

使用STARK-ST50做基准，在LaSOT上评估去掉不同组件的影响。见表七，结论有：

- 编码器比解码器重要
- positional encoding的影响可以忽略不计，不是关键组件。
- 角点头对精度提高很重要
- 分数头也很重要

### 4.5 和其他框架比较

用STARK-ST50作为基准，和其他候选框架做比较，结果见表八。

- 模板图像作为查询。这种设计缺乏从模板到搜索区域的信息流。
- 使用匈牙利算法。
- 更新查询内嵌。
- 同时学习定位和分类。

## 5. 总结

提出了一种新的基于Transformer的跟踪框架，可以同时捕捉时间维度和空间维度的长范围依赖关系。STARK跟踪器去除了超参数敏感的后处理，具有一个简单的推理管线。大量实验表明STARK在五个短时和长时测试基准上达到了最优，并且可以实时运行。期望这项工作能引起人们对Transformer跟踪架构的更多关注。



## 参考文献归类

### 基于卷积的目标跟踪方法

- Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, and Junjie Yan. SiamRPN++: Evolution of siamese visual tracking with very deep networks. In CVPR, 2019.
- Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, and Michael Felsberg. ATOM: Accurate tracking by overlap maximization. In CVPR, 2019
- Paul Voigtlaender, Jonathon Luiten, Philip HS Torr, and Bastian Leibe. Siam R-CNN: Visual tracking by re-detection. In CVPR, 2020.
- Luca Bertinetto, Jack Valmadre, Jo˜ao F Henriques, Andrea Vedaldi, and Philip H S Torr. Fully-convolutional siamese networks for object tracking. In ECCVW, 2016.
- Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning discriminative model prediction for tracking. In ICCV, 2019
- Hyeonseob Nam and Bohyung Han. Learning multi–domain convolutional neural networks for visual tracking. In CVPR, 2016.
- Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, and Xiaoyun Yang. High-performance longterm tracking with meta-updater. In CVPR, 2020
- Huchuan Lu and Dong Wang. Online visual tracking. Springer, 2019
- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R–CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015
- Bin Yan, Xinyu Zhang, Dong Wang, Huchuan Lu, and Xiaoyun Yang. Alpha-refine: Boosting tracking performance by precise bounding box estimation. arXiv preprint arXiv:2012.06815, 2020
- Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, and Huchuan Lu. GradNet: Gradient-guided network for visual object tracking. In ICCV, 2019.
- Guangting Wang, Chong Luo, Xiaoyan Sun, Zhiwei Xiong, and Wenjun Zeng. Tracking by instance detection: A metalearning approach. In CVPR, 2020.
- Tianyu Yang, Pengfei Xu, Runbo Hu, Hua Chai, and Antoni B Chan. ROAM: Recurrently optimizing tracking model. In CVPR, 2020.
- David Held, Sebastian Thrun, and Silvio Savarese. Learning to track at 100 fps with deep regression networks. In ECCV, 2016
- Lianghua Huang, Xin Zhao, and Kaiqi Huang. Globaltrack: A simple and strong baseline for long-term tracking. In AAAI, 2020

### Siamese trackers

- Bo Li, Junjie Yan, Wei Wu, Zheng Zhu, and Xiaolin Hu. High performance visual tracking with siamese region proposal network. In CVPR, 2018
- Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, and Junjie Yan. SiamRPN++: Evolution of siamese visual tracking with very deep networks. In CVPR, 2019
- Zedu Chen, Bineng Zhong, Guorong Li, Shengping Zhang, and Rongrong Ji. Siamese box adaptive network for visual tracking. In CVPR, 2020
- Dongyan Guo, Jun Wang, Ying Cui, Zhenhua Wang, and Shengyong Chen. SiamCAR: Siamese fully convolutional classification and regression for visual tracking. In CVPR, 2020
- Yinda Xu, Zeyu Wang, Zuoxin Li, Ye Yuan, and Gang Yu. SiamFC++: towards robust and accurate visual tracking with target estimation guidelines. In AAAI, 2020
- Bingyan Liao, Chenye Wang, Yayun Wang, Yaonong Wang, and Jun Yin. PG-Net: Pixel to global matching network for visual tracking
- Zhipeng Zhang, Houwen Peng, Jianlong Fu, Bing Li, and Weiming Hu. Ocean: Object-aware anchor-free tracking. In ECCV, 2020
- Zheng Zhu, Qiang Wang, Bo Li, Wei Wu, Junjie Yan, and Weiming Hu. Distractor-aware siamese networks for visual object tracking. In ECCV, 2018

### Online methods

- Lichao Zhang, Abel Gonzalez-Garcia, Joost van de Weijer, Martin Danelljan, and Fahad Shahbaz Khan. Learning the model update for siamese trackers. In ICCV, 2019
- Tianyu Yang and Antoni B Chan. Learning dynamic memory networks for object tracking. In ECCV, 2018
- Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, and Michael Felsberg. ATOM: Accurate tracking by overlap maximization. In CVPR, 2019
- Goutam Bhat, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning discriminative model prediction for tracking. In ICCV, 2019

### 长时跟踪

- Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, and Xiaoyun Yang. High-performance longterm tracking with meta-updater. In CVPR, 2020
- Bin Yan, Haojie Zhao, Dong Wang, Huchuan Lu, and Xiaoyun Yang. ‘Skimming-Perusal’ Tracking: A framework for real-time and robust long-term tracking. In ICCV, 2019.

### Transformer

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. **Attention is all you need**. In NIPS, 2017.

### Transformer在计算机视觉领域的应用

- Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to end object detection with transformers. In ECCV, 2020
- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020
- Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. TrackFormer: Multi-object tracking with transformers. arXiv preprint arXiv:2101.02702, 2021
- Peize Sun, Yi Jiang, Rufeng Zhang, Enze Xie, Jinkun Cao, Xinting Hu, Tao Kong, Zehuan Yuan, Changhu Wang, and Ping Luo. TransTrack: Multiple-object tracking with transformer. arXiv preprint arXiv:2012.15460, 2020

### Transformer在非CV领域的应用

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT, 2019.
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019
- Christoph L¨uscher, Eugen Beck, Kazuki Irie, Markus Kitza, Wilfried Michel, Albert Zeyer, Ralf Schl¨uter, and Hermann Ney. Rwth asr systems for librispeech: Hybrid vs attention– w/o data augmentation. arXiv preprint arXiv:1905.03072, 2019. 
- Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. MaX-DeepLab: End-to-end panoptic segmentation with mask transformers. arXiv preprint arXiv:2012.00759, 2020

### 目标跟踪综述

- Peixia Li, Dong Wang, Lijun Wang, and Huchuan Lu. Deep visual tracking: Review and experimental comparison. PR, 2018.

### 目标跟踪书籍

- Huchuan Lu and Dong Wang. Online visual tracking. Springer, 2019

### 匈牙利算法

- Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 1955