# Introduction
## 分类
1. supervised SR
2. unsupervised SR
3. domain-specific SR

## 不同技术的区别
1. 不同类型的网络架构
2. 不同类型的损失函数
3. 不同类型的学习策略
4. 等等

# Problem Setting and Terminology
## 问题定义
- 一般LR图像可以定义为一个退化的过程：Ix = D(Iy;Delta);其中D是退化映射函数
- 为了模拟退化过程，一般会采用bicubic插值的降采样加上抗锯齿(bicubic interpolation with antialiasing)
- 还有采用多种操作组合的方法来模拟退化；先对HR图像加上一个模糊核(blur kernel)，然后降采样，最后加上高斯白噪声；这种方法更接近现实世界的情况，已经被证明更有用[39]
## 数据集
有些数据集会提供LR-HR图像对；有些只提供HR图像，这时一般会使用MATLAB中的`imresize`方法来得到LR图像。常用数据集：
- BSD300[40]
- BSD500[41]
- DIV2K[42]
- General-100[43]
- L20[44]
- Manga109[45]
- OutdoorScene[46]
- PIRM[47]
- Set5[48]
- Set14[49]
- T91[21]
- Urban100[50]
- ImageNet[51]
- MS-COCO[52]
- VOC2012[53]
- CelebA[54]
- 也有多个数据集一起使用的

## 评价标准(Image Quality Assessment, IQA)
评价标准分主观感受的标准和用公式计算的客观标准。前者一般比较耗时（也不好操作吧）；后者又可以分为三种：
1. full-reference.需要使用参照图像
2. reduced-refrence.
3. no-reference
常用客观评价标准：
### PSNR
虽然PSNR在计算的时候因为没有考虑到人的视觉因素，从而导致较差的效果，但是为了不同论文之间工作的比较，它还是目前最广泛采用的评价标准。
### SSIM
考虑到了人的视觉系统(Human visual system, HVS)。
### Mean Opinion Score
主观打分：1(bad)-5(good)
### Learning-based Perceptual Quality

## Operating Channels
早期的模型都只训练YCbCr空间的Y通道[26][43][78][79]，新的模型更多的使用RGB通道[28][31][57][70]。 
采用不同的通道会使评测结果差别巨大(4 dB)

## Super-resolution Challenges
- NTIRE Challenge[80]. 与CVPR关联。分bicubic downscaling赛道和blind赛道。前者是理想情况，后者是用现实中的情况。
- PIRM Challenge[47][81]。与ECCV关联。

# 有监督SR(Supervised Super-Resolution)
# SR框架
- Pre-upsampling SR.先将LR图像用传统的上采样算法拉伸到HR尺寸，然后再使用深度学习做精细化处理。代表是SRCNN。优点是可以接受任意尺寸和放大比例的输入。缺点是会引入副作用，如噪声放大和模糊。并且因为是在HR的尺寸操作，时间和空间复杂度会比其他框架高。
- Post-upsampling SR[43,84]. 上采样操作通过学习获得。计算复杂度低。主流框架之一。缺点是对于比较大的放大倍数(4, 8等)学习比较困难。通用性差不同的放大倍数需要学习不同的模型。
- Progressive Upsampling SR[27]. 渐进上采样，LapSRN。经过多个阶段将LR图像放大到SR图像。降低了学习困难度，但是模型变得更复杂。
- Iterative Up-andDown Sampling SR. 引入了back-projection[12]技术。如DBPN[57]不断的放大-缩小，再放大-再缩小的迭代过程。该模型可以更好的挖掘LR-HR图像对之间的关系，得到更好的重建结果。还处于研究的早期，具有很大的潜力。

## Upsampling方法
### 基于插值的上采样
- Nearest-neighbor
- Bilinear, BLI.
- Bicubic, BCI[10]. 由于有众多副作用，现在的趋势是使用基于学习的上采样。

### 基于学习的上采样
- Transposed Convolution Layer/deconvolution layer[90,91].会导致checkerboard-like pattern.
- Sub-pixel Layer[84].在不同块的边界会容易产生artifact. PixelTCL[94]进行了改进。
- Meta Upscale Module[95].一个模型适用多种放大倍数。缺点是不稳定。

## 网络设计

### Residual Learning
- Global Residual Learning。通过学习图像的残差映射来恢复缺失的高频部分。模型的复杂度和学习困难度大大降低。[26,55,56,98]
- Local Residual Learning。与ResNet[96]类似，克服退化问题。[70,78,85,99]

### Recursive Learning
用递归的方法多次使用同一个模型。DRCN[82], DRRN[56].会引入vanishing/exploding gradient问题。一般会与其他技术结合使用，如residual learning和multi-supervision

### Multi-path Learning
- Global Multi-path Learning.使用多条路径提取图像的不同方面的特征。LapSRN[27], DSRN[85]
- Local Multi-path Learning, MSRN[99]
- Scale-specific Multi-path learning, MDSR[31]
### Dense Connections
DenseNet[102]

### Attention Mechanism
- Channel Attention.[104]
- Non-local Attention.[106]

### Advanced Convolution
- Dilated Convolution. [107]
- Group Convolution.[98,28]
- Depthwise Separable Convolution[110]

### Region-recursive Learning
[64]
### Pyramid Pooling
[115]
### Wavelet Transformation
将图像分解成高频和低频。[117,118,119]
### Desubpixel
使用逆shuffle操作将LR图像分解成更小的尺寸。[81]在PIRM挑战的智能手机场获得了最高分，具有很快的推理速度和好的性能。
### xUnit
[124]在不损失性能的情况下，把模型大小降低了50%

## 学习策略
### Loss Functions
最开始采用L2损失，但是后来发现不能准确的衡量重建后的图像质量。
- Pixel Loss.  Charbonnier loss[27,125].实践中，L1 loss性能优于L2 loss[28,31,126]。缺点是不能反映图像的质量，丢失高频细节。[25,29,58,74]
- Content Loss.[29,127]
- Texture Loss. EnhanceNet[8]
- Adversarial Loss. 应用对抗网络GAN[24], SRGAN[25]。通过MOS测试，得到更好的视觉上的质量提升，虽然PSNR可能更低一些。缺点是方法不成熟，训练困难不稳定。
- Cycle Consistency Loss.[131]
- Total Variation Loss.[140]消除噪声
- Prior-Based Loss. face image SR[30],通过引入分类网络，为SR提供图像特征层次的先验知识，从而提高性能。

实践中，通常会用加权平均组合多种损失函数[8,25,27,46,141]。但是不同损失函数采用多少权值还是依赖于经验，仍然是个问题。

### Batch Normalization
[145]
### Curriculum Learning
[148] 从简单任务开始训练，然后不断增加难度。比如把放大4倍的SR任务分解成：[116]
1. 1倍到2倍
2. 2倍到4倍
3. denoising
4. debluring

### Multi-supervision
DRCN[82]

## 其他提升
- Context-wise Network Fusion(CNF)[100]，将三个轻量级的SRCNN的结果组合，得到与最先进的模型同等的性能。
- Data Augmentation。对原始数据进行一些变换操作，如裁剪，翻转，缩放，旋转等[27,31,44,56,85,98]
- Multi-task Learning.与其他和SR相关的图像处理任务结合，如去噪，语义分割等
- Network Interpolation. 基于PSNR的模型可以得到更接近原图的结果，但是会引入模糊问题。基于GAN的模型可以得到更好的视觉效果但是会引入不愉快的人工痕迹。将两种模型参数插值。
- Self-Ensemble.[44]

## State-of-the-art SR Models
Table-2
# 无监督SR
## Zero-shot SR
有监督学习，其实学习的是一个提前定义好的退化算法模型。不能很好的符合现实中的情况。 ZSSR[83],优点是效果吊打其他模型；缺点是需要在测试的时候训练模型，也就是说要在用户机器上训练，而不是提前训练好。所以推理时间会很长。
## Weakly-supervised SR
- Learned Degradation.因为预先定义的退化是次优的，所以通过学习不成对的LR-HR数据集来确定退化是一个可行的方向。[159]分两步：先训练出一个生成器模拟退化过程来生成LR-HR对；然后用生成的LR-HR对组成的数据集训练SR模型。
- Cycle-in-cycle Super-resolution. CycleGAN[138]

## Deep Image Prior
[160]

# Domain-Specific Application
## Depth Map SR
[165]
## Face Image SR
可以帮助其他Face相关的任务。CBN[170]
## Hyperspectral Image SR
## Real-world Image SR
[187]
## Video SR

# 总结
## 网络设计
- Combining Local and Global Information
- Combining Low- and High-level Information
- Context-specific Attention
- More Efficient Architectures.
- Upsampling Methods.Neural architecture Search(NAS)[212]

## 学习策略
## 评价标准
