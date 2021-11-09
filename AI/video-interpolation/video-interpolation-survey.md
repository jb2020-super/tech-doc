# 视频插帧初探

视频插帧主要分为两大流派：**基于核**的方法和**基于光流**的方法，还有一些其他小众的方法，比如基于阶段的，或者直接合成的。还有一些有趣的工作是将视频插帧和超分、去糊等结合在一起。

这个领域里面一个比较古老的数据集，里面有很多AI的或者非AI的插帧方法的排行：

- Middlebury测试基准榜单：https://vision.middlebury.edu/flow/eval/results/results-i1.php

## 业界应用

- https://developer.nvidia.com/rtx/ngx，NVIDIA的这个NGX SDK里面实现了智能插帧算法Super Slow-Mo.
- https://developer.nvidia.com/blog/whats-new-in-optical-flow-sdk-3-0/，猜测是非AI的光流法。
- https://www.svp-team.cn.com/wiki/Main_Page ，有非AI插帧和AI插帧RIFE等
- https://github.com/hzwer/arXiv2020-RIFE，旷视科技出品，目前有较多应用。

- https://github.com/AaronFeng753/Waifu2x-Extension-GUI，集成了RIFE，CAIN，DAIN，每次只能插两倍，自定义倍数要交钱-_-||

## 视频插帧领域重要论文

### 2021

- ### [Revisiting adaptive convolutions for video frame interpolation](https://openaccess.thecvf.com/content/WACV2021/html/Niklaus_Revisiting_Adaptive_Convolutions_for_Video_Frame_Interpolation_WACV_2021_paper.html)，2021，WACV，Niklaus，Adobe。【Kernel-Based】使用老方法adaptive separable convolution，通过一组底层优化技术，达到了很好的结果。这些技术对其他任务也有潜力。[Code](https://github.com/sniklaus/revisiting-sepconv), [SepConv++] [Rank 2]; Adaptive Convolution这一技术应用广泛。整体提升1.76dB。性能吊打其他基于光流的方法，除了SoftSplat。delayed padding (+0.37 dB) ，input normalization (+0.30 dB) ，network improvements (+0.42 dB) ，kernel normalization (+0.52 dB) ，contextual training (+0.18 dB) 。https://github.com/sniklaus/revisiting-sepconv 【20 star】![sepconv++-F1](\images\sepconv++-F1.PNG)![SepConv++T5](\images\SepConv++T5.PNG)

- ### [Deep Animation **Video Interpolation** in the Wild](http://openaccess.thecvf.com/content/CVPR2021/html/Siyao_Deep_Animation_Video_Interpolation_in_the_Wild_CVPR_2021_paper.html)，2021，CVPR，L Siyao。针对动画片。https://github.com/lisiyao21/AnimeInterp/ 【200 star】

- ### [Flavr: Flow-agnostic video representations for fast frame interpolation](https://arxiv.org/abs/2012.08512)，2021，CVPR，T Kalluri。3D space-time convolution。端到端学习。解决分线性运动的难题。不需要光流和深度图。推理速度快3倍。有潜力扩展到超分和去糊。 https://github.com/tarun005/FLAVR 【192 star】【[演示](https://tarun005.github.io/FLAVR/)】。U-Net，ResNet-3D 18层，encoder-decoder框架<img src="images\FLAVR-F1.PNG" alt="FLAVR-F1" style="zoom:100%;" /><img src="images\FLAVR-F2.PNG" alt="FLAVR-F2" style="zoom:100%;" /><img src="images\FLAVR-F3.PNG" alt="FLAVR-F3" style="zoom:100%;" />

- ### [CDFI: Compression-Driven Network Design for **Frame Interpolation**](http://openaccess.thecvf.com/content/CVPR2021/html/Ding_CDFI_Compression-Driven_Network_Design_for_Frame_Interpolation_CVPR_2021_paper.html)，2021，CVPR，T Ding。【DNN-based】基于AdaCoF，对模型进行了压缩和优化。https://github.com/tding1/CDFI 【69 star】

- ### [Multi-Level Adaptive Separable Convolution for Large-Motion Video **Frame Interpolation**](https://openaccess.thecvf.com/content/ICCV2021W/PBDL/html/Wijma_Multi-Level_Adaptive_Separable_Convolution_for_Large-Motion_Video_Frame_Interpolation_ICCVW_2021_paper.html)， 2021，针对大运动场景。

- ### [Time Lens: Event-Based Video Frame Interpolation](https://openaccess.thecvf.com/content/CVPR2021/html/Tulyakov_Time_Lens_Event-Based_Video_Frame_Interpolation_CVPR_2021_paper.html)，2021，CVPR，S Tulyakov，2。https://github.com/uzh-rpg/rpg_timelens【424】http://rpg.ifi.uzh.ch/TimeLens.html 

### 2020

- ### [Softmax splatting for video frame interpolation](http://openaccess.thecvf.com/content_CVPR_2020/html/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.html)，2020， CVPR，Niklaus，58。【光流法】

- ### [Zooming slow-mo: Fast and accurate one-stage space-time video super-resolution](http://openaccess.thecvf.com/content_CVPR_2020/html/Xiang_Zooming_Slow-Mo_Fast_and_Accurate_One-Stage_Space-Time_Video_Super-Resolution_CVPR_2020_paper.html)，2020，CVPR，X  Xiang，42。【插帧+超分】比 DAIN+EDVR，DAIN+RBPN效果更好，速度快三倍。https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020【706 star】

- ### [Adacof: Adaptive collaboration of flows for video frame interpolation](http://openaccess.thecvf.com/content_CVPR_2020/html/Lee_AdaCoF_Adaptive_Collaboration_of_Flows_for_Video_Frame_Interpolation_CVPR_2020_paper.html)，2020， CVPR，H Lee，35。https://github.com/HyeongminLEE/AdaCoF-pytorch 【131 star】克服自由度的局限，泛化的 warping module。<img src="images\AdaCof-F1.PNG" alt="AdaCof-F1" style="zoom:70%;" />

  ![AdaCof-F3](images\AdaCof-F3.PNG)

- ### CAIN [Channel attention is all you need for video frame interpolation](https://ojs.aaai.org/index.php/AAAI/article/view/6693)，2020，AAAI，M Choi，34。https://github.com/myungsub/CAIN【201】AIM 2019 Video Temporal Super-Resolution Rank 2。不计算光流。PixelShuffle技术。

  <img src="images\CAIN-F1.PNG" alt="CAIN-F1" style="zoom:80%;" />

  ![CAIN-F2](images\CAIN-F2.PNG)

  ![CAIN-F3](images\CAIN-F3.PNG)<img src="images\CAIN-F4.PNG" alt="CAIN-F4" style="zoom:80%;" />

- ### [Bmbc: Bilateral motion estimation with bilateral cost volume for video interpolation](https://link.springer.com/content/pdf/10.1007/978-3-030-58568-6_7.pdf)，2020， J Park，21. 基于双向运动估计，网络复杂。https://github.com/JunHeum/BMBC【65】![BMBC-F1](images\BMBC-F1.PNG)

- ### [FISR: deep joint frame interpolation and super-resolution with a multi-scale temporal loss](https://ojs.aaai.org/index.php/AAAI/article/view/6788)，2020，SY Kim，18.【插帧+超分】

- ### [**Enhanced quadratic video interpolation**](https://link.springer.com/chapter/10.1007/978-3-030-66823-5_3)，2020，Y Liu，ECCV，6。https://github.com/lyh-18/EQVI 【95】。QVI的增强版。解决QVI在大运动，复杂运动下的缺陷。AIM2020 VTSR冠军。residual contextural synthesis network。![EQVI-T2-T3](images\EQVI-T2-T3.PNG)PSNR第一![EQVI-F7](images\EQVI-F7.PNG)比较纹理丰富和大运动场景，更少重影和人工痕迹。

- ### [RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294), 2020, arxiv, Z Huang, 9。https://github.com/hzwer/arXiv2020-RIFE 【2.1k】Megvii 旷视科技。光流法，实时，推理速度快。![RIFE-F1](images\RIFE-F1.PNG)

  ![RIFE-F2](images\RIFE-F2.PNG)

  ![RIFE-T2](images\RIFE-T2.PNG)

### 2019

- ### DAIN [**Depth**-**aware video frame interpolation**](http://openaccess.thecvf.com/content_CVPR_2019/html/Bao_Depth-Aware_Video_Frame_Interpolation_CVPR_2019_paper.html)，2019，CVPR, W Bao, 180。解决大运动或遮挡。利用深度信息监测遮挡。输入帧，深度图，基于光流的上下文特征，局部插值核。https://github.com/baowenbo/DAIN【7k star】。optical flow estimation + single image depth estimation + context-aware image synthesis + adaptive convolutions.![DAIN-F9](images\DAIN-F9.PNG)大运动幅度

- ### [**Quadratic video interpolation**](https://arxiv.org/abs/1911.00627)，2019，X Xu，43。使用二次插值而不是传统的线性插值，解决曲线运动。结果较好，https://github.com/xuxy09/QVI【10】

### 2018

- ### [Super slomo: High quality estimation of multiple intermediate frames for video interpolation](http://openaccess.thecvf.com/content_cvpr_2018/html/Jiang_Super_SloMo_High_CVPR_2018_paper.html)，2018，CVPR，H Jiang ,327。每年引用增长最快。端到端，计算双向光流，U-Net架构，https://github.com/avinashpaliwal/Super-SloMo【2.7k】高引用高星![superslomo-F4](\images\superslomo-F4.PNG)

- ### [Context-aware synthesis for video frame interpolation](http://openaccess.thecvf.com/content_cvpr_2018/html/Niklaus_Context-Aware_Synthesis_for_CVPR_2018_paper.html),2018, CVPR, Niklaus, 204。双向光流，预训练网络提取上下文信息，合成网络，GridNet。http://web.cecs.pdx.edu/~fliu/project/ctxsyn/【NO Code】![CtxSyn-F2](images\CtxSyn-F2.PNG)

  ![CtxSyn-F6](\images\CtxSyn-F6.PNG)

- [Memc-net: Motion estimation and motion compensation driven neural network for video interpolation and enhancement](https://ieeexplore.ieee.org/abstract/document/8840983/)，2018，W Bao，118。

### 2017

- ### 【DVF】[Video frame synthesis using deep voxel flow](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Video_Frame_Synthesis_ICCV_2017_paper.html)，2017，ICCV，Z Liu，481。特点是不仅可以往中间插，还能往后插。 https://github.com/lxx1991/pytorch-voxel-flow 【120+】，https://liuziwei7.github.io/projects/voxelflow/demo.html![DVF-F1](images\DVF-F1.PNG)![DVF-F4](images\DVF-F4.PNG)

- ### 【SepConv】[Video frame interpolation via adaptive separable convolution](http://openaccess.thecvf.com/content_iccv_2017/html/Niklaus_Video_Frame_Interpolation_ICCV_2017_paper.html)，2017，ICCV，Niklaus，398。解决大运动需要大卷积核的问题，用一维核代替二维核，参数更少。perceptual loss。https://github.com/sniklaus/sepconv-slomo 【955】![sepconv-F2](\images\sepconv-F2.PNG)

- ### [Video frame interpolation via adaptive convolution](http://openaccess.thecvf.com/content_cvpr_2017/html/Niklaus_Video_Frame_Interpolation_CVPR_2017_paper.html)，2017，CVPR，Niklaus，282。首次使用深度学习解决视频插帧问题。



