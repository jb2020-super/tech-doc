# Structure-Preserving Super Resolution with Gradient Guidance
## Abstract
Structures matter in single image super resolution
(SISR). Recent studies benefiting from generative adversarial network (GAN) have promoted the development ofSISR by recovering photo-realistic images. However, there are always undesired structural distortions in the recovered images. In this paper, we propose a structure-preserving super resolution method to alleviate the above issue while maintaining the merits of GAN-based methods to generate perceptual-pleasant details. Specifically, we exploit gradient maps of images to guide the recovery in two aspects. On the one hand, we restore high-resolution gradient maps by a gradient branch to provide additional structure priors for the SR process. On the other hand, we propose a gradient loss which imposes a second-order restriction on the super-resolved images. Along with the previous imagespace loss functions, the gradient-space objectives help generative networks concentrate more on geometric structures. Moreover, our method is model-agnostic, which can be potentially used for off-the-shelf SR networks. Experimental results show that we achieve the best PI and LPIPS performance and meanwhile comparable PSNR and SSIM compared with state-of-the-art perceptual-driven SR methods. Visual results demonstrate our superiority in restoring structures while generating natural SR images.
On the one hand, we restore high-resolution gradient maps by a gradient branch to provide additional structure priors for the SR process. On the other hand, we propose a gradient loss which imposes a second-order restriction on the super-resolved images. Along with the previous imagespace loss functions, the gradient-space objectives help generative networks concentrate more on geometric structures.

## 1. Introduction
使用MSE作为损失函数训练出来的模型，通常会有高的PSNR，但是结果会比较模糊。 
最近流行的基于生成对抗网络(GAN)的方法，如SRGAN, EnhanceNet, ESRGAN, NatSR.虽然生成的结果比较清晰，但是容易引入几何失真(geometric distortion)，改变了原图的语义。  
本文方法：一方面，设计一个梯度分支，将LR图的梯度转换到HR来作为辅助；另一方面，提出了梯度损失(gradient loss)来监督恢复图像的梯度。
## 2. Related Work
- 面向PSNR的方法。SRCNN, DRCN, VDSR, SRResNet, RDN, RCAN
- 感知驱动的方法。SRGAN, ESRGAN.
- 梯度相关方法。

## Approach