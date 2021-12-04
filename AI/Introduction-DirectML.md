# 初识DirectML

## 1. DirectML是什么？

DirectML是微软发布的一套机器学习推理API。具有与DirectX12接口相似的风格，所有与DirectX12兼容的硬件显卡都可以支持。包括：Intel GPU，AMD GPU，NVIDIA GPU等。Windows 10, v1903版本开始，DirectML将随操作系统发布。不过随操作系统发布的版本会稍微老一些。可以将特定版本的DirectML随应用程序发布。最新版本的DirectML可以从[NuGet]([redistributable NuGet package](https://www.nuget.org/packages/Microsoft.AI.DirectML/))中获取。

### 2. DirectML都支持哪些操作？

DirectML不同的版本支持的操作集会有所不同，一般更高的版本会支持更多的操作。具体某个操作是否支持需要根据使用的具体版本号来查询。官方文档会持续更新各版本加入的新操作，可以在这里查询[https://docs.microsoft.com/zh-cn/windows/ai/directml/dml-feature-level-history](https://docs.microsoft.com/zh-cn/windows/ai/directml/dml-feature-level-history)。文档相对于最新版本代码会有些落后，最新版请参考[https://github.com/microsoft/DirectML](https://github.com/microsoft/DirectML)。

### 3. DirectML性能为什么好？

理论上来说，可以自己用HLSL写computer shader实现卷积操作等神经网络操作。但是，DirectML可以做到比手写的computer shader更高效。这种高效来自于Direct3D 12的metacommand特性。Metacommand可以使DirectML访问硬件厂商提供的特殊硬件优化和架构优化，比如，**将卷积操作和激活函数融合在一起**。当然，如果某个操作没有Metacommand的实现，那么就会转换成普通的computer shader实现，性能就会下降。

由于目前DirectML还不够成熟，所有可能某些显卡型号对它的支持不够好，优化做的不好的话，性能就发挥不出来。实测的结果是，**AMD显卡对DirectML整体性能的发挥比NVIDIA显卡做的好**。

### 4.  DirectML适合什么项目？

由于DirectML是比较底层的API，以及和DirectX12之间的亲密关系。所以，对性能有较高要求，或者有实时性要求的项目都可以使用。比如：在游戏或者视频播放中使用实时超分等计算机视觉模型。而且它的跨显卡特性也减轻了开发负担，一套推理代码可以适用多种平台的显卡。当然，如果单纯只考虑Nvidia显卡，那么TensorRT的性能会比DirectML更胜一筹。不过TensorRT也有它的缺点，比如部署起来比较麻烦。

### 5. DirectML vs  Windows ML

Windows ML也是微软发布的一套Windows平台的机器学习推理引擎。DirectML可以为Windows ML提供GPU硬件加速。当然Windows ML也可以在CPU上跑。比起直接使用DirectML的API，Windows ML的API更简单，而且一套代码可以同时支持CPU和GPU上的推理。对于不追求极致性能的程序，WindowsML是更好的选择。

### 6. DirectML vs  ONNXRuntime

[ONNXRuntime](https://onnxruntime.ai/)是一个开源的推理引擎和训练引擎框架。它支持众多的底层推理引擎作为加速器，比如：Intel OpenVINO，DirectML，TensorRT等等。对于DirectML来说，可以看出它是DirectML的一个更高层的封装。ONNXRuntime的强大之处在与，使用同一套API就可以支持多套推理引擎加速器的切换。使推理性能在不同的硬件平台上面达到最优。当然，是否是最优也不是那么的绝对，需要结合实际的应用场景和测试结果。

和直接使用DirectML接口相比，ONNXRuntime不会跑的更快，相反可能会因为框架本身功能的局限性而限制使用的范围。这个时候就只能直接使用DirectML。当然，如果没有这种限制，可以优先考虑使用ONNXRuntime。毕竟实现起来会更容易。

### 7. DirectML vs TensorRT

TensorRT是NVIDIA发布的专门用于自家显卡的推理引擎。可以在自家显卡上达到推理性能的极致。这个是DirectML无法比的。当然，具体性能可以提升多少需要结合实际的应用场景和测试结果。虽然TensorRT的性能很高，但是缺点也很明显：

- 运行时依赖较重量级。相比于DirectML的几十M的DLL大小，TensorRT的众多依赖可以轻松超过1GB。
- TensorRT只能运行与NVIDIA显卡。限制了使用范围。
- TensorRT在部署到桌面端时，第一次运行需要编译Engine文件。这无疑又增加了开发周期和部署时的难度。

