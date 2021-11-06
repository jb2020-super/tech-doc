# DirectML适合什么项目？
# DirectML中的绑定
## Binding 绑定
Binding是指将资源挂载到管线上，以便在初始化和执行机器学习操作时使用。资源可以是输入输出Tensor，也可以是操作需要的临时或持久的资源。
## Dispatchable 可派发的
一个Dispatchable可以是一个操作初始化器或者一个编译好的操作。实现了IDMLDispatchable接口的对象通过调用`IDMLCommandRecorder::RecordDispatch`被录制到命令列表，并派发给GPU。
## 步骤
1. 调用`IDMLDispatchable::GetBindingProperties`来确定需要多少描述子和(临时/永久)资源。
2. 创建D3D12描述子堆(descriptor heap),足够装下所有描述子。然后绑定到管线。
3. 调用`IDMLDevice::CreateBindingTable`创建绑定表来代表绑定到管线的资源。使用DML_BINDING_TABLE_DESC结构描述绑定表。
4. 创建临时/持久资源作为Direct3D 12缓冲区资源，用DML_BUFFER_BINDING和DML_BINDING_DESC结构描述它们，并将它们添加到绑定表中。
5. 如果dispatchable是一个编译过的操作符，那么创建一个张量元素的缓冲区作为Direct3D 12缓冲区资源。填充/上传它，用DML_BUFFER_BINDING和DML_BINDING_DESC结构描述它，并将它添加到绑定表。
6. 调用`IDMLCommandRecorder::RecordDispatch`时，将绑定表作为参数传递
# Residency 驻留
在D3D12中，程序员需要自己管理资源的驻留与去除。应用程序应该避免在短时间内，频繁的让一个资源进出GPU。
