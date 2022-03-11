## IDL

### 指针参数

The ref pointer means that the pointer should not be null. If null, the interface proxy will return marshaling error to thecaller. A unique pointer can be null.

```c++
HRESULT g([in, ref] short *ps);
HRESULT g([in, unique] short *ps);
HRESULT k([in, ptr] short *ps1, [in, ptr] short *ps2);
```

如果接口的设计者希望将同一个指针传给接口k的两个参数，并且希望接收端得到的也是同样的指针地址，那么可以使用[ptr]属性。

```c++
[
	uuid(),
	object,
    pointer_default(unique)
]
interface IDogManager : IUnknown
{
	typedef struct{
		HUMAN *pOwner;
	}DOG;
	HRESULT Take([in] DOG *dog);
}
```



- top-level pointer, 接口参数指针是顶级指针，如DOG*dog指针

- embedded pointer, 类型内部的指针是内嵌指针，如果HUMAN*pOwner指针。

- pointer_default(unique)用来指定内嵌指针的默认属性为unique，不影响顶级指针。

- **COM指针内存管理规则**：对于[in]参数，调用者负责分配。对于[out]和[in,out]参数指针，调用者负责分配顶级指针内存，被调者负责分配内嵌指针内存。被调者需要使用COM唯一指定的专用内存分配器来分配内嵌指针内存。内嵌指针的释放由调用者负责。

  ```c++
  void *CoTaskMemAlloc(DWORD cb);
  void CoTaskMemFree(void *pv);
  ```

  只所以不使用C runtime library的内存分配函数是因为，不同模块的C runtime可能实现不一样。这样就无法保证在一个模块分配的内存，在另一个模块被正确释放。

- [in,out]属性的指针情况更负责。调用者可能会分配好内嵌指针的内存，被调者可以重现按需求分配内嵌指针内存，甚至是将它释放掉。

- 跨模块内存指针传递是怎么实现的？Interface Stub会将数据打包成ORPC相应，然后在被调者进程空间中释放掉内存。数据通过ORPC通道传输给调用者，然后由接口代理分配新的内存。![image-20220224160536450](F:\github\tech-doc\win32\image-20220224160536450.png)

- 可以通过实现IMallocSpy接口来监视COM内存的分配和释放。

### 数组做参数

<Essential COM> p328 - p345

## Apartment

公寓定义了并发和重入的规则。有两种类型的公寓：Single threaded apartment(STA)和multithreaded apartments(MTA)。一个进程可以有多个STA公寓，只能有一个MTA公寓。公寓里可以有多个COM对象。STA公寓只能由创建它的线程访问（整个生命周期）。MTA公寓可以由多个线程进入。Rental threaded 公寓（RTA），同一时间只能被一个线程进入，但是等这个线程出来之后就可以供其他线程进入。而STA公寓一辈子只能供一个线程进入。

一个线程被创建之后，在使用COM之前，必须调用CoInitialize初始化公寓模式为STA，或者调用CoInitializeEx指定公寓模式。

对象必须和客户住在不同的公寓中。

代理是一个对象在另一个公寓中的替身，供其他对象访问它的接口。而实际的执行还是在它自己的公寓中。

每个CLSID可以有四种不同的线程模式。

- ThreadingMode="Free", 对应MTA
- ThreadingMode="Apartment", 对应STA
- ThreadingMode="Both", 表示可以这个类可以执行在STA或MTA模式。
- 未定义表示只能在main STA运行。Main STA是经常中第一个初始化的STA。

In-process servers的情况，在创建对象之前，用户已经调用过了CoInitializeEx。如果用户的公寓和CLSID的兼容，那么在进程内激活请求时，CLSID可以直接在客户的公寓内实例化对象。这是最高效的情况，因为不需要代理。如果公寓模式不兼容，那么就会悄悄的在另一个公寓中实例化对象，并返回一个代理给客户。

- 