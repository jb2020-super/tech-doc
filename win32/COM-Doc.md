## Reference

- https://docs.microsoft.com/en-us/windows/win32/com/component-object-model--com--portal

## 基础概念

COM是一种软件重用的二进制级别的标准。应用程序不需要重新编译就可以使用新的COM组件。

COM对象通过接口的方式将功能特性提供给用户。每个接口都有一个GUID作为唯一的标识，用户需要通过GUID来查询得到想要的接口。接口的定义是动态IDL接口定义语言来实现。IDL使用专门的编译器来生成头文件等。

在C++中，COM接口是通过抽象基类实现，也就是只包含纯虚函数的类。一个类可以实现多个接口。每个COM类也是通过GUID来唯一标识，叫做CLSID。

所有的COM接口都需要继承IUnknown接口。它提供了接口查询和基于引用计数的内存管理功能。

- QueryInterface
- AddRef
- Release

COM类可以驻留在DLL或者EXE中，通常系统会在注册表中存储CLSID到对应组件的映射关系。可以通过调用`CoCreateInstance`来创建COM对象。COM服务（或者叫COM组件）需要提供`IClassFactory`接口的实现，以便用户在创建COM对象的时候被调用。

### IClassFactory

对于一个基于DLL的COM服务，需要导出一个well-known函数给CoGetClassObject调用。这个函数就是DllGetClassObject.

都需要实现IClassFactory接口，因为在创建对象的时候会调用到它的接口。

- CreateInstance。用来创建一个新的COM类对象，并返回指定的接口
- LockServer。这个接口用来管理客户对COM类对象的引用计数。一般用于进程外激活。

```def
LIBRARY NAME
EXPORTS
	DllGetClassObject private
```

private关键字说明这个接口是给系统调用的。调用过程的伪代码：

```c++

HRESULT CoGetClassObject(REFCLSID rclsid, DWORD dwClsCtx, COSERVERINFO *pcsi, REFIIO riid, void **ppv) {
	HRESULT hr = REGOB_E_CLASSNOTREG;
	*ppv = 0;
	if (dwClsCtx & CLSCTX-INPROC) {
        // 进程内激活
        HRESULT (*pfnGCO)(REFCLSIO,REFIIO,void**) 0;
        // 通过查表检测是否已经加载到进程
        pfnGCO = LookupInClassTable(rclsid,dwClsCtx);
        if (pfnGCO == 0) {
            // 从注册表或者类商店里获取DLL文件名
            char szFileName[MAX-PATH];
            hr = GetFileFromClassStoreOrRegistry(rclsid,dwClsCtx, szFileName);
            if (SUCCEEDED(hr)) {
                // 通过文件名加载DLL
                HINSTANCE hInst = LoadLibrary(szFileName);
                if （ChInst == 0)
                    return CO_E_DLLNOTFOUND;
                 // 获取函数入口地址
                pfnGCO = GetProcAddress(hlnst, "DllGetClassObject");
                if (pfnGCO == 0)
                    return CO_E_ERRORINDLL;
                // 将函数地址添加到类表
                InsertInClassTable(rclsid, dwClsCtx, hlnst, pfnGCO);
            }
        }
        // 调用DllGetClassObject函数
        hr = (*pfnGCO)(rclsid, riid, ppv);
    }
    if ((dwClsCtx&(CLSCTX_LOCAL_SERVER | CLSCTX_REMOTE_SERVER))
    && hr == REGDB_E_CLASSNOTREG) {
	    //handle out-of-proc/remote request
    }
	return hr;
}
```

### DShow风格的实现方法

https://docs.microsoft.com/en-us/windows/win32/directshow/factory-template-array

### 使用C#应用程序调用非注册C++ COM组件的方法

- 添加app.manifest文件，右键打开工程属性配置，在Application -> Resources -> Icon and manifest -> Manifest 配置项下面选择创建好的[app.manifest](./app.manifest)文件。关键部分是dependency项目需要写对。

  ```xml
    <dependency >
      <dependentAssembly>
        <assemblyIdentity
            type="x64"
            name="MyComponent"
            version="1.0.0.1"
            processorArchitecture="amd64"          
          />
      </dependentAssembly>
    </dependency>
  ```

- Platform需要和组件保持一致。假如组件的编译平台是x64，那么应用程序也需要时x64。注意不能是`Any CPU`，否则COM对象会创建失败。右键打开工程属性，将Build -> General -> Platform target 修改为x64.

