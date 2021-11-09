## DLL搜索机制

桌面应用程序可以通过多种方法来控制DLL加载路径：

1. 通过[**manifest**](https://docs.microsoft.com/en-us/windows/win32/sbscs/manifests)文件指定。
2. [DLL重定向技术](https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-redirection)可以使LoadLibrary优先在应用程序所在目录查找。
3. 如果上面两种方法都没用使用，则会做一些检查：
   1. 如果同名的DLL已经在内存里，则会直接使用，不会开启搜索。
   2. 如果DLL在知名的DLL列表里（在这个注册表下：**HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\KnownDLLs**），则会使用这些知名DLL，不会开启搜索。
4. 如果以上条件都没有满足，则会开启DLL搜索。
5. 另外，对于DLL依赖的其他DLL，比如a.DLL依赖b.DLL，系统会直接用b.DLL的名字调用LoadLibrary，而不会加入额外的路径。

### DLL标准搜索顺序

DLL标准搜索顺序，分安全搜索模式和非安全搜索模式。默认安全搜索模式是开启的，如果要禁用可以通过创建注册表键值**HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager**\**SafeDllSearchMode**并设为0。本文只讨论安全搜索模式。对于非安全搜索模式，可以参考[官方文档](https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)。

安全模式搜索顺序：

1. 应用程序加载时所在的目录。

2. 系统目录。可以使用`GetSystemDirectory`获取。

   ```c++
   #include <Windows.h>
   #include <iostream>
   int main() {
       const int MAX_BUFFER_SIZE = 1024;
       wchar_t buffer[MAX_BUFFER_SIZE];
       UINT len = GetSystemDirectoryW(buffer, MAX_BUFFER_SIZE);
       if (!len) {
           std::wcout << "GetSystemDirectory failed with error code: " << GetLastError() << std::endl;
       }
       else
       {
           std::wcout << buffer << std::endl;
       }
       return 0;
   }
   ```

   输出：
   `C:\WINDOWS\system32`

3. 16位系统目录。

4. Windows目录。使用`GetWindowsDirectory`获取。

   ```c++
   #include <Windows.h>
   #include <iostream>
   int main() {
       const int MAX_BUFFER_SIZE = MAX_PATH;
       wchar_t buffer[MAX_BUFFER_SIZE];
       UINT len = GetWindowsDirectoryW(buffer, MAX_BUFFER_SIZE);
       if (!len) {
           std::wcout << "GetWindowsDirectory failed with error code: " << GetLastError() << std::endl;
       }
       else
       {
           std::wcout << buffer << std::endl;
       }
       return 0;
   }
   ```

   输出：

   `C:\WINDOWS`

5. 当前目录。使用`GetCurrentDirectory`获取。

   ```c++
   int main() {
       const int MAX_BUFFER_SIZE = MAX_PATH;
       wchar_t buffer[MAX_BUFFER_SIZE];
       DWORD len = GetCurrentDirectoryW(MAX_BUFFER_SIZE, buffer);
       if (!len) {
           std::wcout << "GetCurrentDirectory failed with error code: " << GetLastError() << std::endl;
       }
       else
       {
           std::wcout << buffer << std::endl;
       }
       return 0;
   }
   ```

   这个函数的输出可能和应用程序所在的路径不同。

6. 系统环境变量PATH下的目录。



### DLL候补搜索顺序

当使用`LoadLibraryEx`函数，并且指定**LOAD_WITH_ALTERED_SEARCH_PATH**标记，以及 `lpFileName`参数是绝对路径时，会触发DLL候补搜索顺序。候补搜索顺序和标准搜索顺序的区别只在第一条。

安全模式下DLL候补搜索顺序：

1. 参数`lpFileName`所在的目录。
2. 系统目录。
3. 16位系统目录。
4. Windows目录。
5. 当前目录。
6. 系统环境变量PATH下的目录。

通过调用`SetDllDirectory`函数也可以改变候补搜索顺序：

1. 应用程序加载的目录。
2. 通过`SetDllDirectory`添加的目录。
3. 系统目录。
4. 16位系统目录。
5. Windows目录。
6. 系统环境变量PATH下的目录。

### 自定义搜索顺序

应用程序可以通过指定`LoadLibraryEx`或`SetDefaultDllDirectories`函数中不同的搜索标记来自定义搜索顺序。下面是几个常用标记：

1. **LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR**，DLL所在的目录。
2. **LOAD_LIBRARY_SEARCH_APPLICATION_DIR**，应用程序目录。
3. **LOAD_LIBRARY_SEARCH_USER_DIRS**，用户事先通过接口`AddDllDirectory`或`SetDllDirectory`指定的一个或多个目录。
4. **LOAD_LIBRARY_SEARCH_SYSTEM32**， 系统目录。

### 参考

1. https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
2. https://docs.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa