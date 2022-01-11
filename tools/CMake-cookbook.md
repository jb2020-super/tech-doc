source code

- https://github.com/dev-cafe/cmake-cookbook

continuous integration(CI) services:

- https://travis-ci.org
- https://www.appveyor.com/
- https://circleci.com

## CH 01

### Recipe 1

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

生成的文件：

- CMakeFiles文件夹包含一些临时文件，用于检测操作系统，编译器等。
- cmake_install.cmake，用于处理安装规则的CMake脚本。
- CMakeCache.txt，再次运行CMake时使用的缓存文件。

### Recipe 2: Switch Generator

```bash
cmake --help #列出可用的生成器
cmake -G "Visual Studio 14 2015 Win64" ..
```

### Recipe 3: Generate Library

生成静态或动态库工程

```
add_library(lib_name STATIC lib.cpp) 
```

- STATIC, 静态库
- SHARED, 动态库
- OBJECT，可用于生成动态库和静态库
- MODULE
- IMPORTED
- INTERFACE
- ALIAS

如何使用库

```
target_link_libraries(hello-world lib_name)
```

### Recipe 4: Conditionals

```cmake
set(USE_LIBRARY OFF)
if(USE_LIBRARY)
  #...
else()
  #...
endif()
```

### Recipe 5: Options

```cmake
option(USE_LIBRARY "Help string" OFF)
```

```bash
cmake -D USE_LIBRARY=ON ...
```

```cmake
include(CMakeDependentOption)
cmake_dependent_option(MAKE_STATIC_LIBRARY "Help string" OFF "USE_LIBRARY" ON)
```

cmake_dependent_option的作用是，当USE_LIBRARY为True时，MAKE_STATIC_LIBRARY对用户可见，并且默认值设为OFF;否则，对用户不可见，值设为ON。

### Recipe 6: 指定编译器

推荐使用这种方式，而不是通过暴露环境变量

```bash
cmake -D CMAKE_CXX_COMPILER=clang++
cmake --system-information information.txt # 输出系统信息
```

### Recipe 7: Build Type

multiple-configuration generators

```cmake
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CONFIGURATION_TYPES "Release;Debug")
```

```bash
cmake --build . --config Release
```



## References

- https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/