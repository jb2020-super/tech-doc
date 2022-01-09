## What is CMake?

CMake is a build-system generator. Generate platform-native build system with the same set of CMake scripts.

## Books , Docs and Tutorials

- CMake Cookbook. https://github.com/dev-cafe/cmake-cookbook
- https://cmake.org/documentation/
- https://github.com/boostcon/cppnow_presentations_2017/blob/master/05-19-2017_friday/effective_cmake__daniel_pfeifer__cppnow_05-19-2017.pdf
- https://github.com/TheErk/CMake-tutorial
- https://crascit.com/tag/cmake/
- https://github.com/onqtam/awesome-cmake
- Mastering CMake by Ken Martin and Bill Hoffman, 2015, Kitware Inc.
- Professional CMake by Craig Scott: https://crascit.com/professional-cmake/



1. 创建CMakeLists.txt
2. mkdir build
3. cd build
4. cmake ../
5. cmake --build .

## Picking a generator

```bash
cmake --help
cmake -G "Visual Studio 16 2019" -A 
```

## Setting options

````bash
# common options
-DCMAKE_BUILD_TYPE=Release # Release Debug
-DCMAKE_INSTALL_PREFIX="D:/" # The location to install
-DBUILD_SHARED_LIBS=ON #
-DBUILD_TESTING=
````

