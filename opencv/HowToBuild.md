## How To Build

build 32-bit version

```bash
cd build_folder_x86
cmake -G "Visual Studio 16 2019" -A Win32 -DBUILD_LIST=tracking -DOPENCV_EXTRA_MODULES_PATH=opencv_root_path/opencv_contrib/modules/ opencv_root_path/opencv/
```

build 64-bit version

```bash
cd build_folder_x64
cmake -DBUILD_LIST=tracking -DOPENCV_EXTRA_MODULES_PATH=opencv_root_path/opencv_contrib/modules/ opencv_root_path/opencv/
```

The path of **opencv** and **opencv_contrib** should be absolute path.

