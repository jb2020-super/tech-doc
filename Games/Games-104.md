- 游戏引擎是复杂度的艺术；是艺术创作的生产力工具（工具链）
- 图形学只是游戏引擎中的一小部分

## 2. Layered Architecture of Game Engine

分层架构

- Chain of Editor
  - Digital Content Creation
- Function Layer。
  - 多线程，多核架构
- Resource Layer。Importing,转换成统一的asset file format，GUID，资源的唯一标识符。资源管理，handle系统。
  - 资源管理。生命期管理。内存是有限的。GC，垃圾回收，延迟加载。
- Core Layer
  - 数学库。SIMD
  - 核心数据结构。内存管理。内存碎片。
  - 数据放一起，提高cache命中。顺序访问。按块分配和释放。
- Platform Layer
  - 文件系统差异
  - 图形API