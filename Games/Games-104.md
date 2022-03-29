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

## 3. 如何构建游戏世界

- Dynamic Game Objects
- Static Game Object
- Environment: Sky, Terrain, Vegetation。地形系统。天气系统。日夜变换系统。植被。
- Everything is a Game Object(GO).统一抽象成GO
- 如何描述一个物体。属性+行为。
- Component Base。用组件化思想构建世界，而不是面向对象继承关系。 解决公共继承的问题。
- Object-based Tick。Tick的时序问题。
- Component-based tick. 并行处理，减少cache miss。每一种组件一起tick。流水线。
- 事件机制，处理GO之间的交互（通信）。邮局系统。
- 场景管理。 GO id + position。Hierarchical segmentation。层级结构。BVH(Bounding Volume Hierarchies)，Octree， BSP
- 时序问题？最精妙。
- Q&A。 
  - Tick时间过长？分批延迟处理。