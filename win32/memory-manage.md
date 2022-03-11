## Windows系统内存管理

- 在32位系统上，一个进程的虚拟内存空间默认是2GB，在开启大内存并且修改启动项的情况下，可以达到3GB；64位系统可以达到4GB。
- 虚拟内存空间驻留在物理内存中的那部分子集叫做工作集（working set）。
- 当物理内存不够时，内存管理器复杂将一部分内存页交换到磁盘。
- 使用Address Windowing Extension(AWE)技术，可以使32位进程使用超过4GB的内存。使用步骤如下：

![image-20220310171728605](F:\github\tech-doc\win32\image-20220310171728605.png)

- 1.  使用接口AllocateUserPhysicalPages 或AllocateUserPhysicalPagesNuma 分配物理内存。
  2. 使用接口VirtualAlloc, VirtualAllocEx, or VirtualAllocExNuma（带MEM_PHYSICAL标记）， 在进程的虚拟地址空间中创建一个或多个AWE窗口。
  3. 使用接口MapUserPhysicalPages 或 MapUserPhysicalPagesScatter，将第一步中的AWE 内存和第二步中的AWE窗口建立映射关系。
- 这种方法也有一些限制：
- 1. 页面无法被其他进程共享
  2. 物理页面无法映射到超过一个的虚拟地址
  3. 页面保护限制在read/write, read-only和no access