# TimeCounter.h

Header only simple time counter. The output info is printed to console window.

## How to use

```c++
#include "TimeCounter.h"
int main(){
    TimeCounter tc;
    tc.OpenConsoleWindow();
    tc.StartCount();
    // do some work
    tc.EndCount();
    tc.StandardPrint();
    return 0;
}

```
## Output Format
```
[frame_number][user info]cur: xxx, avg: xxx, fps: xxx, total_time: xx:xx:xx.xx
```
