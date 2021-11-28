## 同时播放多个视频

https://github.com/mpv-player/mpv/issues/3854

同时播放两个视频

mpv.exe a.mp4 --external-file=b.mp4 --lavfi-complex='[vid1] [vid2] hstack [vo]' --loop

| a | b |

同时播放四个视频

mpv.exe a.mp4 --external-file=b.mp4 --external-file=c.mp4 --external-file=d.mp4 --lavfi-complex='[vid1] [vid2] hstack [t1];[vid3] [vid4] hstack [t2]; [t1] [t2] vstack [vo]' --loop

| a | b |

| c | d |



## 常用操作快捷键

单步，, .

调播放速度，[ ]