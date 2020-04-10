# FFmpeg

## 介绍

### 下载

[http://ffmpeg.org/download.html](https://link.jianshu.com/?t=http%3A%2F%2Fffmpeg.org%2Fdownload.html)

### FFmpeg介绍

FFmpeg是根据GNU通用公共许可证获得许可的多媒体处理自由软件项目的名称。 该项目最受欢迎的部分是用于视频和音频编码/解码的ffmpeg命令行工具，其主要特点是速度快，输出质量高和文件大小比较小。 FFmpeg中的“FF”表示 媒体播放器上的表示“快进”的控制按钮，“mpeg”是Moving Pictures Experts Group的缩写。

| ffmpeg   | 快速音频和视频编码器/解码器                |
| -------- | ------------------------------------------ |
| ffplay   | 媒体播放器                                 |
| ffprobe  | 显示媒体文件的特点                         |
| ffserver | 使用HTTP和RTSP协议进行多媒体流的广播服务器 |

| libavcodec    | 各种多媒体编解码器的软件库 |
| ------------- | -------------------------- |
| libavdevice   | 软件库的设备               |
| libavfilter   | 软件库包含过滤器           |
| libavformat   | 媒体格式的软件库           |
| libavutil     | 包含各种实用程序的软件库   |
| libpostproc   | 用于后期处理的软件库       |
| libswresample | 用于音频重采样的软件库     |
| libswscale    | 用于媒体扩展的软件库       |

ffmpeg程序读入内存中指定的任意数量的输入的内容，根据输入的参数或程序的默认值对其进行处理，并将结果写入任意数量的输出。输入和输出可以是计算机文件、管道、网络流、抓取设备等

在代码转换过程中，ffmpeg在**libavformat**库中调用**demuxers**来读取输入，并从数据包中获取编码数据。如果有更多的输入，ffmpeg可以通过跟踪任何活动输入流的最低时间戳来保持它们的同步。然后解码器从编码的数据包中生成未压缩的帧，在可选的过滤后，帧被发送到编码器。编码器产生新的编码包，它被发送到**muxer**并写入到输出。

FFmpeg工具的重要部分是过滤器，它可以被组织成过滤链和**filtergraphs**。Filtergraphs可以是简单的或复杂的。在解码源和编码输出之间实现滤波处理。

### flters**, **filterchains**，**filtergraphs

在多媒体处理中，术语过滤器是指在编码到输出之前修改输入的软件工具。过滤器分为音频和视频过滤器。FFmpeg内置了许多多媒体过滤器，可以通过多种方式组合它们。FFmpeg的过滤API(应用程序编程接口)是libavfilter软件库。

过滤器包括在输入和输出之间使用-vf选项的视频过滤器和-af选项音频过滤器

例如,下一个命令生成一个测试模式顺时针旋转90°使用转置过滤器

```cmd
ffplay -f lavfi -i testsrc -vf transpose=1
```

过滤器通常用于filterchains(逗号分隔的过滤器序列)和filtergraphs(分号分隔的filterchains序列)。

### 选择的媒体流

一些媒体容器如AVI、Matroska、MP4等可以包含多种类型的流，FFmpeg可以识别5种流类型:音频(a)、附件(t)、数据(d)、字幕(s)和视频(v)。

| 说明符形式                  | 描述                                                         |
| --------------------------- | ------------------------------------------------------------ |
| stream_index                | 选择该索引的流(编号)                                         |
| stream_type[:stream_index]  | stream_type为字母a(音频)、d(数据)、s(字幕)、t(附件)或v(视频);如果添加了stream_index，它将选择该类型的流并使用给定的索引，否则它将选择该类型的所有流 |
| p:program_id[:stream_index] | 如果添加了stream_index，那么使用给定的program_id在程序中选择带有stream_index的流，否则将选择该程序中的所有流 |
| stream_id                   | 按格式指定的ID选择流                                         |

```cmd
//设置音频和视频的使用-b选项的比特率
ffmpeg -i input.mpg -b:a 128k -b:v 1500k output.mp4
```

### 显示输出预览

```cmd
.\ffplay -i lin.mp4
```

### Lavfi虚拟设备

|             | 输入设备:lavfi                                               |
| ----------- | ------------------------------------------------------------ |
| 描述        | 从filtergraph的打开的输出pad中处理数据，对于每个输出垫创建一个对应的流，映射到编码。filtergraph由一个-graph选项指定，目前只支持视频输出垫 |
| 语法        | -f lavfi [-graph[ -graph_file]]                              |
|             | lavfi选项的描述                                              |
| -graph      | 作为输入的filtergraph，每个视频的开放输出必须用一个“outN”形式的唯一字符串标记，其中N是一个数字，从0开始，对应于设备生成的映射的输入流。第一个未标记的输出将自动分配给“out0”标签，但是其他所有的输出都需要显式指定。如果没有指定，则默认为输入设备指定的文件名 |
| -graph_file | filtergraph的文件名被读取并发送到其他过滤器，filtergraph的语法与选项-graph指定的语法相同 |

Lavfi通常用于显示测试模式，例如带有命令的SMPTE条:

```cmd
ffplay -f lavfi -i smptebars
```

## 组件和项目

FFmpeg项目由4个命令行工具和9个软件库组成，可供许多公司和软件项目使用。 

### FFplay介绍

FFplay是一个简单的媒体播放器，能够播放ffmpeg工具可以解码的所有媒体格式

| 描述       | 简单的媒体播放器，使用FFmpeg和SDL库，它主要用于测试和开发 |
| ---------- | --------------------------------------------------------- |
| 语法       | ffplay [options] [input_file]                             |
|            | 参数描述                                                  |
| options    | 几乎所有可用于ffmpeg工具的选项都可以与ffplay一起使用      |
| input_file | 输入可以是常规文件，管道，网络流，抓取设备等              |

使用ffplay在lightorange背景上显示各种testsrc视频源

```
ffplay -f lavfi -i testsrc -vf pad=400:300:(ow-iw)/2:(oh-ih)/2:orange
```

#### 键和鼠标控制

| key                    | 描述                                        |
| ---------------------- | ------------------------------------------- |
| q, ESC                 | 退出                                        |
| f                      | 切换全屏                                    |
| p, Spacebar            | 切换暂停                                    |
| a                      | 音频通道                                    |
| v                      | 视频通道                                    |
| t                      | 可用的字幕                                  |
| w                      | 在可用的显示模式选项中循环:视频，rdft，音频 |
| 向左的箭头 /向右的箭头 | 向后/向前拖动10秒钟                         |
| 向下翻页/ 向上翻页     | 向后/向前拖动10分钟                         |
| 点击鼠标               | 查找与宽度部分对应的文件中的百分比          |

#### 显示模式

`rdft`（逆实数离散傅立叶变换）

`waves`（来自滤波器显示波的音频波）

###  FFprobe介绍

ffprobe是一种从多媒体流中收集信息并以人机和机器可读方式打印的实用程序。 它可用于检查多媒体流使用的容器的格式以及其中每个媒体流的格式和类型。 选项用于列出ffprobe支持的一些格式或设置显示哪些信息，并设置ffprobe如何显示它。 其输出易于通过文本过滤器进行分析，并且由`-of`（或`-print_format`）选项指定的选定writer定义的表单的一个或多个部分组成。

FFmpeg组件:ffprobe

| 描述                         | 命令行工具，用于检测多媒体流中的各种数据以进行分析。 它可以单独使用或与文本过滤器一起使用，以获得复杂的处理 |
| ---------------------------- | ------------------------------------------------------------ |
| 语法                         | ffprobe [options] [input_file]                               |
|                              | 参数描述                                                     |
| options                      | 几乎所有的ffmpeg工具可用的选项都可以使用ffprobe              |
| input_file                   | 输入可以是常规文件、管道、网络流、抓取设备等                 |
|                              | 附加的ffprobe选项                                            |
| -bitexact                    | force bit精确输出，用于产生不依赖于特定构建的输出            |
| -count_frames                | 每个流的帧数，并在相应的流部分报告                           |
| -count_packets               | 每个流的数据包数，并以相应的流部分方式报告设置打印格式，w_name是写入器名称，w_options是写入器选项 |
| -of w_name[=w_options]       | 设置打印格式，w_name是写入器名称，w_options是写入器选项      |
| -select_streams str_spec     | 只选择str_spec指定的流，可以是下一个字母:a=audio, d=data, s=subtitle, t=attachment, v=video |
| -show_data                   | 显示有效负载数据，如hex和ASCII转储，加上- show_数据包，它转储数据包的数据，再加上-show_streams，它转储codec extradata |
| -show_error                  | 在探测输入时显示有关发现错误的信息                           |
| -show_format                 | 显示有关输入媒体流的容器格式的信息                           |
| -show_format_entry name      | 像-show_format，但只打印由容器格式信息指定的条目，而不是全部 |
| -show_frames                 | 显示输入媒体流中包含的每个框架的信息                         |
| -show_library_version        | 显示与库版本相关的信息                                       |
| -show_packets                | 输入媒体流中包含的每个数据包的信息如何                       |
| -show_private_data  -private | 显示数据依赖于特定显示元素的格式，选项是默认启用的，但是可以设置为0，例如在创建符合xsd的XML输出时 |
| -show_streams                | 显示输入媒体流中包含的每个媒体流的信息                       |
| -show_versions               | 显示与程序和库版本相关的信息，这相当于设置了-show_program_version和-show_library_version选项 |

### FFserver介绍

ffserver是一个在Linux上运行的多媒体流媒体服务器，官方的Windows二进制文件还不能使用

FFmpeg组件:ffserver

| 描述          | 为音频和视频提供流媒体服务器的实用程序。 它支持多个实时供稿，从文件流式传输并在实时供稿上进行时间转换。 如果在ffserver.conf配置文件中指定了足够的存储源存储，则可以在每个实时供稿中寻找过去的位置。 ffserver默认在守护进程模式下在Linux上运行，这意味着它将自身置于后台并从其控制台分离，除非它以调试模式启动或在配置文件中指定了NoDaemon选项。 |
| ------------- | ------------------------------------------------------------ |
| 语法          | ffserver [options]                                           |
|               | 参数的描述                                                   |
| options       | 几乎所有的ffmpeg工具可用的选项都可以与ffserver一起使用       |
|               | 额外的ffserver选项                                           |
| -d            | 启用调试模式，这会增加日志的冗余性，将日志消息定向到stdout，并导致ffserver在前台运行，而不是作为一个守护进程 |
| -f configfile | 使用configfile而不是/etc/ffserver.conf                       |
| -n            | 启用无启动模式，这将禁用各种部分的所有启动指令，因为ffserver将不会启动任何ffmpeg实例，您将不得不手动启动它们 |

### FFmpeg软件库

#### libavcodec

libavcodec是一个用于解码和编码多媒体的编解码器库

#### libavdevice

libavdevice是一个特殊的设备复用/解复用库，是libavformat库的补充。

它提供了各种平台特定的复用器和解复用器，例如用于抓取设备，音频采集和播放。

#### libavfilter

libavfilter是一个过滤器库，它为FFmpeg和客户端库或应用程序提供媒体过滤层。

#### libavformat

libavformat是一个包含音频/视频容器格式的解复用和复用的库。

#### libavutil

libavutil是包含用于FFmpeg的不同部分的例程的辅助库

#### libpostproc

libpostproc是一个包含视频后处理例程的库。

#### libswresample

libswresample库能够处理不同的采样格式，采样率和不同数量的通道以及不同的通道布局。 它支持直接转换样品格式和一次打包/平面。

#### libswscale

libswscale是一个包含视频图像缩放例程的库，并提供快速模块化缩放界面



## 比特率/帧率/文件大小

比特率和帧速率是视频的基本特征，它们的正确设置对整体视频质量非常重要

### 帧率(频率)的介绍

帧速率是编码成视频文件的每秒帧数（FPS或fps）,人眼需要至少约15 fps来观看连续运动,LCD显示器通常具有60 Hz的频率.

有两种帧速率 - 隔行扫描（在FPS编号后表示为i）和逐行扫描（在FPS编号后表示为p）。

> 隔行扫描：先扫描奇数行，再扫描偶数行。两个‘场’

### 帧率设置 -r

```cmd
//avi文件从25到30 fps值
ffmpeg -i input.avi -r 30 output.mp4
```

### 使用FPS过滤器

另一种设置帧速率的方法是使用fps过滤器，这在过滤链中尤其有用。

```cmd
ffmpeg -i clip.mpg -vf fps=fps=25 clip.webm
```

### 帧率的预定义值

除了数值，设置帧率的两种方法都接受下一个预定义的文本值:

| 缩写               | 精确值     | 相应的FPS（相应的帧） |
| ------------------ | ---------- | --------------------- |
| ntsc-film          | 24000/1001 | 23.97                 |
| film               | 24/1       | 24                    |
| pal, qpal, spal    | 25/1       | 25                    |
| ntsc, qntsc, sntsc | 30000/1001 | 29.97                 |

```cmd
//设置帧速率为29.97 fps，接下来的3个命令给出了相同的结果
ffmpeg -i input.avi -r 29.97 output.mpg
ffmpeg -i input.avi -r 30000/1001 output.mpg
ffmpeg -i input.avi -r ntsc output.mpg
```

### 比特(数据)率的介绍

比特率（也是比特率或数据率）是决定整体音频或视频质量的参数。 它规定了每时间单位处理的位数，在FFmpeg中，位速率以每秒位数表示

| 类型       | 缩写 | 描述                                                         |
| ---------- | ---- | ------------------------------------------------------------ |
| 平均比特率 | ABR  | 平均每秒处理的位数，该值也用于VBR编码，需要时是输出的某个文件大小 |
| 恒定比特率 | CBR  | 每秒处理的比特数是恒定的，这对于存储是不实际的，因为具有快速运动的部分需要比静态比特更多的比特，CBR主要用于多媒体流 |
| 可变比特率 | VBR  | 每秒处理的比特数是可变的，复杂的场景或声音被编码更多的数据并与CBR进行比较，相同尺寸的文件的VBR质量比CBR更好（VBR编码比CBR需要更多的时间和CPU功率 ，但最近的媒体播放器可以充分解码VBR。） |

### 设置比特率 -b

比特率决定了存储1秒编码流的位数，它使用-b选项设置

```cmd
//设置总体1.5 Mbit每秒的比特率
ffmpeg -i film.avi -b 1.5M film.mp4
```

如果可能的话，ffmpeg使用一个可变比特率（VBR），并对比具有快速运动的部分具有更少比特的静态部分进行编码。 ffmpeg通常用于使用高级编解码器来降低输出文件的比特率和相应的文件大小，

```cmd
ffmpeg -i input.avi -b:v 1500k output.mp4
```

### 固定比特率(CBR)设置

例如视频会议之类的**实时视频流**，可以使用固定的比特率，因为传输的数据不能被缓冲。

为了设置输出的恒定比特率，三个参数必须具有相同的值:比特率(-b选项)、最小速率(-minrate)和最大速率(-maxrate)。对于minrate和maxrate选项可以添加一个流指示符，maxrate选项需要设置一个-bufsize选项(比特的速率控制缓冲区大小)。例如，要设置0.5 Mbit/s的CBR，我们可以使用以下命令:

```cmd
ffmpeg -i in.avi -b 0.5M -minrate 0.5M -maxrate 0.5M -bufsize 1M out.mkv
```

### 设置输出文件的最大尺寸 -fs

为了使输出文件的大小保持一定的值，我们使用-fs选项（文件大小的缩写）,以字节为单位

```cmd
//指定10兆字节的最大输出文件大小
ffmpeg -i input.avi -fs 10MB output.mp4
```

### 文件的大小计算

编码输出的最终文件大小是音频和视频流大小的总和。以字节为单位的视频流大小的方程是(由比特到字节的转换为8):

```
video_size = video_bitrate * time_in_seconds / 8
```

如果音频未压缩，其大小由公式计算:

```
 audio_size = sampling_rate * bit_depth * channels * time_in_seconds / 8
```

要计算压缩音频流的文件大小，我们需要知道它的比特率和方程。

```
audio_size = bitrate * time_in_seconds / 8。
```

例如，用1500 kbits/s的视频比特率和128 kbits/s音频比特率计算10分钟视频剪辑的最终大小，我们可以使用这些公式:

```
file_size = video_size + audio_size
file_size = (video_bitrate + audio_bitrate) * time_in_seconds / 8
file_size = (1500 kbit/s + 128 kbits/s) * 600 s
file_size = 1628 kbit/s * 600 s
file_size = 976800 kb = 976800000 b / 8 = 122100000 B / 1024 = 119238.28125 KB
file_size = 119238.28125 KB / 1024 = 116.443634033203125 MB ≈ 116.44 MB
```

- 1 byte (B) = 8 bits (b)
- 1 kilobyte (kB or KB) = 1024 B
- 1 megabyte (MB) = 1024 KB, 等.
   最终文件的大小比计算的要大一些，因为包含了一个muxing开销和文件元数据

## 调整缩放视频 scale

在FFmpeg中调整视频的大小意味着可以通过一个选项改变其宽度和高度，而缩放则意味着使用一个具有高级功能的**scale filter**来改变帧的大小.

### 调整视频 -s

输出视频的宽度和高度可以在输出文件名之前设置-s选项。视频分辨率以wxh格式输入，其中w为像素宽度，h为像素高度。例如，要将初始分辨率的输入调整为320x240，我们可以使用以下命令:

```cmd
ffmpeg -i input_file -s 320x240 output_file
```

### 预定义的视频帧大小

FFmpeg工具没有输入视频宽度和高度的精确数字，而是提供了在下一页的表中列出的预定义视频大小。下面两个命令的结果相同:

```cmd
ffmpeg -i input.avi -s 640x480 output.avi
ffmpeg -i input.avi -s vga output.avi
```

FFmpeg中视频大小的缩写

| 大小       | 缩写        | 典型用法                 |
| ---------- | ----------- | ------------------------ |
| 128x96     | sqcif       | 手机                     |
| 160x120    | qqvga       | 手机                     |
| 176x144    | qcif        | 手机                     |
| 320x200    | cga         | 旧的CRT显示器            |
| 320x240    | qvga        | 手机、摄像头             |
| 352x288    | cif         | 手机                     |
| 640x350    | ega         | 旧的CRT显示器            |
| 640x480    | vga         | 显示器,摄像头            |
| 704x576    | 4cif        | 官方数字视频大小的电视。 |
| 800x600    | svga        | 显示器                   |
| 852x480    | hd480, wvga | 摄像机                   |
| 1024x768   | xga         | 显示器,摄像头            |
| 1280x720   | hd720       | 高清电视,摄像机          |
| 1280 x1024 | sxga        | 显示器                   |
| 1366x768   | wxga        | 显示器                   |
| 1408x1152  | 16cif       | 设备使用CIF              |
| 1600x1024  | wsxga       | 显示器                   |
| 1600x1200  | uxga        | 显示器，摄像机           |
| 1920x1080  | hd1080      | 高清电视,摄像机          |
| 1920x1200  | wuxga       | 宽屏显示器               |
| 2048x1536  | qxga        | 显示器                   |
| 2560x1600  | woxga       | 显示器                   |
| 2560x2048  | qsxga       | 显示器                   |
| 3200x2048  | wqsxga      | 显示器                   |
| 3840x2400  | wquxga      | 显示器                   |
| 5120x4096  | hsxga       | 显示,显微镜相机          |
| 6400x4096  | whsxga      | 显示器                   |
| 7680x4800  | whuxga      | 显示器                   |

### 调整大小时的注意事项

奈奎斯特 -Shannon采样定理

视频通常被调整为比来源更小的分辨率，这被称为下采样，主要用于便携式设备，通过互联网流媒体等,在较小的尺寸中，一些细节将会丢失.

### 专业的扩大滤波器

将视频调整为更大的帧大小比较少见，因为该功能几乎可以提供所有媒体播放器，但由此产生的图像有时并不清晰，特别是当源解析度非常小时。 用于平滑放大的源的特殊滤波器是super2xsai滤波器

将**128x96**视频从移动电话放大到分辨率**256x192**像素，可以使用下一个命令:

```cmd
 ffmpeg -i phone_video.3gp -vf super2xsai output.mp4
```

### 调整过滤器scale

当使用-s选项更改视频帧大小时，会在相关滤镜图片的末尾插入缩放视频滤镜。 要管理缩放过程开始的位置，可以直接使用缩放过滤器。

视频过滤器:缩放

| 描述        | 通过更改输出样本宽高比来缩放源，显示宽高比保持不变。 |
| ----------- | ---------------------------------------------------- |
| 语法        | scale=width:height[:interl={1\|-1}]                  |
| **********  | 变量表示宽度和高度参数。                             |
| iw or in_w  | 输入的宽度                                           |
| ih or in_h  | 输入的高度                                           |
| ow or out_w | 输出的宽度                                           |
| oh or out_h | 输出的高度                                           |
| a           | 纵横比，与iw/ih相同。                                |
| sar         | 输入样本纵横比，与dar/a相同。                        |
| dar         | 输入显示纵横比，与*sar相同。                         |
| hsub        | 水平色度子样本值，为yuv422p像素格式为2。             |
| vsub        | 垂直色度子样本值，为yuv422p像素格式为1。             |
| **********  | 可选的interl参数的可用值。                           |
| 1           | 应用交错感知扩展。                                   |
| -1          | 如果源被标记为交错的，应用是交错的意识扩展。         |

例如，下面两个命令的结果相同:

```
ffmpeg -i input.mpg -s 320x240 output.mp4
ffmpeg -i input.mpg -vf scale=320:240 output.mp4
```

scale filter的优点是，对于框架设置，可以使用上面表中描述的其他参数。

```cmd
//创建一个半大小的视频
ffmpeg -i input.mpg -vf scale=iw/2:ih/2 output.mp4
```

## 裁剪视频 crop

### 基础知识

裁剪视频意味着从输入到输出中选择想要的矩形区域而没有余数。 裁剪通常用于调整大小，填充和其他编辑

视频过滤器:裁剪

| 描述         | 将输入视频帧的宽度和高度从x和y值表示的位置裁剪到指定的宽度和高度;x和y是输出的左上角坐标，协调系统的中心是输入视频帧的左上角。如果使用了可选的keep_aspect参数，将会改变输出SAR(样本宽比)以补偿新的DAR(显示长宽比) |
| ------------ | ------------------------------------------------------------ |
| 语法         | crop=ow[:oh[:x[:y[:keep_aspect]]]]                           |
| ************ | 用于ow和oh参数的表达式中的可用变量                           |
| x, y         | 对x的计算值(从左上角水平方向的像素个数)和y(垂直像素的数量)，对每个帧进行评估，x的默认值为(iw - ow)/2, y的默认值为(ih - oh)/2 |
| in_w, iw     | 输入的宽度                                                   |
| in_h, ih     | 输入的高度                                                   |
| out_w, ow    | 输出(裁剪)宽度，默认值= iw                                   |
| out_h, oh    | 输出(裁剪)高度，默认值= ih                                   |
| a            | 纵横比，与iw/ih相同                                          |
| sar          | 输入样本比例                                                 |
| dar          | 输入显示宽比，等于表达式a*sar                                |
| hsub, vsub   | 水平和垂直的色度子样本值，对于像素格式yuv422p, hsub的值为2,vsub为1 |
| n            | 输入框的数目，从0开始                                        |
| pos          | 位置在输入框的文件中，如果不知道NAN                          |
| t            | 时间戳以秒表示，如果输入时间戳未知                           |

```cmd
//在输入框的左三、中三和右三
ffmpeg -i input -vf crop=iw/3:ih:0:0 output
ffmpeg -i input -vf crop=iw/3:ih:iw/3:0 output
ffmpeg -i input -vf crop=iw/3:ih:iw/3*2:0 output
```

**ow/oh 表示输出宽高，x/y表示裁剪起点坐标。**

**x默认值：（iw - ow）/2**

**y默认值：（ih - oh）/2**

### 自动检测裁剪区域

为了自动检测出裁剪的非黑色区域，我们可以使用crop检测过滤器，如下表所示。当输入视频中包含一些黑条时，这种自动裁剪是有用的，通常是在从第4:3到16:9的转换之后，反之亦然。

| 描述         | 检测作物过滤器的作物大小，结果是由参数确定的输入帧的非黑色区域 |
| ------------ | ------------------------------------------------------------ |
| 语法         | cropdetect[=limit[:round[:reset]]] all parameters are optional |
| ************ | 参数的描述                                                   |
| limit        | 阈值，从0(无)到255 (all)，默认值= 24                         |
| round        | -即使是整数，宽度和高度也必须是可分割的 - 4:2 2视频需要一个2的值，它只给出了维度 -偏移量自动更改为中心帧 -默认值为16，它是许多编解码器的最佳值 |
| reset        | 计数器决定了多少帧crop探测将重置之前检测到的最大视频区域并重新开始检测当前最优的作物区域。默认值为0。当通道标识扭曲了视频区域时，这是很有用的。0表示永远不会重置和返回在回放期间遇到的最大区域 |

limit参数指定了选择了多少深颜色的输出，零值意味着只有完整的黑色被裁剪。例如，要裁剪非黑输出，我们可以使用以下命令:

```cmd
ffmpeg -i input.mpg -vf cropdetect=limit=0 output.mp4
```

### 时间的裁剪

媒体播放器通常有一个进度条，显示经过的秒数，但大多数只有在鼠标指针停止并在特定持续时间后隐藏时才会显示。 FFmpeg包含一个包含定时器的testsrc视频源，

## 填充视频 pad

填充视频意味着向视频帧添加额外的区域以包含额外的内容

### 基础知识

对于视频填充，我们使用表格中描述的填充过滤器。

| 描述          | 在输入视频帧中添加彩色填充，该帧位于协调系统中的[x，y]点，其中输出帧的左上角是[0,0]。 输出的大小由宽度和高度参数设置。 |
| ------------- | ------------------------------------------------------------ |
| 语法          | pad=width[:height[:x[:y[:color]]]] 中括号里面的参数都是可选的 |
| ***********   | 参数的描述                                                   |
| color         | 十六进制形式的RGB颜色值：0xRRGGBB [@AA]，其中AA的范围是（0,1）中的十进制值或任何有效的颜色名称，如白色，蓝色，黄色等，默认值为黑色，请参见颜色有关详细信息，请参阅[FFmpeg基本介绍](https://www.jianshu.com/p/7f675764704b)章节中的名称部分 |
| width, height | 带填充的输出帧的宽度和高度，宽度的值可以从高度导出，反之亦然，两个参数的默认值都是0 |
| x, y          | 输入左上角的坐标（偏移量）与输出帧的左上角有关，两个参数的默认值均为0 |
| ***********   | 参数的高度，宽度，x, y的表达式的可用变量                     |
| a             | 纵横比，与iw/ih相同                                          |
| dar           | 输入显示宽比，与*sar相同                                     |
| hsub, vsub    | 水平和垂直的色度子样本值，对于像素格式yuv422p, hsub的值为2,vsub为1 |
| in_h, ih      | 输入的高度                                                   |
| in_w, iw      | 输入的宽度                                                   |
| n             | 输入框的数目，从0开始                                        |
| out_h, oh     | 输出高度，默认值=高度                                        |
| out_w, ow     | 输出宽度，默认值=宽度                                        |
| pos           | 位置在输入框的文件中，如果不知道NAN                          |
| sar           | 输入样本比例                                                 |
| t             | 时间戳以秒表示，如果输入时间戳未知                           |
| x, y          | x和y的偏移量由x和y表示，或者NAN如果没有指定                  |

```cmd
//在一个svga大小的照片周围创建一个30像素宽的粉红色框架
ffmpeg -i photo.jpg -vf pad=860:660:30:30:pink framed_photo.jpg
```

### 从4:3到16:9的填充视频

有些设备只能以16：9的宽高比播放视频，而4：3宽高比的视频必须在两种尺寸的水平方向上进行填充。 在这种情况下，高度保持不变，宽度等于高度值乘以16/9。 x值（输入视频帧水平偏移量）从表达式（output_width - input_width）/ 2开始计数，因此填充的语法为：

```cmd
ffmpeg -i input -vf pad=ih*16/9:ih:(ow-iw)/2:0:color output
```

### 从16:9到4:3的填充视频

为了显示在4:3高宽比的显示中所创建的视频，我们应该垂直地填充两个大小的输入。因此，宽度保持不变，高度为宽* 3/4。y值(输入视频帧垂直偏移)是从表达式(output_height - input_height)/2中计算的，填充的语法为:

```cmd
ffmpeg -i input -vf pad=iw:iw*3/4:0:(oh-ih)/2:color output
```

## 翻转和旋转视频

### 水平翻转 hflip

```cmd
//测试testsrc视频源的水平翻转，我们可以使用以下命令
ffplay -f lavfi -i testsrc -vf hflip
```

### 垂直翻转 vflip

```cmd
ffplay -f lavfi -i rgbtestsrc -vf vflip
```

### 旋转过滤器 transpose

| 描述        | 将行与输入的列进行转置，如果选择，也会翻转结果           |
| ----------- | -------------------------------------------------------- |
| 语法        | transpose={0, 1, 2, 3} one from the values 0 - 3 is used |
| *********** | 描述可用的值                                             |
| 0           | 输入由90°逆时针旋转,垂直翻转                             |
| 1           | 输入是顺时针旋转90°                                      |
| 2           | 输入是逆时针旋转90°                                      |
| 3           | 输入是顺时针旋转90°,垂直翻转                             |

```cmd
ffplay -f lavfi -i smptebars -vf transpose=0
ffplay -f lavfi -i smptebars -vf transpose=2,vflip
```

请注意，转置滤波器的值0和3在视频帧上同时提供两个操作——旋转和垂直翻转。这意味着值0的使用包括两个过滤器的效果，以上两个命令的结果相同

## 模糊，锐化和降噪

包含各种噪声的视频输入可以使用去噪滤波器和选项来增强。 在视频编码之前，去噪是视频预处理的一部分

### 模糊 boxblur

模糊效果用于提高图像（视频帧）中某些类型的噪声的质量，其中每个输出像素值是根据相邻像素值计算的。 

| 描述        | 使用均值模糊算法在输入上创建一个模糊效果                     |
| ----------- | ------------------------------------------------------------ |
| 语法        | boxblur=luma_r:luma_p[:chroma_r:chroma_p[:alpha_r:alpha_p]] filter expects 2 or 4 or 6 parameters, r =半径, p = 权重，程度，功率 |
| **********  | 参数                                                         |
| alpha_r     | -用于模糊相关输入平面(以像素为单位)的盒子的半径 - value是下面描述的变量的表达式 -默认值来源于luma_radius和luma_power |
| alpha_p     | - alpha功率，确定过滤器被应用到相关平面的次数 -默认值来源于luma_radius和luma_power |
| chroma_r    | -用于模糊相关输入平面(以像素为单位)的box的色度半径 - value是下面描述的变量的表达式 -默认值来源于luma_radius和luma_power |
| chroma_p    | -色度功率，确定过滤器被应用到相关平面的次数 -默认值来源于luma_radius和luma_power |
| luma_r      | -用于模糊相关输入平面(以像素为单位)的box的半径 - value是下面描述的变量的表达式 |
| luma_p      | - luma功率，确定过滤器被应用到相关平面的次数                 |
| *********** | 在表达式中，对阿尔法，色度和luma半径的变量                   |
| w,h         | 输入宽度和像素高度                                           |
| cw, ch      | 输入色度图像的像素宽度和高度                                 |
| hsub        | 水平色度子样本值，为yuv422p像素格式为2                       |
| vsub        | 垂直色度子样本值，为yuv422p像素格式为1                       |
|             | 半径是一个非负数，并且不能大于luma和阿尔法平面的表达式min(w,h)/2的值，以及对chroma平面的min(cw,ch)/2的值 |

例如，在输入视频中，当luma半径值为1.5,luma功率值为1时，我们可以使用下一个命令:

```cmd
ffmpeg -i input.mpg -vf boxblur=1.5:1 output.mp4
```

另一个FFmpeg模糊效果过滤器与是`smartblur`过滤器在表中描述:

| 描述        | 模糊输入而不影响轮廓                                         |
| ----------- | ------------------------------------------------------------ |
| 语法        | smartblur=luma_r:luma_s:luma_t[:chroma_r:chroma_s:chroma_t] parameters in [] are optional, r = radius, p = power, t = threshold |
| *********** | 参数的描述                                                   |
| chroma_r    | 色度(颜色)半径，从0.1到5.0的浮点数，它指定用于模糊图像的高斯滤波器的方差(如果更大) |
| chroma_s    | 色度强度，在范围-1.0到1.0之间的浮点数，配置模糊;从0.0到1.0的值将模糊图像，从-1.0到0.0的值将增强图像 |
| chroma_t    | chrominance treshold，一个从-30到30的整数，它被用作一个系数来决定一个像素是否应该被模糊;0的值将过滤所有的图像，0到30的值将过滤平坦区域，从-30到0的值将过滤边缘 |
| luma_r      | 亮度(亮度)半径，从0.1到5.0的浮点数，指定用于模糊图像的高斯滤波器的方差(如果更大的话，会更慢) |
| luma_s      | 亮度强度，从-1.0到1.0的浮动数值，配置模糊;从0.0到1.0的值将模糊图像，从-1.0到0.0的值将增强图像 |
| luma_t      | 亮度treshold，一个整数，范围从-30到30，作为一个系数来决定一个像素是否应该被模糊;0的值将过滤所有图像，0到30的值将过滤平面区域，从-30到0的值将过滤edgese图像 |
|             | 如果色度参数没有设置，则使用luma参数来实现像素的色度         |

例如，为了改进半色调图像，我们将luma半径设为最大值5，亮度强度为0.8，亮度阈值为0，因此整个图像是模糊的:

```cmd
ffmpeg -i halftone.jpg -vf smartblur=5:0.8:0 blurred_halftone.png
```

### 锐化视频 unsharp 

为了锐化或模糊视频帧，我们可以使用表中描述的不清晰的过滤器。

| 描述                     | 根据指定的参数增加或模糊输入视频                             |
| ------------------------ | ------------------------------------------------------------ |
| 语法                     | l_msize_x:l_msize_y:l_amount:c_msize_x:c_msize_y:c_amount all parameters are optional, if not set, the default is 5:5:1.0:5:5:0.0 |
| ***********              | 参数的描述                                                   |
| l_msize_x,luma_msize_x   | luma矩阵水平尺寸，3和13之间的整数，默认值为5                 |
| l_msize_y,luma_msize_y   | luma矩阵的垂直大小，整数在3和13之间，默认值是5               |
| l_amount,luma_amount     | luma效应强度，介于-2.0和5.0之间的浮点数，负值创建模糊效果，默认值为1 |
| c_msize_x,chroma_msize_x | 色度矩阵的水平大小，整数在3和13之间，默认值是5               |
| c_msize_y,chroma_msize_y | 色度矩阵的垂直大小，整数在3和13之间，默认值为5               |
| c_amount,chroma_amount   | chroma效果强度，-2.0和5.0之间的浮动值，负值创建模糊效果，默认值为0.0 |

锐化滤波器可以作为普通的不锐掩模和高斯模糊。例如，要使用默认值锐化输入，我们可以使用该命令。

```cmd
ffmpeg -i input -vf unsharp output.mp4
```

### 降噪 denoise3d

视频过滤器denoise3d减少了噪音，它是mp过滤器的一部分(来自MPlayer项目)。

| 描述           | 生成质量更好的平滑视频帧，并尝试提高可压缩性。               |
| -------------- | ------------------------------------------------------------ |
| 语法           | mp=denoise3d[=luma_spatial[:chroma_spatial[:luma_tmp[:chroma_tmp]]]] (所有参数都是可选的) |
| ***********    | 参数的描述                                                   |
| luma_spatial   | 空间luma强度，非负浮点数，默认值为4.0                        |
| chroma_spatial | 空间色度强度，非负浮点数，默认值为3.0                        |
| luma_tmp       | 时间luma强度，非负浮点数，默认值为6.0                        |
| chroma_tmp     | 时间色度强度，非负浮点数，默认值为luma_tmp*chroma_spatial/luma_spatial |

例如，要使用denoise3d过滤器的默认值来增强输入，我们可以使用该命令

```cmd
ffmpeg -i input.mpg -vf mp=denoise3d output.webm
```

### 降噪 hqdn3d

denoise3d过滤器的高级版本是hqdn3d过滤器，它已经在libavfilter库中，是一个本地的FFmpeg过滤器。过滤器的名称是高质量的denoise三维过滤器的缩写，它在表中描述

| 描述           | 生产高质量的平滑视频帧，并尝试提高压缩率，它是一个增强版的denoise3d过滤器 |
| -------------- | ------------------------------------------------------------ |
| 语法           | hqdn3d=[luma_spatial[:chroma_spatial[:luma_tmp[:chroma_tmp]]]] |
| ***********    | 参数的描述                                                   |
| luma_spatial   | 空间luma强度，非负浮点数，默认值为4.0                        |
| chroma_spatial | 空间色度强度，非负浮点数，默认值为3.0*luma_spatial/4.0       |
| luma_tmp       | 时间luma强度，非负浮点数，默认值为6.0*luma_spatial/4.0       |
| chroma_tmp     | 时间色度强度，非负浮点数，默认值为luma_tmp*chroma_spatial/luma_spatial |

例如，为了减少视频输入中带有默认hqdn3d值的噪声，我们可以使用以下命令:

```cmd
ffmpeg -i input.avi -vf hqdn3d output.mp4
```

### 使用nr选项进行降噪

如何减少视频输入中的噪音的其他方法是-nr（降噪）选项。 它的值是一个从0到100000的整数，其中0是默认值，范围1-600对公用内容有用。 如果视频包含强烈的噪音，请尝试使用更高的值。 由于此选项比denoise3d和hqdn3d过滤器使用的计算机资源少得多，因此当速度很重要时，它是消噪的首选方式。 例如，在较旧的计算机上，我们可以使用以下命令改善观看稍微噪声的视频：

```cmd
ffplay -i input.avi -nr 500
```



## 画中画 overlay

overlay视频技术经常被使用，常见的例子是放置在电视屏幕上的电视频道标志，通常位于右上角，以标识特定的频道。

### 介绍

视频overlay是一种技术，它可以在(通常是较大的)背景视频或图像上显示前景视频或图像。我们可以使用在表格中描述的覆盖视频过滤器:

| 描述        | 在指定位置上覆盖第一个输入                                   |
| ----------- | ------------------------------------------------------------ |
| 语法        | overlay[=x:y[[:rgb={0, 1}]] 参数x和y是可选的，其默认值为0 rgb参数是可选的，其值为0或1 |
| *********** | 参数的描述                                                   |
| x           | 从左上角的水平坐标，默认值为0                                |
| y           | 从左上角的垂直坐标，默认值为0                                |
| rgb         | rgb = 0…输入的颜色空间不改变，默认值 rgb = 1…输入的颜色空间设置为RGB |
| *********** | 变量，可以用在x和y的表达式中                                 |
| W           | 主要输入宽度                                                 |
| H           | 主要输入高度                                                 |
| w           | overlay输入宽度                                              |
| h           | overlay输入高度。                                            |

视频覆盖命令的结构如下，input1是视频背景，input2是前景:

```cmd
ffmpeg -i input1 -i input2 -filter_complex overlay=x:y output
```

请注意，不是使用-vf选项，而是使用-filter_complex选项，因为现在有两个输入源(通常是视频文件或图像)。

### 一个角落的logo

```cmd
//左上角
ffmpeg -i pair.mp4 -i logo.png -filter_complex overlay pair1.mp4
//右上角
ffmpeg -i pair.mp4 -i logo.png -filter_complex overlay=W-w pair2.mp4
```

### Logo显示在指定的时刻

在某些情况下，例如当视频包含一个特别的介绍时，可以在一个时间间隔后加上一个-itsoffset选项来添加标识(或其他源到覆盖)。例如，在开始的5秒后，在蓝色背景上添加一个红色标志，我们可以使用以下命令:

```cmd
ffmpeg -i video_with_timer.mp4 -itsoffset 5 -i logo.png ^ -filter_complex overlay timer_with_logo.mp4
```

## 添加文字 drawtext

### 介绍

如何将一些文本添加到视频输出中的两种常用方法是使用字幕或叠加技术(overlay)。 具有许多可能性的最高级选项是使用表中描述的抽象滤镜：

| 描述                   | 从文本文件或字符串在视频中添加文本，并使用各种参数进行修改。 文本从文本文件参数指定的文件中加载，或直接使用文本参数输入。 其他必需参数是指定选定字体的字体文件。 文本位置由x和y参数设置。 |
| ---------------------- | ------------------------------------------------------------ |
| Syntax                 | drawtext=fontfile=font_f:text=text1[:p3=v3[:p4=v4[...]]] p3，p4 ...表示参数＃3，参数＃4等 |
| **********             | 参数的描述                                                   |
| box                    | 如果box=1，在文本周围绘制一个方框，颜色由boxcolor参数设置，默认值为0 |
| boxcolor               | 颜色为box参数，颜色名称或0xRRGGBB[AA]格式(详见第1章的颜色名称)，默认值为白色 |
| draw                   | 表达式指定如果表达式求值为0时，是否应该绘制文本，则不绘制文本，默认为“1”。它用于指定只在特定条件下绘制文本。接受的变量和函数将在下一页和本章的内置数学函数中描述 |
| fix_bounds             | 如果是true，文本坐标是固定的，以避免剪切                     |
| fontcolor              | 用于绘制字体、颜色名称或0xRRGGBB[AA]格式的颜色，默认为黑色   |
| fontfile               | 字体文件用于绘制文本的正确路径，强制参数                     |
| fontsize               | 要绘制的文本字体大小，默认值为16                             |
| ft_load_flags          | 用于加载字体的标志，默认值是“render”;更多信息在FT_LOAD_* libfreetype标志的文档中 |
| shadowcolor            | 在绘制的文本、颜色名称或0xRRGGBB[AA]格式后面绘制阴影的颜色，可能后面跟着一个alpha说明符，默认值是黑色 |
| shadowx, shadowy       | x和y抵消了文本阴影位置对文本位置的影响，它们可以是正的，也可以是负值，两者的默认值是“0” |
| tabsize                | 用于呈现选项卡的空间大小，默认值为4                          |
| timecode               | hh:mm:ss[:;]ff格式，可以使用或不使用文本参数，但必须指定timecode_rate参数 |
| timecode_rate, rate, r | timecode帧率(仅限时间)                                       |
| text                   | 要绘制的文本字符串，必须是UTF-8编码的字符序列，如果没有指定textfile参数，该参数是必需的 |
| textfile               | 文本文件与要绘制的文本，文本必须是一个UTF-8编码字符序列;如果不使用文本参数，则该参数是强制性的;如果指定了文本和文本文件参数，则显示一条错误消息 |
| x, y                   | x和y值是表示文本将在视频帧中绘制的偏移量的表达式;它们相对于左上角，而x和y的默认值为“0”;下面描述了接受的变量和函数 |
| ***********            | 接受变量和函数表达式中的x和y参数                             |
| dar                    | 输入显示纵横比，与(w / h) * sar相同                          |
| hsub, vsub             | 水平和垂直的色度子样本值。例如，像素格式的“yuv422p”hsub是2，而vsub是1 |
| line_h, lh             | 每个文本行的高度                                             |
| main_h, h, H           | 输入的高度                                                   |
| main_w, w, W           | 输入的宽度                                                   |
| max_glyph_a, ascent    | 从基线到最高/上格坐标的最大距离，用于放置一个字形轮廓点，用于所有呈现的字形;一个正值，由于网格 |
| max_glyph_d, descent   | 从基线到最低网格坐标的最大距离，用于放置一个字形轮廓点，用于所有呈现的字形;一个负值，由于网格 |
| max_glyph_h            | 最大字形高度，即所呈现文本中所包含的所有字形的最大高度，相当于上升下降 |
| max_glyph_w            | 最大的字形宽度，这是在呈现的文本中所包含的所有字形的最大宽度 |
| n                      | 输入框的数目，从0开始                                        |
| rand(min, max)         | 返回最小值和最大值之间的随机数                               |
| sar                    | 输入样本比例                                                 |
| t                      | 时间戳以秒表示，如果输入时间戳未知                           |
| text_h or th           | 呈现文本的高度                                               |
| text_w or tw           | 渲染文本的宽度                                               |
| x, y                   | x和y坐标，在这里文本被绘制，这些参数允许x和y表达式相互引用，所以你可以指定y=x/dar |

例如，要在白色背景上使用黑色字体的Arial字体绘制一个受欢迎的消息(默认位于左上角)，我们可以使用该命令(字符在一行上键入):

```
ffplay -f lavfi -i color=c=white ^ -vf drawtext=fontfile=/Windows/Fonts/arial.ttf:text=Welcome
```

### 动态文本

要将文本横向移动到视频帧中，我们将t变量包括到x参数的表达式中，例如，在右向左方向以n个像素每一秒移动提供的文本，我们使用x=w-t*n的表达式。为了将运动改变为从左到右的方向，使用x=w+t*n的表达式。例如，要显示“动态RTL文本”字符串在顶部移动，我们使用命令。

```
ffmpeg -f lavfi -i color=c=#abcdef -vf drawtext=^ "fontfile=arial.ttf:text='Dynamic RTL text':x=w-t*50
```

## 格式之间转换

ffmpeg工具的最常见用法是从一种音频或视频格式转换为另一种相关的格式。

### 介绍

媒体格式是能够存储音频或视频数据的特殊文件类型。 其中一些能够存储更多类型的数据与多个流，这些被称为容器。并可以使用命令`ffmpeg -formats`进行显示。

媒体容器是特定类型的包装文件，用于存储多媒体流和相关元数据的特殊文件格式。由于音频和视频可以通过各种方法（算法）进行编码和解码，容器提供了将各种媒体流存储在一个文件中的简单方法。 一些容器只能存储音频（AIFF，WAV，XMF等），一些只能存储图片（TIFF ...），但大多数容器存储音频，视频，字幕，元数据等。

如果只更改容器并保留编解码器，我们可以使用`-c copy`

```
ffmpeg -i input.avi -q 1 -c copy output.mov
```

### 转码和转换

将输入文件使用ffmpeg处理成输出文件称为转换，它可以包括格式之间的转换或者仅修改某些数据，输出媒体格式保持不变的转码。 数据包可以被编码压缩或解压缩，压缩包括使用特定的编解码器。 转码过程可以分为几个部分：

- 解复用（demultiplexing） - 基于文件扩展名（.avi，mpg等）被选中来自libavformat库的最好的解复用（解复用器），从输入文件生成编码数据包

- 解码-数据包是由一个适当的解码器解码，产生未压缩的帧;如果使用`-c copy`(或`-codec copy`)选项，则不会发生解码(也不进行过滤)。
- 可选的过滤器 - 解码的帧可以通过指定的过滤器进行修改
- 编码 - 未压缩的帧由选定的编码器编码为数据包
- 复用（multiplexing） - 将数据包复用（multiplexed）为选定的媒体格式。

ffmpeg中转换的可用选项被划分为通用的和私有的。可以为任何容器、编解码器或设备设置通用选项，私有选项针对所选的编解码器、容器或设备。

### 编解码器介绍

codec的名字来源于单词编码解码器(或编码解码器)，它表示一个设备或软件工具，用于编码和解码一个被压缩的视频或音频流。FFmpeg编解码器定义是一种媒体比特流格式。下一个命令显示可用的编解码器:

```cmd
ffmpeg -codecs //显示编解码器
ffmpeg -decoders //只显示解码器
ffmpeg -encoders //只显示编码器
```

可以为输入和输出文件指定编解码器，如果输出包含多个流，则每个流可以使用不同的编解码器。 如果我们在没有编解码器的情况下指定输出格式，则ffmpeg会选择默认编解码器，常见媒体格式的默认编解码器列表如下：

| 格式  | 编解码器   | 其他数据                                                     |
| ----- | ---------- | ------------------------------------------------------------ |
| .avi  | mpeg4      | mpeg4 (Simple profile), yuv420p; audio: mp3                  |
| .flv  | flv1       | yuv420p; audio: mp3                                          |
| .mkv  | h264       | h264 (High), yuvj420p; audio: vorbis codec, fltp sample format |
| .mov  | h264       | h264 (High), yuvj420p; audio: aac (mp4a)                     |
| .mp4  | h264       | h264 (High), yuvj420p; audio: aac (mp4a)                     |
| .mpg  | mpeg1video | yuv420p; audio: mp2                                          |
| .ogg  | theora     | yuv422p, bitrate very low; audio excluded during conversion  |
| .ts   | mpeg2video | yuv422p; audio: mp2                                          |
| .webm | vp8        | yuv420p; audio: vorbis codec, fltp sample format             |

| 格式  | 编解码器  | 额外数据                                           |
| ----- | --------- | -------------------------------------------------- |
| .aac  | aac       | libvo_aacenc, bitrate 128 kb/s                     |
| .flac | flac      | FLAC (Free Lossless Audio Codec), bitrate 128 kb/s |
| .m4a  | aac       | mp4a, bitrate 128 kb/s                             |
| .mp2  | mp2       | MPEG Audio Layer 2, bitrate 128 kb/s               |
| .mp3  | mp3       | libmp3lame, bitrate 128 kb/s                       |
| .wav  | pcm_s16le | PCM (Pulse Code Modulation), uncompressed          |
| .wma  | wmav2     | Windows Media Audio                                |

## 时间操作

多媒体处理包括改变输入持续时间，设置延迟，仅从输入中选择特定部分等。这些时间操作接受2种格式的时间规格：

- `[-]HH:MM:SS[.m...]`
- `[-]S+[.m...]`
  `HH`是小时数，`MM`是分钟数，`SS`或`S`是秒数，`m`是毫秒数。

### 音频和视频的持续时间 

#### - t选项设置

要设置媒体文件的持续时间，我们可以使用`-t`选项，其值是以秒为单位的时间或格式为`HH：MM：SS.milliseconds`的时间。 例如，要为music.mp3文件设置3分钟的持续时间，我们可以使用该命令

```cmd
ffmpeg -i music.mp3 -t 180 music_3_minutes.mp3
```

#### 通过帧数设置

在某些情况下，通过指定具有可用选项的帧数来设置录制的持续时间可能很有用：

- 音频: `-aframes number`  或者 `-frames:a number` 
- 数据: `-dframes number` 或者 `-frames:d number` 
- 视频: `-vframes number` 或者 `-frames:v number`

```cmd
  ffmpeg -i video.avi -vframes 15000 video_10_minutes.avi
```

### 从开始设置延迟

要从指定时间开始记录输入，我们可以使用`-ss`（从开始搜索）选项，其值是以秒或`HH：MM：SS.milliseconds`格式表示的时间。 该选项既可以在输入文件和输出文件之前使用，也可以在输出文件之前使用，编码更精确。 例如，要从第10秒开始转换，我们可以使用以下命令：

```cmd
ffmpeg -i input.avi -ss 10 output.mp4
```

### 从媒体文件中提取特定部分

要从音频或视频文件中剪辑特定部分，我们同时使用`-ss`和`-t`选项，ffplay在左下角显示当前时间，可以使用空格键或P键暂停/开启播放。 例如，要从文件video.mpg保存第5分钟（4x60 = 240秒），我们可以使用以下命令：

```cmd
ffmpeg -i video.mpg -ss 240 -t 60 clip_5th_minute.mpg
```

## 元数据和字幕

媒体文件中的元数据包含艺术家，作者，日期，流派，发布者，标题等附加信息，并且不会显示在视频帧中。 字幕是文本数据，通常包含在单独的文件中，并显示在视频帧底部附近，尽管一些容器文件格式（如VOB）支持包含字幕文件。

### 元数据介绍

元数据通常用于MP3文件，媒体播放器通常在其中显示诸如歌曲标题，艺术家，专辑等的项目

要显示位于Windows 7的Sample Music文件夹中的文件Kalimba.mp3的元数据

```
ffplay -i "/Users/Public/Music/Sample Music/Kalimba.mp3"
```

### 创建元数据

元数据被包含在带有-元数据选项的媒体文件中，后跟一个键=值对，其中的键或值必须是双引号，如果包含空格。当需要输入更多的密钥时，可以使用几个元数据选项，例如:

```cmd
ffmpeg -i input -metadata artist=FFmpeg -metadata title="Test 1" output
```

### 保存和加载文件的元数据

为了保存媒体文件中包含的元数据，我们可以使用-f选项指定的ffmetadata格式，在该文本文件的名称之前存储元数据。例如，从视频中保存元数据。在前面的示例中创建的wmv文件，我们可以使用该命令。

```
ffmpeg -i video.wmv -f ffmetadata data.txt
```

### 删除元数据

要删除不是实际的元数据，我们可以使用设置为负值的`-map_metadata`选项，例如从文件`input.avi`中删除所有元数据，我们可以使用以下命令：

```cmd
ffmpeg -i input.avi -map_metadata -1 output.mp4
```

### 字幕的介绍

字幕是包含在视频帧底部附近的文本数据，用于提供附加信息，如将口语外语翻译为本地语言，提高识字率的相同语言字幕等。字幕可以分为两种主要类型：

- 外部媒体播放器在播放期间包含在独立文件中并且包含在视频帧中的优点是可以在没有视频的情况下进行编辑和分发
- 内部的，包含在具有视频和音频流的媒体文件容器中

其他部分包括在实况视频广播期间同时创建的准备好的字幕和实况字幕。 其他排序将字幕分为打开和关闭 - 打开或关闭字幕和字幕等关闭字幕时，不能关闭打开的字幕。

支持的字幕编解码器和文件格式列表位于表格中，支持列D表示此格式可以解码，E表示编码的可用性（dvb_teletext和eia_608尚未指定）。 例如，要将SRT格式的字幕转换为ASS格式，我们可以使用以下命令：

```cmd
ffmpeg -i subtitles.srt subtitles.ass
```

### 直接编码到视频的字幕

例如，如果我们想要将一个字幕视频包含到网页中，我们需要将字幕编码到视频流中，2个过滤器可以做到:ass(只编码ass格式)和在表中描述的字幕过滤器:
 视频过滤器:字幕

| 描述          | 包括使用libass库的输入视频的字幕   |
| ------------- | ---------------------------------- |
| 语法          | subtitles=filename[:original_size] |
|               | 描述的选项                         |
| f, filename   | 包含字幕的文件的名称               |
| original_size | 原始视频的大小，当输入被调整时需要 |

请注意，并非所有的字幕格式都由所有的容器支持，大多数容器(AVI, Matroska, MP4, MPG，等等)支持ASS和SRT。

```cmd
ffmpeg -i video.avi -vf subtitles=titles.srt video.mp4
```

## 数字音频

“数字音频”一词与“数字视频”一词相比，它是一种处理和显示移动图像的技术，而音频则与声音有关。数字音频是一种技术，用于捕获、记录、编辑、编码和复制声音，这些声音通常由脉冲编码调制(PCM)进行编码。FFmpeg支持许多音频格式，包括AAC、MP3、Vorbis、WAV、WMA等。

### 音频量化和采样

由于人类听觉系统的生理限制，压力波的连续值可以用有限的一系列值代替，这些值可以作为数字存储在计算机文件中。 计算机使用二进制数字，所以常见的音频位深度（音频分辨率）是两个幂：

| 位深度 | 值计算              | 描述                                          |
| ------ | ------------------- | --------------------------------------------- |
| 8 bit  | 2^8 =256            | 用于电话，旧设备                              |
| 12 bit | 2^12 =4,096         | DV(数字视频)的标准，用于数码相机等            |
| 14 bit | 2^14 =16,384        | 用于NICAM压缩，电视立体声，等等               |
| 16 bit | 2^16 =65,536        | 标准音频CD和DAT(数字音频磁带)，是当今最常见的 |
| 20 bit | 2^20 =1,048,576     | 附加标准的超级音频CD和DVD音频                 |
| 24 bit | 2^24 =16,777,216    | 标准的超级音频CD和DVD音频                     |
| 32 bit | 2^32 =4,294,967,296 | 专业设备,蓝光技术                             |

### 音频文件格式

量化和采样音频被保存在不同的媒体文件格式，下一个表描述特定的文件格式，仅用于音频(MP3格式支持也包括图像):

| 未压缩的                  | 无损压缩   | 有损压缩     |
| ------------------------- | ---------- | ------------ |
| ALAC                      | AIFF (PCM) | AAC          |
| AU                        | ALS        | AC-3         |
| BWF                       | ATRAC      | AMR          |
| PCM (raw, without header) | FLAC       | MP2, MP3     |
| WAV (PCM)                 | WavPack    | Musepack     |
|                           | WMA        | Speex        |
|                           |            | Vorbis (OGG) |

