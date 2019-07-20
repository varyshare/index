> 特征点提取与匹配在计算机视觉中是一个很重要的环节。比如人脸识别，目标跟踪，三维重建，等等都是先提取特征点然后匹配特征点最后执行后面的算法。因此学习特定点提取和匹配是计算机视觉中的基础。本文将介绍FAST特征点提取与匹配算法的原理，并使用Python不调用OpenCV包实现FAST特征点提取算法。

# 特征点提取到底是提取的是什么？
答：**首先，提取的是角点，边缘**。提取角点可以进行跟踪，提取边就可以知道图像发生了怎样的旋转。反正都是提取的是那些周围发生颜色明显变化的那些地方。这个也很容易想通，要是它周围全一样的颜色那肯定是物体的内部，一来没必要跟踪。二来它发生了移动计算机也无法判断，因为它周围都一样颜色计算机咋知道有没有变化。**其次，提取的是周围信息（学术上叫做：描述子）**。我们**只要提到特征点提取就一定要想到提取完后我们是需要匹配的**。为了判断这个点有没有移动，我们需要比较前后两帧图片中相同特征点之间是否有位移。为了判断是否是相同特征点那就要进行比对（匹配）。**怎么比较两个特征点是否是同一个**？**这就需要比较这两个特征点周围信息是否一样。周围信息是一样那就认为是同一个特征点**。那么怎么比较周围信息呢？一般会把周围的像素通过一系列计算方式变成一个数字。然后比较这个数字是否相同来判断周围信息是否相同。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718145644785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)

# 所有特征提取与匹配算法通用过程
1. 找到那些周围有明显变化的像素点作为特征点。如下图所示，那些角点和边缘这些地方明显颜色变化的那些像素点被作为特征点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718152825942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)
2. **提取这些特征点周围信息。一般是在当前这个点周围随机采样选几个像素点作为当前特征点的周围信息，或者画个圈圈进行采样**。不同采样方法构成了不同算法。反正你想一个采样方法那你就创建了一种算法。下面是三个出名的算法采样周围像素点的方法。现在你就看看，大概知道是这么个意思就可以，不用太在乎这些复杂的图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718151318564.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)
3. 特征点匹配。比如我要跟踪某个物体，我肯定是要先从这个物体提取一些特征点。然后看下一帧相同特征点的位置在哪，计算机就知道这个物体位置在哪了。怎么匹配？前面提到了我们第2步有提取当前特征点周围信息，只要周围信息一样那就是相同特征点。特征匹配也有很多种算法，最土的是前后两帧图片上的特征点一个一个的比对。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718152119359.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)
**记住学习任何特征提取与匹配算法都要时刻想起上面提到的三步骤。这样你不会太陷入那些书里面的细节中而学了很久都不懂。或者学完就忘。事实上那些算法非常简单，只不过你不知道他们各个步骤之间的联系是什么为什么这么设计。不知道这些当然就看不懂了。**
# FAST特征点提取算法
FAST (Features from Accelerated Segment Test)是一个特征点提取算法的缩写。这是一个点提取算法。它原理非常简单，**遍历所有的像素点，判断当前像素点是不是特征点的唯一标准就是在以当前像素点为圆心以3像素为半径画个圆（圆上有16个点），统计这16个点的像素值与圆心像素值相差比较大的点的个数。超过12个差异度很大的点那就认为圆心那个像素点是一个特征点**。那么什么叫做差异度很大呢？答：就是像素值相减取绝对值，然后我们设置一个数字只要前面那个绝对值大于这个数字，那就认为差异大。比如我**设置阈值是3。第1个像素点的像素值是4，中间圆心像素值是10，然后10-4=6，这是大于阈值3的。所以第1个像素点算所一个差异度较大的像素点**。就这样**统计1~16个中有多少个是和圆心相比差异度比较大的点。只要超过12个那就认为圆心是一个特征点。**是不是很简单？其实这些算法只要你知道他们想干嘛，你也可以设计一个不错的算法的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718160218250.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)
现在稍微有一丝丝难度的是怎么画圆。因为这个圆它是一个像素一个像素的画。这个圆其实你自己可以随便设计一个算法画圆。今天我们要讲FAST算法当然还是介绍下他是怎么画圆的。他就用了最普通的图形学画圆算法（[Bresenham 画圆法](http://en.wikipedia.org/wiki/Midpoint_circle_algorithm) ）。

**其实到这里FAST算法我们就介绍完了。为了节省大家的时间（你的赞和关注是支持我分享的动力）**，我把Bresenham 画圆法也讲讲。
## Bresenham 布雷森汉姆算法画圆的原理与编程实现教程
注意：Bresenham的圆算法只是中点画圆算法的优化版本。区别在于Bresenham的算法只使用整数算术，而中点画圆法仍需要浮点数。所以我先介绍中点画圆法。
### 中点画圆法
看下面这个图，这就是一个像素一个像素的画出来的。我们平常的圆也是一个像素一个像素的画出来的，你可以试试在“画图”这个软件里面画一个圆然后放大很多倍，你会发现就是一些像素堆积起来的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718192037816.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071819203165.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)
我们看出来圆它是一个上下左右都对称，而且也是中心对称的。所以我们只用画好八分之一圆弧就可以，其他地方通过对称复制过去就好。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718192849587.png)
看下面这幅图，绿线夹住的那部分就是八分之一圆弧。**注意我们是逆时针画圆的（即从水平那个地方即(r,0)开始画因为一开始我们只知道水平位置的像素点该放哪其他地方我们都不知道）**。Bresenham 算法画完一个点(x,y)后`注意x,y都是整数。他们代表的是x,y方向上的第几个像素。`，它下一步有两个选择(x,y+1),(x-1,y+1)。也就是说y一定增加,但是x要么保持不变要么减一（你也可以让x一定增加y要么不变要么加一，其实差不多的）。当程序画到粉红色那个像素点的时候，程序选择下一步要绘制的点为(x-1，y+1)。当程序画到黄色的那个像素点时候，程序选择下一步要绘制的点为(x,y+1)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718193719108.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_7F1FFF,t_70)
我们看看粉色的那个点的下一步是如何抉择的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718201205329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)

我们把上面那个图进一步的放大，如下图所示。粉红色点坐标为(x,y)，绿色坐标为（x-1,y+1），紫色坐标为（x,y+1），红色方块坐标为绿色点和紫色点的中点。我们知道中点到两个端点的距离是一样远的，而现在绿色点和紫色点所确定的线段的中点在圆弧外侧。这意味着圆弧离绿色那个点更近（为什么？判断远近的标准是线段与圆弧的交点，即下图灰色那个点与绿色和紫色点的远近）。绿色点和紫色点是粉红色点的下一步待选点。由于绿色的点离圆弧更近，所以确定下一步走到绿色点。回忆下我们是怎么判断远近的？我们是根据中点到底是在圆弧内还是圆弧外进行判断的。而中点坐标是(x-0.5,y+1)。判断是在圆内还是圆外的方法是$p_{k+1}=(x-0.5)^2+(y+1)^2-r^2$是大于0还是小于0，注意这里我们作了一个假设即圆心坐标为（0,0）.$p_{k+1}=(x-0.5)^2+(y+1)^2-r^2$是大于0则中点在圆外，小于0则在圆内。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190718212448628.png)
好了现在中点画圆法已经介绍完了，我们已经可以画出一个完整的圆了。但是由于它计算中点是有小数运算，我们知道浮点数运算是比整数运算满非常多的。于是Bresenham这个人就对中点画圆法进行了改进。
## Bresenham画圆法是怎么改进中点画圆法的呢？
Bresenham也是根据待选的两个点哪个离圆弧近就下一步选哪个。但是它不通过中点到底是在圆内还是圆外。那它是怎么判断的呢？这两个点一定有一个在圆弧内一个在圆弧外。到底选哪个？Bresenham的方法就是直接计算两个点离圆弧之间的距离，然后判断哪个更近就选哪个。如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190719115849411.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZhcnlzaGFyZQ==,size_16,color_FFFFFF,t_70)
那么怎么用数学量化他们离圆弧的距离呢？
答：前面我们提到了，当前粉红色这个点坐标为$(x_k,y_k)$，下一步它有两种可能的走法绿色$(x_k-1,y_k+1)$，紫色坐标为$(x_k,y_k+1)$
$d1 = (x_k-1)^2+(y_k+1)^2-r^2$
$d2 = (x_k)^2+(y_k+1)^2-r^2$
注意：$d1 = (x_k-1)^2+(y_k+1)^2-r^2$小于0的，因为绿色那个点一定在圆内侧。$d2 = (x_k)^2+(y_k+1)^2-r^2$一定是大于等于0的，因为紫色那个点一定在圆外侧。

**所以我们只用比较$P_k = d1+d2$到底是大于0还是小于0就能确定选哪个点了。大于0选绿色$(x_k-1,y_k+1)$那个点（因为紫色那个点偏离圆弧程度更大）。小于0则选紫色$(x_k,y_k+1)$那个点**。

**好了Bresenham画圆法我讲完了**。

你或许会问，不对啊。我在网上看到的关于Bresenham画圆法的博客还有其他公式。确实我还有一个小细节没讲。**你用上面的方法是已经可以画圆了，剩下的就是一些提高计算效率的小细节**。

$P_k = d1+d2= (x_k-1)^2+(y_k+1)^2-r^2+(x_k)^2+(y_k+1)^2-r^2$这个公式走到下一步时候$P_{k+1} = d1+d2$又要重新计算。为了提高效率。人们就想能不能通过递推的方式来算$P_{k+1}$，即能不能找一个这样的公式$P_{k+1}=P_k+Z$提高计算效率。

这个也很简单，这个递推公式关键在于求Z。而我们变换下公式$P_{k+1}=P_k+Z$得到$Z=P_{k+1}-P_k$。
注意：$P_k= d1+d2= (x_k-1)^2+(y_k+1)^2-r^2+(x_k)^2+(y_k+1)^2-r^2$我们已知的，而$P_{k+1}$这个根据$P_k$大于0还是小于0也可以算出来。
1. 当$P_k>=0$则证明靠近外侧的那个待选点$(x_k,y_k+1)$离圆弧更远，所以我们下一步选的点是另外一个靠近内侧圆弧的那个点$(x_k-1,y_k+1)$。也就是说第k+1步那个点$(x_{k+1},y_{k+1})=(x_k-1,y_k+1)$。
$Z=P_{k+1}-P_k= (x_{k+1}-1)^2+(y_{k+1}+1)^2-r^2+(x_{k+1})^2+(y_{k+1}+1)^2-r^2 
-[ (x_k-1)^2+(y_k+1)^2-r^2+(x_k)^2+(y_k+1)^2-r^2] \\
= (x_k-1-1)^2+(y_k+1+1)^2-r^2+(x_k-1)^2+(y_k+1+1)^2-r^2 
-[ (x_k-1)^2+(y_k+1)^2-r^2+(x_k)^2+(y_k+1)^2-r^2]\\
=-4x_k+4y_k+10$。所以$P_{k+1}=P_k-4x_k+4y_k+10$
2. 当$P_k<0$时，们下一步选的点是另外一个靠近内侧圆弧的那个点是$(x_k,y_k+1)$。也就是说第k+1步那个点$(x_{k+1},y_{k+1})=(x_k,y_k+1)$。我们看看现在的Z是多少。
$Z=P_{k+1}-P_k= (x_{k+1}-1)^2+(y_{k+1}+1)^2-r^2+(x_{k+1})^2+(y_{k+1}+1)^2-r^2 
-[ (x_k-1)^2+(y_k+1)^2-r^2+(x_k)^2+(y_k+1)^2-r^2] \\
= (x_k-1)^2+(y_k+1+1)^2-r^2+(x_k)^2+(y_k+1+1)^2-r^2 
-[ (x_k-1)^2+(y_k+1)^2-r^2+(x_k)^2+(y_k+1)^2-r^2]\\
=4y_k+6$。所以$P_{k+1}=P_k+4y_k+6$

现在是真的完全讲完了。
使用OpenCV库中的FAST特征点检测算法
```python
import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('simple.jpg',0)
fast = cv2.FastFeatureDetector()
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))
cv2.imwrite('fast_true.png',img2)
```

参考文献：
[1] https://medium.com/software-incubator/introduction-to-feature-detection-and-matching-65e27179885d
[2] https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html#fast
[3] https://medium.com/software-incubator/introduction-to-fast-features-from-accelerated-segment-test-4ed33dde6d65
[4] https://www.youtube.com/watch?v=1Te8U_JR8SI
