



# 人脸识别与疲劳检测

# 简介

1、技术栈：Python + TensorFlow + OpenCV 

2、基本功能需求：人脸检测 + 人脸验证  ==》人脸考勤管理 

3、扩展功能需求：表情检测 + 语音提醒  ==》疲劳驾驶提醒

# 使用流程

先使用`record.py`进行人脸捕获，再使用`facecnn.py`对CNN进行训练，最后使用`face.py`与`face_find.py`进行人脸识别。



`face.py`主要的人脸识别程序，识别人脸并输出预测值并进行打卡。

`face_find.py`是驾驶员疲劳检测功能，通过眼睛的EAR值判断眼睛的状态，闭眼时间超过一定的次数即进行语音提示。

`facecnn.py`用于训练识别人脸所需的CNN

`record.py`用于人脸捕获，输入名字后，将会开启摄像头自动进行人脸捕获。

# 主要功能

`face.py`:通过OpenCV捕获摄像头视频流，在检测到人脸后，将人脸图像保存到指定路径，同时在图像上绘制文本表示保存状态，最终实现了简单的人脸识别功能。

`face_find.py`:利用OpenCV捕获摄像头视频流，在检测到人脸后计算眼睛的EAR值，根据阈值判断眼睛的状态，并在图像上绘制相关信息，最终实现了简单的眨眼次数统计和提示功能。

`facecnn.py`:

这段代码实现了一个基于卷积神经网络（CNN）的人脸识别模型。主要分为数据处理和模型训练两大部分。

1. **数据处理(`Data`类)**：

   - 初始化时指定图片大小和图片路径。
   - `process_img`方法用于图片预处理，包括调整大小、灰度转换和边缘填充。
   - `read_path`方法遍历指定路径下的所有图片，对每张图片进行处理，并收集标签。
   - `one_hot`方法对标签进行独热编码处理。
   - `split`方法将处理后的数据集分为训练集和测试集，并进行归一化和形状调整以适配CNN输入。

2. **CNN模型(`CNN`类)**：

   - 使用`tensorflow.keras`构建序贯模型，包括两个卷积层、两个最大池化层、一个全连接层和一个输出层，使用ReLU激活函数和Softmax输出。
   - `train`方法用于编译模型（使用Adam优化器和交叉熵损失函数），训练模型，并在测试集上评估模型性能。

3. **主程序**：

   - 实例化`Data`类和`CNN`类，加载数据，然后使用加载的数据训练CNN模型，并评估其性能。

   `record.py`:

   这段代码是一个基于OpenCV的人脸捕获程序。主要功能包括：

   1. 导入所需的库：`cv2` 用于OpenCV操作，`numpy` 用于数值计算，`PIL` 用于图像处理，`os` 用于文件操作。
   2. 定义了一个名为 `CatchFromVideo` 的函数，用于从视频流中捕获人脸。该函数接受姓名、保存路径、最大捕获数量、摄像头索引和分类器路径等参数。
   3. 在 `CatchFromVideo` 函数中：
      - 初始化摄像头和加载人脸分类器。
      - 循环处理视频流中的帧。
      - 使用分类器检测每一帧中的人脸。
      - 如果检测到人脸，则保存人脸图像文件，并更新计数器。
      - 在视频流上显示检测到的人脸，并在人脸周围绘制边界框。
      - 当捕获的人脸数量达到指定的最大值或用户关闭窗口时停止。
   4. 定义了一个辅助函数 `Draw_I`，用于在图像上绘制文本。该函数在需要时在OpenCV图像和PIL图像之间进行转换。
   5. 在主函数部分：
      - 提示用户输入姓名。
      - 根据输入的姓名创建一个目录以保存人脸图像。
      - 调用 `CatchFromVideo` 函数开始捕获人脸。

   