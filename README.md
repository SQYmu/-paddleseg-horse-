# -paddleseg-horse-基于PaddleSeg套件进行马的分割

对于已经分割好的数据集进行训练，实现图片中马与背景的分离

# 一、项目背景

对百度提供的分割数据集进行训练以较好的完成分割任务，熟悉使用百度提供的paddle框架与其开发套件。具体的效果如下：

![](https://ai-studio-static-online.cdn.bcebos.com/9e3ab2ecdd8e4b55b3271e7cdee618d316a684e2bb8843aca85e3a8327382363)

# 二、数据集简介

百度提供的已经分割好的关于马的图片和标记共328张图片

## 1.数据加载和预处理


```python
# 数据的加载和预处理
进行label图片的二值化
!python parse_horse_label.py

# 训练数据集
!python PaddleSeg/train.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--batch_size 4\
--iters 2000\
--learning_rate 0.01\
--save_interval 200\
--save_dir PaddleSeg/output\
--seed 2021\
--log_iters 20\
--do_eval\
--use_vdl

# 评估数据集
!python PaddleSeg/val.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams
```

训练集样本量: 328，验证集样本量: 328


## 2.数据集查看


```python
查看数据集的格式
!tree segDataset -L 2
├── horse
│   ├── Annotations
│   └── Images
其中annotations是已经标记好的标记文件，images是原图片

```


# 三、模型选择和开发

本项目直接使用了较为简单和轻量化的模型bisenet。bisenet是旷视科技提出的一种双向深度学习网络模型，具体的结构如下：

## 1.模型组网

![](https://ai-studio-static-online.cdn.bcebos.com/be16167ce67e402783220f7e23bfbe0940376d512e8d4654bd042d9ef289085b)


Spatial Path：Spatial Path 包含三层，每层包含一个步幅（stride）为 2 的卷积，随后是批归一化和 ReLU。因此，该路网络提取相当于原图像 1/8 的输出特征图。由于它利用了较大尺度的特征图，所以可以编码比较丰富的空间信息。

Context Path：先使用Xception快速下采样，尾部接一个全局pooling，然后类似u型结构容和特征，只上采样两次，这里用到ARM（即注意力优化模块），用于refine特征。之后上采样（双线性插值）到spatial path分路特征相同大小。

最后在 Spatial Path 和 Context Path 的基础上提出 BiSeNet，以实现实时语义分割，如图 2(a) 所示。本文把预训练的 Xception 作为 Context Path 的 backbone，把带有步幅的三个卷积层作为 Spatial Path；接着融合这两个组件的输出特征以做出最后预测，它可以同时实现实时性能与高精度。

## 2.模型训练


```python
# 配置优化器、损失函数、评估指标

#优化器使用sgd模型
optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5
  
#learning rate 设置在0.01，追求更高的精度可以将lr设置的低一些，同时随着训练次数的增加lr逐渐减少以获得最佳效果
lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  end_lr: 0
  power: 0.9

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1, 1, 1, 1, 1]

#不使用预训练模型
model:
  type: BiSeNetV2
  pretrained: Null
              
# 设置训练轮数
batch_size: 4 #一次训练四张图片
iters: 1000 #共训练1000次
```

    2021-08-13 11:20:32 [INFO]	[TRAIN] epoch: 1, iter: 20/2000, loss: 3.0412, lr: 0.009914, batch_cost: 0.1093, reader_cost: 0.01300, ips: 36.6094 samples/sec | ETA 00:03:36
    2021-08-13 11:20:34 [INFO]	[TRAIN] epoch: 1, iter: 40/2000, loss: 2.4766, lr: 0.009824, batch_cost: 0.0941, reader_cost: 0.00007, ips: 42.4983 samples/sec | ETA 00:03:04
    2021-08-13 11:20:36 [INFO]	[TRAIN] epoch: 1, iter: 60/2000, loss: 2.1365, lr: 0.009734, batch_cost: 0.0918, reader_cost: 0.00008, ips: 43.5563 samples/sec | ETA 00:02:58
    2021-08-13 11:20:38 [INFO]	[TRAIN] epoch: 1, iter: 80/2000, loss: 2.3350, lr: 0.009644, batch_cost: 0.0921, reader_cost: 0.00008, ips: 43.4516 samples/sec | ETA 00:02:56
    2021-08-13 11:20:40 [INFO]	[TRAIN] epoch: 2, iter: 100/2000, loss: 2.0317, lr: 0.009553, batch_cost: 0.0961, reader_cost: 0.00369, ips: 41.6407 samples/sec | ETA 00:03:02
    2021-08-13 11:20:42 [INFO]	[TRAIN] epoch: 2, iter: 120/2000, loss: 1.9597, lr: 0.009463, batch_cost: 0.0945, reader_cost: 0.00007, ips: 42.3217 samples/sec | ETA 00:02:57
    2021-08-13 11:20:43 [INFO]	[TRAIN] epoch: 2, iter: 140/2000, loss: 1.9451, lr: 0.009372, batch_cost: 0.0924, reader_cost: 0.00025, ips: 43.3077 samples/sec | ETA 00:02:51
    2021-08-13 11:20:45 [INFO]	[TRAIN] epoch: 2, iter: 160/2000, loss: 1.9334, lr: 0.009282, batch_cost: 0.0922, reader_cost: 0.00007, ips: 43.3627 samples/sec | ETA 00:02:49
    2021-08-13 11:20:47 [INFO]	[TRAIN] epoch: 3, iter: 180/2000, loss: 1.8718, lr: 0.009191, batch_cost: 0.0964, reader_cost: 0.00380, ips: 41.4850 samples/sec | ETA 00:02:55
    2021-08-13 11:20:49 [INFO]	[TRAIN] epoch: 3, iter: 200/2000, loss: 1.8275, lr: 0.009100, batch_cost: 0.0935, reader_cost: 0.00009, ips: 42.7880 samples/sec | ETA 00:02:48      
    2021-08-13 11:20:49 [INFO]	Start evaluating (total_samples: 328, total_iters: 328)...
    328/328 [==============================] - 8s 24ms/step - batch_cost: 0.0244 - reader cost: 1.3448e-0
    2021-08-13 11:20:57 [INFO]	[EVAL] #Images: 328 mIoU: 0.6850 Acc: 0.8342 Kappa: 0.6238 
    2021-08-13 11:20:57 [INFO]	[EVAL] Class IoU: 
    [0.7831 0.587 ]
    2021-08-13 11:20:57 [INFO]	[EVAL] Class Acc: 
    [0.9593 0.6266]
    2021-08-13 11:20:57 [INFO]	[EVAL] The model with the best validation mIoU (0.6850) was saved at iter 200.
    


## 3.模型评估测试


```python
# 进行单独模型评估。
!python PaddleSeg/val.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams #模型路径
```

    2021-08-13 11:33:30 [INFO]	Loading pretrained model from PaddleSeg/output/best_model/model.pdparams
    2021-08-13 11:33:30 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-13 11:33:30 [INFO]	Loaded trained params of model successfully
    2021-08-13 11:33:30 [INFO]	Start evaluating (total_samples: 328, total_iters: 328)...
    328/328 [==============================] - 8s 24ms/step - batch_cost: 0.0237 - reader cost: 5.7794e-
    2021-08-13 11:33:38 [INFO]	[EVAL] #Images: 328 mIoU: 0.8621 Acc: 0.9425 Kappa: 0.8494 
    2021-08-13 11:33:38 [INFO]	[EVAL] Class IoU: 
    [0.9254 0.7988]
    2021-08-13 11:33:38 [INFO]	[EVAL] Class Acc: 
    [0.9564 0.9015]


## 4.模型预测

### 批量预测

使用predict.py来完成对大量数据集的批量预测。

```python
# 进行预测操作
!python PaddleSeg/predict.py\
--config PaddleSeg/configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
--model_path PaddleSeg/output/best_model/model.pdparams\
--image_path segDataset/horse/Images\ #可以选择文件夹或者单张图片
--save_dir PaddleSeg/output/horse
```

    2021-08-13 11:32:53 [INFO]	Number of predict images = 328
    2021-08-13 11:32:53 [INFO]	Loading pretrained model from PaddleSeg/output/best_model/model.pdparams
    2021-08-13 11:32:53 [INFO]	There are 356/356 variables loaded into BiSeNetV2.
    2021-08-13 11:32:53 [INFO]	Start to predict...
    328/328 [==============================] - 16s 49ms/ste


# 四、效果展示

将准备好的数据集用bisenet模型进行训练之后，可以预测到图片中马的位置并能生成mask，即掩膜文件：

![](https://ai-studio-static-online.cdn.bcebos.com/ea376b33c3f34092b5ff2117596ac2a7b3a8fd39c6e44e0d8385c844c2834219)

![](https://ai-studio-static-online.cdn.bcebos.com/5acb3d49c0de4d97a9f67706fc3873b5d2eeb60c76a341a6b31f356d95dad9a1)

# 五、总结与升华

1. 数据集文件的命名：数据集文件的命名不能含有空格，否则在训练时不能读取。解决方法是写bat文件批量去空格，或者修改配置文件（这种方法会出bug，但是理论上可行）
2. 数据集的路径：默认是以paddleseg的路径为根路径，否则路径不能正常读取
3. lr的设置：不能设置的过大，否则loss会出现NAN
4. class_num的设置：需要考虑背景，即要识别的物体种类+1，否则会出现错误
5. resize的设置：要在速度与精度之间寻求平衡

项目的难度很低，但是这是一个学习的过程，我在这之中受益匪浅。

# 个人简介

SQY木：

<https://aistudio.baidu.com/aistudio/personalcenter/thirdview/834686>
