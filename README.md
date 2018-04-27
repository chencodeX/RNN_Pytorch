# RNN_Pytorch
使用pytorch框架搭建一个encoder-forecaster结构的图像时序预测模型，ConvGRU内核

##文件说明

    BMSELoss.py 一个对不同降雨级别进行加权损失的loss函数。
    ConvGRUCell.py 利用卷积操作实现的ConvGRU内核，单Cell
    ConvLSTM.py 利用卷积操作实现的ConvLSTM内核，包含了单Cell，多层Cell的实现方式，以及一个实验用的RNN模型
    encoder.py 序列编码结构
    forecaster.py 序列预测结构
    HKO_EF.py 尝试一个试验性质的训练方式
    HKO_model.py HKO-7模型的搭建和训练
    RNN.py 一般的RNN模型
    RNN_train.py RNN模型的训练

##模型说明 

    本项目参考的HKO模型为论文  [Deep learning for precipitation nowcasting: A benchmark and a new model](http://papers.nips.cc/paper/7145-deep-learning-for-precipitation-nowcasting-a-benchmark-and-a-new-model)
    encoder和forecaster 结构皆参照此论文编写
    本文的数据样本原始大为20*1*477*477，训练过程中会缩放或随机切割到120*120大小。
    原始数据为0-70之间的DBz值，在输入网络前已经映射到（0-255）/255.所以代码中的BMSELoss的阈值也经过了调整。