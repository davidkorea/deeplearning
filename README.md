skipgram

# deeplearning

> **Kaggle kernel has GPU, Paddlepaddle kernel doe not!!** 

# Issue 1 GPU

1、在构造tf.Session()时候通过传递tf.GPUOptions作为可选配置参数的一部分来显式地指定需要分配的显存比例，如下所示：

    # 假如有12GB的显存并使用其中的4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


per_process_gpu_memory_fraction指定了每个GPU进程中使用显存的上限，但它只能均匀作用于所有GPU，无法对不同GPU设置不同的上限

2、尝试如下设置：

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

当allow_growth设置为True时，分配器将不会指定所有的GPU内存，而是根据需求增长

3、在执行训练脚本前使用：

    export CUDA_VISIBLE_DEVICES=1

# Issue 2 Glorot Initialization(glorot_uniform, glorot_normal)

Dense(), Conv2d() etc. all have its initial weights, and could be diviided into 2 categories.
1. RandomNormal 正态分布初始化

```keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))```

        mean：均值
        stddev：标准差
        seed：随机数种子
        
2. RandomUniform 均匀分布初始化

```keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)```
        
        minval：均匀分布下边界 
        maxval：均匀分布上边界
        seed：随机数种子
        
Reference: [Keras_网络配置 » 初始化方法Initializers](http://keras-cn.readthedocs.io/en/latest/other/initializations/)

# Issue 3 CNN Visualization

Reference: [Fashion MNIST - CNN with Keras](https://www.kaggle.com/bugraokcu/cnn-with-keras)

# Issue 4 - Numpy.nonzero()

Reference: [Numpy.nonzero() 详解 numpy module中 nonzero（）函数](https://blog.csdn.net/roler_/article/details/42395393)
```
>>> x = np.array([[1,0,0], [0,2,0], [1,1,0]])
>>> x
array([[1, 0, 0],
       [0, 2, 0],
       [1, 1, 0]])
>>> np.nonzero(x)
(array([0, 1, 2, 2]), array([0, 1, 0, 1])) # non zero row index array, no zero column index array
```

```
>>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> a > 3
array([[False, False, False],
       [ True,  True,  True],
       [ True,  True,  True]])
>>> np.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
```
