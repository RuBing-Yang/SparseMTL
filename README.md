# SparseMTL

多算子稀疏格式预测器:

- 为每个算子任务设置单独的输出层，不同算子之间放置矩阵先验，融合后根据损失函数反向传播。
- 用硬参数共享的方式，对不同分类的交叉熵损失用融合公式进行权重求和。
- 将多线性关系网络引入稀疏格式预测领域，通过协方差矩阵计算不同任务先验关系并补充到损失函数中.

## 代码组织

- 数据预处理
  - 下载`SuiteSparse`数据集真实矩阵，存储在`data`文件夹中
  - 由`get_dataset.py`文件进行数据预处理
- 标注数据
  - 代码在`label`文件夹中，输入放在`label/data`文件夹中
  - 使用`taco`稀疏编译工具运行`spmv`和`sddmm`算子，
  - 运行时间存储在`label/output`文件夹中，作为数据标签
- 输入采样
  - 在`sample`文件夹中
  - 密度采样和直方图采样两种方式
- 训练模型
  - 单算子：密度采样对应的`SingleNet`，直方图采样对应的`DoubleNet`
  - 多算子：`MTLDataSet`划分数据集，`MTLNet`为采用多线性关系网络的模型
  - `tensorboard`的输出默认在`logs`文件夹中：
  
    ```bash
    tensorboard --logdir=./logs --port=8080
    ```
    
## 运行方式：

- 单个算子spmv

  - density

    - sample
        ```bash
        python sample/density.py data/suite-sparse/no_all_zero_list.txt data/npz/spmv_no_all_zero.npz

        python sample/density.py data/operator/spmv_list.txt data/npz/spmv_density.npz

        python sample/density.py data/operator/spmv_crop_list.txt data/npz/spmv_crop_density.npz
        ```
    - gen data 
        ```bash
        python model/genData.py density data/npz/spmv_crop_density.npz 4
        ```
    - train
        ```bash
        python model/SingleNet.py train ../data/npz/train_density.npz ../data/npz/test_density.npz ../data/model/ ../data/npz/result.npz
        ```
  - histograom

    - sample
        ```bash
        python sample/histogram.py data/suite-sparse/spmv_sddmm_list.txt data/npz/spmv_histogram.npz

        python sample/histogram.py data/operator/spmv_sddmm_crop_list.txt data/npz/spmv_sddmm_crop_histogram.npz
        
        python sample/histogram.py data/operator/active_spmv_sddmm_crop_list.txt data/npz/active_spmv_sddmm_crop_histogram.npz
        
        python sample/my_data.py data/npz/my_data_histogram.npz
        ```
    - gen data
        ```bash        
        python model/genData.py histogram data/npz/spmv_sddmm_crop_histogram.npz 3

        python model/genData.py histogram data/npz/active_spmv_sddmm_crop_histogram.npz 3

        python model/genData.py histogram data/npz/my_data_histogram.npz 3
        ```
    - train
        ```bash
        python model/DoubleNet.py train ../data/npz/train_histogram.npz ../data/npz/test_histogram.npz ../data/model/histogram/ ../data/npz/result.npz
        ```

  - sptfs

    - sample
        ```bash        
        python sample/histogram.py data/operator/spmv_sddmm_crop_list.txt data/npz/spmv_sddmm_crop_sptfs.npz
        
        python sample/histogram.py data/operator/active_spmv_sddmm_crop_list.txt data/npz/active_spmv_sddmm_crop_sptfs.npz
        ```
    - flat
        ```bash
        python model/genData.py flat data/npz/spmv_crop_sptfs.npz 4

        python model/DoubleNet.py train ../data/npz/train_flatten_imgs.npz ../data/npz/test_flatten_imgs.npz ../data/model/flat/ ../data/npz/result_flatten_imgs.npz
        ```
    - map
        ```bash
        python model/genData.py map data/npz/spmv_sddmm_crop_sptfs.npz 3

        python model/genData.py map data/npz/active_spmv_sddmm_crop_sptfs.npz 3

        python model/DoubleNet.py train ../data/npz/train_map_imgs.npz ../data/npz/test_map_imgs.npz ../data/model/map/ ../data/npz/result_map_imgs.npz
        ```
- 多算子 spmv & sddmm

  - histogram


  - sample

  - gen data
        ```bash
      python model/genData.py histogram data/npz/spmv_sddmm_crop_histogram.npz 3

      python model/genData.py histogram data/npz/active_spmv_sddmm_crop_histogram.npz 3
        ```      

  - train
        ```bash
      python model/MTLNet.py train ../data/npz/train_histogram.npz ../data/npz/test_histogram.npz ../data/model/histogram/ ../data/npz/result.npz
        ```

  - sptfs

    - sample
        ```bash
        python sample/histogram.py data/operator/active_spmv_sddmm_crop_list.txt data/npz/active_spmv_sddmm_crop_sptfs.npz
        ```
    - gen data

        ```bash
        python model/genData.py map data/npz/spmv_sddmm_crop_sptfs.npz 3

        python model/genData.py map data/npz/active_spmv_sddmm_crop_sptfs.npz 3
        ```
    - train
        ```bash
        python model/MTLNet.py train ../data/npz/train_map_imgs.npz ../data/npz/test_map_imgs.npz ../data/model/map/ ../data/npz/result_map_imgs.npz
        ```