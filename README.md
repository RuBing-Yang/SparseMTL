
# tensorboard

    tensorboard --logdir=/home/rubing/SparseMTL/logs --port=8080

# 单个算子spmv

## density

- sample

    python sample/density.py data/suite-sparse/no_all_zero_list.txt data/npz/spmv_no_all_zero.npz

    python sample/density.py data/operator/spmv_list.txt data/npz/spmv_density.npz

    python sample/density.py data/operator/spmv_crop_list.txt data/npz/spmv_crop_density.npz

- gen data 

    python model/genData.py density data/npz/spmv_crop_density.npz 4

- train

    python model/SingleNet.py train ../data/npz/train_density.npz ../data/npz/test_density.npz ../data/model/ ../data/npz/result.npz

## histograom

- sample

    python sample/histogram.py data/suite-sparse/spmv_sddmm_list.txt data/npz/spmv_histogram.npz

    python sample/histogram.py data/operator/spmv_sddmm_crop_list.txt data/npz/spmv_sddmm_crop_histogram.npz
    
    python sample/histogram.py data/operator/active_spmv_sddmm_crop_list.txt data/npz/active_spmv_sddmm_crop_histogram.npz
    
    python sample/my_data.py data/npz/my_data_histogram.npz

- gen data
    
    python model/genData.py histogram data/npz/spmv_sddmm_crop_histogram.npz 3

    python model/genData.py histogram data/npz/active_spmv_sddmm_crop_histogram.npz 3

    python model/genData.py histogram data/npz/my_data_histogram.npz 3

- train

    python model/DoubleNet.py train ../data/npz/train_histogram.npz ../data/npz/test_histogram.npz ../data/model/histogram/ ../data/npz/result.npz


## sptfs

- sample
    
    python sample/histogram.py data/operator/spmv_sddmm_crop_list.txt data/npz/spmv_sddmm_crop_sptfs.npz
    
    python sample/histogram.py data/operator/active_spmv_sddmm_crop_list.txt data/npz/active_spmv_sddmm_crop_sptfs.npz

- flat

    python model/genData.py flat data/npz/spmv_crop_sptfs.npz 4

    python model/DoubleNet.py train ../data/npz/train_flatten_imgs.npz ../data/npz/test_flatten_imgs.npz ../data/model/flat/ ../data/npz/result_flatten_imgs.npz

- map

    python model/genData.py map data/npz/spmv_sddmm_crop_sptfs.npz 3

    python model/genData.py map data/npz/active_spmv_sddmm_crop_sptfs.npz 3

    python model/DoubleNet.py train ../data/npz/train_map_imgs.npz ../data/npz/test_map_imgs.npz ../data/model/map/ ../data/npz/result_map_imgs.npz

# 多算子 spmv & sddmm

## histogram


- sample

- gen data

    python model/genData.py histogram data/npz/spmv_sddmm_crop_histogram.npz 3

    python model/genData.py histogram data/npz/active_spmv_sddmm_crop_histogram.npz 3
    

- train

    python model/MTLNet.py train ../data/npz/train_histogram.npz ../data/npz/test_histogram.npz ../data/model/histogram/ ../data/npz/result.npz


## sptfs

- sample

    python sample/histogram.py data/operator/active_spmv_sddmm_crop_list.txt data/npz/active_spmv_sddmm_crop_sptfs.npz

- gen data


    python model/genData.py map data/npz/spmv_sddmm_crop_sptfs.npz 3

    python model/genData.py map data/npz/active_spmv_sddmm_crop_sptfs.npz 3

- train

    python model/MTLNet.py train ../data/npz/train_map_imgs.npz ../data/npz/test_map_imgs.npz ../data/model/map/ ../data/npz/result_map_imgs.npz

# test

    cd 3d-tensor

    python Dl3dNet.py test <train data> <test data> <model data> <result data>

# calculate precision/speedup
    
    python calPrecision.py

    python calcSpeedup.py
