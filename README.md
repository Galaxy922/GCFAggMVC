# GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering


This repo contains the code and data of our CVPR'2023 paper GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering. 


> [GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.pdf)

<img src="https://github.com/Galaxy922/GCFAggMVC/blob/main/figs/Framework.png"  width="897" height="317" />

## Requirements

python=3.7.1

pytorch=1.12.1

numpy>=1.21.6

scikit-learn>=1.0.2

## Datasets

The Synthetic3d, Prokaryotic, and MNIST-USPS datasets are placed in "data" folder. The others dataset could be downloaded from [cloud](https://pan.baidu.com/s/1XNWW8UqTcPMkw9NpiKqvOQ).

## Usage

The code includes:

- an example implementation of the model,
- an example clustering task for different missing rates.

```bash
python run.py --dataset 0 --devices 0 --print_num 100 --test_time 5
```

You can get the following output:

```bash
Epoch : 100/500 ===> Reconstruction loss = 0.2819===> Reconstruction loss = 0.0320 ===> Dual prediction loss = 0.0199  ===> Contrastive loss = -4.4813e+02 ===> Loss = -4.4810e+02
view_concat {'kmeans': {'AMI': 0.5969, 'NMI': 0.6106, 'ARI': 0.6044, 'accuracy': 0.5813, 'precision': 0.4408, 'recall': 0.3835, 'f_measure': 0.3921}}
Epoch : 200/500 ===> Reconstruction loss = 0.2590===> Reconstruction loss = 0.0221 ===> Dual prediction loss = 0.0016  ===> Contrastive loss = -4.4987e+02 ===> Loss = -4.4984e+02
view_concat {'kmeans': {'AMI': 0.6575, 'NMI': 0.6691, 'ARI': 0.6974, 'accuracy': 0.6593, 'precision': 0.4551, 'recall': 0.4222, 'f_measure': 0.4096}}
Epoch : 300/500 ===> Reconstruction loss = 0.2450===> Reconstruction loss = 0.0207 ===> Dual prediction loss = 0.0011  ===> Contrastive loss = -4.5115e+02 ===> Loss = -4.5112e+02
view_concat {'kmeans': {'AMI': 0.6875, 'NMI': 0.6982, 'ARI': 0.8679, 'accuracy': 0.7439, 'precision': 0.4586, 'recall': 0.444, 'f_measure': 0.4217}}
Epoch : 400/500 ===> Reconstruction loss = 0.2391===> Reconstruction loss = 0.0210 ===> Dual prediction loss = 0.0007  ===> Contrastive loss = -4.5013e+02 ===> Loss = -4.5010e+02
view_concat {'kmeans': {'AMI': 0.692, 'NMI': 0.7027, 'ARI': 0.8736, 'accuracy': 0.7456, 'precision': 0.4601, 'recall': 0.4451, 'f_measure': 0.4257}}
Epoch : 500/500 ===> Reconstruction loss = 0.2281===> Reconstruction loss = 0.0187 ===> Dual prediction loss = 0.0008  ===> Contrastive loss = -4.5018e+02 ===> Loss = -4.5016e+02
view_concat {'kmeans': {'AMI': 0.6912, 'NMI': 0.7019, 'ARI': 0.8707, 'accuracy': 0.7464, 'precision': 0.4657, 'recall': 0.4464, 'f_measure': 0.4265}}
```

## Citation

If you find our work useful in your research, please consider citing:

```latex
@InProceedings{Yan_2023_CVPR,
    author    = {Yan, Weiqing and Zhang, Yuanyang and Lv, Chenlei and Tang, Chang and Yue, Guanghui and Liao, Liang and Lin, Weisi},
    title     = {GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19863-19872}
}
```
## For any problems, contact me via zhangyuanyang922@gmail.com.

<!-- # This is the sources codes for the paper: Weiqing Yan, Yuanyang Zhang, Chenlei Lv, Chang Tang, Guanghui Yue, Weisi Lin. GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering, IEEE CVPR 2023 Accept.

# Paper:
  https://openaccess.thecvf.com/content/CVPR2023/html/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.html

# To test the trained model, run:
  python test.py

# To train a new model, run:
  python train.py
  
# The experiments were conducted on Linux with i9-10900K CPU, 62.5GB RAM and 3090Ti GPU.

# citation
```
@InProceedings{Yan_2023_CVPR,
    author    = {Yan, Weiqing and Zhang, Yuanyang and Lv, Chenlei and Tang, Chang and Yue, Guanghui and Liao, Liang and Lin, Weisi},
    title     = {GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {19863-19872}
}
```

# If you find the code is useful, please cite the above paper. For any problems, contact me via zhangyuanyang922@gmail.com.
 -->
