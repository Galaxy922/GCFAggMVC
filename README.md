## GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering


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

- an example for train a new model：

```bash
python run.py
```

- an example  for test the trained model:

```bash
python test.py
```

You can get the following output:

```bash
Epoch 290 Loss:15.420288
Epoch 291 Loss:15.431067
Epoch 292 Loss:15.417261
Epoch 293 Loss:15.436375
Epoch 294 Loss:15.398655
Epoch 295 Loss:15.406467
Epoch 296 Loss:15.413018
Epoch 297 Loss:15.419146
Epoch 298 Loss:15.419894
Epoch 299 Loss:15.389602
Epoch 300 Loss:15.377309
---------train over---------
Clustering results:
ACC = 0.9700 NMI = 0.8713 PUR=0.9700 ARI = 0.9126
Saving model...
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

If you have any problems, contact me via zhangyuanyang922@gmail.com.


