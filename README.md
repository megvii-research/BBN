## BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition
Boyan Zhou, Quan Cui, Xiu-Shen Wei*, Zhao-Min Chen

This repository is the official PyTorch implementation of paper [BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition](https://arxiv.org/abs/1912.02413).

## Main requirements

  * **PyTorch ≥1.0**
  * **torchvision ≥0.2.2_post3**
  * **TensorboardX**
  * **Python 3**

## Pretrain models for iNaturalist

We provide the BBN pretrain models of both 1x scheduler and 2x scheduler for iNaturalist 2018 and iNaturalist 2017.

iNaturalist 2018: [Baidu Cloud](https://pan.baidu.com/s/1olDppTptZ5HYWsgQsMCPLQ), [Google Drive](https://drive.google.com/open?id=1B9ZEfMHqE-KQRKX6nQLQRm8ErFrnHaoE)

iNaturalist 2017: [Baidu Cloud](https://pan.baidu.com/s/1soxsHKKblhapew_wuEdKPQ), [Google Drive](https://drive.google.com/open?id=1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU)

## Usage
```bash
# To train long-tailed CIFAR-10 with imbalanced ratio of 50:
python main/train.py  --cfg configs/cifar10.yaml     

# To validate with the best model:
python main/valid.py  --cfg configs/cifar10.yaml

# To debug with CPU mode:
python main/train.py  --cfg configs/cifar10.yaml   CPU_MODE True
```

You can change the experimental setting by simply modifying the parameter in the yaml file.

## Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/home/BBN/iNat18/images/train_val2018/Plantae/7477/3b60c9486db1d2ee875f11a669fbde4a.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 7477
                    },
                    ...
                   ]
    'num_classes': 8142
}
```
You can use the following code to convert from the original format of iNaturalist. 
The images and annotations can be downloaded at [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md) and [iNaturalist 2017](https://github.com/visipedia/inat_comp/blob/master/2017/README.md)

```bash
# Convert from the original format of iNaturalist
python tools/convert_from_iNat.py --file train2018.json --root /home/iNat18/images --sp /home/BBN/jsons
```


## Citing this repository
If you find this code useful in your research, please consider citing us:
```
@article{zhou2019BBN,
	title={{BBN}: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition},
	author={Boyan Zhou and Quan Cui and Xiu-Shen Wei and Zhao-Min Chen},
	journal={arXiv preprint arXiv:1912.02413},
	pages={1--14},
	year={2019}
}
```

## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: weixs.gm@gmail.com

Boyan Zhou: zhouboyan94@gmail.com

Quan Cui: cui-quan@toki.waseda.jp
