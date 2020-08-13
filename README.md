# attribute-guided-image-generation-from-layout
This is the code repository complementing the paper "attribute-guided image generation from layout". The pretrained model can be downloaded here: (TODO)

![Model Pipeline](demo/pipeline.pdf)

## Dependencies and Setup
- Python 3.6.8
- [SPADE](https://github.com/NVlabs/SPADE)
- numpy==1.16.3
- torch==1.1.0
- torchvision==0.3.0
- PIL==6.0.0
- imageio==2.5.0
- tensorboardX==1.8
- sklearn==0.20.3
- h5py==2.8.0

## Experiments
### Datasets
[Visual Genome (VG)](https://visualgenome.org/)
Download: 
```bash
cd data/Dataset/vg
chmod +x download_vg.sh
./download_vg.sh
```
Preprocessing: 
```bash
cd data
python preprocess_vg.py
```

### Train 
To train 64x64 (or 128x128) from scratch, run
```bash
python train64.py
```
The script will check if pretrained models exsit.

Training 128x128 model is computationally expensive. You might consider using `torch.DataParallel`.

### Test
To test 64x64 (or 128x128) model, run 
```bash
python test64.py
```
Note that you can modify attributes as you wish in the test64.py (test128.py) script

