# IFSO

## Get Started
#### 1. Create the environment
* Create a conda environment
```shell
conda create -n ifso python=3.7
conda activate ifso
```

* Intall cudatoolkits, PyTorch and torchvision
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
```
#### 2. Download our code
```shell
git clone https://github.com/hrcheng1066/IFSO.git
cd IFSO
```
#### 3. Build the ``mmcv-full``
```shell
 git clone https://github.com/open-mmlab/mmcv.git
 cd mmcv
 MMCV_WITH_OPS=1 pip install -e .
```
#### 4. Replace some files in mmcv with ours
```shell
replace IFSO/mmcv/mmcv/runner/base_runner.py with IFSO/replace/base_runner.py
replace IFSO/mmcv/mmcv/runner/epoch_base_runner.py with IFSO/replace/epoch_base_runner.py
replace IFSO/mmcv/mmcv/runner/hooks/chenckpoint.py with IFSO/replace/checkpoint.py
```

#### 5. Pre-train, Prune and Fine-tune
Go to IFSO/tools/ and release the corresponding part of experiments.sh
```shell
bash experiments.sh
```


