# COLING2022-TAKE: Topic-shift Aware Knowledge sElection for Dialogue Generation
This repository contains the code for COLING-2022 full paper:[TAKE: Topic-shift Aware Knowledge sElection for Dialogue Generation](http://baidu.com).

![TAKE model pic](https://github.com/iie-ycx/COLING2022-TAKE/raw/main/fig/take-pic.png)

Please contact Chenxu Yang (sduycx@163.com) if you have any question.
## Requirements
- transformers==4.15.0
- python 3.9
- pytorch 1.10.1
## Datasets
We use the Wizard of Wikipedia datasets preprocessed by [Meng et al](https://dl.acm.org/doi/10.1145/3404835.3462824). You can download the datasets from [here](https://share.weiyun.com/rpmIidMZ). After downloading, please create folder ./datasets in the root directory and put the files in it.
## Running TAKE Codes
### Inference
We provide pretrained checkpoints to save your time, and you can acquire them [here](https://share.weiyun.com/aFmsVVcb).
You need to download the corresponding checkpoints and put them in the folder ./knowSelect/output/TAKE_WoW/model/ (KS) or the folder ./dialogen/output/TAKE_WoW/model/ (DG).
After that, you can run the infer_bash.sh in in the root directory to get the evaluation results. 
### Retraining
Of course, you can also retrain and evaluate TAKE by running the train_bash.sh in the root directory.
## Citation
Please cite our paper if you use use source code of TAKE in your work:
> not ready