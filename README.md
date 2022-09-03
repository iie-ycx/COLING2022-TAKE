# COLING2022-TAKE: Topic-shift Aware Knowledge sElection for Dialogue Generation
This repository contains the code for COLING-2022 full paper:[TAKE: Topic-shift Aware Knowledge sElection for Dialogue Generation](http://baidu.com).

![TAKE model pic](https://github.com/iie-ycx/COLING2022-TAKE/raw/main/fig/take-pic.png)

Please contact Chenxu Yang (yangchenxu@iie.ac.cn) if you have any question.
## Requirements
- transformers==4.15.0
- python 3.9
- pytorch 1.10.1
## Datasets
We use the Wizard of Wikipedia datasets preprocessed by [Meng et al](https://dl.acm.org/doi/10.1145/3404835.3462824). You can download the datasets from [here](https://share.weiyun.com/rpmIidMZ). After downloading, please put the files in the following 3 dirs:

./knowSelect/datasets/wizard_of_wikipedia/

./dialogen/datasets/wizard_of_wikipedia/

./dialogen/datasets/wow_gpt2/

## Running TAKE Codes
### Inference
We provide pretrained checkpoints to save your time, and you can acquire them [here](https://share.weiyun.com/zqoSPsF7).
You need to download the corresponding checkpoints and put them in the folder ./knowSelect/output/TAKE_WoW/model/ (KS) or the folder ./dialogen/output/TAKE_WoW/model/ (DG) according to their name.
After that, you can run the infer_bash.sh in in the root directory to get the evaluation results. 
### Retraining
Of course, you can also retrain and evaluate TAKE by running the train_bash.sh in the root directory.
## Citation
Please cite our paper if you use use source code of TAKE in your work:
> not ready