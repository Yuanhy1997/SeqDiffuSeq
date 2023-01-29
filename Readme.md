This is the official repo for the paper [SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers](https://arxiv.org/pdf/2212.10325.pdf)

Parts of our codes are modified from [DiffusionLM](https://github.com/XiangLi1999/Diffusion-LM) and [minimaldiffusion](https://github.com/madaan/minimal-text-diffusion) repos. 

## Environment 

Before running our code, you may setting the environments using the following lines.

```{bash}
conda create -n seqdiffuseq python=3.8
conda install mpi4py
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0
pip install -r requirements.txt
```

## Preparing dataset

For the non-translation tasks, we follows [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) for the dataset settings.

For IWSLT14 and WMT14, we follow the data preprocessing from [fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/translation), we also provide the processed datasets in the following links.


## Training
To run the code, we use iwslt14 en-de as an illustrative example: 
1. Prepare the data of iwslt14 under ./data/iwslt14/ directory;  
2. Learning the BPE tokenizer by
```{bash}
python ./tokenizer_utils.py train-byte-level iwslt14 10000 
```
3. To train with the following line:
```{bash}
mkdir ckpts
bash ./train_scripts/iwslt_en_de.sh 0 de en
#(for en to de translation) bash ./train_scripts/iwslt_en_de.sh 0 en de
```
You may modify the scripts in ./train_scripts for your own training settings.

## Inference

After training accomplish, you can run the following line for inference:
```{bash}
bash ./inference_scrpts/iwslt_inf.sh path-to-ckpts/ema_0.9999_280000.pt path-to-save-results path-to-ckpts/alpha_cumprod_step_260000.npy
```
The ema_0.9999_280000.pt file is the model weights and alpha_cumprod_step_260000.npy is the saved noise schedule. You have to use the most recent .npy schedule file saved before .pt model weight file.

## Other Comments

Note that for all the training experiments, we all set the maximum training steps and warmups to 1000000 and 10000. For different datasets, it is needless to stop training until maximum training steps. IWSLT14 use checkpoint around 300000 training steps, WMT15 around 500000 train steps and non-translation task around 100000 train steps. 

You can change the hyperparameter setting for your own experiments, maybe increasing the training batches or modify the training schedule will bring some improvements. 

## Citation 

If you find our work and codes interesting and useful, please cite:
```bibtex
@article{Yuan2022SeqDiffuSeqTD,
  title={SeqDiffuSeq: Text Diffusion with Encoder-Decoder Transformers},
  author={Hongyi Yuan and Zheng Yuan and Chuanqi Tan and Fei Huang and Songfang Huang},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.10325}
}
```