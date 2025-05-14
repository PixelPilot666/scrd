<div align="center">

## Seek Common Ground While Reserving Differences: Semi-supervised Image-Text Sentiment Recognition

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![License](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/exped1230/S2-VER/blob/main/LICENSE)
  
</div>

This is the official implement of our paper **Seek Common Ground While Reserving Differences: Semi-supervised Image-Text Sentiment Recognition**.



## Abstract
Multi-modal sentiment analysis has attracted extensive research attention as more and more users share images and texts to express their emotions and opinions on social media. Collecting large amounts of labeled sentiment data is an expensive and challenging task due to the high cost of labeling and unavoidable label ambiguity. To address this issue, semi-supervised learning is explored to utilize the extensive unlabeled data. However, different from the common recognition tasks, the conflicting sentiment between image and text leads to the sub-optimal of the semi-supervised learning algorithms. To this end, we propose a novel semi-supervised image-text sentiment recognition framework (SCDR) for capturing common and specific information of the single modal. Specifically, we introduce decoupling networks to separate modal shared and private features and use the private features to train uni-modal classifiers. Furthermore, considering the complex relation between modalities, we devise a modal selection attention (MSA) module that adaptively assesses the dominating sentiment modality at sample-level, which guides the fusion of multi-modal representations. Furthermore, to prevent the model predictions from overly relying on common features under the guidance of multi-modal labels, we design a pseudo-label filtering (PLF) strategy based on modality selection and loss values. Extensive experiments and comparisons on five publicly available datasets demonstrate that SCRD outperforms state-of-the-art methods.

## Training
To perform SCRD on MVSA-Single with 600 labels, run:

```
python main.py \
--data_dir 'dataset/MVSA_Single' \
--train_data_dir 'dataset/MVSA_Single' \
--test_data_dir 'dataset/MVSA_Single' \
--gpu 0 \
--save_dir './saved_models-single/n600' \
--lr 1e-4 \
--batch_size 2 \
--num_labels 600 \
--threshold 0.95 \
--num_train_iter 512 ```