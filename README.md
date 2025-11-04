# CFASR

## PyTorch implementation of “Dual-domain divide-and-conquer for scene text image super-resolution (CFASR)”.

## The source code will be released soon.

## Introduction
### Abstract

Scene Text Image Super-Resolution (STISR) aims to reconstruct high-quality text images from low-resolution
inputs to improve downstream text recognition accuracy. A critical problem lies in the fact that different regions
suffer from different degradations, e.g., text, backgrounds, and particularly their transitional boundaries (text
edges) undergo distinct degradation processes. Text edges frequently suffer from severe blurring effects that gen-
erate ambiguous "soft boundaries", which critically degrade recognition performance. To address this problem,
we propose a novel dual-domain divide-and-conquer super-resolution framework to achieve different treatments
of degraded differences. Specifically, we propose a frequency divide-and-conquer module (FDC) to decouple the
high- and low-frequency features in the frequency domain and perform differentiated processing. For the spatial
domain, we propose a spatial divide-and-conquer module (SDC) to achieve divide-and-conquer processing in the
spatial domain. Experimental results on the TextZoom benchmark dataset show that the recognition accuracy
of the model in the downstream scene text recognition task reaches the SOTA level. 

### 1. Requirements

Simply run the following command to install dependencies (note: different scikit-image or torchvision versions may slightly affect results):

conda install --yes --file requirements.txt

### 2. Datasets

We conduct experiments on the following benchmark datasets:

TEXTZoom Text Image Super-Resolution Dataset

Please refer to the corresponding dataset webpages or contact the authors for download links.

### 3. Evaluate the Pre-trained Models

We provide pre-trained CFASR models and evaluation results on the above datasets.
Download pre-trained weights from Google Drive (Coming Soon)
 and place them under the ./pretrained/ directory.

Then, run the following script to perform testing:

./test.sh


Before testing, make sure that the input LR images are placed in the ./input/ folder. The reconstructed HR results will be saved to the ./results/ directory.

### 4. Train Models

To train CFASR from scratch, run:

./train.sh


You can modify the configuration file in ./configs/ to change the dataset path, batch size, or training hyperparameters.



### 5. Acknowledgements

We thank all contributors for insightful discussions and related open-source works that inspired this project.

Our implementation is partially inspired by the following repositories:

RTSRN

