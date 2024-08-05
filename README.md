# LabelEngine: A General Framework for Interactive Object Counting, Detection, and Segmentation
Pytorch implementation for paper [LabelEngine: A General Framework for Interactive Object Counting, Detection, and Segmentation](https://arxiv.org), work in progress. <br>
Qin Liu, Nurislam Tursynbek, Elizabeth Brunk, Marc Niethammer <br>
UNC-Chapel Hill

#### [Paper](https://arxiv.org/) | [Project]() | [Demos]()

## Installation
The code is tested with ``python=3.10``, ``torch=2.2.0``, ``torchvision=0.17.0``.
```
git clone https://github.com/uncbiag/LabelEngine
cd LabelEngine
```
Now, create a new conda environment and install required packages accordingly.
```
conda create -n engine python=3.10
conda activate engine
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Getting Started

