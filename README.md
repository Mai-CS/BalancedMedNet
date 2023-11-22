# Balanced Contrastive Representation Learning for Long-tailed Medical Image Classification

The long-tailed data distribution problem in machine learning causes model bias since the majority of the data is concentrated in a few classes, leading to misclassifications in the minority classes. In medical datasets, this problem is particularly challenging as it can have serious consequences. To address this problem, we propose **B**alanced **M**edical **N**et (BMN), a novel approach that balances supervised contrastive learning by using class averaging and class complement to alleviate the problem of long-tailed distribution in medical datasets. In BMN, class averaging takes the average of instances of each class in a minibatch, thus reducing the contribution of head classes and emphasizing the importance of tail classes, and class complements are introduced to have all classes represented in the minibatch. We evaluate the effectiveness of our approach on two long-tailed medical datasets, [ISIC2018](https://challenge.isic-archive.com/landing/2018/) and [APTOS2019](https://www.kaggle.com/c/aptos2019-blindness-detection), and found that it outperformed or matched the performance of state-of-the-art methods in terms of classification accuracy and F1-score. Our proposed method has the potential to improve diagnosis and treatment for patients with possibly fatal diseases and addresses an important issue in medical dataset.

This is a PyTorch implementation of our project:
<img src="img/BMN_Horizontal (1).png" width="500" style="background-color:white;"/>

We adopt the codebase of [BCL](https://github.com/FlamieZhu/Balanced-Contrastive-Learning).

Weights for best models can be found [here](https://drive.google.com/drive/folders/1XsvUILrVPL2zEVpIf6RPz6vgO7tMHgSo?usp=sharing).

## Datasets

#### [ISIC2018](https://challenge.isic-archive.com/data/#2018)

Download the data related to the 3rd task (Training Data, Training Ground Truth, Validation Data, and Validation Ground Truth). The text files in <code>data/ISIC2018</code> are the processed labels ready for the Dataset class to read.

#### [APTOS2019](https://www.kaggle.com/c/aptos2019-blindness-detection)

Download the train images and the train.csv file from the kaggle competition. The text files that include the train/val split we adopted can be found here <code>data/APTOS2019</code>.

## Dependencies

Run the following command to install dependencies before running the code: <code>pip install -r requirements.txt</code>

## Code Structure

- <code>dataset/</code>:
    - <code>dataset.py</code>: defines the dataset class used for both datasets
- <code>loss/</code>:
    - <code>contrastive.py</code>: includes the definition of the <code>SCL</code> (supervised contrastive loss) and <code>BalSCL</code> (balanced contrastive loss) classes
    - <code>logitadjust.py</code>: includes the definition of the losses used in the classification branch, e.g. <code>LogitAdjust</code>, <code>FocalLoss</code>, <code>EQLv2</code>, and <code>LabelSmoothingCrossEntropy</code>
- <code>models/</code>:
    - <code>resnext.py</code>: defines the <code>BCLModel</code>, as well as the <code>ResNet</code> and <code>ResNeXt</code> backbones
- <code>randaugment.py</code>: includes the implementation of <code>AutoAugment</code> and <code>RandAugment</code>
- <code>main.py</code>: main file that is ran for training
- <code>utils.py</code>: includes the definitions of some util functions
- <code>train-isic.sh</code>: includes the command to run the training for the isic dataset with all the arguments that match our best experiment. The arguments <code>data</code>, <code>val_data</code>, <code>txt</code>, and <code>val_txt</code> are the paths to the training images, validation images, training labels, and validation labels. You need to specify these directories. Another argument that has to be specified is <code>user_name</code> which is the wandb username where the experiments will be logged.
- <code>train-aptos.sh</code>: includes the command to run the training for the aptos dataset with all the arguments that match our best experiment. The arguments <code>data</code>, <code>val_data</code>, <code>txt</code>, and <code>val_txt</code> are the paths to the training images, validation images, training labels, and validation labels. You need to specify these directories. Another argument that has to be specified is <code>user_name</code> which is the wandb username where the experiments will be logged.