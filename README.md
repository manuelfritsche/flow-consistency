## Semi-Supervised Learning of Semantic Segmentation from Video

This is the PyTorch implementation of my project "Semi-Supervised Learning of Semantic Segmentation from Video".
The project was supervised by Yuhua Chen and Stamatios Georgoulis.

**Abstract**

Training semantic segmentation models typically requires a large amount of pixel-wise annotations. Collecting
these annotations is a laborious and expensive process. On the other hand, unlabeled videos can be
collected at a much lower cost and contain rich information, such as how objects move.
This thesis aims to leverage such information as an additional source of supervision. We investigate the
problem in a semi-supervised setting, where only a small number of frames are annotated, while a large
number of unlabeled frames are provided. We observe that pixels corresponding to the same object should
have similar velocity. Motivated by this observation, we introduce a regularization loss, which uses the
optical flow between two frames. It is designed to penalize inconsistencies between the optical flow and
the prediction from the semantic segmentation model. This loss does not require any ground truth labels
and can utilize the temporal information in unlabeled videos. In addition, the semantic segmentation is also
supervised by annotated data through a commonly used cross entropy term.
We evaluate our method on the Cityscapes dataset and compare it to a baseline that only uses cross
entropy loss. Experiments are made with different amounts of annotated images, where we achieve significant
performance gains over the baseline when using additional unlabeled data. The results demonstrate the
effectiveness of the proposed method in cases where annotated data is limited.


### Installation


**Python Requirements**
```shell
pip install -r requirements.txt
```
Some of the dependencies (like `pytorch`) might need to be installed with conda.

**Optical Flow**

Follow the steps in [Pyflow](https://github.com/pathak22/pyflow) to install the optical flow.
Make sure to save the repository in the main folder.

### Data

* Download the Cityscapes dataset from [here](https://www.cityscapes-dataset.com/).
* Extract the zip / tar and modify the path appropriately in the `.yml` files in `configs/`.
* The following folders of the dataset are required:
    - `gtFine`
    - `leftImg8bit`
    - `leftImg8bit_sequence`

### Pre-trained Models
We provide [pre-trained models](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints.tar.gz)
with and without Flow Consistency Loss (FCL) where different fractions of the annotated data were used for training.

Alternatively, the individual models be downloaded from the following links:

| | | | | |
|---:|:---:|:---:|:---:|:---:
| FCN-8s|[100%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s_100.pkl)| [50%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s_50.pkl)|    [10%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s_10.pkl) |    [1%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s_1.pkl)|
| +FCL|[100%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s+fcl_100.pkl)| [50%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s+fcl_50.pkl)| [10%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s+fcl_10.pkl)| [1%](https://data.vision.ee.ethz.ch/yuhchen/share/project/semi-sup-sseg-video/checkpoints/fcn8s+fcl_1.pkl)
| | | | | |

### Usage

**To pre-compute the optical flow:**

```shell
python calculate_pyflow.py --config [CONFIG] 
```
 * `--config`: The configuration file to use. 
This file only affects the images that are used. 
All provided configuration files will yield the same results.


**To train the model:**

```shell
python train.py --config [CONFIG] 
```
 * `--config`: The configuration file to use:
    - `configs/fcn8s_x.yml` will use FCN8s without Flow Consistency Loss and *x* percent of the data used.
    - `configs/fcn8s+fcl_x.yml` will use FCN8s with Flow Consistency Loss and *x* percent of the labeled data used.
 The rest of the data will be used as unlabeled data for Flow Consistency Loss.


**To validate the model :**
```shell
usage: validate.py --config [CONFIG] --model_path [MODEL_PATH] [--eval_flip]
```
 * `--config`: Config file to be used (.yml file)
 * `--model_path`: Path to the saved model (.pkl file)
 * `--eval_flip`: Enable evaluation with flipped images



**To visualize the output of the model:**

```shell
python visualize.py --config [CONFIG]  --model_path [MODEL_PATH]
```
 * `--config`: Config file to be used (.yml file)
 * `--model_path`: Path to the saved model (.pkl file)


### Acknowledgements
This code is based on [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
