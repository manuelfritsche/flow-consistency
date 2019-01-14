## Semi-Supervised Video Semantic Segmentation

This is the PyTorch implementation of my project "Semi-Supervised Video Semantic Segmentation".


### Installation


**Python Requirements**
```shell
pip install -r requirements.txt
```

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


### Usage

**To precompute the optical flow:**

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

