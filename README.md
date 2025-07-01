
<h1 align="center"> Learning to Be a Transformer to Pinpoint Anomalies (IEEE Access) </h1> 


<br>

:rotating_light: This repository contains download links to the datasets, code snippets, and checkpoints of our work "**Learning to Be a Transformer to Pinpoint Anomalies**",  [IEEE Access](https://ieeeaccess.ieee.org/)
 
by [Alex Costanzino*](https://alex-costanzino.github.io/), [Pierluigi Zama Ramirez*](https://pierlui92.github.io/), [Giuseppe Lisanti](https://www.unibo.it/sitoweb/giuseppe.lisanti), and [Luigi Di Stefano](https://www.unibo.it/sitoweb/luigi.distefano).  \* _Equal Contribution_

University of Bologna


<div class="alert alert-info">


<h2 align="center"> 

[Project Page](https://alex-costanzino.github.io/learning_to_be_a_transformer/) | [Paper](https://ieeexplore.ieee.org/document/11048772)
</h2>


## :bookmark_tabs: Table of Contents

1. [Introduction](#clapper-introduction)
2. [Datasets](#file_cabinet)
3. [Checkpoints](#inbox_tray)
4. [Code](#memo-code)
6. [Contacts](#envelope-contacts)

</div>

## :clapper: Introduction
To efficiently deploy strong, often pre-trained feature extractors, recent Industrial Anomaly Detection and Segmentation (IADS) methods process low-resolution images, e.g., 224x224 pixels, obtained by downsampling the original input images. However, while numerous industrial applications demand the identification of both large and small defects, downsampling the input image to a low resolution may hinder a method's ability to pinpoint tiny anomalies.

We propose a novel Teacher-Student paradigm to leverage strong pre-trained features while processing high-resolution input images very efficiently. The core idea concerns training two shallow MLPs (the Students) by nominal images so as to mimic the mappings between the patch embeddings induced by the self-attention layers of a frozen vision Transformer (the Teacher). Indeed, learning these mappings sets forth a challenging pretext task that small-capacity models are unlikely to accomplish on out-of-distribution data such as anomalous images.

Our method can spot anomalies from high-resolution images and runs way faster than competitors, achieving state-of-the-art performance on MVTec AD and the best segmentation results on VisA. We also propose novel evaluation metrics to capture robustness to defect size, i.e., the ability to preserve good localisation from large anomalies to tiny ones. Evaluating our method also by these metrics reveals its neatly superior performance.

<h4 align="center">

</h4>

<img src="./assets/architecture.jpg" alt="Alt text" style="width: 800px;" title="architecture">

:fountain_pen: If you find this code useful in your research, please cite:

```bibtex
@article{costanzino2025learning2be,
  author    = {Costanzino, Alex and Zama Ramirez, Pierluigi and Lisanti, Giuseppe, and Di Stefano, Luigi},
  title     = {Learning to Be a Transformer to Pinpoint Anomalies},
  journal   = {IEEE Access},
  year      = {2025},
}
```

<h2 id="file_cabinet"> :file_cabinet: Datasets </h2>

In our experiments, we employed two datasets featuring rgb images: [VisA](https://github.com/amazon-science/spot-diff) and [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad).


<h2 id="inbox_tray"> :inbox_tray: Checkpoints </h2>

Here, you can download the weights of the student networks employed in the results our paper.

To use these weights, please follow these steps:

1. Create a folder named `checkpoints/checkpoints_visa` in the project directory;
2. Download the weights [[Download]]();
3. Copy the downloaded weights into the `checkpoints_visa` folder.


## :memo: Code

<div class="alert alert-info">

**Warning**:
- The code utilizes `wandb` during training to log results. Please be sure to have a wandb account. Otherwise, if you prefer to not use `wandb`, disable it in `A_train_VisA.py` with the `flag mode = 'disabled'`.

</div>


### :hammer_and_wrench: Setup Instructions

**Dependencies**: Ensure that you have installed all the necessary dependencies. The list of dependencies can be found in the `./requirements.txt` file.


### :rocket: Inference L2BT

The `B_infer_VisA.py` script test the framework. It can be used to generate anomaly maps as well.

You can specify the following options:
   - `--dataset_path`: Path to the root directory of the dataset.
   - `--checkpoint_folder`: Path to the directory of the checkpoints, i.e., `checkpoints/checkpoints_visa`.
   - `--class_name`: Class on which the framework was trained, i.e., `candle`.
   - `--epochs_no`: Number of epochs used in framework optimization.
   - `--batch_size`: Number of samples per batch employed for framework optimization.
   - `--qualitative_folder`: Folder on which the anomaly maps are saved.
   - `--quantitative_folder`: Folder on which the metrics are saved.
   - `--visualize_plot`: Flag to visualize qualitatived during inference.
   - `--produce_qualitatives`: Flag to save qualitatived during inference.

If you haven't downloaded the checkpoints yet, you can find the download links in the **Checkpoints** section above.

### :rocket: Train L2BT

The `A_train_VisA.py` script train the framework.

You can specify the following options:
   - `--dataset_path`: Path to the root directory of the dataset.
   - `--checkpoint_savepath`: Path to the directory on which checkpoints will be saved, i.e., `checkpoints/checkpoints_visa`.
   - `--class_name`: Class on which the framework is trained, i.e., `candle`.
   - `--epochs_no`: Number of epochs for framework optimization.
   - `--img_size`: Resolution employed during training.
   - `--batch_size`: Number of samples per batch for framework optimization.
   - `--label`: A label to tell apart the experiments.


## :envelope: Contacts

For questions, please send an email to alex.costanzino@unibo.it.
