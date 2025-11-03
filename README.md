# Automated individual identification of Spotted Bowerbirds 🪶  

## Introduction 📚

Spotted Bowerbirds (_Chlamydera maculata_) are medium-sized, brown passerines with characteristic spots on their wings and back. Males of this species are known for building and decorating elaborate structures called bowers, which are considered extensions of the males' phenotypes and serve solely for sexual signalling, to attract females. During the breeding season male bower owners spend much of their time in and around their bowers, which makes them ideal focal points for long-term monitoring through camera traps.

Since 2018, the [Fusani Lab](https://fusanilab.org/) has studied a population of banded bowerbirds in Australia using motion-triggered camera traps. Their research has explored topics such as long-term male partnerships (Spezie & Fusani, 2022, 2023), and transfer learning (Knoester et al., 2024). A crucial component of these studies is the reliable identification of individual birds over time. Until now, re-identification has been done manually by visually inspecting videos to observe the birds’ leg bands. Each video is then annotated in a spreadsheet with details such as the individual ID, the displayed behavior, and the number of visible birds. Across three breeding seasons, they collected around 200,000 videos (each lasting 30 seconds to 2 minutes) from approximately 50 individuals. Manually reviewing and annotating this volume of data is extremely time consuming.

To address this challenge, several studies have used Convolutional Neural networks (CNNs) to automate individual re-identification (Li et al., 2019; Ferreira et al., 2020). CNNs are deep learning architectures that learn hierarchical image features, allowing them to recognize subtle and complex patterns (Schmidhuber, 2014), such as  animals' natural markings, that might distinguish an individual from another. 

In this study, we aimed to develop a CNN based pipeline for automatic identification of individual Spotted Bowerbirds from camera trap footage. Specifically, our objectives were to:

1. Train a CNN to identify individual bowerbirds with an F1-score of 0.85 or higher, testing the hypothesis that these birds show enough individual visual variation for reliable identification.
2. Determine the minimal amount of training data required per individual to reach this performance level, to assess the practicality of applying this method to new individuals for whom data is typically limited.
3. Evaluate the impact of viewpoint on the performance of the CNN, hypothesising that side and back viewpoints, where the birds' characteristic spots are visible from, would be more informative for individual recognition.
4. Evaluate the impact of viewpoint on the minimal data requirement, hypothesising that training on only the most informative viewpoints would reduce the amount of data needed per individual to achieve this performance.

## Dataset

All data used in this study was collected by Giovanni Spezie (PhD candidate at the University of Veterinary Medicine Vienna, supervised by Prof. Leonida Fusani) in Taunton National Park (Scientific), Queensland, Australia, during the 2021 breeding season, between July and November. The dataset comprised 76,645 videos, with a total of 16 individual birds. All videos were manually scored by several students prior to the realisation of this study. 

## Structure of the repo

- **`1_Data_processing/`** →  Video filtering, frame sampling, detection and segmentation of the birds, data split
- **`2_Individual_classifier/`** →  Training the individual classifier on all available data
- **`3_Minimal_data_requirement/`** → Training the individual classifier on increasingly larger subsets of all available data (50-1000 instances)
- **`4_Viewpoint_and_classifier_performance/`** → Training the individual classifier on viewpoint-specific datasets
- **`5_Viewpoint_and_data_requirement/`** → Training the individual classifier on increasingly larger subsets of the viewpoint-specific datasets 

## Prerequisites

### Software 

Approach 1: Install dependencies in a dedicated virtual environment from the requirements.txt file.
```
conda create -n env_name python=version 
conda activate env_name
pip install -r lists compatible versions of all necessary libraries and packages.txt
```

Approach 2: Directly install the environment from the bird_id_env.YAML file.
```
conda env create -f bird_id_env.yml --name env_name
```

### Hardware 

Our ResNet50 model was implemented in PyTorch (1.13.1) and Torchvision (0.14.1). Training was conducted on a computer partition from the Vienna Scientific Cluster, on an NVIDIA A40 GPU (NVIDIA Corporation) and 8 CPU cores from a node equipped with 256 GB of RAM, using the CUDA framework for GPU acceleration on an AlmaLinux (8.5) operating system.

## Data pre-processing 🎞️

Model weights (.pth files) are included when pre-trained models are required. 

### 1. Video filtering 

Ground truth for individual identity was established through manual coding of all recorded videos. This process involved visual inspection to document whether the videos contained visible birds, whether they were banded and unbanded individuals, and to determine the identity of banded individuals. Videos were subsequently filtered from this initial scoring to include only those featuring a single owner bowerbird. 
We excluded videos featuring multiple individuals (banded or unbanded), no birds, single unbanded individuals, individuals of different species, and single banded non-owners. 

### 3.1. Frame sampling

Before frame extraction, a 10% subset of the videos from each bird was held out for testing. Afterwards, frames from the remaining videos were extracted using the OpenCV library, at a fixed interval. After extracting all possible frames, the extractions are limited to 10 frames per video, using the image similarity index and removing frames where no bird is visible, through a pre-trained YOLOv11 detection model.

### 3.2. Frame processing

A multistage image processing pipeline was applied to the extracted raw frames aiming to standardise the size and position of the bird within the frames.

* Object detection: First, a pre-trained YOLOv11 (Ultralytics) model was used to detect the birds in each frame (Figure 3a). For frames in which a bird was detected, the frame was cropped to the bounding box with the highest confidence score, isolating the detected bird (Figure 3b). Frames in which no bird was detected were excluded from further analysis.

* Cropping of the bird in the image: Crops the frame to keep only the area within the highest-confidence bounding box. 

* Bird masking: If there was a detection, the YOLOv11 segmentation model generates a pixel-wise mask of the bird. The mask is then processed using connected component analysis, which groups neighbouring pixels into distinct regions. Each region should represent a separate detected object. However, sometimes, due to lower-quality detections, parts of an object are detected and masked separately. To reduce these lower-quality detections, the script skips regions of too few pixels (currently MIN_BLOB_PIXELS = 5000). If, after this filtering, no region remains, the frame is skipped. This ensures that only frames with clear, identifiable birds are kept.

* Removing leg bands: Leg bands were removed to prevent them from being identified as features for classification by the classifier. each mask was processed to digitally remove the leg band to prevent the classifier from overfitting the training data. This was done by iterating through each pixel row from the lower one-third portion of the mask, i.e. the fraction of the mask expected to contain the birds' legs, to identify narrow vertical structures, i.e. with a width less than or equal to 100 pixels and setting those regions to the background value (black or '0').

### 3.3. Training-validation data split

This script splits masked bird images into training and validation sets (currently test_size=0.3, therefore there is a 70-30 train-val split). The results of the split are logged in processed_bird_ids.log.

## 3.4. Training and evaluating the classifier 💪🏼

Our ResNet50 model was implemented in PyTorch (1.13.1) and Torchvision (0.14.1). Training was conducted on a computer partition from the Vienna Scientific Cluster, on an NVIDIA A40 GPU (NVIDIA Corporation) and 8 CPU cores from a node equipped with 256 GB of RAM, using the CUDA framework for GPU acceleration on an AlmaLinux (8.5) operating system.

The model was initialised with weights pre-trained on the ImageNet dataset. The final fully connected layer was replaced with a classification head to output 16 classes, corresponding to the individual birds, and used a softmax activation function for probability distribution across classes.

Input images were resized to 512×512 pixels and normalized using the standard ImageNet mean and standard deviation values. During training, data augmentation was applied in the form of random horizontal flips (probability = 0.5). The model was trained using Stochastic Gradient Descent with a momentum of 0.9, an initial learning rate of 1×10−3, and a batch size of 32. The learning rate was reduced by a factor of 0.1 every 7 epochs. The model was trained for a total of 20 epochs, with performance on the validation set monitored after each epoch to select the best performing checkpoint. The entire training process required 298 minutes (~5 hours). This selected model was then used for final evaluation on a test set, obtained from the held-out test videos, corresponding to 10% of the original videos.

The performance of the model was assessed both on the validation and on the test sets, through the F1-score, which is calculated as the harmonic mean of precision and recall, combining the two into a single performance metric. Precision is defined as the ratio of true positives (TP) to the sum of true positives and false positives (FP), and recall , also known as sensitivity, is the ratio of true positives (TP) to the sum of true positives and false negatives (FN). The F1-score performance cutoff of 0.85 was established as a study-specific benchmark, as no single cutoff can be applied across studies.

## 3.5. Running automated classification

The script determines the most likely bird ID by following these steps:

1️⃣ Counting how many times each class is predicted for each frame in a subfolder. If no subfolder is provided but simply frames showing the same bird, all frames are processed as one.

2️⃣ Computing a list of probabilities (confidence scores) assigned to the predicted classes, and summing the confidence scores only for the most common class. The sum is divided by most_common_count to get the average confidence. This provides inforamtion about how sure the model is about its most frequent prediction.

3️⃣ Printing the top 3 most frequently predicted classes and how many frames they were predicted for.
This gives information about whether other birds appear similarly frequently and whether there’s classification uncertainty.

## 4. Minimal training data

To determine the minimal number of instances per individual bird required to achieve an acceptable classification performance, we compared the performance of models trained and validated on an increasing number of instances, i.e. increasingly large subsets. For clarity, each subset refers to the total number of instances per individual. Each subset was further randomly split into training, validation and test subsets, corresponding to 70%, 20%, and 10%, respectively. Data from each subset was used to train and validate the classifier, and the performance of the model was evaluated solely on the validation set, and not on the test set.

## 5. Results 

Our study demonstrated the feasibility of training a ResNet50 classifier with camera trap footage to identify individual Spotted Bowerbirds, achieving a mean F1-score of 0.9877 on the validation set and a mean F1-score of 0.926 on the test set. This high performance empirically supports the existence of consistent and learnable inter-individual variations in this species. A key finding was that an F1-score of ≥0.85 could be attained with a relatively modest dataset of 400 instances per individual (using a 70:30 training/validation split).

* Performance of the classifier

The classifier achieved a mean F1-score of 0.9877 on the valdiaiton set, and a mean F1-score of 0.926 on the test set. Birds with fewer training instances showed slightly lower scores. 

Classification report

| Class (bird) | Valid videos | Total frames | Training set | Validation set | Test set | F1-score (Validation) | F1-score (Test) |
|--------------|-------------:|-------------:|-------------:|---------------:|---------:|----------------------:|----------------:|
| B02 | 3 341 | 10 349 | 7 244 | 3 105 | 35 | 0.99 | 0.886 |
| B03 | 1 750 | 5 482 | 3 837 | 1 645 | 77 | 0.99 | 0.960 |
| B04 | 4 333 | 11 806 | 8 264 | 3 542 | 6 | 0.99 | 0.705 |
| B05 | 7 779 | 5 366 | 3 756 | 1 610 | 77 | 0.98 | 0.939 |
| B07 | 3 407 | 7 124 | 4 986 | 2 138 | 92 | 0.99 | 0.962 |
| B11 | 5 606 | 5 319 | 3 723 | 1 596 | 45 | 0.99 | 0.967 |
| B18 | 1 581 | 1 638 | 1 146 | 492 | 75 | 0.97 | 0.966 |
| B23 | 3 291 | 5 627 | 3 938 | 1 689 | 88 | 0.99 | 0.982 |
| B26 | 1 635 | 1 763 | 1 234 | 529 | 131 | 0.97 | 0.935 |
| B29 | 4 476 | 3 356 | 2 349 | 1 007 | 33 | 0.98 | 0.876 |
| B30 | 3 033 | 2 898 | 2 028 | 870 | 8 | 0.98 | 0.875 |
| B31 | 3 940 | 3 445 | 2 411 | 1 034 | 7 | 0.98 | 0.933 |
| B47 | 3 124 | 2 706 | 1 894 | 812 | 95 | 0.99 | 0.945 |
| B49 | 1 826 | 3 821 | 2 674 | 1 147 | 196 | 0.99 | 0.992 |
| B50 | 4 079 | 7 807 | 5 464 | 2 343 | 72 | 0.99 | 0.915 |
| B52 | 1 601 | 4 899 | 3 429 | 1 470 | 140 | 0.99 | 0.968 |
| **Total / Mean** | **54 802** | **83 406** | **58 377** | **25 029** | **1 117** | **0.98** | **0.926** |

Confusion matrices showing theprediction performance on the model on validaiton and test sets

The following confusion matrices shoe the distribution of predicted classes against the ground truth. The elements across the diagonal represent correctly classified instances for each bird, and the elements outside of the diagonal correspond to misclassifications.

![Confusion matrix](confusion_matrices_bw.svg)

* Minimal training data

The following graph shows the model performance on the validation set across increasing subset sizes. The learning curve indicates a general trend of increasing F1-score with larger subset sizes. The F1-score reached approximately 0.85 with a subset of 400 instances. Beyond this point, the F1-score continued to improve but the rate of improvement diminished until reaching a 0.944 F1-score at 1000 instances.

![Model performance across increasing subset sizes](Performance_across_subsets.svg)

### 6. Challenges encountered and solutions implemented

### 7. Potential improvements and future work:
* Standardising bird posture: In our current dataset, birds appear in various poses. An approach could be to train a pose estimation model, e.g. through key point detection, to filter frames based on the bird's position, e.g. keeping only frames were the bird's back is visible.
* Leg band removal: Birds' legs are not always positioned vertically in the image. Thus, narrow structure filtering was not always successful, and there are instances where leg bands are still visible. Colour segmentation could be implemented alone or as a separate step to detect coloured bands. This approach was attempted but discarded due to the orange tones present in the background and as part of the bird's body.  A more thorough approach, not so focused on orange tones, may work better, e.g. converting the frames to different colour spaces first and then applying colour segmentation.

### Ethics

Ethical approval for this study was obtained from the Animal Ethics Committee of the Department of Agriculture and Fisheries (AEC reference number: CA 2018/04/1185), and field activities at Taunton National Park (Scientific) were approved by Queensland Wildlife and Parks Service (PTU18-001089; NPS18-001090).

### Acknowledgements

The computational results of this work have been achieved using the Life Science Compute Cluster (LiSC) of the University of Vienna. Data were collected thanks to a grant of the Austrian Science Fund (FWF: W1262-B29 [https://doi.org/10.55776/W1262]).

This project would not have been possible without the continuous support and feedback provided by Dr. Cliodhna Quigey, Dr. Leonida Fusani, Job Knoester, MSc. and Dr. Giovani Spezie. They are all truly great collaborators.

### References

Ferreira AC, Silva LR, Renna F, Brandl HB, Renoult JP, Farine DR, et al. Deep learning‐based methods for individual recognition in small birds. Methods Ecol Evol. 2020 Sep;11(9):1072–85.   

Knoester J, Spezie G, Mann DC, Fusani L. Do social interactions predict similarities in audio-visual courtship signals in spotted bowerbirds? In: Proceedings of the 10th Convention of the European Acoustics Association Forum Acusticum 2023 [Internet]. Turin, Italy: European Acoustics Association; 2024. Available from: https://dael.euracoustics.org/confs/fa2023/data/articles/000411.pdf 

Li S, Li J, Tang H, Qian R, Lin W. ATRW: A benchmark for Amur Tiger Re-identification in the Wild [Internet]. arXiv [cs.CV]. 2019. Available from: http://arxiv.org/abs/1906.05586

Schmidhuber J. Deep learning in neural networks: An overview [Internet]. arXiv [cs.NE]. 2014. Available from: http://arxiv.org/abs/1404.7828
  
Spezie G, Fusani L. Male-male associations in spotted bowerbirds (Ptilonorhynchus maculatus) exhibit attributes of courtship coalitions. Behav Ecol Sociobiol. 2022 Jul 7;76(7):97. 

Spezie G, Fusani L. Sneaky copulations by subordinate males suggest direct fitness benefits from male-male associations in spotted bowerbirds (Ptilonorhynchus maculatus). Ethology. 2023 Jan;129(1):55–61. 
