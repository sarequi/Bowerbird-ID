# Bowerbird Individual Identification  

## 1. Introduction 

Since 2018, the Fusani Lab's team has been filming a population of spotted bowerbirds in Australia. The videos were captured in the wild using motion-triggered camera traps placed in front of the bowers, which are structures male bowerbirds build to attract mates. 

Several birds were already identified with uniquely coloured leg bands. However, identifying the birds on video through the leg bands is time-consuming and can be cumbersome due to physical occlusions. Furthermore, a significant number of birds, predominantly females, remain unbanded, making it difficult to track their behaviour and interactions over time. Automated identification of individual birds would ease the research of their behaviour. 

The project aims to automate the identification of individual bowerbirds using machine learning. A ResNet-type algorithm was trained to classify 16 birds on a rich dataset of video frames from 2018, and achieved a performance of 0.9877 mAP. While the scope of this project is to identify and classify birds only within the 16 classes, this model could ultimately be able to create new categories for unbanded birds.

This documentation covers: 
* Required environment and dependencies
* Computational resources, with emphasis on accessing the Vienna Scientific Cluster for GPU-powered processing 
* Step-by-step data pre-processing 
* Model training and validation + How to use the final model
* Lessons learned

## 2. Prerequisites

### Software requirements

A requirements.txt file is provided, listing compatible versions of all necessary libraries and packages.

The recommended approach would be to create a dedicated virtual environment, e.g. using conda, to avoid conflicts with other Python projects, and install dependencies directly from the requirements.txt file.

```
conda create -n env_name python=version 
conda activate env_name
pip install -r requirements.txt
```
A yaml file is also provided.
```
conda env create -f bowerbird_id_env.yml --name your_custom_env_name
```

### Hardware requirements 
 
During the initial stages of development, the functionality of the scripts for video frame extraction and preprocessing was tested on a sample of the dataset (with up to 50 instances per bird) on CPU-only machines. This approach validated the data processing pipeline without requiring substantial computing power. Once validated, the project was scaled up to the full 2018 dataset, using computational power from to the Vienna Scientific Cluster (VSC).

Both pipelines (CPU- and GPU-based) are provided on the project's GitHub repository. Detailed information on how to navigate the VSC is included in the supplementary material. For CPU-processing, the scripts are provided as notebook files, as these are more interactive. Meanwhile, the files requried for GPU-processing are provided as Python files, and with a matching SLURM file, as they needed to be ran through scheduled jobs at the VSC.

## 3. Data pre-processing

### Video sampling

In the reference study, pictures were captured as the birds sat on a perch, triggered by an RFID identifier attached to each bird. This ensured relatively controlled shots, as the birds were always at a standardised distance from the camera, and in a relatively uniform posture. In contrast, for this project we extracted frames from continuous video footage fro a motion-triggered camera. Firstly, this meant that there could be videos featuring multiple birds, or no birds at all, and that the capture birds in a variety of postures, at varying distances from the camera, and performing multiple behaviours, including copulation events. 

To reduce the variation across extracted frames, video sampling was done considering certain behavioural and contextual criteria, obtained from a scoring spreadsheet (`2018.xlsx`). Filtering was applied to ensure that  the extracted videos contained only one bird, and that that bird was the owner of the bower. The name of videos that met this criteria were saved in a dictionary as a json file (`valid_videos.json`), listing each bird ID along with a corresponding array of video IDs.

### Frame sampling

Frames are opened and extracted from the validated videos through OpenCV. The script iterates through the videos, sampling frames at a fixed interval. After it is done extracting frames from a video, it limiting extractions to a maximum of ten frames per video. The extracted frames are written into an output directory, logging the bird ID, video name, and timestamp into a metadata file (`extracted_frames_metadata.csv`). 

Then, Ultralytics' YOLOv11 (`yolo11m-seg.pt`) was used to filter out frames that do not feature a bird. To minimize redundant data, **PIL’s `imagehash`** was used to filter out near-duplicate frames, with a similarity threshold of ____. 

### Frame processing

Objectives
 Standardising the images by ensuring birds are the same size regardless of their position in the scene, e.g. to avoid birds positioned further away from the camera from appearing smaller. It is also meant to reduce the presence of noise and artifacts present in the background. The aim of pre processing is to enhance the relevant features of the brids to ease model training and improve classification.

Leg bands were removed from the scene as they may have interfered with feature learning.

Filtering out frames with no brids in them 

Birds changing position in the scene: 

Flow chart of frame pre processing pipeline:

## Object detection
Mask R-CNN setup and configuration - Steps for training or fine-tuning the model - Addressing performance issues with manual annotations 

## 4. Model training and evaluation

CNN architecture and design
Training process and hyperparameter tuning
Techniques for improving classification accuracy 
Metrics used for model evaluation

## 5. Automated classification



## 6. Results and discussion

The best model's accuracy was **0.9877**

Classification report

| Class (bird) | F1-score | Support |
|--------------|----------|---------|
| B02          |  0.99    | 3408    |
| B03          | 0.99     | 1869    |
| B04          | 0.99     | 3963    |
| B05          | 0.98     | 1746    |
| B07          | 0.99     | 2390    |
| B11          |0.99      | 1761    |
| B18          | 0.97     | 540     |
| B23          | 0.99     | 1873    |
| B26          | 0.97     | 576     |
| B29          | 0.98     | 1104    |
| B30          | 0.98     | 938     |
| B31          | 0.98     | 1155    |
| B47          | 0.99     | 903     |
| B49          | 0.99     | 1254    |
| B50          | 0.99     | 2561    |
| B52          | 0.99     | 1600    |

where:
* Precision: Ratio of correctly predicted positive observations to the total predicted positives.
-  Recall: Ratio of correctly predicted positive observations to the actual positives.
-  F1-score: The weighted average of Precision and Recall.
-  Support: The number of actual occurrences of the class in the dataset.

![Training progress](.assets/validation_curves.png)

![Confusion matrix](.assets/confusion_matrix.png)

Key findings 
Challenges encountered and solutions implemented
Potential improvements and future work 
* Birds appear in various poses. A method for normalising posture could be to train a pose estimation model, e.g. through key point detection, to filter frames based on the bird'S position, e.g. keeping only frames wwere the brids' back is visible.
* Leg band removal: Colour segmentation could be implemented either alone or as a separate step to detect coloured bands. This approach was attempted but given upon on due to the orange tones present in the background and as part of the birds' body.  A more thorough approach, not so focused on orange tones, may work better, perhaps by converting the frames to different colour spaces first and then applying colour segmentation.

## 7. Reproducibility
Structure and usage of `.yaml`
 Python files 
Organization of training and test datasets 

## 8. Supplementary material

### Working on the VSC

Running scripts with GPU support is possible without scheduling jobs, and can be done in the following way through the local terminal (seems not to work on the NoMachine terminal)

### General access to the VSC

resource pool  = VSC-5
login_server   = vsc5.vsc.ac.at
username       = username
project name   = Bowerbird-ID
project_id     = 72607 # this is also your Linux group id, aka my folders within the home and data directories 
home directory = /home/fs72607/username          (size is 100.0 GiB)
data directory = /gpfs/data/fs72607/username     (size is 9.8 TiB)

The home & data directories are also available via environment variables: `$HOME` and `$DATA`
The optimal way to copy files into the $DATA directory is using FileZilla
The optimal GUI to explore files on $DATA is NoMachine

1. Activate VPN ()BIG-IP Edge Client: Ensure that the connection is "Full tunel"

2. In the terminal:
* Access the VSC server through the SSH key:  ```ssh username@login_server```
* Enter required SSH passphrase
* Enter sms-set OTP 

## Running scripts through VSC's JupyterLab

Move to the  directory where the script to run is
```
cd Bowerbird-ID/3_Frame_sampling
```

## Allocate GPU
The cluster's configuration requires to either submit the job via sbatch or request full nodes with GPUs, since salloc cannot allocate partial nodes. To request a full node with 2 GPUs: (ONLY ONCE)
```
salloc -N 1 -p zen3_0512_a100x2 --qos zen3_0512_a100x2 --gres=gpu:2

```
Check which node you were allocated
```
squeue -u $USER

```
SSH into the allocated node
```
ssh n3071-001
```
Load CUDA
```
spack unload
spack load cuda@11.8.0%gcc@9.5.0/ananl33
```
Check wheter the GPU is accessible
```
nvidia-smi
```
## Create an environment on the VSC server

Load Miniconda module to enable conda commands
module load miniconda3  
```
module load miniconda3
```
Load Miniconda module to enable conda commands
module load miniconda3  
```
eval "$(conda shell.bash hook)"
```
Activate the Conda environment 
```
conda activate sarah_env
```

## Schedule job

-   Save the Script as a .slrm file
    
    `nano ~/Bowerbird-ID/3_Frame_sampling/Frame_sampling.slrm` 
    
-   **Submit the Job**:
    
    `sbatch ~/Bowerbird-ID/3_Frame_sampling/Frame_sampling.slrm` 
    
-   **Monitor the Job**:
    
    `squeue -u $(whoami)` 
    
-   **Check Logs**:
    
    bash
    
    Copy code
    
    `less frame_sampling_1.log`

