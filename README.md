# ELogger
Here is a repo for our paper: End-to-End Log Statement Generation at Block-Level


## Project Structure
```
├─whether_to_log  # whether-to-log component. ELogger and the variant of ELogger based on UniXcoder.
├─www_log         # www-log component. ELogger and the variant of ELogger based on UniXcoder.     
├─dataset    
├─outputs     
├─CodeT5p         # The variant of ELogger based on CodeT5p
└─
```

## Step 1: Check Python Dependencies
To install ELogger dependencies, please run:

```shell
pip install -r requirements.txt
```

## Step 2: Prepare Datasets

The original dataset we used is provided by LACNE. The original dataset can be found at this link: https://drive.google.com/drive/folders/1D12y-CIJTYLxMeSmGQjxEXjTEzQImgaH?usp=sharing

The full processed datasets can be found at: https://drive.google.com/drive/folders/1N06FIgEuE4l3NXwqrg3xbffHOfRhPhdN?usp=share_link

The whether_to_log example dataset is under ```/examples/whether_to_log_dataset```
The www_log example dataset is under ```/examples/www_to_log_dataset```

If you want to run ELogger on the full datasets, you should download the data from the above websites, put the corresponding files in ```/dataset``` 

## Step 3: Fine-tune or Inference
ELogger and the variant of ELogger based on UniXcoder: You can use the shell provided in /whether_to_log/README.md and /www_log/README.md to fine-tune or inference. 
The variant of ELogger based on CodeT5p: You can use the shell provided in /CodeT5p/README.md to fine-tune or inference. 

The outputs of ELogger and baselines can be  found at this link: https://drive.google.com/drive/folders/17gzLW7gBoX3GNjWKHdzMzkmOAOAgjp7c?usp=share_link
