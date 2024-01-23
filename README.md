# ELogger
Here is a repo for our paper: End-to-End Log Statement Generation at Block-Level


## Dataset
The original dataset we used is provided by LACNE. The original dataset can be found at this link: https://drive.google.com/drive/folders/1D12y-CIJTYLxMeSmGQjxEXjTEzQImgaH?usp=sharing

## Project Structure
```
├─whether_to_log  # whether-to-log component. ELogger and the variant of ELogger based on UniXcoder.
├─www_log         # www-log component. ELogger and the variant of ELogger based on UniXcoder.     
├─dataset    
├─outputs     
├─CodeT5p         # The variant of ELogger based on CodeT5p
└─

## Step 1: Check Python Dependencies

To install ELogger dependencies, please run:

```shell
pip install -r requirements.txt
```

## Step 2: Prepare Datasets
The whether_to_log example dataset is under ```/examples/whether_to_log_dataset```

The www_log example dataset is under ```/examples/www_to_log_dataset```

## Step 3: Fine-tune or Inference
ELogger and the variant of ELogger based on UniXcoder: You can use the shell provided in /whether_to_log/README.md and /www_log/README.md to fine-tune or inference. 
The variant of ELogger based on CodeT5p: You can use the shell provided in /CodeT5p/README.md to fine-tune or inference. 
