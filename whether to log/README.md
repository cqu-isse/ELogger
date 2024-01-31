### Data Format
train.txt/valid.txt/test.txt stored in the following format:  
	code_block	label

### Fine-tune

#### shell for ELogger
```
mkdir saved_models_for_Elogger
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --do_eval \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --epoch 4 \
    --code_length 512 \
    --data_flow_length 32 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
```


#### shell for the variant of ELogger based on UniXcoder
```
mkdir saved_models_for_UniXcoder
python run_4_UniXcoder.py \
    --output_dir saved_models \
    --model_name_or_path microsoft/unixcoder-base \
    --do_train \
    --train_data_file dataset/whether_to_log/train.txt \
    --eval_data_file dataset/whether_to_log/valid.txt \
    --num_train_epochs 4 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 
```


### Inference

#### shell for ELogger
```
python run.py \
    --output_dir=saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_test \
    --test_data_file=dataset/test.txt \
    --epoch 4 \
    --code_length 512 \
    --data_flow_length 32 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log
```


#### shell for the variant of ELogger based on UniXcoder
```
python run_4_UniXcoder.py \
    --output_dir saved_models \
    --model_name_or_path microsoft/unixcoder-base \
    --do_test \
    --test_data_file dataset/whether_to_log/test.txt \
    --num_train_epochs 4 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 
```
