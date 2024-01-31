

### Fine-tune

#### shell for ELogger
```
lr=3e-5
batch_size=32
beam_size=10
source_length=512
target_length=512
output_dir=saved_models_for_ELogger/ELogger_www_log
train_file=dataset/www_log/train_source.code,dataset/www_log/train_target.code
dev_file=dataset/www_log/valid_source.code,dataset/www_log/valid_target.code
epochs=30 
pretrained_model=microsoft/graphcodebert-base

mkdir -p $output_dir

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name microsoft/graphcodebert-base --config_name microsoft/graphcodebert-base --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/train.log
```

#### shell for the variant of ELogger based on UniXcoder
```
mkdir saved_models_for_UniXcoder
python run_4_UniXcoder.py \
	--do_train \
	--do_eval \
	--model_name_or_path /home/fuying/CodeBERT/UniXcoder/unixcoder-base \
	--train_filename dataset/www_log/train_source.code,dataset/www_log/train_target.code \
	--dev_filename dataset/www_log/valid_source.code,dataset/www_log/valid_target.code \
	--output_dir saved_models_for_UniXcoder/UniXcoder_www_log \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 10 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 3e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30 
```


### Inference

#### shell for ELogger
```
batch_size=32
beam_size=10
source_length=512
target_length=512
output_dir=saved_models_for_ELogger/ELogger_www-log
pretrained_model=microsoft/graphcodebert-base
test_file=dataset/www_log/test_source.code,dataset/www_log/test_target.code
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin


python run.py --do_test --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name microsoft/graphcodebert-base --config_name microsoft/graphcodebert-base --load_model_path $load_model_path --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size 2>&1| tee $output_dir/test.log

```

#### shell for the variant of ELogger based on UniXcoder
```
python run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename dataset/www_log/test_source.code,dataset/www_log/test_target.code \
	--output_dir saved_models_for_UniXcoder/UniXcoder_www-log \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 8 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--learning_rate 3e-5 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 30 
```
