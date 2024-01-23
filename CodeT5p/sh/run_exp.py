#!/usr/bin/env python
import os
import argparse


def get_cmd(task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch, warmup,
            model_dir, summary_dir, res_fn, max_steps=None, save_steps=None, log_steps=None):
    if max_steps is None:
        cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s' % \
                  (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
                   warmup, model_dir, summary_dir, res_fn)
    else:
        cmd_str = 'bash exp_with_args.sh %s %s %s %d %d %d %d %d %d %d %d %d %s %s %s %d %d %d' % \
                  (task, sub_task, model_tag, gpu, data_num, bs, lr, source_length, target_length, patience, epoch,
                   warmup, model_dir, summary_dir, res_fn, max_steps, save_steps, log_steps)
    return cmd_str


def get_args_by_task_model(task, sub_task, model_tag):
    if task == 'translate':
        # java-cs: Read 10300 examples, avg src len: 13, avg trg len: 15, max src len: 136, max trg len: 118
        # [TOKENIZE] avg src len: 45, avg trg len: 56, max src len: 391, max trg len: 404
        src_len = 320
        trg_len = 256
        epoch = 100
        patience = 5
    elif task == 'summarize':
        # ruby: Read 24927 examples, avg src len: 66, avg trg len: 12, max src len: 501, max trg len: 146
        # [TOKENIZE] avg src len: 100, avg trg len: 13, max src len: 1250, max trg len: 161
        # Python: Read 251820 examples, avg src len: 100, avg trg len: 11, max src len: 512, max trg len: 222
        # [TOKENIZE] avg src len: 142, avg trg len: 12, max src len: 2016, max trg len: 245
        # Javascript: Read 58025 examples, avg src len: 114, avg trg len: 11, max src len: 512, max trg len: 165
        # [TOKENIZE] avg src len: 136, avg trg len: 12, max src len: 3016, max trg len: 177
        src_len = 256
        trg_len = 128
        epoch = 15
        patience = 2
    elif task == 'www_log':
        if sub_task == 'java':
            src_len = 512
            trg_len = 512
        epoch = 30
        patience = 5
    elif task == 'concode':
        # Read 100000 examples, avg src len: 71, avg trg len: 26, max src len: 567, max trg len: 140
        # [TOKENIZE] avg src len: 213, avg trg len: 33, max src len: 2246, max trg len: 264
        src_len = 320
        trg_len = 150
        epoch = 30
        patience = 3
    elif task == 'defect':
        # Read 21854 examples, avg src len: 187, avg trg len: 1, max src len: 12195, max trg len: 1
        # [TOKENIZE] avg src len: 597, avg trg len: 1, max src len: 41447, max trg len: 1
        src_len = 512
        trg_len = 3
        epoch = 10
        patience = 2
    elif task == 'whether_to_log':
        src_len = 512
        trg_len = 5
        epoch = 4
        patience = 2

    if 'codet5_small' in model_tag:
        bs = 32
        if task == 'summarize' or task == 'translate' or (task == 'www_log' and sub_task == 'small'):
            bs = 64
        elif task == 'whether_to_log':
            bs = 32
    elif 'codet5_large' in model_tag:
        bs = 8
    else:
        bs = 10
        if task == 'translate':
            bs = 25
        elif task == 'summarize':
            bs = 48
        elif task == 'whether_to_log':
            if model_tag in ['codebert', 'roberta', 'codet5p']:
                bs = 16
            else:
                bs = 10
    lr = 5
    if task == 'concode':
        lr = 10
    elif task == 'defect':
        lr = 2
    return bs, lr, src_len, trg_len, patience, epoch


def run_one_exp(args):
    bs, lr, src_len, trg_len, patience, epoch = get_args_by_task_model(args.task, args.sub_task, args.model_tag)
    print('============================Start Running==========================')
    cmd_str = get_cmd(task=args.task, sub_task=args.sub_task, model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=src_len, target_length=trg_len,
                      patience=patience, epoch=epoch, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/{}_{}.txt'.format(args.res_dir, args.task, args.model_tag))
    
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def run_multi_task_exp(args):
    # Total train data num = 1149722 (for all five tasks)
    if 'codet5_small' in args.model_tag:
        bs, lr, max_steps, save_steps, log_steps = 60, 5, 600000, 20000, 100
    else:
        bs, lr, max_steps, save_steps, log_steps = 25, 5, 800000, 20000, 100

    if args.data_num != -1:
        max_steps, save_steps, log_steps = 1000, 200, 50
    print('============================Start Running==========================')
    cmd_str = get_cmd(task='multi_task', sub_task='none', model_tag=args.model_tag, gpu=args.gpu,
                      data_num=args.data_num, bs=bs, lr=lr, source_length=-1, target_length=-1,
                      patience=-1, epoch=-1, warmup=1000,
                      model_dir=args.model_dir, summary_dir=args.summary_dir,
                      res_fn='{}/multi_task_{}.txt'.format(args.res_dir, args.model_tag),
                      max_steps=max_steps, save_steps=save_steps, log_steps=log_steps)
    print('%s\n' % cmd_str)
    os.system(cmd_str)


def get_sub_tasks(task):
    if task == 'summarize':
        sub_tasks = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
    elif task == 'translate':
        sub_tasks = ['java-cs', 'cs-java']
    elif task == 'www_log':
        sub_tasks = ['java']
    elif task == 'whether_to_log':
        sub_tasks = ['java', 'medium']
    elif task in ['concode', 'multi_task']:
        sub_tasks = ['none']
    return sub_tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default='codebert',
                        choices=['roberta', 'codet5_small', 'codet5_base', 'codet5_large','codet5p'])
    parser.add_argument("--task", type=str, default='whether_to_log', choices=['summarize', 'concode', 'translate',
                                                                          'www_log', 'whether_to_log', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='java')
    parser.add_argument("--res_dir", type=str, default='results', help='directory to save fine-tuning results')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard', help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1, help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    
    assert args.sub_task in get_sub_tasks(args.task)
    
    if args.task != 'multi_task':
        run_one_exp(args)
    else:
        run_multi_task_exp(args)