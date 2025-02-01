import numpy as np
import torch
import os
from tqdm.auto import tqdm
import argparse
from torch import cuda
import time
from utils import *
from torch.cuda.amp import autocast as autocast
import pickle
import random
import json
from datasets.nextqa import NEXTQADataset
from torch.backends import cudnn
from utils.utils import *
from utils.optims import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default='experiments')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    parser.add_argument('--word_size', default=1, help="n_gpus")
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--grounding_lr', type=float, default=7e-5) # 3e-5

    parser.add_argument('--use_schedule', type=bool, default=False)
    parser.add_argument('--warmup_start_lr', type=float, default=1e-8)
    parser.add_argument('--min_lr', type=float, default=5e-6, help='min_lr for consine annealing')
    parser.add_argument('--max_T', type=int, default=30, help='epoches for lr->min_lr / min_lr->lr')
    parser.add_argument('--eval_step', type=int, default=1, help="eval every 1/eval_step epoch")
    parser.add_argument('--save_ckpt', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default='nextqa', choices=['nextqa'])
    parser.add_argument('--frame_count', type=int, default=32)
    parser.add_argument('--mode', type=str, default='grounding', choices=['grounding', 'uniform', 'oracle'])

    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--width', type=float, default=0.2)
    parser.add_argument('--use_spatial', type=bool, default=True)
    parser.add_argument('--model', type=str, default='t5-xl', choices=['t5-xl'])
    parser.add_argument('--use_vit', type=bool, default=False)
    parser.add_argument('--use_lora', type=bool, default=False)

    args = parser.parse_args()
    return args


def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def prepare_inputs(args, data):

    video_ids = data["video_ids"]
    qids = data["qids"]
    types = data["types"]

    questions = data["questions"] 
    answers = data["answers"]

    mc_prompt = "Considering the information presented in the frame, select the correct answer from the options."

    # if args.dataset == 'intentqa':
    options_a0 = data["options_a0"]
    options_a1 = data["options_a1"]
    options_a2 = data["options_a2"]
    options_a3 = data["options_a3"]
    options_a4 = data["options_a4"]
    text_input =  ['Question: ' + question + f'\nOptions: \nA: {option_a0} \nB: {option_a1} \nC: {option_a2} \nD: {option_a3} \nE: {option_a4}' + '\nAnswer: ' for question, option_a0, option_a1, option_a2, option_a3, option_a4 in zip(questions, options_a0, options_a1, options_a2, options_a3, options_a4)]

    text_output = answers

    if args.dataset == 'nextqa':
        return text_input, text_output, questions, options_a0, options_a1, options_a2, options_a3, options_a4


@torch.no_grad()
def eval(args, val_loader, model):
    model.eval()

    val_loss = 0
    val_vqa_loss = 0
    val_reg_loss = 0
    val_info_loss = 0
    val_acc = 0
    overall_acc = 0

    acc_records = []
    
    for step, data in enumerate(tqdm(val_loader, desc=f"validation")):

        if args.dataset == 'nextqa':
            text_input, text_output, questions, options_a0, options_a1, options_a2, options_a3, options_a4 = prepare_inputs(args, data)
            samples = {
                    "text_input": text_input,
                    "text_output": text_output,
                    "questions": questions,
                    "options_a0": options_a0,
                    "options_a1": options_a1,
                    "options_a2": options_a2,
                    "options_a3": options_a3,
                    "options_a4": options_a4,
                    "frame_features": data["frame_features"].cuda(args.local_rank),
                    "answers_text": data["answers_text"],
                    "answers_id": data["answers_id"]
                }

        generate_kwargs = {
            "do_sample": True,
            "num_beams": 5, 
            "min_length": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 30,
            "temperature":1,
            "top_p":0.9,
            "repetition_penalty":1,
            "length_penalty":1
            }

        with torch.cuda.amp.autocast(enabled=True, dtype=model.dtype): # Enable autocast before and after
            with torch.no_grad():
                outputs = model(samples)
                pred_texts = model.generate(samples, **generate_kwargs)

        for i in range(args.eval_bs):
            qid = data['qids'][i]
            video_id = data['video_ids'][i]
            type = data['types'][i]
            input_text = text_input[i]
            label = text_output[i]
            pred = pred_texts[i]

            acc_records.append({
                'qid': qid,
                'video_id': video_id,
                'type': type,
                'input': input_text,
                'label': label,
                'pred': pred
                })

        loss = outputs['loss']
        val_loss += loss.item() 
        val_vqa_loss += outputs['vqa_loss'].item() 
        val_reg_loss += outputs['regression_loss'].item() 
        val_info_loss += outputs['infoNCE_loss'].item() 
        val_acc += compute_acc(bs = args.eval_bs, labels = text_output, preds = pred_texts)

        if step<=4:
            for i in range(len(text_input)):
                print()
                print("---------------------eval-------------------------")
                print("---------------------ids-------------------------")
                print("video_id: " + data["video_ids"][i] + "  qid: " + data["qids"][i])
                print("---------------------type-------------------------")
                print(data["types"][i])
                print("---------------------input-------------------------")
                print(text_input[i])
                print("---------------------preds-------------------------")
                print(pred_texts[i])
                print("--------------------answers------------------------")
                print(text_output[i])
                print()


    
    def compute_acc_nextqa(data):
        """
        Compute the overall and class-wise accuracy for the given data.

        Args:
        data (list): A list of dictionaries, where each dictionary contains the keys:
                    'qid', 'video_id', 'type', 'input', 'label', 'pred'.

        Returns:
        tuple: A tuple containing the overall accuracy and a dictionary with class-wise accuracies.
        """
        total_samples = len(data)
        correct_predictions = 0
        class_counts = {'C': 0, 'T': 0, 'D': 0}
        class_correct = {'C': 0, 'T': 0, 'D': 0}

        for sample in data:
            if sample['pred'] == sample['label']:
                correct_predictions += 1
                class_correct[sample['type'][0]] += 1
            class_counts[sample['type'][0]] += 1 

        overall_accuracy = correct_predictions / total_samples
        class_accuracies = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}

        return overall_accuracy, class_accuracies

    if args.dataset == 'nextqa':
        overall_acc, class_acc = compute_acc_nextqa(acc_records)
        print('Overall Acc: ', overall_acc)
        print('Class Acc: ', class_acc)

    # Average the evaluation metrics across different processes
    val_loss = round(val_loss/len(val_loader), 4)
    val_vqa_loss = round(val_vqa_loss/len(val_loader), 4)
    val_reg_loss = round(val_reg_loss/len(val_loader), 4)
    val_info_loss = round(val_info_loss/len(val_loader), 4)
    val_acc = round(val_acc/len(val_loader), 4)
    model.train()
    # return val_loss, val_vqa_loss, val_reg_loss, val_info_loss, val_acc, overall_acc
    return val_loss, val_acc, overall_acc

def train(args, train_dataset, val_dataset, model):

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_bs, pin_memory=True, shuffle=False, drop_last=True, num_workers=8)


    if args.mode == 'grounding':
        ignored_params = list(map(id, model.grounding.parameters())) # Returns the memory addresses of the parameters
        base_params = filter(lambda p: p.requires_grad and id(p) not in ignored_params, model.parameters()) 
        optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': model.grounding.parameters(), 'lr': args.grounding_lr}], 
        lr = args.lr, betas=(0.9, 0.999), weight_decay=0.02)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr = args.lr, betas=(0.9, 0.999), weight_decay=0.02)


    val_loss, val_acc, overall_acc = eval(args, val_loader, model)
    print(f'val_loss:{val_loss} val_acc:{val_acc}')

    # val_loss, val_vqa_loss, val_reg_loss, val_info_loss, val_acc, overall_acc = eval(args, val_loader, model)


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'nextqa':
        train_dataset = NEXTQADataset(anno_path='../nextqa/annotations_mc/train.csv', frame_count=args.frame_count)
        val_dataset = NEXTQADataset(anno_path='../nextqa/annotations_mc/val.csv', frame_count=args.frame_count)
        test_dataset = NEXTQADataset(anno_path='../nextqa/annotations_mc/test.csv', frame_count=args.frame_count)


    from models.blip2_t5_instruct import Blip2T5Instruct
   
    if 't5' in args.model:
        model = Blip2T5Instruct(
            dtype=torch.bfloat16,
            frame_num=args.frame_count,
            mode = args.mode,
            window_size = args.window_size,
            use_spatial = args.use_spatial,
            model = args.model,
            temperature = args.temperature,
            width = args.width,
            use_vit = args.use_vit,
            use_lora = args.use_lora
        )        
    
    model.load_state_dict(torch.load('./experiments/t5-xl_nextqa_9_0.0.pth', map_location='cpu'))

    device = torch.device('cuda', args.local_rank)
    init_seeds(args.seed)
    torch.cuda.set_device(device)
    model = model.cuda(args.local_rank)

    print(get_parameter_number(model))
    print("trian_num: ", len(train_dataset), " val_num: ", len(val_dataset),  " test_num: ", len(test_dataset))
    print(args)

    train(args, train_dataset, test_dataset, model)