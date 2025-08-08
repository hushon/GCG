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


q_template = """Categorize the following question into one of (a) causal question, (b) temporal question or (c) descriptive question. 
- Explanations about each question type: 
Causal questions are designed to explain actions, either uncovering the intentions of the previously occurring actions 
or stating causes for subsequent actions. Questions in the causal group ask either why the objects act in a certain manner 
or how (what did they do) to bring about an observed effect. (e.g., "**why** was the toddler in red crying at the end of the video?", 
"**how** did the lady help the toddler who fell at the end?") 
Temporal questions assess the model’s capability of reasoning about temporal relationships between actions. Temporal actions, 
while related to causality, are determined only by order of occurrence. Hence, questions of this type ask about the previous 
(what ... do before ...), present (what... doing when/while/as ...) or next actions (what/how ... do/react after ...). 
(e.g., "what was the lady doing **before** the toddler in red fell off the stone?", "how did the lady react **after** the toddler in red 
fell off the stone?", "what was the boy doing **when** the toddler fell backwards from the stone?") 
Descriptive questions focus on scene description of the videos (e.g., the places, objects / attributes, and main actions 
/ events). These questions complement causal and temporal questions to make up a holistic video comprehension and also 
allow for comparison between different types of questions. Specifically, the questions cover binary choices (yes/no, or 
the answers are indicated in the questions, e.g., “... tired or energetic?”), location (where), counting (how many) and 
other free-form questions. (e.g., "did the toddler in red cry in the video?", "where was this video taken?", "how many kids 
are shown in the video?", "what is this video about?") 
- Instructions for your answer: prompt "C" if causal, "T" if temporal, and "D" if descriptive. 
- Question: {}"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--eval_bs', type=int, default=4)
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

    # Gradio 인터페이스 생성
    import gradio as gr
    from typing import List
    #from PIL.Image import Image
    from PIL import Image
    from utils.utils import image_transform

    image_processor = image_transform(image_size=224)
    
    def inference_wrapper(filepaths, questions, options_a0_input, options_a1_input, options_a2_input, options_a3_input, options_a4_input):
        # filepaths: list of str (파일 경로)
        imgs = [Image.open(fp) for fp in filepaths]
        imgs = torch.stack([image_processor(x) for x in imgs], dim=0)
        imgs = imgs[None, ...] # create batch axis
        assert imgs.shape == (1, 32, 3, 224, 224), f"{imgs.shape=}"
        # result = keyframe_select_inference(imgs, text)
        # keyframes = result["keyframes"]
        # scores = result["scores"]
        # score_table = [[i, s] for i, s in enumerate(scores)]
        # return keyframes, score_table

        """
        "text_input": ['Question: how did the lady in grey direct the dog on what it should do?\nOptions: \nA: pull the leash \nB: brush the dog \nC: caress the dog \nD: carry it in her arm \nE: hand gestures and pointing\nAnswer: ', 'Question: what is the lady in jeans doing?\nOptions: \nA: singing \nB: stand up \nC: lift up her leg \nD: mopping the floor \nE: playing\nAnswer: ', 'Question: why are the group of the people in the mountains?\nOptions: \nA: finding dog \nB: it is winter \nC: hiking trip \nD: scared \nE: campfire\nAnswer: ', 'Question: why did the baby reach for the boy near the end?\nOptions: \nA: happy and laughing \nB: help the baby fix the toy \nC: support the baby \nD: play with it \nE: fall and pounce on the boy\nAnswer: ']
        "text_output": ['E', 'D', 'C', 'E']
        "questions": ['how did the lady in grey direct the dog on what it should do?', 'what is the lady in jeans doing?', 'why are the group of the people in the mountains?', 'why did the baby reach for the boy near the end?']
        "options_a0": ['pull the leash', 'singing', 'finding dog', 'happy and laughing']
        "options_a1": ['brush the dog', 'stand up', 'it is winter', 'help the baby fix the toy']
        "options_a2": ['caress the dog', 'lift up her leg', 'hiking trip', 'support the baby']
        "options_a3": ['carry it in her arm', 'mopping the floor', 'scared', 'play with it']
        "options_a4": ['hand gestures and pointing', 'playing', 'campfire', 'fall and pounce on the boy']
        "answers_text": ['hand gestures and pointing', 'mopping the floor', 'hiking trip', 'fall and pounce on the boy']
        "answers_id": tensor([4, 3, 2, 4])
        """

        # text_input, text_output, questions, options_a0, options_a1, options_a2, options_a3, options_a4 = prepare_inputs(args, data)
        samples = {
                "text_input": [f'Question: {questions}\nOptions: \nA: {options_a0_input} \nB: {options_a1_input} \nC: {options_a2_input} \nD: {options_a3_input} \nE: {options_a4_input}\nAnswer: '],
                "text_output": ['A'],
                "questions": [questions],
                "options_a0": [options_a0_input],
                "options_a1": [options_a1_input],
                "options_a2": [options_a2_input],
                "options_a3": [options_a3_input],
                "options_a4": [options_a4_input],
                # "frame_features": data["frame_features"].cuda(args.local_rank),
                "answers_text": [options_a0_input],
                "answers_id": torch.tensor([0]),
                "types": ['nothing'],
                "pixel_values": imgs.cuda() # [bs, frame_count, 3, 224, 224]
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
            "length_penalty":1,
            "return_dict_in_generate": True,
            "output_scores": True,
            # "output_logits": True,
            }

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32): # Enable autocast before and after
            with torch.no_grad():
                prompts = [q_template.format(q) for q in samples['questions']]
                pred_texts, sequences_scores = model.text_generate(prompts, **generate_kwargs)  # classify the QA type of the given question
                samples['pred_types'] = pred_texts  # append the predicted QA type to the input sample data

                # outputs = model(samples)  # loss 계산 ('loss', 'vqa_loss', 'regression_loss', 'infoNCE_loss')
                pred_texts, sequences_scores = model.generate(samples, **generate_kwargs)  # answer generation

                indicators = model.grounding.indicators  # like [[5, 11, 16, 24]]
                indicators = indicators[0]
                keyframe_imgs = [Image.open(filepaths[idx]) for idx in indicators]

        return pred_texts, keyframe_imgs


    with gr.Blocks() as demo:
        gr.Markdown("## Keyframe Selection Inference Demo")
        with gr.Row():
            # 파일 경로를 반환하도록 type='filepath' 설정
            file_input = gr.File(label="Upload Images", file_count="multiple", type='filepath')
            with gr.Column():
                questions = gr.Textbox(label="Enter a question", value='how do the two man play the instrument')
                options_a0_input = gr.Textbox(label="Option A", value='roll the handle')
                options_a1_input = gr.Textbox(label="Option B", value='tap their feet')
                options_a2_input = gr.Textbox(label="Option C", value='strum the string')
                options_a3_input = gr.Textbox(label="Option D", value='hit with sticks')
                options_a4_input = gr.Textbox(label="Option E", value='pat with hand')
                submit_button = gr.Button("Submit")
            with gr.Column():
                text_output = gr.Textbox(label="Generated Answer", interactive=False)
                gallery_output = gr.Gallery(label="Selected Keyframes")
                # score_output = gr.Dataframe(headers=["Index", "Score"], label="Scores")

        submit_button.click(inference_wrapper, inputs=[file_input, questions, options_a0_input, options_a1_input, options_a2_input, options_a3_input, options_a4_input], outputs=[text_output, gallery_output])

    demo.launch()


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
   
    assert args.use_vit, "args.use_vit must be enabled so that the model expects raw image inputs instead of image features"

    if 't5' in args.model:
        model = Blip2T5Instruct(
            # dtype=torch.bfloat16,
            dtype=torch.float32,
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
    
    missing_keys, unexpected_keys = model.load_state_dict(torch.load('./experiments/t5-xl_nextqa_5_0.0.pth', map_location='cpu'), strict=False)
    # model.load_state_dict(torch.load('./experiments_uniform/t5-xl_nextqa_1_0.0.pth', map_location='cpu'))

    device = torch.device('cuda', args.local_rank)
    init_seeds(args.seed)
    torch.cuda.set_device(device)
    model = model.cuda(args.local_rank)

    print(get_parameter_number(model))
    print("trian_num: ", len(train_dataset), " val_num: ", len(val_dataset),  " test_num: ", len(test_dataset))
    print(args)

    train(args, train_dataset, test_dataset, model)