import heapq
import json
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--extract_feature_model', type=str, default='blip', help='blip/clip/sevila')
    parser.add_argument('--score_path', type=str, default='./outscores/longvideobench/blip/scores.json')
    parser.add_argument('--frame_path', type=str, default='./outscores/longvideobench/blip/frames.json')
    parser.add_argument('--max_num_frames', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--t1', type=int, default=0.8)
    parser.add_argument('--t2', type=int, default=-100)
    parser.add_argument('--all_depth', type=int, default=5)
    parser.add_argument('--output_file', type=str, default='./selected_frames')

    return parser.parse_args()

def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    i = 0
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        i += 1
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score)//2]
            score2 = score[len(score)//2:]
            fn1 = fn[:len(score)//2]
            fn2 = fn[len(score)//2:]                       
            split_scores.append(dict(score=score1, depth=depth+1))
            split_scores.append(dict(score=score2, depth=depth+1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn

    return all_split_score, all_split_fn

def select_keyframes_for_sample(itm_out, fn_out, max_num_frames, ratio, t1, t2, all_depth):
    nums = int(len(itm_out) / ratio)
    if nums == 0 and len(itm_out) > 0:
        new_score = [itm_out[0]]
        new_fnum = [fn_out[0]]
    elif nums == 0 and len(itm_out) == 0:
        new_score = []
        new_fnum = []
    else:
        new_score = [itm_out[num * ratio] for num in range(nums)]
        new_fnum = [fn_out[num * ratio] for num in range(nums)]
    
    score = new_score
    fn = new_fnum
    num = max_num_frames

    if len(score) == 0:
        return []

    if len(score) >= num:
        score_np = np.array(score)
        min_score = np.min(score_np)
        max_score = np.max(score_np)
        if max_score == min_score:
            normalized_data = np.zeros_like(score_np) if len(score_np) > 0 else []
        else:
            normalized_data = (score_np - min_score) / (max_score - min_score)
        
        a, b = meanstd(len(normalized_data), [dict(score=normalized_data, depth=0)], num, [fn], t1, t2, all_depth)
        out_sample = []
        for s, f_segment in zip(a, b):
            if not s['score'].size:
                continue
            f_num_to_select = int(num / 2**(s['depth']))
            if f_num_to_select == 0 and len(f_segment) > 0:
                f_num_to_select = 1
            
            f_num_to_select = min(f_num_to_select, len(s['score']))

            if f_num_to_select > 0:
                topk_indices = heapq.nlargest(f_num_to_select, range(len(s['score'])), s['score'].__getitem__)
                f_nums_selected = [f_segment[t] for t in topk_indices]
                out_sample.extend(f_nums_selected)
        out_sample.sort()
        return out_sample
    else:
        return fn

def select_keyframes_from_scores(itm_outs_all, fn_outs_all, max_num_frames, ratio, t1, t2, all_depth):
    outs = []

    for itm_out, fn_out in zip(itm_outs_all, fn_outs_all):
        selected_frames_sample = select_keyframes_for_sample(itm_out, fn_out, max_num_frames, ratio, t1, t2, all_depth)
        outs.append(selected_frames_sample)
    return outs

def main(args):
    max_num_frames = args.max_num_frames
    ratio = args.ratio
    t1 = args.t1
    t2 = args.t2
    all_depth = args.all_depth

    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    if not os.path.exists(os.path.join(args.output_file, args.dataset_name)):
        os.makedirs(os.path.join(args.output_file, args.dataset_name), exist_ok=True)
    out_score_path = os.path.join(args.output_file, args.dataset_name, args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.makedirs(out_score_path, exist_ok=True)

    selected_frames_list = select_keyframes_from_scores(itm_outs, fn_outs, max_num_frames, ratio, t1, t2, all_depth)

    output_filepath = os.path.join(out_score_path, 'selected_frames.json')
    with open(output_filepath, 'w') as f:
        json.dump(selected_frames_list, f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)