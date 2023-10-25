#like group-vit and denseclip
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import os.path as osp
import argparse
from re import L
import torch
# from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import scipy.io
import PIL.Image
# from skimage.draw import polygon
from pathlib import Path
from PIL import Image
# from sklearn.metrics import jaccard_score as iou
from modules.eval_metrics import mean_iou, mean_dice, intersect_and_union
# from model.ZeroCLIP import CLIPTextGenerator
import clip 
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
import torchvision.transforms as T
from get_args import get_args, draw_part_cosegmentation
from sentence_transformers import SentenceTransformer, util
from collections import Counter,defaultdict
import torch
import torch.nn.functional as F

from utils.utils import compute_cosine_sim,kl_div_uniformtest
from utils.logging import Logger
import sys

from natsort import natsorted
import cv2
from tqdm import tqdm
import pandas as pd

from utils.imagenet_templates import ness_template,full_imagenet_templates as FULL_IMAGENET_TEMPLATES



# legacy
def to_binary_map(segmap, gt=False):
    binary_map = None
    count = 0
    class_id = []
    # print('!! ', segmap.min(),segmap.max())
    for i in range(0,segmap.max()+1):
        # print(i)
        bm = (segmap==i)
        if bm.sum()>0:
            if binary_map is None:
                binary_map = np.expand_dims(bm, axis=0)
            else:
                binary_map = np.concatenate((binary_map, np.expand_dims(bm, axis=0)))
            class_id.append(i-1)
            count += 1
    return binary_map.astype(int), class_id

def to_binary_map_noskip(segmap, gt=False):
    binary_map = None
    count = 0
    class_id = []
    # print('!! ', segmap.min(),segmap.max())
    for i in range(0,segmap.max()+1):
        # print(i)
        bm = (segmap==i)
        
        if binary_map is None:
            binary_map = np.expand_dims(bm, axis=0)
        else:
            binary_map = np.concatenate((binary_map, np.expand_dims(bm, axis=0)))
        class_id.append(i-1)
        count += 1
    return binary_map.astype(int), class_id

def to_parts_img(segmap):
    parts_img = np.zeros_like((segmap[0]))
    for i in range(segmap.shape[0]):
        parts_img[segmap[i]==1] = i+1
    return parts_img



def encode_text(texts,prompts=[],model_name='sbert'):
    global models
    
    if len(prompts) > 0: 
        n_text = len(texts)
        texts = sum([ [ p.format(text) for p in prompts]  for text in texts ],[])
    with torch.no_grad(): 
        if  model_name == 'sbert':
            text_embeddings = models[model_name].encode(texts,convert_to_tensor=True)
        elif model_name == 'clip':
            tokenized_texts = clip.tokenize(texts).cuda()  # .cuda()
            text_embeddings = models[model_name].encode_text(tokenized_texts)
    if len(prompts) > 0:
        text_embeddings = text_embeddings.reshape(n_text,len(prompts),text_embeddings.shape[-1])  # (nt, np, c)
        text_embeddings = text_embeddings.mean(dim=1) # Mean text embeddeding (nt, np, c) --> (nt,c)
#         print(  np.array(texts)[torch.arange(len(texts)).reshape(n_text,len(prompts)).tolist()])
        
    return text_embeddings #.cpu()









parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default="pascal", choices=['coco','pascal','pascal_context_59','pascal_context_459']) # coco, pascal, pascal_context
parser.add_argument('--subset', '-s', type=str, default="val", choices=['val','train']) # coco, pascal, pascal_context

parser.add_argument('--prediction_dir', type=str, default="results/v3/s_114-23_-gb2")
parser.add_argument("--text_encoder",  type=str,  default='sbert', help='[sbert, clip]')





parser.add_argument('--mode', type=str, default="t-t") 
parser.add_argument('--text_thres', type=float, default=None) 
parser.add_argument('--clip_thres', type=float, default=None) 
parser.add_argument('--max_idx', type=int, default=None) 


parser.add_argument("--apply_merge",   default=True, action=argparse.BooleanOptionalAction )
parser.add_argument("--apply_grounding",   default=False, action=argparse.BooleanOptionalAction )



parser.add_argument('--prompt_opt', type=str, default='mini', choices=['mini','imagenet','ness']) 







parser.add_argument('--output_dir', type=str, default="results_eval/",
                    help="Path to the output directory of instance maps")
args = parser.parse_args()


if args.dataset == 'coco':
    image_dir = "../data/coco/images/val/val2017/"
    input_label_dir = "../data/coco/stuffthingmaps_trainval2017/val2017/"
    labels_dir = "../data/coco/stuffthingmaps_trainval2017/labels.txt"
    class_emb_dir = '../data/coco/label_embeddings.pt' 
elif args.dataset == 'pascal':
    if args.subset == 'val': image_dir = "../data/voc2012/val/image_val/"
    else: image_dir = "../data/voc2012/train/image_train/"
    input_label_dir = "../data/voc2012/VOC2012/SegmentationClass/"
    labels_dir = "../data/voc2012/labels.txt"
    class_emb_dir = '../data/voc2012/label_embeddings_withprompt.pt'
elif 'pascal_context' in args.dataset:
    if args.subset == 'val': image_dir = "../data/pascal_context/val/images/"
    elif args.subset == 'train': image_dir = "../data/pascal_context/train/images/"
    
    input_label_dir = "../data/pascal_context/segment_anno/"
    labels_dir = "../data/pascal_context/labels.txt" # 459
    labels_dir_59 = "../data/pascal_context/labels_59.txt"




else: print('unknown dataset'); exit()

class_emb_dir = None

print("prediction_dir at {}".format(args.prediction_dir))

# opening the file in read mode
all_cls_names = []; clsname2ids = {}; id2clsname = {}
if labels_dir is not None:
    with open(labels_dir,'r') as f:
        for l in f.read().splitlines():
            cls_id,cls_name = l.split(': ')
            cls_id = int(cls_id)
            all_cls_names += [cls_name]
            clsname2ids[cls_name] = cls_id
            id2clsname[cls_id] = cls_name


#

# unlabeled_classes =  [ 'unlabeled', 'ambiguous' ]  + ['bg','amb']


unlabeled_classes =  [ 'unknown']   + ['bg','amb']


if args.dataset == 'pascal' :
    # bg is also unlabeled class in pascal
    # clsname2ids['unlabeled'] = 0
    # clsname2ids['ambiguous'] = 255

    clsname2ids['bg'] = 0
    clsname2ids['amb'] = 255



    id2clsname = { cid: cls for cls,cid in clsname2ids.items()  }


    all_cls_names =  sorted(list(clsname2ids.keys()))
    clsname2idx = { cls: idx for idx,cls in enumerate(all_cls_names)}

if 'pascal_context'in  args.dataset :
    if  args.dataset == 'pascal_context_59' :
        print('using 59 classes')
        selected_cls_names_59 = []
        with open(labels_dir_59,'r') as f:
            for l in f.read().splitlines():
                cls_id,cls_name = l.split(': ')
                cls_id = int(cls_id)
                selected_cls_names_59 += [cls_name]
        clsname2ids['unknown'] = 431  # 431 ---> 999
        for cls in list(clsname2ids.keys()):
            # 'unknown' == 431   is not in 59 in the first place
            if cls not in selected_cls_names_59: 
                # clsname2ids[cls]  = clsname2ids['unknown']
                unlabeled_classes += [cls]
                del clsname2ids[cls]
        
    else: print('using 459 classes')

    id2clsname = { cid: cls for cls,cid in clsname2ids.items()  }
    
    all_cls_names =  sorted(list(clsname2ids.keys()))
    clsname2idx = { cls: idx for idx,cls in enumerate(all_cls_names)}




if args.dataset == 'coco' :
    # add background
    # coco  0: unlabeled
    clsname2ids = { cls : cid-1 for cls,cid in clsname2ids.items() }
    clsname2ids['unlabeled'] = 255


    id2clsname = { cid: cls for cls,cid in clsname2ids.items()  }

    all_cls_names =  sorted(list(clsname2ids.keys()))
    clsname2idx = { cls: idx for idx,cls in enumerate(all_cls_names)}

filtered_cls_names = [ c for c in all_cls_names if c not in unlabeled_classes ]

print(f'class name: {all_cls_names}')
print(f'cls id: {id2clsname}')
print(len(all_cls_names),len(all_cls_names))

# prompt selection
if args.prompt_opt == 'mini':
    prompts = ['Image of a {}.']
elif  args.prompt_opt == 'imagenet':
    prompts = FULL_IMAGENET_TEMPLATES
elif args.prompt_opt == 'ness':
    prompts = ness_template
print(f'prompt:\n{prompts}')






device = 'cuda'
models = {}
models['clip'],_ = clip.load('ViT-L/14@336px', device)
models['sbert'] = SentenceTransformer('all-mpnet-base-v2')


models['sbert'].eval()


models['clip'].to(device)
models['sbert'].to(device)


log_feats = {}
if class_emb_dir is None: 

    # no prompt
    # gt_class_feats = torch.stack([encode_text(cls,model_name=args.text_encoder) for cls in all_cls_names]).to('cpu')
    gt_class_feats = encode_text(all_cls_names,model_name=args.text_encoder)

    log_feats['all_gt_class_feats_sbert'] = encode_text(all_cls_names,model_name='sbert')
    log_feats['all_gt_class_feats_clip'] = encode_text(all_cls_names,model_name='clip')


    gt_class_feats_with_prompt_clip = encode_text(all_cls_names,prompts=prompts,model_name='clip')

prediction_log_dir = osp.join(args.prediction_dir,'log')
prediction_log_paths = natsorted([ osp.join(prediction_log_dir, p) for p in os.listdir(prediction_log_dir) ] )

if args.mode == 'textsim':
    mask_and_crop_prediction_log_dir  = 'results_data/context/gt_mask/459_crop_and_mask/log'
    mask_and_crop_prediction_log_paths = natsorted([ osp.join(prediction_log_dir, p) for p in os.listdir(prediction_log_dir) ] )

# create output directory
if not osp.exists(args.output_dir ): os.makedirs(args.output_dir,exist_ok=True)

output_fname =  osp.basename(args.prediction_dir) if    osp.basename(args.prediction_dir) else   osp.basename(osp.dirname(args.prediction_dir)) # '_'.join(args.prediction_dir.split('/')) 

sys.stdout = Logger(osp.join(args.output_dir, f'{output_fname}_log.txt'),append=False,overwrite=True)  #TODO:  overwrite as a config 

print(f"run log will be saved at  {osp.join( args.output_dir, f'{output_fname}_log.txt')  }  ")
print(f"result will be saved at: {osp.join( args.output_dir, f'{output_fname}.txt')  } ")


exp_tags = []
# thres = [ '0.1','0.5','0.9']
thres = [ '0.9']

exp_tags += thres
exp_tags += [ f'{t}_merge' for t in  thres ]
# exp_tags += [ f'{t}_all' for t in  thres ]
exp_tags += [ f'{t}_mergeR' for t in  thres ]



# exp_tags += [ f'{t}_all' for t in exp_tags]
# exp_tags += [ f'{t}_merge' for t in exp_tags]

# exp_tags = [ '0.1','0.1_merge']

# exp_tags = ['0.1','0.3', '0.5', '0.7','0.9']

print(f'experiment threshold tag: {exp_tags} ')


scores_overall = defaultdict() # overall experiment score 

scores_images = defaultdict(lambda: defaultdict(list)) # 1 img/ 1 value

scores_images_thres =  defaultdict(lambda: defaultdict((lambda: defaultdict(list))))


# [exp_tag][rep][cls][TP/FN][thres]
recall_global_stats = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: defaultdict(lambda: defaultdict(int))))))

# [exp_tag][rep][iou][cls]
per_class_stats = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: defaultdict(list)))))


# scores_images = defaultdict(list)



    
# for recall
decimal_precision = 0.01
min_iou_thres = 0.0; max_iou_thres = 1.00
iou_thresholds = np.arange(min_iou_thres,max_iou_thres+decimal_precision,decimal_precision)



gt_clip_scores_dist = [] 

with torch.no_grad():


    prediction_log_paths = prediction_log_paths[:args.max_idx]  if args.max_idx is not None else prediction_log_paths

    num_prediction = len(prediction_log_paths) # if it's not finish, this will be less than max_idx

    for idx, pred_log_path in tqdm(enumerate(prediction_log_paths),total=len(prediction_log_paths)):
        # load prediction log

        img_name = osp.basename(pred_log_path).split('.')[0]
        print('\nimage: ', img_name)


        prediction_log = torch.load(pred_log_path)

        if args.mode == 'textsim':

            tag = 'gt'
            # print(prediction_log.keys())
            if len(prediction_log.keys())==0: continue

            tag = '0.9'  # hack
            pred_open_vocabs = prediction_log[tag]['captions'] 

            # pred_open_vocabs = prediction_log['all']['captions'] 
            gt_class = prediction_log['gt_cls']


            # filter invalid class
            # invalid_class = ['bg','amb', 'unknown', 'unlabeled']

            # old run
            # if args.dataset == 'pascal' and gt_class[-1] != 'amb': gt_class += ['amb']

            is_valid_class =  [  not (c in  unlabeled_classes) for c in gt_class   ]
            pred_open_vocabs = np.array(pred_open_vocabs)[is_valid_class].tolist()
            gt_class = np.array(gt_class)[is_valid_class].tolist()
            

            

            for model_name in ['sbert','clip']:
                if model_name == 'sbert': thres = 0.5
                # elif model_name == 'clip': thres = 0.1
                pred_open_vocab_feats = encode_text(pred_open_vocabs,model_name=model_name)

                gt_class_feats = encode_text(gt_class,model_name=model_name)
    
                # cosine sim
                text_similarity = compute_cosine_sim(pred_open_vocab_feats,gt_class_feats,pairwise=True) 
                if model_name == 'sbert':
                    text_similarity[text_similarity>=thres] = 1
                    text_similarity[text_similarity<thres] = 0
                scores_images[tag][f'text_similarity_{model_name}'] += [text_similarity.mean().item()]

                print(f'text similarity {model_name}')
                for i,(pred_ov,gt_cls) in enumerate(zip(pred_open_vocabs,gt_class)):  print(f'{pred_ov} --> {gt_cls} : {text_similarity[i]}')

                # top1 acc
                all_gt_cls_feats= log_feats[f'all_gt_class_feats_{model_name}']

                text_similarity_all = compute_cosine_sim(pred_open_vocab_feats,all_gt_cls_feats) 
                pred_class_indices = torch.argmax(text_similarity_all,dim=1)  # top 1



                # print(pred_class_indices)
                # print( torch.tensor([ clsname2ids[cls] for cls in gt_class ] ))
                # print( pred_class_indices == torch.tensor([ clsname2ids[cls] for cls in gt_class ] ))
                scores_images[tag][f'top1_acc_{model_name}'] +=  [  ( pred_class_indices == torch.tensor([ clsname2ids[cls] for cls in gt_class ] ).to(device)   ).int().sum().item()/len(gt_class) ]
                
                if model_name == 'clip': 
                    
                    # for clip score:  always use crop and mask img_feat 
                    # mask_and_crop_prediction_log_path = mask_and_crop_prediction_log_paths[idx]
                    # assert osp.basename(mask_and_crop_prediction_log_path) == osp.basename(pred_log_path)
                    # crop_and_mask_prediction_log = torch.load(mask_and_crop_prediction_log_path)

                    # crop_and_mask_img_feats = torch.stack(crop_and_mask_prediction_log['all']['img_feats'])[is_valid_class].to(device)
                    
                    img_feats = torch.stack(prediction_log[tag]['img_feats'])[is_valid_class].to(device)
                    # img_feats = torch.stack(prediction_log['all']['img_feats'])[is_valid_class].to(device)


                    # pred_open_vocab_feats_with_prompt = encode_text(pred_open_vocabs,model_name=model_name,prompts=prompts)
                    gt_class_feats_with_prompt = encode_text(gt_class,model_name='clip',prompts=prompts)

                    # clip_scores = compute_cosine_sim(pred_open_vocab_feats_with_prompt,crop_and_mask_img_feats,pairwise=True) 
                    clip_scores = compute_cosine_sim(gt_class_feats_with_prompt,img_feats,pairwise=True) 

                    gt_clip_scores_dist += [  (gt_class[i],clip_scores[i].item() ) for i in range(len(gt_class) )    ]

                    for i in range(len(gt_class) ):
                        print((gt_class[i],clip_scores[i].item() ) )


                    
                    print(f'clip score')
                    for i,(pred_ov,gt_cls) in enumerate(zip(pred_open_vocabs,gt_class)):  print(f'{pred_ov} --> {gt_cls} : {clip_scores[i]}')


                    scores_images[tag][f'clip_score']  += [clip_scores.mean().item()]

                    # ref clip  ==   Hmean (  2.5 * max( clip_score, 0) , max( text_similarity, 0 )  )
                    w = 2.5

                    clip_scrores_c =  w* torch.clip(clip_scores,0)
                    text_similarity_c = torch.clip(text_similarity,0)
                    scores_images[tag][f'clip_score_ref'] +=   [ ( (2 * clip_scrores_c * text_similarity_c) / (clip_scrores_c + text_similarity_c) ).mean().item() ] # harmonic mean 


                    for k in ['all',3,10]:
                        sorted_indices = torch.argsort(clip_scores,descending=True).tolist()
                        topk_clip_scores = clip_scores[sorted_indices] if k == 'all' else  clip_scores[sorted_indices[:k]] 
                        scores_images[tag][f'kl.{k}'] += [kl_div_uniformtest(topk_clip_scores).cpu().item()]
                        scores_images[tag][f'kl.{k}/1'] += [kl_div_uniformtest(topk_clip_scores/topk_clip_scores[0]).cpu().item()]




                print(f'summary score for {img_name}')
                for score_name in  scores_images[tag]:
                    print(f'{score_name} : {scores_images[tag][score_name][-1]} ')
                print(20*'#')


            continue





        # load image
        image = io.imread(image_dir + (img_name + '.jpg'))
        pil_image = PIL.Image.fromarray(image).convert('RGB')#.resize((parts_img.shape[1],parts_img.shape[0]))


        # load grountruth mask
        if args.dataset == 'coco':
            gt_mask = Image.open(input_label_dir + img_name + '.png').convert('L')
            
        elif args.dataset == 'pascal': 
            gt_mask = Image.open(input_label_dir + img_name + '.png')     
            # gt_mask[gt_mask==255] = 0 # edge --> ambiguous region
        elif 'pascal_context' in args.dataset:
            gt_mask = scipy.io.loadmat(input_label_dir + img_name + '.mat')['LabelMap']
            # gt_mask[gt_mask==255] = 0  # edge --> ambiguous region
        
        gt_mask = np.array(gt_mask).astype(np.int32)
        gt_mask = torch.tensor(gt_mask,device=device)


      
        
        gt_cls_within_img =  [ id2clsname[c_id] for c_id in  Counter( torch.tensor(gt_mask).flatten().tolist()).keys() if ( c_id in id2clsname and  id2clsname[c_id] not in unlabeled_classes ) ] # 255 == background
        t =  [ clsname2ids[c] for c in gt_cls_within_img ]

        # gt_cls_idx_within_img = [ clsname2idx[c] for c in gt_cls_within_img ]  # for indexing features 


        # gt_masks, class_ids = to_binary_map(gt_mask)

        # print(Counter(torch.tensor(gt_mask).flatten().tolist() ))
        for exp_tag in exp_tags:
            if not exp_tag in prediction_log: continue
            if len(prediction_log[exp_tag]['captions']) == 0: 
                print(f'empty log: {exp_tag} .. skip')
                continue


            print(f'exp_tag : {exp_tag}')

            pred_masks = prediction_log[exp_tag]['masks'].to(device)
            pred_open_vocabs = prediction_log[exp_tag]['captions']

            # area of each mask 

            if pred_masks.ndim == 2 : maskid_area = { i : len(torch.where(pred_masks==i)[0]) for i in range(int(pred_masks.max()+1)) }
            elif pred_masks.ndim == 3:  maskid_area = {  i : pred_masks[i].sum().item() for i in range(len(pred_masks)) }
            areas = torch.tensor([ maskid_area[i] for i in sorted(maskid_area.keys()) ])


            # infer with no prompt
            print(pred_open_vocabs)
            pred_open_vocab_feats =  encode_text(pred_open_vocabs,model_name=args.text_encoder) # torch.randn(len(pred_open_vocabs),768)  # 
           
            # selected class name 
            if args.apply_grounding: cls_names = gt_cls_within_img
            else: cls_names = filtered_cls_names  # eg: no bg and ambiguous class

            cls_name_idx = [ clsname2idx[c] for c in  cls_names  ] # for sorted
            
            new_clsname2idx = {  c: i  for i, c in enumerate(cls_names)   } 



            print(f'pred_open_vocabs: {pred_open_vocabs}')
            print(f'gt_cls_within_img: {gt_cls_within_img}')
            
            
            img_feats = torch.stack(prediction_log[exp_tag]['img_feats']).to(device)

            # assign classes to masks (zero-shot transfer: aligning either pred_open_vocab (t-t) or visual mask (i-t) with all the class labels )
            if args.mode == 't-t':
                class_feats =  gt_class_feats[ cls_name_idx  ]
                similarity_mat = compute_cosine_sim(pred_open_vocab_feats,class_feats )

            elif args.mode == 'i-t':
                class_feats =  gt_class_feats_with_prompt_clip[ cls_name_idx  ]
                similarity_mat = compute_cosine_sim(img_feats,class_feats)

            pred_class_indices = torch.argmax(similarity_mat,dim=1).tolist()

            predcls_mask_ids = defaultdict(list)

            # gt label assignment & filter out by threshold ( lower than threshold --> assume unlabel )
            for i,idx in enumerate(pred_class_indices): 
                if args.mode == 't-t' and args.text_thres is not None and  similarity_mat[i][idx] < args.text_thres: 
                    print('less than threshold')
                    print(f'open: {pred_open_vocabs[i]} --->  {cls_names[idx]} text score: {similarity_mat[i][idx]} <<  { args.text_thres}')
                    continue
                if args.mode == 'i-t' and args.clip_thres is not None and  similarity_mat[i][idx] < args.clip_thres: continue 
                print(f'open: {pred_open_vocabs[i]} --->  {cls_names[idx]} text score: {similarity_mat[i][idx]} ')
                predcls_mask_ids[cls_names[idx]] +=  [i]


            
            # final mask of that class that will be used in evaluation
            tag_predcls2bestmaskid =  defaultdict(dict);  
            for rep_tag in ['union','argmax']:
                tag_predcls2bestmaskid[rep_tag] = {}  # so that there will always be penalty if there's no one pass threshold 
                
            for pred_cls,mask_ids in predcls_mask_ids.items():
                # merge every masks that passes threshold
                if args.apply_merge:
                    tag_predcls2bestmaskid['union'][pred_cls] = mask_ids # predcls_bestmask_id = mask_ids
                # argmax



                
                # argmax 
                if  args.mode == 'i-t': 
                    gt_cls_feat_with_prompt_clip = gt_class_feats_with_prompt_clip[new_clsname2idx[pred_cls]] # with prompt
                    # gt_cls_feat_with_prompt_clip = gt_class_feats_with_prompt_clip[clsname2idx[pred_cls]] # with prompt
                    clip_scores =   compute_cosine_sim( gt_cls_feat_with_prompt_clip[None], img_feats[mask_ids])[0]

                    predcls_bestmask_idx = torch.argmax(clip_scores).item()  

                    # if even the best clip score is less than the clip threshold ---> discard the prediction 
                    best_score = clip_scores[predcls_bestmask_idx]
                    if args.clip_thres is not None and best_score < args.clip_thres: 
                        # print(f'{pred_cls} : clip_score {clip_scores[predcls_bestmask_idx][0]}')
                        continue
                    n_best = (clip_scores == best_score).int().sum()
                    if  n_best > 1 : 
                        print('argmax with equal similarity --> select the largest mask')
                        area_map =  torch.tensor([ area    if score == best_score   else -1  for area,score in  zip(areas[mask_ids],clip_scores )  ])
                        predcls_bestmask_idx = torch.argmax(area_map).item()
                    
                elif args.mode == 't-t':
                    # gt_cls_feat = class_feats[clsname2idx[pred_cls]]
                    gt_cls_feat = class_feats[new_clsname2idx[pred_cls]]

                    text_sim = compute_cosine_sim(gt_cls_feat[None], pred_open_vocab_feats[mask_ids])[0]
                    predcls_bestmask_idx = torch.argmax(text_sim).item()

    
                    best_score = text_sim[predcls_bestmask_idx]
                    if best_score < args.text_thres:
                        continue

                    n_best = (text_sim == best_score).int().sum().item()
                    if  n_best > 1 : 
                        print('argmax with equal similarity --> select the largest area mask')
                        area_map =  torch.tensor([ area    if score == best_score   else -1  for area,score in  zip(areas[mask_ids],text_sim )  ])
                        predcls_bestmask_idx = torch.argmax(area_map).item()

                tag_predcls2bestmaskid['argmax'][pred_cls] = mask_ids[predcls_bestmask_idx]

            # within img score
            recall_stats = defaultdict(lambda: defaultdict(int) ) # { 'thres' : 'TP/FN': num } # within img score
            tag2scores_within_img = defaultdict(lambda: defaultdict(list) )
            for rep_tag,predcls2bestmaskid in tag_predcls2bestmaskid.items():
                print(f'#### represent: {rep_tag}  ####')
                # set of grounding class union predicted class 
                for cls in set(gt_cls_within_img + list(predcls2bestmaskid.keys())):
                    # print(f'cls: {cls}')


                    # ground truth segment pre-process
                    gt_class_id = clsname2ids[cls]
                    
                    gt_mask_bi = (gt_mask==gt_class_id).int()
                    
                    gt_amb_bi = torch.full(gt_mask_bi.shape,False,device=device)           
                    # for amb_cls in unlabeled_classes:
                    #     if amb_cls in clsname2ids:
                    #         amb_id = clsname2ids[amb_cls]
                    #         gt_amb_bi = gt_amb_bi | (gt_mask==amb_id).to(device)
 
                    if 'amb' in clsname2ids:
                        amb_id = clsname2ids['amb']
                        gt_amb_bi = (gt_mask==amb_id).to(device)

                    # # if cls not in grounding cls --> not consider the unlabeled part in IoU computation
                    if cls not in gt_cls_within_img:
                        # bg for pascal2012, unlabeled for pascal context
                        for unlabeled_cls in  ['bg']:
                            if unlabeled_cls in clsname2ids:
                                unlabel_id = clsname2ids[unlabeled_cls]
                                gt_amb_bi = gt_amb_bi | (gt_mask==unlabel_id).to(device)




                    # predicted segments pre-process
                    if cls in predcls2bestmaskid:
                        mask_ids = predcls2bestmaskid[cls]
                        # pred_masks = prediction_log[exp_tag]['masks'].to(device)
                        # many masks (combine, mergez: just union)
                        if isinstance(mask_ids,list):
                            # np.logical_or.reduce
                            if  pred_masks.ndim == 3: 
                                # for all tag  ( usually over-lap segments)
                                pred_mask_bi = torch.any(pred_masks[mask_ids].bool(), dim=0 ) # merge 
                            else:
                                pred_mask_bi = torch.isin(pred_masks, torch.tensor(mask_ids).to(device))    
                            pred_mask_bi = pred_mask_bi.int()
    
                        else:  
                            mask_id = mask_ids
                            if isinstance ( pred_masks ,list) or pred_masks.ndim == 3 : 
                                pred_mask_bi =  pred_masks[mask_id]
                            else: 
                                pred_mask_bi = (pred_masks==mask_id).int()
                    else: 
                        pred_mask_bi  = torch.zeros(gt_mask_bi.shape,device=device,dtype=torch.uint8)

                    # Mode mode='nearest' matches buggy OpenCV’s INTER_NEAREST interpolation algorithm.
  
                    pred_mask_bi = F.interpolate(pred_mask_bi[None][None].float(), size=gt_mask_bi.shape, mode='nearest')[0][0]

                    pred_mask_bi = torch.where(gt_amb_bi,0,pred_mask_bi)

                    pred_mask_bi = pred_mask_bi.cpu() #.numpy()
                    gt_mask_bi = gt_mask_bi.cpu() #.numpy()
                    
                    # index 1 is the interested cls ( 0 is not the focus)
                    tag2scores_within_img[rep_tag]['iou'] += [ mean_iou([pred_mask_bi],[gt_mask_bi],num_classes= 2, ignore_index=None)['IoU'][1]]
                    tag2scores_within_img[rep_tag]['dice'] += [ mean_dice([pred_mask_bi], [gt_mask_bi],num_classes=2,ignore_index= None)['Dice'][1] ]
                    
                    
                    iou_score = tag2scores_within_img[rep_tag]['iou'][-1]

                    area_intersect, area_union, area_pred_label, area_label = intersect_and_union(pred_mask_bi,gt_mask_bi,num_classes=2,ignore_index=None)
                    area_gt, _, _, _ = intersect_and_union(gt_mask_bi,gt_mask_bi,num_classes=2,ignore_index=None) # hack area gt
                    area_pred, _, _, _ = intersect_and_union(pred_mask_bi,pred_mask_bi,num_classes=2,ignore_index=None) # hack area gt



                    ioa_gt_score = area_intersect[1]/area_gt[1]
                    ioa_pred_score = area_intersect[1]/area_pred[1]


                    print(f" {cls} - iou  : {iou_score}")
                    print(f" {cls} - ioa_gt  : {ioa_gt_score}")
                    print(f" {cls} - ioa_pred  : {ioa_pred_score}")

                    tag2scores_within_img[rep_tag]['ioa_gt'] += [ioa_gt_score]
                    tag2scores_within_img[rep_tag]['ioa_pred_score'] += [ioa_pred_score]




                    # compute Recall


                    per_class_stats[exp_tag][rep_tag]['iou'][cls] += [iou_score]

                    # threshold 
                    for iou_thres in iou_thresholds:
                        if iou_score >= iou_thres:
                            recall_stats['TP'][iou_thres] += 1 
                            recall_global_stats[exp_tag][rep_tag][cls]['TP'][iou_thres] += 1 # class stats
                            
                        # if T-T or S-T <= Thres ---> IoU is 0 automatically here 
                        else: 
                            recall_stats['FN'][iou_thres] += 1 
                            recall_global_stats[exp_tag][rep_tag][cls]['FN'][iou_thres] += 1 # class stats

                        
                # tag2scores_within_img[rep_tag]['TP'] = [TP]  # sum of TP within images # 1 score already , mean of 1 elements == itself
                # tag2scores_within_img[rep_tag]['FN'] = [FN] 

                # single image score recall :  mean of 1 datapoint == itself
                tag2scores_within_img[rep_tag]['recall'] =  [ np.mean( [ recall_stats['TP'][iou_thres]/(recall_stats['TP'][iou_thres]+recall_stats['FN'][iou_thres])   for iou_thres in iou_thresholds ]) ]
                # tag2scores_within_img[rep_tag]['recall_ioa'] =  [ np.mean( [ recall_stats['TP_ioa'][iou_thres]/(recall_stats['TP_ioa'][iou_thres]+recall_stats['FN_ioa'][iou_thres])   for iou_thres in iou_thresholds ]) ]

            

            # compute recall (no rep: no union, no argmax, can be used with argmax iou ) (use each segment from DINO directily)
            # compute using all segments from DINO: if one of the predicted class have IOU >= iou_thres ---> TP =  1 else 0 
            is_compute_recall = True

            recall_stats = defaultdict(lambda: defaultdict(int) ) # re init
            if is_compute_recall  : # and  ('merge' not in exp_tag) :  

                # set of grounding class union predicted class 
                for cls in set(gt_cls_within_img + list(predcls_mask_ids.keys())):
                    # print(f'cls: {cls}')


                    # ground truth segment pre-process
                    gt_class_id = clsname2ids[cls]
                    
                    gt_mask_bi = (gt_mask==gt_class_id).int()
                    
                    gt_amb_bi = torch.full(gt_mask_bi.shape,False,device=device)
                    # for amb_cls in unlabeled_classes:
                    #     if amb_cls in clsname2ids:
                    #         amb_id = clsname2ids[amb_cls]
                    #         gt_amb_bi = gt_amb_bi | (gt_mask==amb_id).to(device)

                    if 'amb' in clsname2ids:
                        amb_id = clsname2ids['amb']
                        gt_amb_bi = (gt_mask==amb_id).to(device)


                    # if cls not in grounding cls --> not consider the unlabeled part in IoU computation
                    if cls not in gt_cls_within_img:
                        # bg for pascal2012, unlabeled for pascal context
                        for unlabeled_cls in  ['bg']:
                            if unlabeled_cls in clsname2ids:
                                unlabel_id = clsname2ids[unlabeled_cls]
                                gt_amb_bi = gt_amb_bi | (gt_mask==unlabel_id).to(device)



                    # predicted segments pre-process


                    # per class per img
                    iou_scores = []; ioa_gt_scores = []; ioa_pred_scores =[]

                    if cls in predcls_mask_ids:
                        mask_ids = predcls_mask_ids[cls]
                        
                        for mask_id in mask_ids:
                            if isinstance ( pred_masks ,list) or pred_masks.ndim == 3 : 
                                pred_mask_bi =  pred_masks[mask_id]
                            else: 
                                pred_mask_bi = (pred_masks==mask_id).int()

                            pred_mask_bi = F.interpolate(pred_mask_bi[None][None].float(), size=gt_mask_bi.shape, mode='nearest')[0][0]

                            pred_mask_bi = torch.where(gt_amb_bi,0,pred_mask_bi)
                            pred_mask_bi = pred_mask_bi.cpu() #.numpy()
                            gt_mask_bi = gt_mask_bi.cpu() #.numpy()

                            iou_scores +=  [ mean_iou([pred_mask_bi],[gt_mask_bi],num_classes= 2, ignore_index=None)['IoU'][1] ]
                            
                            

                            area_intersect, area_union, area_pred_label, area_label = intersect_and_union(pred_mask_bi,gt_mask_bi,num_classes=2,ignore_index=None)
                            area_gt, _, _, _ = intersect_and_union(gt_mask_bi,gt_mask_bi,num_classes=2,ignore_index=None) # hack area gt
                            area_pred, _, _, _ = intersect_and_union(pred_mask_bi,pred_mask_bi,num_classes=2,ignore_index=None) # hack area gt


                            ioa_gt_scores += [area_intersect[1]/area_gt[1]]
                            ioa_pred_scores += [area_intersect[1]/area_pred[1]]




                    else: 
                        # those that does not pass alignment threshold
                        # -1 is definitely < 0  --> won't pass even if thres_iou = 0
                        iou_scores += [-1]
                        ioa_gt_scores += [-1]
                        ioa_pred_scores += [-1]

                        print(cls)
                        print(gt_cls_within_img)

                    # only max is fine  (argmax iou ) 
                    max_iou_scores = max( iou_scores )
                    max_ioa_gt_scores = max( ioa_gt_scores )
                    max_ioa_pred_scores = max( ioa_pred_scores )

                    for iou_thres in iou_thresholds:
                        if max_iou_scores >= iou_thres  : 
                            recall_stats['TP'][iou_thres] += 1 
                            recall_global_stats[exp_tag]['allseg'][cls]['TP'][iou_thres] += 1 # class stats
                        # if T-T or S-T <= Thres ---> IoU is 0 automatically here 
                        else: 
                            recall_stats['FN'][iou_thres] += 1 
                            recall_global_stats[exp_tag]['allseg'][cls]['FN'][iou_thres] += 1 # class stats

                        if max_ioa_gt_scores >= iou_thres  : 
                            recall_stats['TP_ioa_gt'][iou_thres] += 1 
                            recall_global_stats[exp_tag]['allseg'][cls]['TP_ioa_gt'][iou_thres] += 1 # class stats
                        # if T-T or S-T <= Thres ---> IoU is 0 automatically here 
                        else: 
                            recall_stats['FN_ioa_gt'][iou_thres] += 1 
                            recall_global_stats[exp_tag]['allseg'][cls]['FN_ioa_gt'][iou_thres] += 1 # class stats
                        
                        if max_ioa_pred_scores >= iou_thres :
                            recall_stats['TP_ioa_pred'][iou_thres] += 1 
                            recall_global_stats[exp_tag]['allseg'][cls]['TP_ioa_pred'][iou_thres] += 1 # class stats
                        # if T-T or S-T <= Thres ---> IoU is 0 automatically here 
                        else: 
                            recall_stats['FN_ioa_pred'][iou_thres] += 1 
                            recall_global_stats[exp_tag]['allseg'][cls]['FN_ioa_pred'][iou_thres] += 1 # class stats                         


                tag2scores_within_img['allseg']['recall_avgthres'] =  [ np.mean( [ recall_stats['TP'][iou_thres]/(recall_stats['TP'][iou_thres]+recall_stats['FN'][iou_thres])   for iou_thres in iou_thresholds ]) ]

                tag2scores_within_img['allseg']['recall_area'] =  [ np.trapz( [ recall_stats['TP'][iou_thres]/(recall_stats['TP'][iou_thres]+recall_stats['FN'][iou_thres])   for iou_thres in iou_thresholds ] , dx= decimal_precision) ]
                tag2scores_within_img['allseg']['recall_ioa_gt_area'] =  [ np.trapz( [ recall_stats['TP_ioa_gt'][iou_thres]/(recall_stats['TP_ioa_gt'][iou_thres]+recall_stats['FN_ioa_gt'][iou_thres])   for iou_thres in iou_thresholds ] , dx= decimal_precision) ]
                tag2scores_within_img['allseg']['recall_ioa_pred_area'] = [ np.trapz( [ recall_stats['TP_ioa_pred'][iou_thres]/(recall_stats['TP_ioa_pred'][iou_thres]+recall_stats['FN_ioa_pred'][iou_thres])   for iou_thres in iou_thresholds ]) ]



                for iou_thres in iou_thresholds:
                    tag2scores_within_img['allseg'][f'recall_T{iou_thres:.2f}'] = recall_stats['TP'][iou_thres]/(recall_stats['TP'][iou_thres]+recall_stats['FN'][iou_thres])
                    tag2scores_within_img['allseg'][f'recall_ioa_gt_T{iou_thres:.2f}'] = recall_stats['TP_ioa_gt'][iou_thres]/(recall_stats['TP_ioa_gt'][iou_thres]+recall_stats['FN_ioa_gt'][iou_thres])
                    tag2scores_within_img['allseg'][f'recall_ioa_pred_T{iou_thres:.2f}'] = recall_stats['TP_ioa_pred'][iou_thres]/(recall_stats['TP_ioa_pred'][iou_thres]+recall_stats['FN_ioa_pred'][iou_thres])





            # 1 img/ 1 score 
            for rep_tag,scores_within_img in tag2scores_within_img.items():
                for i, (score_name,scores) in enumerate(scores_within_img.items()):
                    
                    # img_score = np.array( scores ).mean()  if scores is not None else float('nan')
                    img_score = np.nanmean(scores ) # ignore nan
                    img_score_med = np.nanmedian(scores ) # ignore nan


                    score_name_tag = f"D{exp_tag}_{args.mode}" # exp_tag

                    score_name_tag_thres = f"{args.mode}" # no threshold specified within tag

                    if  args.mode == 't-t':
                        score_name_tag += f'_T{args.text_thres}'
                        score_name_tag_thres += f'_T{args.text_thres}'

                    if rep_tag == 'union':
                        score_name_tag += '_U'
                        score_name_tag_thres += '_U'

                    elif rep_tag == 'argmax':
                        score_name_tag += f'_C{args.clip_thres}'
                        score_name_tag_thres += f'_C{args.clip_thres}'

                    elif rep_tag == 'allseg':
                        score_name_tag += '_A'
                        score_name_tag_thres += '_As'
                    
                    scores_images[score_name_tag][score_name] +=  [ img_score ] 
                    scores_images_thres[score_name_tag_thres][score_name][exp_tag] += [img_score]


                    # scores_images_thres[score_name_tag_thres][f'{score_name}_med'][exp_tag] += [img_score_med]


                    # do not print threshold or select only specific thresholds
                    if 'recall' in score_name: continue

                    print(f'{score_name_tag}-{score_name} : {img_score}')

            print(20*'#' +'\n'+ 20*'#' + '\n' )     


# print('Threshold selection')
# anchor_scorenames =   [] # ['iou']  
# compute_kl = False                 
# if compute_kl:
#     anchor_scorenames += [f'kl{k}' for k in kl_topk] + [f'kl{k}/1' for k in kl_topk]   + ['avgclip1']
#     anchor_scorenames += [f'kl{k}_med' for k in kl_topk] + [f'kl{k}/1_med' for k in kl_topk]  + ['avgclip1_med']

anchor_scorenames =  [] #['iou']
print(f'anchor_scorenames: {anchor_scorenames}') 
for overall_tag,tag2scorenames in scores_images_thres.items():
    print(f'overall tag: {overall_tag}') # eg D0.1_t-t_T0.3_C0.15
    if '_U' in overall_tag:
        print(f'representation tag: {overall_tag} is not supported for computed KL')
        continue
    for anchor_scorename in anchor_scorenames:
        print(f'anchor score: {anchor_scorename}')

        anchor_scores =  torch.tensor([ tag2scorenames[anchor_scorename][d_thres]   for d_thres in exp_tags])
        best_tag_indices = torch.argmax(anchor_scores,dim=0)
        best_tag_indices = best_tag_indices.tolist()


        for scorename in  tag2scorenames:
            print(f'score name: {scorename}')
            # the score of D_threshold that maximize the anchor score
            scores_overall[f'{scorename}.{overall_tag}.best{anchor_scorename}'] = np.nanmean([ tag2scorenames[scorename][exp_tags[best_tag_idx]][i] for i,best_tag_idx in enumerate(best_tag_indices) ])
            print(f" scores_overall[f'{scorename}.{overall_tag}.Dbest{anchor_scorename}'] :  { scores_overall[f'{scorename}.{overall_tag}.best{anchor_scorename}']} ")







print(20*'#')
print('overall scores:')

for tag,score_names in scores_images.items():
    for score_name in score_names:
        scores = scores_images[tag][score_name]
        # overall_score = np.array([ e for e in scores if e != float('nan') ]).mean() 
        overall_score = np.nanmean( scores )
        
        score_name_tag = f'{score_name}_{tag}'

        scores_overall[score_name_tag] = overall_score

    print(f'{tag}: {overall_score}')



# recall score (there's iou threshold )
for exp_tag in exp_tags:
    for rep_tag in ['argmax','union']:
        # non-weight avg between all classes

        ## recall: there's iou threshold 
        mean_each_thres  = []
        for iou_thres in iou_thresholds:
            mean_each_thres += [ np.nanmean( [ recall_global_stats[exp_tag][rep_tag][cls]['TP'][iou_thres]/(recall_global_stats[exp_tag][rep_tag][cls]['TP'][iou_thres]+recall_global_stats[exp_tag][rep_tag][cls]['FN'][iou_thres])  for cls in recall_global_stats[exp_tag][rep_tag] ]) ]

        overall_score = np.nanmean( mean_each_thres )

for exp_tag in exp_tags:
    for rep_tag in ['argmax','union']:
        for score_name in  per_class_stats[exp_tag][rep_tag]:
            for cls in per_class_stats[exp_tag][rep_tag][score_name]:
                tag_rep = 'U' if 'union' else 'A'
                tag_score_name = f'{cls}.C{score_name}.{tag_rep}'
                scores_overall[tag_score_name] = np.nanmean(per_class_stats[exp_tag][rep_tag][score_name][cls])



# write txt
with open(osp.join( args.output_dir, f'{output_fname}.txt'), 'w' ) as f:
        for score_name,score in scores_overall.items():
            f.write(f'{score_name}: {score}\n')
            print(f'{score_name}: {score}\n')

# write csv
scores_overall['num_pred'] = num_prediction
serie = pd.Series(scores_overall)
series = [serie]
df = pd.DataFrame(series, index=[output_fname])
df.to_csv(osp.join(args.output_dir,f'{output_fname}.csv' ),index_label='experiment')


if args.mode == 'textsim':
    torch.save(gt_clip_scores_dist, osp.join( args.output_dir, f'{output_fname}_gt_clip_scores_dist.pt'))
   
print(f"run log saved at  {osp.join( args.output_dir, f'{output_fname}_log.txt')  }  ")
print(f"result saved at: {osp.join( args.output_dir, f'{output_fname}.txt')  } ")
print(f"save eval result at: {osp.join(args.output_dir,f'{output_fname}.csv' )}")
                





        






