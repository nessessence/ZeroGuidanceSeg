
#
import argparse
from pathlib import Path
from pydoc import describe
from numpy import outer
import torch
# import clip as clip
import clip_v3_nb_mask as clip
from PIL import Image


import skimage.io as io
from skimage.measure import label, regionprops
from skimage.transform import resize
import PIL.Image
from model.ZeroCLIP_v2 import CLIPTextGenerator
# from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
import numpy as np
from torchvision.utils import save_image, draw_segmentation_masks

import scipy.io
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks


import nltk

import os
import os.path as osp
from collections import defaultdict,Counter

from tqdm import tqdm
from utils.utils import compute_cosine_sim,kl_div_uniformtest

from sentence_transformers import SentenceTransformer, util
Sbert = SentenceTransformer('all-mpnet-base-v2')


from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

# os.environ['CURL_CA_BUNDLE'] = ''

# nltk.download('punkt')
is_noun = lambda pos: pos[:2] == 'NN'

clip_size = 336#224#288
clip_patch = 14#32
attention_dim = 24
clip_dim = 768


plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)


# draw images and its segment
fig, ax = plt.subplots()
ax.axis('off')

color_list = ["cyan", "yellow", "blue", "lime", "darkviolet", "magenta", "red", "brown",
            "orange", "lightseagreen", "slateblue", "deepskyblue", "indianred", "tan", "olive", "plum", "palegreen", "bisque"]




def save_image(masks,captions,scores,save_dir,img_name,original_img=None,log_data=None,is_binary=False,save_pil=True):

    fig, ax = plt.subplots()
    ax.axis('off')


    ax.clear()
    legend = ax.get_legend()

    if legend is not None: legend.remove()

    if original_img is not None: 
        img = original_img.permute(1,2,0).numpy()
        orginal_mask = np.concatenate((img*255, 255. * np.ones_like(masks).astype(np.uint8)[..., None]),axis=-1)

        ax.imshow(orginal_mask.astype(np.int32), vmin=0, vmax=255)
        # save original
        # fig.savefig(save_dir / f'input.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)
        ax.axis('off')
        fig.savefig(save_dir / f'input.png', bbox_inches = 'tight', pad_inches = 0)

    n_segment = int(masks.max())+1
    cmap = 'jet' if n_segment > 17 else ListedColormap(color_list[:n_segment])

    
    if len(captions) > 0 and not is_binary: 

        
        mask_list = [ (masks==i).astype(bool) for i in range(n_segment) ]

        # print(cmap)
        # exit()
        # cmap = plt.cm.tab10(np.arange(len(mask_list)))

        cmap = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, n_segment))

        # print( len(mask_list), len(captions))
        img = PIL.Image.fromarray( (original_img.permute(1,2,0).numpy()* 255).astype(np.uint8) )

        if len(scores) > 0:
            captions = [ f'{c} : {s:.3f}' for c,s in zip(captions,scores)]

        fig = overlay_masks(img, mask_list, labels=captions, colors=cmap, mask_alpha=0.5)
        fig.savefig(save_dir / f'{img_name}.png', bbox_inches="tight")#, dpi=300)

    else: 
        n_segment = int(masks.max())+1
        cmap = 'jet' if n_segment > 17 else ListedColormap(color_list[:n_segment])
        ax.imshow(masks, cmap=cmap, vmin=0, vmax=n_segment-1, alpha=0.5, interpolation='nearest')

    # plot with captions
        hd = []
        for s_id,caption in enumerate(captions):
            # plot with sccore
            if len(scores) > 0:
                if is_binary: 

                    hd.append(mpatches.Patch(color=color_list[1], label=f'{caption} ({scores[s_id]:.4f})'))
                    print(f's_id:{1} - caption:{caption} - color:{color_list[0]}')
                else:
                    hd.append(mpatches.Patch(color=color_list[s_id], label=f'{caption} ({scores[s_id]:.4f})'))
                    print(f's_id:{s_id} - caption:{caption} - color:{color_list[s_id]}')
            else:
                if is_binary: 
                    hd.append(mpatches.Patch(color=color_list[1], label=f'{caption}'))
                    print(f's_id:{1} - caption:{caption} - color:{color_list[0]}')
                else:
                    hd.append(mpatches.Patch(color=color_list[s_id], label=f'{caption}'))
                    print(f's_id:{s_id} - caption:{caption} - color:{color_list[s_id]}')

        # if masks.shape[1] > masks.shape[0]*1.2:
        ax.legend(handles=hd)

        # else:
        #     ax.legend(handles=hd, bbox_to_anchor=(1.3, 1.1))
        ax.axis('off')
        fig.savefig(save_dir / f'{img_name}.png', bbox_inches = 'tight', pad_inches = 0)

    
    print(f"save image at {save_dir} / {img_name}.png")

    # fig.savefig(save_dir / f'{img_name}.png',  transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.close('all')

def save_label(masks,captions,save_dir,img_name='label',log_data=None,is_binary=False):

    ax.clear()
    legend = ax.get_legend()
    if legend is not None: legend.remove()


    n_segment = int(masks.max())+1
    cmap = 'jet' if n_segment > 17 else ListedColormap(color_list[:n_segment])
    ax.imshow(masks, cmap=cmap, vmin=0, vmax=n_segment-1, alpha=0.5, interpolation='nearest')


    hd = []
    # print(len(captions)))
    # if len(color_list) == len(captions):
    for s_id,caption in enumerate(captions):
        if is_binary: 
            hd.append(mpatches.Patch(color=color_list[1], label=f'{caption} '))
            print(f's_id:{1} - caption:{caption} - color:{color_list[0]}')
        else:
            hd.append(mpatches.Patch(color=color_list[s_id%17], label=f'{caption} '))
            print(f's_id:{s_id} - caption:{caption} - color:{color_list[s_id%17]}')

    if masks.shape[1] > masks.shape[0]*1.2:
        ax.legend(handles=hd)

    else:
        ax.legend(handles=hd, bbox_to_anchor=(1.3, 1.1))
    print(f"save image at {save_dir} / {img_name}.png")

    fig.savefig(save_dir / f'{img_name}.png')

    plt.close('all')
    # exit()


def to_binary_map(segmap, gt=False, offset=0):
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
            class_id.append(i+offset)
            count += 1
    return binary_map.astype(int), class_id



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
        
    return text_embeddings #.cpu()

def normalize(t):
    return t / (t.norm(dim=-1, keepdim=True))

device = 'cuda'
models = {}
models['clip'],_ = clip.load('ViT-L/14@336px', device)
models['sbert'] = SentenceTransformer('all-mpnet-base-v2')


models['sbert'].eval()


models['clip'].to(device)
models['sbert'].to(device)




def run(args,text_generator, img_path, ):
    # text_generator.zero_grad()


    img_name = osp.basename(img_path).split('.')[0]

    # print(f'args.data_annotation_path: {args.data_annotation_path}')
    # if not osp.exists(f'{args.data_annotation_path}/{img_name}/cluster_log.pt'):
    #     print('cluster log not exist')
    #     return
    
    annotation_log = torch.load(f'{args.data_annotation_path}/{img_name}/cluster_log.pt')



    cls_names = []; clsname2ids = {}
    if args.clsid_dir is not None:
        with open(args.clsid_dir,'r') as f:
            for l in f.read().splitlines():
                cls_id,cls_name = l.split(': ')
                cls_id = int(cls_id)

                if '/' in cls_name:
                   cls_names +=  [c for c in cls_name.split('/')]
                else: cls_names += [cls_name]

                clsname2ids[cls_name] = cls_id
    print(cls_names)




    if  args.data == 'gt_pascal':
        my_file = open(args.labels_dir, "r")
        data = my_file.read()
        labels = data.split("\n")[:-1]
        print(labels[0],labels[-1])
        my_file.close()

        class_emb = torch.load(args.class_emb_dir)
        num_class = len(labels)


        gt_map = PIL.Image.open(args.gt_path + img_name + '.png')       
        gt_map = np.array(gt_map)
        # gt_map[gt_map==255] = -1




        _, class_id = to_binary_map(gt_map, offset=-1)
        class_id = class_id[1:-1]
        mask = torch.from_numpy(gt_map).float()

        counter = Counter(mask.flatten().tolist())
        n_segment = len(counter.keys())

        for i,cid in enumerate(sorted(list(counter.keys()))):
            mask[mask==cid] = i
    
        mask_size = mask.shape


    elif  args.data == 'gt_coco':
        my_file = open(args.labels_dir, "r")
        data = my_file.read()
        labels = data.split("\n")[1:-1]
        print(labels[0],labels[-1])
        labels = labels + ['255: unlabeled']
        my_file.close()

        class_emb = torch.load(args.class_emb_dir)
        num_class = len(labels)


        gt_map_pil = PIL.Image.open(args.gt_path + img_name + '.png').convert('L')   
        gt_map = np.array(gt_map_pil)
        gt_map[gt_map==255] = 182
        # print(gt_map.min(), gt_map.max())
        # print(0, (gt_map==0).sum())
        # print(1, (gt_map==1).sum())
        # print(22, (gt_map==22).sum())
        # print(123, (gt_map==123).sum())
        # print(255, (gt_map==255).sum())
        # print('all', (gt_map<1000).sum())


        _, class_id = to_binary_map(gt_map)
        mask = torch.from_numpy(gt_map).float()

        counter = Counter(mask.flatten().tolist())
        n_segment = len(counter.keys())
        # print(counter.keys())
        # exit()

        for i,cid in enumerate(sorted(list(counter.keys()))):
            mask[mask==cid] = i
    
        mask_size = mask.shape

    elif  args.data == 'gt_pascal_context':
        my_file = open(args.labels_dir, "r")
        data = my_file.read()
        labels = data.split("\n")[:-1]
        labels = [z.split(": ")[1] for z in labels]
        print(labels[0],labels[-1])
        my_file.close()
        # unknown at 431

        my_file = open(args.labels_dir_59, "r")
        data = my_file.read()
        labels_59 = data.split("\n")[:-1]
        labels_59 = [z.split(": ")[1] for z in labels_59]
        print(labels_59[0],labels_59[-1])
        my_file.close()


        # class_emb = torch.load(args.class_emb_dir)
        num_class = len(labels)


        gt_map_459 = scipy.io.loadmat(args.gt_path + img_name + '.mat')['LabelMap']       
        gt_map_459 = np.array(gt_map_459)
        # gt_map[gt_map==255] = -1
        # print(img_name)
        # print(gt_map.min(), gt_map.max())
        idx_59_to_idx_469 = {}
        for idx, l in enumerate(labels_59):
            idx_59_to_idx_469[idx+1] = labels.index(l) + 1

        # print(idx_59_to_idx_469)

        c59 = False
        if c59:
            gt_map = np.zeros_like(gt_map_459, dtype=np.uint8)
            for idx, l in enumerate(labels_59):
                gt_map[gt_map_459 == idx_59_to_idx_469[idx+1]] = idx + 1
            labels = ['unknown'] + labels_59
            ofs = 0
        else:
            gt_map = gt_map_459
            ofs = -1
        
        _, class_id = to_binary_map(gt_map, offset=ofs)
        # print(class_id)
        mask = torch.from_numpy(gt_map.astype(np.float))#.float()

        counter = Counter(mask.flatten().tolist())
        n_segment = len(counter.keys())
        # print(list(counter.keys()))
        # exit()

        for i,cid in enumerate(sorted(list(counter.keys()))):
            mask[mask==cid] = i
    
        mask_size = mask.shape




    if 'gt' not in args.data:
        mask_size = annotation_log[args.thres_list[0]]['mask'].shape

    output_path = args.output_path
    save_dir = Path(f"{output_path}/{img_name}/")
    save_dir_images = Path(f"{output_path}/images/")
    save_dict_dir = Path(f'{output_path}/log/')


    # in case refill
    if osp.exists(save_dict_dir / f'{img_name}.pt'): 
        print(f'{img_name}.pt  is already computed .. skip')
        return

    save_dir.mkdir(exist_ok=True, parents=True)
    save_dict_dir.mkdir(exist_ok=True, parents=True)


    log_data ={}

    pil_img = io.imread(img_path)

    pil_img = PIL.Image.fromarray(pil_img).convert('RGB')

    pil_img_clip = pil_img.resize((clip_size,clip_size))
    
    
    
    img_masksize = T.ToTensor()(pil_img.resize((mask_size[-1],mask_size[-2])))

    sd_global_subtraction = args.sd_global_subtraction
    

    for ith, thres in enumerate(args.thres_list):

        apply_merge = args.apply_merge
        compute_all_nodes =  args.compute_all_nodes


        print(f'threshold: {thres}')

        annotation = annotation_log[thres]


        if 'gt' in  args.data:
            mask = mask # define above

            n_segment = int(mask.max() +1)
            mask_tree =  [cid for cid in range(n_segment)]
        else: 
            mask = annotation['mask'].float()
            mask_tree = annotation['tree'].tolist()


        n_segment =  len(Counter(mask.flatten().tolist()).keys())

        # fix mask (bug from clustering)
        if int(mask.max() +1) > n_segment:
            print('fixing mask')
            print(f'before:  {Counter(mask.flatten().tolist())} ')
            # fixed_masks = np.full(masks.shape,-1)
            for i in range(n_segment):
                if i not in mask:
                    mask[mask==(i+1)] = i
            print(f'after:  {Counter(mask.flatten().tolist())} ')
            

        # n_segment = int(mask.max() +1)


        
        sid_area = {}
        for sid in range(n_segment):
            sid_area[sid] = (mask==sid).int().sum().item()
        print(sid_area)


        print(f'{thres} n-segment: {n_segment}')
        


            

        print(f'tree len {mask_tree}')

        # generate captions
        captions_list = []
        clip_score_list = []

        # cid = child_node_id in th tree
        queue_s_ids = [ (cid,[s_id]) for cid,s_id in zip( mask_tree, (range(n_segment)))]
        tree_merger = defaultdict(dict)  #  { cid:  { s_ids, clip_score, text} }
        merge_log = []


        cid_mergeR_sids =defaultdict(list) # high treshold text_sids

        cls_bestcid = {}
        cls_bestcid_predcls_clipscore ={}
 
        predcls_cids = {} 
        predcls2cid_union = {} 
        cls_union_predcls_clipscore = {}
        cid2predcls_union = {}


        cls2selected_greedycls_sids = {}
        cls2queue_greedycls_sids  = {}
        cls2selected_greedycls_cid = {}
        cls2greedycls_clip_score = defaultdict(int)

        clip_score_records = []


        while len(queue_s_ids) > 0:
            print(f'queue:  {queue_s_ids}')
            cid,s_ids = queue_s_ids.pop(-1)

            print(f'cid: {cid}, s_ids: {s_ids}')
           
            # the binary mask of mask_clip
            mask_attention_binary = torch.zeros((1,1,attention_dim,attention_dim))
            for s_id in s_ids:
                # mask_attention_binary[mask_attention==s_id] = 1

                mask_bi = (mask==s_id)
                mask_bi = mask_bi.float()[None][None]
                # mask_attention_binary = torch.nn.Upsample(size=(attention_dim,attention_dim), mode='bilinear')(mask_bi)
                cur_mask_attention_binary = F.interpolate(mask_bi, size=(attention_dim,attention_dim), mode='area')
                # ambiguous_region = (cur_mask_attention_binary%1!=0)
                # cur_mask_attention_binary[ambiguous_region] = 0.

                mask_attention_binary += cur_mask_attention_binary
            mask_attention_binary.clamp(max=1)
            # ambiguous_region = (mask_attention_binary%1!=0)
            # mask_attention_binary[mask_attention_binary<1] = 0.

            if mask.sum() <= 0:
                captions_list.append('None')
                clip_score_list.append(-1)
                continue


            if args.crop_and_mask:
                lbl_0 = label(mask_bi[0,0]) 
                props = regionprops(lbl_0)
                # print(mask_bi[0][0].shape, mask_size)
                # torch_save_im(mask_bi[0], 'crop_mask.png')

                if len(props) == 0:
                    captions_list.append('-')
                    clip_score_list.append(-1)
                    continue

                im_feature = None
                x1 = mask_size[-2] -1
                y1 = mask_size[-1] -1
                x2, y2 = 0, 0
                # print(x1,y1,x2,y2)
                for i, prop in enumerate(props):
                    bbox = prop.bbox
                    # print('bb ', bbox)
                    if x1 > bbox[0]: x1 = bbox[0]
                    if y1 > bbox[1]: y1 = bbox[1]
                    if x2 < bbox[2]: x2 = bbox[2]
                    if y2 < bbox[3]: y2 = bbox[3]
                    # print(x1,y1,x2,y2)

                crop_img = img_masksize[:, x1:x2, y1:y2]
                crop_mask = mask_bi[0,:, x1:x2, y1:y2]
                print(crop_img.shape)
                crop_img = T.ToPILImage()(crop_img*crop_mask)
                # save_img = crop_img.save(f"crop.png")
                # print(crop_img.size)
                # exit()
                image_features = text_generator.get_img_feature_all([crop_img.resize((clip_size,clip_size))], None,None)[0]
                
            else:
                image_features = text_generator.get_img_feature_all([pil_img_clip], None,mask_attention_binary, sd_global_subtraction)[0]

            print('apply projection')
                
            normalized_image_features = normalize( image_features).to(text_generator.device)

            # image_features = text_generator.get_img_feature_all([pil_img_clip], None,mask_attention_binary, sd_global_subtraction)[0]
            # normalized_image_features = normalize( image_features).to(text_generator.device)

            
            # text optimization 
            args.prompts = ['Image of a {}.',
            ]
            # args.prompts = ['The {}.',
            # ]

            args.prompts_unformat = ['Image of a',
            ]

      
            captions = []
            captions_with_prompt = []
            for prompt,prompt_uf in zip( args.prompts, args.prompts_unformat):
                pred_open_vocabs = text_generator.run(normalized_image_features[None], prompt_uf, beam_size=args.beam_size)  #  give 5 candidate captions (beam_size == 5 )
                captions += pred_open_vocabs
                captions_with_prompt += [ prompt.format(c) for c in pred_open_vocabs]
            print(f'captions_with_prompt: {captions_with_prompt}')


            with torch.no_grad():
                
                encoded_captions =  encode_text(captions,model_name='clip',prompts=args.prompts)  #   [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device))[0] for c in captions_with_prompt]
                normalized_encoded_captions =  normalize(encoded_captions).float()   # [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]

                # clip_scores = (normalized_encoded_captions @ normalized_image_features.t()).squeeze()
                clip_scores = compute_cosine_sim(encoded_captions,image_features)

                print(f'score: {clip_scores.tolist()}')
                
                # best clip score 
                best_clip_idx = clip_scores.argmax().item()
                clip_score = clip_scores[best_clip_idx].item()


                # select the best caption
                # caption = captions[best_clip_idx][:-1].lower().replace('.','')
                caption = captions[best_clip_idx][:-1].lower().replace('.','').strip()

                
                # compute anyway    
                sbert_encoded_captions =  encode_text(captions,model_name='sbert',prompts=args.prompts)  
                sbert_encoded_captions =  normalize( sbert_encoded_captions  )


                if args.text_encoder == 'sbert': 
                    text_feat_tag = 'text_feat_sbert'
                    print('using SBERT to encode text')
                

                    
                else: text_feat_tag = 'text_feat'

                best_encoded_captions = encoded_captions[best_clip_idx].clone().cpu()
                best_sbert_encoded_captions = sbert_encoded_captions[best_clip_idx].clone().cpu()



                image_features = image_features.cpu()



            # original caption (only original child leaf)
            if cid in mask_tree:
                captions_list = [caption] + captions_list  # reverse order !!!
                clip_score_list = [clip_score] + clip_score_list

            # merging
            if  (apply_merge or args.compute_all_nodes) and   (cid*2)+1 in tree_merger and (cid*2)+2 in tree_merger: 
                # one of the parition got higher score than its parent --> partition 
                # if max( tree_merger[(cid*2)+1]['clip_score'] , tree_merger[(cid*2)+2]['clip_score'] ) < clip_score : 

                score_threshold = args.merge_thres

                cosine_sim = None 

               

                if args.merge_choice == 'image':
                    cosine_sim = compute_cosine_sim(tree_merger[(cid*2)+1]['img_feat'], tree_merger[(cid*2)+2]['img_feat']).item()
                    is_merge = cosine_sim   >=  score_threshold

                    log = f" image cos_sim ( { tree_merger[(cid*2)+1]['caption']},  {tree_merger[(cid*2)+2]['caption']} ): {cosine_sim}  -----  threshold =  {score_threshold} "


                elif args.merge_choice == 'text':
                    cosine_sim = compute_cosine_sim(tree_merger[(cid*2)+1][text_feat_tag] , tree_merger[(cid*2)+2][text_feat_tag]).item()
                    is_merge = cosine_sim   >=  score_threshold

                    log = f" text cos_sim ( { tree_merger[(cid*2)+1]['caption']},  {tree_merger[(cid*2)+2]['caption']} ): {cosine_sim}  -----  threshold =  {score_threshold} "


                elif args.merge_choice == 'clip_max':
                    is_merge = max( tree_merger[(cid*2)+1]['clip_score'] , tree_merger[(cid*2)+2]['clip_score'] ) < clip_score

                    log = f" clip max:( { tree_merger[(cid*2)+1]['caption']},  {tree_merger[(cid*2)+2]['caption']} )   vs   {clip_score}  "

                elif args.merge_choice == 'clip_mean':
                    is_merge = 0.5*( tree_merger[(cid*2)+1]['clip_score'] + tree_merger[(cid*2)+2]['clip_score'] ) < clip_score

                    log = f" clip mean:( { tree_merger[(cid*2)+1]['caption']},  {tree_merger[(cid*2)+2]['caption']} )   vs   {clip_score}  "

                elif args.merge_choice == 'mix':
                    print(f'Text-Visual Merging using text encoder {text_feat_tag}')
                    # alpha = 0.5
                    alpha = args.merge_weight


                    cosine_sim_img =  compute_cosine_sim(tree_merger[(cid*2)+1]['img_feat'],tree_merger[(cid*2)+2]['img_feat']).item()
                    cosine_sim_text = compute_cosine_sim(tree_merger[(cid*2)+1][text_feat_tag] , tree_merger[(cid*2)+2][text_feat_tag]).item()

                    cosine_sim = alpha*cosine_sim_img + (1-alpha)*cosine_sim_text

                    log = ''
                    log += f"{tree_merger[(cid*2)+1]['caption']} -- {tree_merger[(cid*2)+2]['caption']}\n"
                    log += f'img sim {cosine_sim_img}  ---> {alpha*cosine_sim_img} \n'
                    log += f'text sim {cosine_sim_text} ---> {(1-alpha)*cosine_sim_text} \n'
                    log += f" {alpha} img + {1-alpha} text cos_sim ( { tree_merger[(cid*2)+1]['caption']},  {tree_merger[(cid*2)+2]['caption']} ): {cosine_sim}  -----  threshold =  {score_threshold} "


                    is_merge = cosine_sim >= score_threshold

                elif args.merge_choice == 'mix_latent':
                    alpha = 0.5

                    mix_latent_1 = 0.5*(tree_merger[(cid*2)+1]['img_feat'] + tree_merger[(cid*2)+1][text_feat_tag])
                    mix_latent_2 = 0.5*(tree_merger[(cid*2)+2]['img_feat'] + tree_merger[(cid*2)+2][text_feat_tag])

                    cosine_sim =  compute_cosine_sim(mix_latent_1,mix_latent_2).item()


                    log = ''
                    log += f"{tree_merger[(cid*2)+1]['caption']} -- {tree_merger[(cid*2)+2]['caption']}\n"
                    log += f"mix latent { tree_merger[(cid*2)+1]['caption']},  {tree_merger[(cid*2)+2]['caption']} ): {cosine_sim}  -----  threshold =  {score_threshold} "


                    is_merge = cosine_sim >= score_threshold

                print(log)
                merge_log += [log]
                
                if  is_merge:
                    is_exclusive = tree_merger[(cid*2)+1]['is_exclusive'] and  tree_merger[(cid*2)+2]['is_exclusive']
                    
                    tree_merger[cid] = {'s_ids': s_ids, 'clip_score': clip_score, 'caption': caption, 'img_feat': image_features, 'text_feat':best_encoded_captions, 'text_feat_sbert':best_sbert_encoded_captions, 'is_exclusive': is_exclusive }

                    print(f'merging sucess {(cid*2)+1} + {(cid*2)+2}  --->  {cid} !!! ')
                    merge_log += [ f'merging success {(cid*2)+1} + {(cid*2)+2}  --->  {cid} !!! ']

                    if is_exclusive: tree_merger[(cid*2)+1]['is_exclusive'] = False;  tree_merger[(cid*2)+2]['is_exclusive'] = False
                    # del tree_merger[(cid*2)+1], tree_merger[(cid*2)+2];

                else:
                    print(f'merging fail !!!')
                    merge_log += [f'merging fail !!!']

                    if compute_all_nodes:
                        tree_merger[cid] = {'s_ids': s_ids, 'clip_score': clip_score, 'caption': caption, 'img_feat': image_features, 'text_feat':best_encoded_captions,   'text_feat_sbert':best_sbert_encoded_captions, 'is_exclusive': False }

            else:
                is_exclusive = apply_merge or compute_all_nodes
                tree_merger[cid] = {'s_ids': s_ids, 'clip_score': clip_score, 'caption': caption, 'img_feat': image_features, 'text_feat':best_encoded_captions,'text_feat_sbert':best_sbert_encoded_captions, 'is_exclusive': is_exclusive }

            if (apply_merge or compute_all_nodes):
                if cid in tree_merger and cid % 2 == 0 and cid-1 in tree_merger: 
                    if tree_merger[cid-1]['is_exclusive'] or compute_all_nodes:
                        queue_s_ids += [(int((cid-1)//2), s_ids + tree_merger[cid-1]['s_ids']) ]
                        print(f'adding parent: {cid}&{cid-1} --> {(cid-1)//2} ')
                elif cid in tree_merger and cid % 2 != 0 and cid+1 in tree_merger: 
                    if tree_merger[cid+1]['is_exclusive'] or compute_all_nodes:
                        queue_s_ids += [(int((cid-1)//2), s_ids + tree_merger[cid+1]['s_ids']) ]
                        print(f'adding parent: {cid}&{cid+1} --> {(cid-1)//2} ')
                        print(queue_s_ids[-1])


            for cid in tree_merger: print(f"cid: {cid}  - s_ids: {tree_merger[cid]['s_ids']}  caption: {tree_merger[cid]['caption']} -score {tree_merger[cid]['clip_score']} --- exclusive:{tree_merger[cid]['is_exclusive']}") 
            

            if  len(queue_s_ids) == 0 and args.apply_merge_refine:
                print(20*'#'+'\n')
                print(f'using Merge Refine threshold : {args.merge_refine_thres} ')
                print(20*'#'+'\n')
                
                cids = []
                text_feats = []; img_feats = []
                areas = [] 
                sids_ = []
                captions = []

                if len(cid_mergeR_sids) == 0:
                    print('\nfilling existing node')
                    for i,(cid,data) in enumerate(list(tree_merger.items())[::-1]):

                        is_apply_merge_refine =  len(data['s_ids']) == 1  if  args.merge_refine_standalone else data['is_exclusive']
                        # based on original merge
                        if is_apply_merge_refine: 
                        # if len(data['s_ids']) == 1: 

                            cid_mergeR_sids[cid] = [data['s_ids']]

                            cids += [cid]
                            text_feats += [data['text_feat']];  img_feats += [data['img_feat']]
                            areas +=  [sum([ sid_area[sid] for sid in data['s_ids'] ])]
                            sids_ +=  [ data['s_ids']]
                            captions += [data['caption']]
                else:
                    for cid in cid_mergeR_sids.keys():
                        data = tree_merger[cid]
                        cids += [cid]
                        text_feats += [data['text_feat']];  img_feats += [data['img_feat']]
                        areas +=  [sum([ sid_area[sid] for sid in data['s_ids'] ])]
                        sids_ +=  [ data['s_ids']]
                        captions += [data['caption']]


                areas = torch.tensor(areas)       
                text_feats = torch.stack(text_feats)
                img_feats = torch.stack(img_feats)

                
                if args.merge_refine_choice == 'text'  :
                    similarity = compute_cosine_sim(text_feats,text_feats)
                
                elif args.merge_refine_choice == 'mix'  :
                    alpha = 0.5 
                    similarity = (alpha)*compute_cosine_sim(text_feats,text_feats) +  (1-alpha)*compute_cosine_sim(img_feats,img_feats)

                elif args.merge_refine_choice == 'mix_latent'  :
                    alpha = 0.5 
                    mean_latent = (alpha)*text_feats + (1-alpha)*img_feats
                    similarity = compute_cosine_sim(mean_latent,mean_latent) 


                for i in range(len(similarity)): similarity[i][i] = -1  # set sim of itself to -1
                
                max_current_cid_intree = max(tree_merger)
                dummy_counter = 1111

                max_sim = torch.max(similarity)
                max_sim_indices = (similarity==max_sim).nonzero()[0]  # return (i,j) of max sim 
                i,j = max_sim_indices

                # lower than threshold
                # 1 merge iterative
                # if lower than threshold --> zero queue --> end right here
                if max_sim >= args.merge_refine_thres : 
                    
                    merge_log += [f'Merge Refine: {captions[i]}:{sids_[i]}- {captions[j]}:{sids_[j]} -- {max_sim} ']
                    print(merge_log[-1])

                    # merge to bigger segments
                    # did not use in 1 time merge
                    if areas[i] >= areas[j] : 
                        areas[j] += areas[i]
                            
                    else:
                        areas[i] += areas[j]

                    next_cid = max_current_cid_intree + dummy_counter

                    queue_s_ids += [(next_cid,sids_[i]+sids_[j])]
                    
                    
                    cid_mergeR_sids[next_cid] = sids_[i]+sids_[j]
                    del cid_mergeR_sids[cids[i]]  
                    del cid_mergeR_sids[cids[j]] 

                    merge_log+= [f'adding cid: {next_cid} sd: {cid_mergeR_sids[next_cid]} to queue']
                    print(merge_log[-1])

                else:
                    
                    merge_log+= [f"\n{captions[i]}:{sids_[i]}- {captions[j]}:{sids_[j]}   :  {max_sim} < {args.merge_refine_thres}  --> End Merge Refine \n Max similarity is lower than the threshold  \n\n"]
                    print(merge_log[-1])

                apply_merge = False;  compute_all_nodes = False

        print('Merge Refine summary:')
        for cid in cid_mergeR_sids.keys():
            print(f"{ tree_merger[cid]['caption']} :  {tree_merger[cid]['s_ids']}")




        print(f'cls_names: {cls_names}')  # all classes in dataset  
        with torch.no_grad():
            # all_cls_name_feats = encode_text(cls_name_with_prompt,model_name='clip')   #    torch.stack([text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device))[0] for c in cls_name_with_prompt]).cpu()

            all_cls_name_feats = encode_text(cls_names,model_name='clip',prompts=args.prompts)   #    torch.stack([text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device))[0] for c in cls_name_with_prompt]).cpu()
            all_img_feats = torch.stack( [tree_merger[cid]['img_feat'] for cid in list(tree_merger.keys())[::-1] ] ).to(device) # reverse

            all_clip_scores = compute_cosine_sim(all_img_feats,all_cls_name_feats)
            clip_sort_idx = torch.argsort(all_clip_scores,dim=1,descending=True).tolist()   # ( n_mask, n_total_class )
            clip_argmax = torch.argmax(all_clip_scores,dim=1).tolist()   # ( n_mask, n_total_class )




        for i, cid in  enumerate(list(tree_merger.keys())[::-1]):
            tree_merger[cid]['gt_class_align'] = (  cls_names[clip_argmax[i]] , all_clip_scores[i][clip_argmax[i]].item()   )
            print( f"{cls_names[clip_argmax[i]]} : {all_clip_scores[i][clip_argmax[i]]}")




        log = defaultdict(lambda: defaultdict(list))

        log['']['masks'] =  np.array(mask)
        
        log['merge']['masks'] = np.array(mask) 
        log['mergeR']['masks'] = np.array(mask) 


        log['all']['masks'] = [ np.zeros_like(mask,dtype=np.int32) for _ in range(len(tree_merger)) ]

        log['merge_cls']['masks'] = [ np.zeros_like(mask,dtype=np.int32) for _ in range(len(cls_bestcid)) ]
        log['merge_cls_greedy']['masks'] = [ np.zeros_like(mask,dtype=np.int32) for _ in range(len(cls2selected_greedycls_cid)) ]
        log['merge_cls_union']['masks'] = [ np.zeros_like(mask,dtype=np.int32) for _ in range(len(predcls2cid_union)) ]
        

        print(f'cid2predcls_union:\n {cid2predcls_union}')
        print(f'tree_merger cid: {tree_merger.keys()}')








            # for analysis of CLIP score distribution: keep data only merge
            # if tree_merger[cid]['is_exclusive']: 



        # Note: reverse order !!!!
        for i,(cid,data) in enumerate(list(tree_merger.items())[::-1]):
 

            is_exclusive = data['is_exclusive']
            s_ids = data['s_ids']
            # merge
            if is_exclusive :
                for s_id in data['s_ids']: log['merge']['masks'][mask==s_id] = len(log['merge']['captions']) # new_s_id
            # merge refine text
            if cid in cid_mergeR_sids: 
                for s_id in data['s_ids']: log['mergeR']['masks'][mask==s_id] = len(log['mergeR']['captions']) # new_s_id

            # all 
            for s_id in s_ids:  log['all']['masks'][i][mask==s_id] = 1 


            for data_name in  ['clip_score','caption', 'img_feat','text_feat','text_feat_sbert','gt_class_align']:
                if cid in mask_tree:    log[''][f'{data_name}s'] += [data[data_name] ]  # original children from  DINO threshold 
                if is_exclusive : log['merge'][f'{data_name}s'] += [data[data_name]]
                if cid in cid_mergeR_sids : log['mergeR'][f'{data_name}s'] += [data[data_name]]


                    
                log['all'][f'{data_name}s'] += [data[data_name]]
            
            log['all'][f'cids'] += [cid] # used in re-structure tree

        # alignment to class existing in the dataset






        # print(torch.tensor(clip_argmax).shape)
        
        # exit()



        # print(f'all_clip_scores: {all_clip_scores}')



        # TODO: len(segment) == 0 ---> filter out in the first place
        if 'None' in captions_list: continue

        print(f'n original segment: {np.array(mask).max() + 1}')
        print(f"captions: {captions_list}")
        
        print('\n'+30*'#')
        print(f'merge log:')
        for l in merge_log: print(l)
        print('\n'+30*'#')


        print(f"cids: {log['all']['cids']}")
        # exit()
        if  args.data == 'gt_pascal':
            gt_captions = ['bg'] + [labels[z].split(": ")[1] for z in class_id] + ['amb']
            save_label(log['']['masks'] ,gt_captions,save_dir)
        elif args.data == 'gt_coco':
            print(len(labels), class_id) 
            gt_captions = [labels[z].split(": ")[1] for z in class_id]
            save_label(log['']['masks'] ,gt_captions,save_dir)
        elif  args.data == 'gt_pascal_context':
            gt_captions = [labels[z] for z in class_id]
            save_label(log['']['masks'] ,gt_captions,save_dir)

        save_img_tags =  ['', 'merge', 'all' ] 
        if args.apply_merge_refine:  save_img_tags += ['mergeR'] # ' merge_cls, merge_cls_greedy', 'merge_cls_union',]:


        for tag in save_img_tags:
            if 'gt' in args.data and tag not in ['', 'all']: continue
            print(tag)
            
            captions = log[tag]['captions'] 
            masks = log[tag]['masks'] 
            scores = log[tag]['clip_scores'] 


            # remove " "
            captions = [ c.replace('''"''', '') for c in captions ]

            # sort score: 
            print(len(captions))
            # print(captions)
            print(Counter(torch.tensor(masks).flatten().tolist()).keys())
            print(Counter(torch.tensor(masks).flatten().tolist()))
            
            if tag in ['', 'merge', 'mergeR']:
                sort_idx = torch.argsort(torch.tensor(scores),descending=True).tolist()
                sorted_scores = [ scores[si] for si in sort_idx]
                sorted_captions =  [ captions[si] for si in sort_idx]
                sorted_masks = np.full(masks.shape,-1)
                for i, si in enumerate(sort_idx):
                    sorted_masks[masks==si] = i
                
                display_captions = sorted_captions
                display_scores = sorted_scores
                display_masks = sorted_masks
            else: 
                display_captions = captions
                display_scores = scores
                display_masks = masks






            if tag == 'all':
                # non-mutually exlusive
                save_dir_i = save_dir/f'{thres:.1f}_all'
                os.makedirs(save_dir_i,exist_ok=True)
                for i in range(len(masks)):
                    print(f'all {i}')
                    save_img_name = f'{i}'
                    
                    display_captions = [ captions[i]] 
                    display_scores = [scores[i]] 

                    # top k classes that maximize clip scores with the mask


                    save_image(masks[i],display_captions ,display_scores,save_dir_i,save_img_name, original_img=img_masksize,is_binary=True)

          


                    # kl
                    # kl = kl_div_uniformtest(torch.tensor([all_clip_scores[i][j] for j in clip_sort_idx[i][:k]]))
                    # kl_top1base = kl_div_uniformtest(torch.tensor([all_clip_scores[i][j] for j in clip_sort_idx[i][:k]])/all_clip_scores[i][clip_sort_idx[i][0]]  )
                    # display_captions += ['kl', 'kl1']
                    # display_scores += [kl,kl_top1base ]

                    k = 1
                    display_captions = [  cls_names[j] for j in clip_sort_idx[i][:k] ]
                    display_scores = [all_clip_scores[i][j] for j in clip_sort_idx[i][:k]]


                    save_dir_i = save_dir/'all_with_gtscore'
                    os.makedirs(save_dir_i,exist_ok=True)

                    save_image(masks[i],display_captions ,display_scores,save_dir_i,save_img_name, original_img=img_masksize,is_binary=True)




            elif tag == 'mergeR':
                # print(display_captions)
                # print(display_masks)
                # print(Counter(torch.tensor(display_masks).flatten().tolist()))

                save_img_name = f'{thres:.1f}_mergeR'
                save_dir_i = save_dir/'mergeR'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)

                save_dir_i = save_dir/'mergeR_withscore'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/'mergeR_png'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,[],[],save_dir_i,save_img_name,original_img=img_masksize)        

                
                # save_img_name = f'{img_name}'

                save_dir_i = save_dir_images/f'mergeR/{img_name}' 
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)

                save_dir_i = save_dir/f'mergeR_withscore/{img_name}'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/f'mergeR_png/{img_name}'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,[],[],save_dir_i,save_img_name,original_img=img_masksize)   
            
            elif tag == 'merge':
                save_img_name = f'{thres:.1f}_merge'
                save_dir_i = save_dir/'merge'
                os.makedirs(save_dir_i,exist_ok=True)
                # print(display_captions)
                # exit()
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)

                save_dir_i = save_dir/'merge_withscore'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/'merge_png'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,[],[],save_dir_i,save_img_name,original_img=img_masksize)


                save_dir_i = save_dir_images/f'merge/{img_name}' 
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)

                save_dir_i = save_dir/f'merge_withscore/{img_name}'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/f'merge_png/{img_name}'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,[],[],save_dir_i,save_img_name,original_img=img_masksize)   

                # relabel with gt class (argmax only )
                # s-t

                # save_dir_i = save_dir/'merge_gt_withscore'
                # os.makedirs(save_dir_i,exist_ok=True)
                # save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)

                # save_dir_i = save_dir/f'merge_gt_withscore/{img_name}'
                # os.makedirs(save_dir_i,exist_ok=True)
                # save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)





            # orginal threshold
            else:
                save_img_name = f'{thres:.1f}' 
                save_image(display_masks,display_captions,[],save_dir,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/'thres'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)

                save_dir_i = save_dir/'thres_withscore'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/'thres_png'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,[],[],save_dir_i,save_img_name,original_img=img_masksize)




                # also save in merge (easy to compare)
                save_dir_i = save_dir/'merge'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)
                 
                # also save in merge (easy to compare)
                save_dir_i = save_dir/'mergeR'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)




                save_dir_i = save_dir_images/f'thres/{img_name}' 
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,[],save_dir_i,save_img_name,original_img=img_masksize)

                save_dir_i = save_dir/f'thres_withscore/{img_name}'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,display_captions,scores,save_dir_i,save_img_name,original_img=img_masksize)
                
                save_dir_i = save_dir/f'thres_png/{img_name}'
                os.makedirs(save_dir_i,exist_ok=True)
                save_image(display_masks,[],[],save_dir_i,save_img_name,original_img=img_masksize)  


            # log

            log_tag = None 
            if tag in ['merge','','mergeR']: 
                log_tag = f'{thres:.1f}_{tag}' if tag else  f'{thres:.1f}'
                log[tag]['masks'] = torch.tensor(masks) # mutually exclusive

            elif tag == 'all': 
                log_tag = f'{thres:.1f}_all'
                log[tag]['masks'] = torch.stack([ torch.tensor(m) for m in masks] ) # non-mutually exclusive 

            if log_tag is not None:
                log_data[log_tag] = dict(log[tag]) # remove lambda (it could not be picked)
            
            if  'gt_' in args.data: log_data['gt_cls'] = gt_captions

    img_name = osp.basename(img_path).split('.')[0]
    torch.save(log_data, save_dict_dir / f'{img_name}.pt')
    torch.save(log_data, save_dir/'log.pt')

    print(f"save log at: {save_dict_dir / f'{img_name}.pt'}")
    print(f"save log at: {save_dir/'log.pt'}")




            


    
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--prompts", type=str, default=["Image of a"])
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.1, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--multi_gpu", action="store_true")

    parser.add_argument("--output_path",'-o',type=str, default="output/test/", help="output_result")


    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_path", type=str, default='example_images/captions/COCO_val2014_000000008775.jpg',
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])

    parser.add_argument("--data", "-d",  type=str, default='coco')
    parser.add_argument('--subset', '-s', type=str, default="val", choices=['val','train', 'val_dinov2']) # coco, pascal, pascal_context


    parser.add_argument("--mask_layer_list", "-ml",  type=str, default='17-24', help='1-2,6-10,19-24 --> [1, 2, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24]')


    parser.add_argument("--merge_thres", type=float, default=0.8)
    parser.add_argument("--merge_choice", type=str, default='image')  # mix_latent

    parser.add_argument("--crop_and_mask",   default=False, action=argparse.BooleanOptionalAction )


    parser.add_argument("--img_idx", "-idx",  type=str,  default='0-100', help='only selected index will be processed')


    parser.add_argument("--n_gpus",'-ng',  type=int,  default=None, help='only selected index will be processed')
    parser.add_argument("--gpu_idx", "-g_idx",  type=int,  default=None, help='only selected index will be processed')



    parser.add_argument("--text_encoder",  type=str,  default=None, help='[sbert]')

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument("--compute_all_nodes",  default=True, action=argparse.BooleanOptionalAction)


    parser.add_argument("--apply_merge",   default=True, action=argparse.BooleanOptionalAction )
    parser.add_argument("--merge_weight",   default=0.5, type=float, help='merge_score = (alpha)img_sim + (1-alpha)text_sim' )



    parser.add_argument("--apply_merge_cls",   default=False, action=argparse.BooleanOptionalAction )
    parser.add_argument("--apply_merge_cls_union",   default=False, action=argparse.BooleanOptionalAction )
    parser.add_argument("--apply_merge_cls_greedy",   default=False, action=argparse.BooleanOptionalAction )

    parser.add_argument("--apply_merge_refine",   default=False, action=argparse.BooleanOptionalAction )
    parser.add_argument("--merge_refine_choice",   default='mix', choices=['mix','text','mix_latent'] )
    parser.add_argument("--merge_refine_thres",   type=float, default=0.8 )
    parser.add_argument("--merge_refine_standalone",    default=False, action=argparse.BooleanOptionalAction )



    parser.add_argument("--already_computed_filter",   default=False, action=argparse.BooleanOptionalAction )

    parser.add_argument('--text_thres', type=float, default=None)
    parser.add_argument('--sd_global_subtraction', "-sd", type=float, default=None) 


    parser.add_argument("--cls_greedy_min_diff_percent",  type=float,  default= 0.01 )










    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()


    args.reset_context_delta = True 


    # args.apply_merge = True 
    print(f'apply merge: {args.apply_merge}')
    print(f'apply merge cls {args.apply_merge_cls}')
    print(f'apply merge cls union: {args.apply_merge_cls_union}')
     # global param hack

    # args.compute_all_nodes = True
    print(f'compute all nodes: {args.compute_all_nodes}')


    


    l = []
    for s in args.mask_layer_list.split(','):
        a,b = [int(e) for e in s.split('-') ]
        l += list(range(a,b+1))
    args.mask_layer_list = l

    # args.mask_layer_list = list(range(19,24))

    text_generator = CLIPTextGenerator(**vars(args))

    

    # print(f'selected img indices: {args.img_idx}')


    # args.mask_layer_list = [23]


    args.clsid_dir = None

    if args.data == 'coco':
        print('using coco')
        if args.subset == 'val':  
            args.data_path = Path('../data/coco/images/val/val2017') # dataset path 
            args.data_annotation_path = 'results_dino2/coco_val/vits16_numpart20'
        elif args.subset == 'train':  
            args.data_path = Path('../data/coco/images/train/train2017') # dataset path 
            args.data_annotation_path = 'results_dino2/coco_val/train_vits16_numpart20'
        

    elif args.data == 'pascal':
        print('using pascal')
        args.data_path = Path('../data/voc2012/val/image_val')
        args.data_annotation_path = 'results_dino2/pascal2012/vits16_numpart20'
        args.clsid_dir = "../data/voc2012/labels.txt"


    elif args.data == 'pascal_context':
        print('using pascal-context')

        if args.subset == 'val':  
            args.data_path = Path("../data/pascal_context/val/images/")
            args.data_annotation_path = 'results_dino2/pascal_context/vits16_numpart20'
        elif args.subset == 'train':  
            args.data_path = Path("../data/pascal_context/train/images/")
            args.data_annotation_path = 'results_dino2/pascal_context/train_vits16_numpart20'
        elif args.subset == 'val_dinov2':  
            args.data_path = Path("../data/pascal_context/val/images/")
            args.data_annotation_path = 'results_dinov2/context_val/pascal_context_59_DINOB_l11'
        # args.data_path = Path('../data/pascal_context/val/images')
        # args.data_path = Path('../data/pascal_context/train/images')
        
        args.clsid_dir = "../data/pascal_context/labels.txt"

    elif args.data == 'pop':
        print('using pascal-context')
        # args.data_path = Path('../data/pascal_context/val/images')
        args.data_path = Path('../data/popculture/images')
        args.data_annotation_path = 'results_dino2/popculture/vits16_numpart20'
        args.clsid_dir = "../data/pascal_context/labels.txt"

    elif args.data == 'vistec':
        print('using pascal-context')
        # args.data_path = Path('../data/pascal_context/val/images')
        args.data_path = Path('../data/vistec/images')
        args.data_annotation_path = 'results_dino2/vistec/vits16_numpart20'
        args.clsid_dir = "../data/pascal_context/labels.txt"

    elif args.data == 'snake':
        print('using snake')
        args.data_annotation_path = '../../godsom/dino-vit-features-main/snake5_srb20'
        main_path = Path('../../godsom/dino-vit-features-main/images/set_snake5/snake')

    elif args.data == 'gt_pascal':
        print('using Pascal groundtruth')
        args.data_path = Path('../data/voc2012/val/image_val')
        args.data_annotation_path = 'results_dino2/pascal2012/vits16_numpart20'
        args.gt_path = "../data/voc2012/VOC2012/SegmentationClass/"
        args.labels_dir = "../data/voc2012/labels.txt"
        args.class_emb_dir = '../data/voc2012/label_embeddings_withprompt.pt'

    elif args.data == 'gt_coco':
        print('using coco groundtruth')
        args.data_path = Path('../data/coco/images/val/val2017')
        # args.data_path = Path('../data/coco/images/train/train2017')
        args.data_annotation_path = 'results_dino2/coco_val/vits16_numpart20'
        args.gt_path = "../data/coco/stuffthingmaps_trainval2017/val2017/"
        args.labels_dir = "../data/coco/stuffthingmaps_trainval2017/labels.txt"
        args.class_emb_dir = '../data/coco/label_embeddings.pt' 
    
    elif args.data == 'gt_pascal_context':
        print('using pascal_context groundtruth')
        args.data_path = Path('../data/pascal_context/val/images')
        # args.data_path = Path('../data/pascal_context/train/images')
        args.data_annotation_path = 'results_dino2/pascal_context/vits16_numpart20'
        args.gt_path = "../data/pascal_context/segment_anno/"
        args.labels_dir = "../data/pascal_context/labels.txt"
        args.labels_dir_59 = "../data/pascal_context/labels_59.txt"
        args.class_emb_dir = '../data/pascal_context/label_embeddings_459.pt'


    
    img_paths = [x for x in args.data_path.iterdir()]
    img_paths.sort()

    # seed = 0
    # random_state = np.random.RandomState(seed)
    # random_state.shuffle(img_paths)

    img_names = [ img_path.name for img_path in  img_paths ]
    # torch.save(img_names,f'playground/log/img_names_seed{seed}.pt')
    # torch.save(img_paths,f'playground/log/img_paths_seed{seed}.pt')


    # print(f'shuffle with seed: {seed}')


    print(f'total dataset: {len(img_paths)} imgs')
    # im_path = im_path[0:100] # genimage th 
    
    already_computed_idx = []
    if args.already_computed_filter: 
        save_dict_dir = Path(f'{args.output_path}/log/')

        filted_img_paths = []
        
        for i,img_path in enumerate(img_paths):
            if osp.exists(save_dict_dir / f'{img_path.name}.pt'): 
                already_computed_idx += [i]
            else: filted_img_paths += [img_path]
        
        img_paths = filted_img_paths
    
    print(f'already computed index ({len(already_computed_idx)}): {already_computed_idx}')
        

    img_idx = []
    for s in args.img_idx.split(','):
        a,b = [int(e) for e in s.split('-') ]
        img_idx += [e for e in list(range(a,b+1)) if e not in already_computed_idx ] 
    args.img_idx = img_idx

    selected_idx = args.img_idx
    selected_idx = np.array(selected_idx)[np.array(selected_idx)<len(img_paths)].tolist()
    print(f'before partition selected indices: ( {min(selected_idx)}, {max(selected_idx)} )  ')
    # img_paths = np.array(img_paths)[seslected_idx].tolist()

    if args.n_gpus is not None and args.gpu_idx is not None:
        print(f'index partition: {args.gpu_idx+1}/{args.n_gpus}')
        selected_idx =  np.array_split(selected_idx,args.n_gpus)[args.gpu_idx]
        print(f"selected img partition indices:\n{selected_idx}")

    print(f'selected idx: {selected_idx}')
    img_paths = np.array(img_paths)[selected_idx].tolist()


    args.thres_list = [ 0.95]
    # args.thres_list = [ 0.1]

    if 'gt' in args.data: args.thres_list = [0.1 ]


    print(f'threshold list: { args.thres_list}')



    # img_paths = [ p for p in img_paths if '2008_004414' in p.name]
    # print(im_path)
    print('image to segment ', len(img_paths))

    
    for i,img_path in tqdm(enumerate(img_paths),total=len(img_paths)):
        print(f'{i}-{args.img_idx[i]} {img_path.name}')
        print(img_path)
        # selected_img = [ '2008_000015', '2008_000019', '2008_000008']

        # if  not np.any([ img_name in img_path.name  for img_name in  selected_img]): continue
        run(args, text_generator, img_path=img_path)

