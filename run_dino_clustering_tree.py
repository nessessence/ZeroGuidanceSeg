import argparse
import torch
from pathlib import Path

import torchvision.transforms
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple
import pydensecrf.densecrf as dcrf
from sklearn.metrics.pairwise import cosine_similarity as cosine
from matplotlib.colors import ListedColormap
from torchvision.utils import save_image


# from sklearn.cluster import AgglomerativeClustering
from dino_modules.feature_extractor import ViTExtractor
from dino_modules.agglomerative_clustering import AgglomerativeClustering

def find_part_cosegmentation(image_paths: List[str], elbow: float = 0.975, load_size: int = 224, layer: int = 11,
                             facet: str = 'key', bin: bool = False, thresh: float = 0.065,
                             model_type: str = 'dino_vits8', stride: int = 4, votes_percentage: int = 75,
                             sample_interval: int = 100, low_res_saliency_maps: bool = True, num_parts: int = 4,
                             num_crop_augmentations: int = 0, three_stages: bool = False,
                             elbow_second_stage: float = 0.94, save_dir: str = None) -> Tuple[List[Image.Image],
                                                                                              List[Image.Image]]:
    """
    finding cosegmentation of a set of images.
    :param image_paths: a list of paths of all the images.
    :param elbow: elbow coefficient to set number of clusters.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param votes_percentage: the percentage of positive votes so a cluster will be considered salient.
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param low_res_saliency_maps: Use saliency maps with lower resolution (dramatically reduces GPU RAM needs,
    doesn't deteriorate performance).
    :param num_parts: Number of parts of final output.
    :param num_crop_augmentations: number of crop augmentations to apply on input images. Increases performance for
    small sets with high variations.
    :param three_stages: If true, uses three clustering stages - fg/bg, non-common objects, and parts. Increases
    performance for small sets with high variations.
    :param elbow_second_stage: elbow coefficient for clustering in the second stage.
    :param save_dir: optional. if not None save intermediate results in this directory.
    :return: a list of segmentation masks and a list of processed pil images.
    """
    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    # model_type = 'dino_vitb16'
    # layer = 11
    extractor = ViTExtractor(model_type, stride, device=device)
    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    if low_res_saliency_maps:
        saliency_extractor = ViTExtractor(model_type, stride=8, device=device)
    else:
        saliency_extractor = extractor
    image_paths.sort()
    if save_dir is not None:
        save_dir = Path(save_dir)

    # create augmentations if needed
    if num_crop_augmentations > 0:
        augmentations_image_paths = []
        augmentations_dir = save_dir / 'augs' if save_dir is not None else Path('augs')
        augmentations_dir.mkdir(exist_ok=True, parents=True)
        for image_path in image_paths:
            image_batch, image_pil = extractor.preprocess(image_path, load_size)
            # print(image_batch.shape, image_pil.shape)
            # assert False
            flipped_image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            crop_size = (int(image_batch.shape[2] * 0.95), int(image_batch.shape[3] * 0.95))
            random_crop = torchvision.transforms.RandomCrop(size=crop_size)
            for i in range(num_crop_augmentations):
                random_crop_file_name = augmentations_dir / f'{Path(image_path).stem}_resized_aug_{i}.png'
                random_crop(image_pil).save(random_crop_file_name)
                random_crop_flipped_file_name = augmentations_dir / f'{Path(image_path).stem}_resized_flip_aug_{i}.png'
                random_crop(flipped_image_pil).save(random_crop_flipped_file_name)
                augmentations_image_paths.append(random_crop_file_name)
                augmentations_image_paths.append(random_crop_flipped_file_name)
        image_paths = image_paths + augmentations_image_paths

   
    # image_paths = image_paths[:200]
    num_images = len(image_paths)
    for image_path in image_paths:
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        print(load_size)
        print(image_path)
        # assert False
        image_pil_list.append(image_pil)
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin).cpu().numpy()
        print(descs.shape) #1 1 3025 384
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size #(55, 55)  (224,224)
        # print(curr_num_patches, curr_load_size) # 109 164 | 224 334
        exit()
        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        descriptors_list.append(descs)

    
    # cluster all images using k-means:
    all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
    normalized_all_descriptors = all_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
    sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
    all_sampled_descriptors = np.ascontiguousarray(np.concatenate(sampled_descriptors_list, axis=2)[0, 0])
    normalized_all_sampled_descriptors = all_sampled_descriptors.astype(np.float32)
    faiss.normalize_L2(normalized_all_sampled_descriptors)  # in-place operation
    # print(normalized_all_descriptors.shape, normalized_all_sampled_descriptors.shape) # (30500, 384) (620, 384)
     
    
    X = normalized_all_descriptors.astype(np.float32)
    print('AGGO', X.shape)
    part_labels_all = AgglomerativeClustering(n_clusters=num_parts).fit_predict(X)
        # exit()
    # print(part_labels_all.shape) #6050
    # print(num_patches_list)
    # print(load_size_list)
    # exit()

    part_labels_list = []  
    start = 0
    for npl in num_patches_list:
        num_pix = npl[0]*npl[1]
        # print(start+num_pix)
        part_labels_list.append(part_labels_all[:,start:start+num_pix])

        start = start+num_pix

    part_num_labels = num_parts*2-2 

    centroids = np.zeros((part_num_labels, X.shape[1]))
    for i in range(part_num_labels):
        centroids[i] = np.sum(X*part_labels_all[i].reshape(-1,1), axis=0)/part_labels_all[i].sum()
    #     # print(ss.shape)
    # exit()
    sim_mat  = cosine(centroids, centroids)
    # print(sim_mat.shape)
    # exit()


    part_segmentations_ori = []
    for id in range(len(image_pil_list)):
        print('~~~', id)
        part_labels = part_labels_list[id]
        print(part_labels.shape)
        part_labels = np.reshape(part_labels, (1,part_num_labels, num_patches_list[id][0], num_patches_list[id][1])).astype(np.float32)
        upsample = torch.nn.Upsample(size=load_size_list[id], mode='bilinear')
        part_labels = upsample(torch.from_numpy(part_labels)).numpy()
        print(part_labels.shape)
    
        part_labels = np.reshape(part_labels, (part_num_labels, load_size_list[id][0], load_size_list[id][1]))

    # for i in range(len(image_pil_list)):
        part_segmentations_ori.append(part_labels)

    centroids_overlap = np.reshape(centroids, (1,part_num_labels, X.shape[1]))
    thres_list = [0.0,0.1,0.2,0.3,0.4, 0.50, 0.6,0.7,0.8,0.9]
    # thres_list = [-0.5, -0.25, -0.1, 0, 0.05, 0.15]
    # thres_list = [0.9]
    part_labels = np.zeros((num_patches_list[0][0], num_patches_list[0][1]))
    pll = np.reshape(part_labels_list[0], (part_num_labels, num_patches_list[0][0], num_patches_list[0][1]))
    part_segmentations = []
    tree = []
    centroids_list = []
    for thres in thres_list:
        count = 0
        part_list = []
        parent_list = []
        tree_id_list = [] 
        for p in range(0,args.num_parts,2):
            mask_bi1 = torch.from_numpy(pll[p])#.to(dtype=torch.bool) 
            mask_bi2 = torch.from_numpy(pll[p+1])#.to(dtype=torch.bool) 
            # mask_bi1[mask_bi1>=0.001] = 1.0
            # mask_bi2[mask_bi2>=0.001] = 1.0
            mask_bi1 = mask_bi1.to(dtype=torch.bool)
            mask_bi2 = mask_bi2.to(dtype=torch.bool)
            # print(p,'parent list ', parent_list)
            if p == 0 or sim_mat[p,p+1] < thres:
                # print(p,p+1)

                if p == 0:
                    part_labels[mask_bi1] = count
                    part_list.append(p)
                    part_labels[mask_bi2] = count+1
                    part_list.append(p+1)

                    parent_list.append(None)
                    parent_list.append(None)
                    tree_id_list.append(p+1)
                    tree_id_list.append(p+2)
                    count = count+2


                else:
                    best_match = -1
                    best_sum = -1
                    # print('@ ',p)#, mask_bi1.sum(), mask_bi2.sum())
                    for idp, pr in enumerate(part_list):
                        mask_pr = torch.from_numpy(pll[pr]).to(dtype=torch.bool)
                        mix = torch.logical_or(mask_bi1, mask_bi2) 
                        # print('^^', mix.sum(), mask_bi1.sum(), mask_bi2.sum())
                        sum_intersect = (mix==mask_pr).sum()
                        # print('si', sum_intersect)
                        # sum_intersect = torch.logical_and(mix,mask_pr).sum()
                        if sum_intersect > best_sum:
                            best_sum = sum_intersect
                            best_match = idp
                    #     print(p,idp,sum_intersect)
                    # exit()
                    # print(best_sum, X.shape)
                    # print('best_sum ', best_sum, best_match)
                    if best_sum == X.shape[0]:
                        # print('best_sum ', best_sum, best_match)
                        cu_parent = part_list[best_match]
                        parent_list.append(cu_parent)
                        parent_list.append(cu_parent)
                        tree_id_list.append(tree_id_list[best_match]*2+1)
                        tree_id_list.append(tree_id_list[best_match]*2+2)

                        part_labels[mask_bi1] = count
                        part_list.append(p)
                        part_labels[mask_bi2] = count+1
                        part_list.append(p+1)
                        count = count+2
                        # print('append ', tree_id_list[-2:])

        centroids = np.zeros((count, X.shape[1]))
        # for eu, pl in enumerate(part_list):
        #     centroids[0,eu] = centroids_overlap[0,pl]
        count_use = 0
        new_part_list = []
        new_tree_id_list = []
        for eu in range(count):
            pl = np.zeros_like(part_labels)
            pl[part_labels==eu] = 1

            if(pl.sum()>0):
                # centroids[count_use] = np.sum(descriptors_list[0][0,0]*pl.reshape(-1,1), axis=0)/pl.sum()
                centroids[count_use] = np.sum(X*pl.reshape(-1,1), axis=0)/pl.sum()
                count_use = count_use+1
                new_part_list.append(part_list[eu])
                new_tree_id_list.append(tree_id_list[eu])
                # save_image(torch.from_numpy(pl)[None], f'./dino_feature/{tree_id_list[eu]}_{part_list[eu]}.png')
        centroids = np.reshape(centroids[:count_use], (1,count_use, X.shape[1]))
        tree.append(new_tree_id_list)


        for img, num_patches, load_size, descs in zip(image_pil_list, num_patches_list, load_size_list, descriptors_list):
            cur_part_num_labels = centroids.shape[1]
            # bg_centroids = tuple(i for i in range(algorithm.centroids.shape[0]) if not i in salient_labels)
            curr_normalized_descs = descs[0, 0].astype(np.float32)
            faiss.normalize_L2(curr_normalized_descs)  # in-place operation


            dist_to_parts = ((curr_normalized_descs[:, None, :] - centroids) ** 2
                            ).sum(axis=2)
            
            
            
            d_to_cent = dist_to_parts.reshape(num_patches[0], num_patches[1], cur_part_num_labels)
            # print(np.max(d_to_cent, axis=-1)[..., None].shape)#55 55 1
            d_to_cent = d_to_cent - np.max(d_to_cent, axis=-1)[..., None]
            
            upsample = torch.nn.Upsample(size=load_size)
            u = np.array(upsample(torch.from_numpy(d_to_cent).permute(2, 0, 1)[None, ...])[0].permute(1, 2, 0)).astype(np.float32)
            # print(u.shape) #224 224 26
            d = dcrf.DenseCRF2D(u.shape[1], u.shape[0], u.shape[2])
            d.setUnaryEnergy(np.ascontiguousarray(u.reshape(-1, u.shape[-1]).T))
            # compat = [50, 15]
            compat = [25, 100]

            d.addPairwiseGaussian(sxy=1, compat=compat[0], kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseBilateral(sxy=5, srgb=13, rgbim=np.array(img), compat=compat[1], kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q = d.inference(10)

            final = np.argmax(Q, axis=0).reshape(load_size)
            
            # print('www ',cc)
            parts_float = final.astype(np.float32)
            # parts_float = part_labels[cc].astype(np.float32)
            # print(parts_float[:6])
            parts_float[parts_float == cur_part_num_labels] = np.nan
            part_segmentations.append(parts_float)
            # print(parts_float.shape) # 224 224
            print(parts_float.min(), parts_float.max())
            # print(centroids.shape)
            # print(tree)
            exit()
        centroids_list.append(centroids[0])
            
    # print(len(tree))
    # print(tree)
    # print(len(centroids_list))
    # print(centroids_list[0].shape)
    # print(centroids_list[0])
    # # print(len(part_segmentations), len(tree))
    # exit()
    return [part_segmentations], image_pil_list, sim_mat, [tree], [centroids_list]


def draw_part_cosegmentation(num_parts: int, segmentation_parts: List[np.ndarray], pil_images: List[Image.Image]) -> List[plt.Figure]:
    """
    Visualizes part cosegmentation results on chessboard background.
    :param num_parts: number of object parts in all part cosegmentations.
    :param segmentation_parts: list of binary segmentation masks
    :param pil_images: list of corresponding images.
    :return: list of figures with fg segment of each image and chessboard bg.
    """
    
    figures = []
    for parts_seg_all, pil_image in zip(segmentation_parts, pil_images):
        f = []
        # for parts_seg_all in parts_seg_all:
        #     print(parts_seg_all)
        # print(len(parts_seg_all))
        for parts_seg in parts_seg_all:
        # for i in range(parts_seg.shape[0]):
            current_mask = ~np.isnan(parts_seg)  # np.isin(segmentation, segment_indexes)
            stacked_mask = np.dstack(3 * [current_mask])
            masked_image = np.array(pil_image)#np.array(pil_image.resize((224,224)))
            # print(stacked_mask.shape)
            # print(masked_image.shape)
            # print(parts_seg.shape)
            # exit()
            masked_image[~stacked_mask] = 0
            masked_image_transparent = np.concatenate((masked_image, 255. * current_mask.astype(np.uint8)[..., None]),
                                                    axis=-1)
            # create chessboard bg
            checkerboard_bg = np.zeros(masked_image.shape[:2])
            checkerboard_edge = 10
            checkerboard_bg[[x // checkerboard_edge % 2 == 0 for x in range(checkerboard_bg.shape[0])], :] = 1
            checkerboard_bg[:, [x // checkerboard_edge % 2 == 1 for x in range(checkerboard_bg.shape[1])]] = \
                1 - checkerboard_bg[:, [x // checkerboard_edge % 2 == 1 for x in range(checkerboard_bg.shape[1])]]
            checkerboard_bg[checkerboard_bg == 0] = 0.75
            checkerboard_bg = 255. * checkerboard_bg

            # show
            fig, ax = plt.subplots()
            ax.axis('off')
            color_list = ["cyan", "yellow", "blue", "lime", "darkviolet", "magenta", "red", "brown",
                      "orange", "lightseagreen", "slateblue", "deepskyblue", "indianred", "tan", "olive", "plum", "palegreen", "bisque"]
            cmap = ListedColormap(color_list)
            # cmap = ListedColormap(["cyan", "yellow"])
            ax.imshow(checkerboard_bg, cmap='gray', vmin=0, vmax=255)
            ax.imshow(masked_image_transparent.astype(np.int32), vmin=0, vmax=255)
            print('----',parts_seg.min(),parts_seg.max())
            ax.imshow(parts_seg, cmap=cmap, vmin=0, vmax=25, alpha=0.5, interpolation='nearest')
            f.append(fig)
        figures.append(f)
    return figures


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor cosegmentations.')
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    parser.add_argument('--save_dir', type=str, default='result_clustering/test/', help='The root save dir for image sets results.')
    parser.add_argument('--load_size', default=None, type=int, help='load size of the input images. If None maintains'
                                                                    'original image size, if int resizes each image'
                                                                    'such that the smaller side is this number.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--thresh', default=0.065, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--elbow', default=0.975, type=float, help='Elbow coefficient for setting number of clusters.')
    parser.add_argument('--votes_percentage', default=75, type=int, help="percentage of votes needed for a cluster to "
                                                                         "be considered salient.")
    parser.add_argument('--sample_interval', default=100, type=int, help="sample every ith descriptor for training"
                                                                         "clustering.")
    parser.add_argument('--outliers_thresh', default=0.7, type=float, help="Threshold for removing outliers.")
    parser.add_argument('--low_res_saliency_maps', default='True', type=str2bool, help="using low resolution saliency "
                                                                                       "maps. Reduces RAM needs.")
    parser.add_argument('--num_parts', default=4, type=int, help="Number of common object parts.")
    parser.add_argument('--num_crop_augmentations', default=0, type=int, help="If > 1, applies this number of random "
                                                                              "crop augmentations taking 95% of the "
                                                                              "original images and flip augmentations.")
    parser.add_argument('--three_stages', default=False, type=str2bool, help="If true, use three clustering stages "
                                                                             "instead of two. Useful for small sets "
                                                                             "with a lot of distraction objects.")
    parser.add_argument('--elbow_second_stage', default=0.94, type=float, help="Elbow coefficient for setting number "
                                                                               "of clusters.")


    parser.add_argument('--selected_img_paths', default=None, type=str) #           playground/log/img_names_seed0.pt                                                               
    parser.add_argument('--f', default=0, type=int)
    parser.add_argument('--t', default=1, type=int)


    parser.add_argument('--start_img_idx', default=0, type=int)
    parser.add_argument('--end_img_idx', default=1, type=int)


    args = parser.parse_args()

    selected_img_paths = None 
    if args.selected_img_paths is not None:
        selected_img_paths = torch.load(args.selected_img_paths)

    with torch.no_grad():

        # prepare directories
        root_dir = Path(args.root_dir)
        sets_dir = [x for x in root_dir.iterdir() if x.is_dir()]
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        for set_dir in tqdm(sets_dir):
            print(f"working on {set_dir}")



            if selected_img_paths is not None: 
                curr_images = selected_img_paths

            else: 
                curr_images = [x for x in set_dir.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                curr_images.sort()
            # curr_images = curr_images[478:]
            print('all_im: ',len(curr_images))

            curr_images = curr_images[args.f: args.t]






            for a in range(len(curr_images)):
                curr_image = curr_images[a]

                print(f'clustering on img idx-{args.start_img_idx+a}: {curr_image.name}')


                curr_save_dir = save_dir / curr_image.name[:-4]
                curr_save_dir.mkdir(parents=True, exist_ok=True)

                parts_imgs, pil_images, sim_mat, trees, centroids = find_part_cosegmentation([curr_image], args.elbow, args.load_size, args.layer,
                                                                args.facet, args.bin, args.thresh, args.model_type,
                                                                args.stride, args.votes_percentage, args.sample_interval,
                                                                args.low_res_saliency_maps, args.num_parts,
                                                                args.num_crop_augmentations, args.three_stages,
                                                                args.elbow_second_stage, curr_save_dir)

                part_figs = draw_part_cosegmentation(args.num_parts, parts_imgs, pil_images)
  
                thres_list = [0.0,0.1,0.2,0.3,0.4, 0.50, 0.6,0.7,0.8,0.9]
    
                save_dict = {}
                for parts_img,part_fig, tree, centroid in zip(parts_imgs,part_figs, trees, centroids):
     
                    for th in range(len(thres_list)):
                        cur_dict = {}
                        tree_array = np.array(tree[th])
                        centroid_array = centroid[th]
                        ths = thres_list[th]
                        cur_dict['mask'] = torch.from_numpy(parts_img[th].astype(np.int32))
                        cur_dict['dino_simmat'] = torch.from_numpy(sim_mat[th])
                        cur_dict['tree'] = torch.from_numpy(tree_array)
                        cur_dict['centroids'] = torch.from_numpy(centroid_array)

                        print(curr_save_dir / f'{ths:.2f}_{Path(curr_image).stem}_vis.png')
                        part_fig[th].savefig(curr_save_dir / f'{ths:.2f}_{Path(curr_image).stem}_vis.png', bbox_inches='tight', pad_inches=0)
                        save_dict[float(f'{ths:.2f}')] = cur_dict
                torch.save(save_dict, curr_save_dir / 'cluster_log.pt')
                plt.close('all')
