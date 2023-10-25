v20
CUDA_VISIBLE_DEVICES=0 python run_dino_clustering_tree.py --root_dir ../data/pascal_context/train --save_dir ./results_dino2/pascal_context/train_vits16_numpart20/ --load_size 224 --num_parts 20 --stride 2 --f 0 --t 500
v22
CUDA_VISIBLE_DEVICES=0 python run_dino_clustering_tree.py --root_dir ../data/voc2012/train --save_dir ./results_dino2/pascal2012/train_vits16_numpart20/ --load_size 224 --num_parts 20 --stride 2 --f 0 --t 500

v15
CUDA_VISIBLE_DEVICES=2 python run_dino_clustering_tree.py --root_dir ../data/coco/images/train --save_dir ./results_dino2/coco_val/train_vits16_numpart20/ --load_size 224 --num_parts 20 --stride 2 --f 0 --t 300
v20
CUDA_VISIBLE_DEVICES=0 python run_dino_clustering_tree.py --root_dir ../data/coco/images/train --save_dir ./results_dino2/coco_val/train_vits16_numpart20/ --load_size 224 --num_parts 20 --stride 2 --f 300 --t 500



CUDA_VISIBLE_DEVICES=0 python run_dino_clustering_tree.py --root_dir ../data/pascal_context/train --save_dir ./results/dino_cluster/pascal_context/train_vits16_numpart20/ --load_size 224 --num_parts 20 --stride 2 --f 0 --t 500

