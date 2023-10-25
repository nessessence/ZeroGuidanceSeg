CUDA_VISIBLE_DEVICES=0 python run_autotext_segment_merge.py -o 'results_data/context/merge/l20-23_-gb5' -idx 0-5000 -d pascal_context -s val --no-compute_all_nodes   --apply_merge   --text_encoder clip --merge_choice mix --merge_thres 0.8  -sd 5

