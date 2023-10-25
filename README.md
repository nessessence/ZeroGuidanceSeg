# Zero-guidance Segmentation
Official Pytorch Implementation of Zero-guidance Segmentation Using Zero Segment Labels


A ICCV 2023 paper ( [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Rewatbowornwong_Zero-guidance_Segmentation_Using_Zero_Segment_Labels_ICCV_2023_paper.pdf), [site](https://zero-guide-seg.github.io/), [5-min video](https://www.youtube.com/watch?v=sIK3ExE0HnU) ):



[<img src="https://img.youtube.com/vi/sIK3ExE0HnU/hqdefault.jpg" width="1280" height="720"
/>](https://www.youtube.com/embed/sIK3ExE0HnU)


### Overall Pipeline

<br>
<img src='figures/overall_pipeline.png'/>
<br>






### Output Examples
<br>
<img src='figures/examples.png'/>
<br>


## Run Zero Guidance Segmentation 
We provide example scripts for each stage
### DINO Clustering
```bash run_dino_clustering.sh ```
### Text Optimization and Merging
```bash run_zeroguidance.sh```
### Evaluate Results
```bash run_eval.sh```





