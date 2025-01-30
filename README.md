# Underwater_Image_Segmentation


### **Underwater Image Segmentation with USIS-SAM & YOLOv8**  
**Comparing USIS-SAM-based segmentation with YOLOv8-seg on underwater datasets**  

---

## 📌 **Project Overview**  
This project implements and evaluates an underwater image segmentation model based on the paper:  
[📄 Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset](https://arxiv.org/abs/2406.06039).  

We retrained the SAM-based model (USIS-SAM) on the **large-scale Underwater Salient Instance Segmentation dataset (USIS10K)**, which comprises **10,632 underwater images** with pixel-level annotations across **seven categories** from diverse underwater scenes. Additionally, we fine-tuned the **YOLOv8-seg model** for comparison, aiming to enhance segmentation performance in underwater environments.
---

## **Repository Structure**  
```
Underwater_Image_Segmentation/
│── src/                 # Python scripts for training & testing
│   ├── coco_to_yolo.py   # Converts COCO annotations to YOLO format
│   ├── inference.py      # Runs inference on test images
│   ├── compare_results.py # Compares SAM & YOLOv8-seg results
│   ├── job.sh            # train the model on HPC
│── results/             # visualization images
│── README.md            # Project documentation


---

## **HPC Training with SLURM **
To efficiently train the model, we used a High-Performance Computing (HPC) cluster with SLURM for job scheduling.
``` bash
sbatch job.sh  
```
---

## **Dataset**  
- The dataset consists of **10,000 underwater images** with instance segmentation annotations.  
- The dataset follows the **COCO format**, which we converted into **YOLO format** using `coco_to_yolo.py`.  

---


##  **Model Training**  
### **Train the SAM-based Model**  
```bash
python tools/train.py project/our/configs/multiclass_usis_train.py
```
  
### **Fine-Tune YOLOv8-Seg Model**  
```bash
yolo segment train data=dataset.yaml model=yolov8-seg.pt epochs=50 imgsz=640
```

---

## 🔍 **Inference & Testing**  
### **Run Inference**  
```bash
python src/vis_infer.py
python src/inference.py --model yolov8 --image test_image.jpg
```

### **Compare Results Against Ground Truth**  
```bash
python src/maks_visualisation.py 
```


---

##  **References**  
- **Original Research Paper:** [📄 Diving into Underwater](https://arxiv.org/abs/2406.06039)  
- **SAM Model:** [Meta's Segment Anything Model](https://github.com/facebookresearch/segment-anything)  
- **YOLOv8-Seg:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  

