# Custom Object Detection System

## Dataset

I selected the Construction Equipment Detection dataset from Roboflow Universe. 
## Training Configuration

* Model: YOLOv8 Nano (`yolov8n.pt`)
* Epochs: 5
* Batch Size: 4
* Image Size: 320 × 320

## Final Evaluation Metrics

The model was evaluated after training using the validation dataset. The evaluation included Precision, Recall, mAP@50, and mAP@50-95 
to measure the detection performance.

## Challenges and Improvements

Training on a CPU took a long time, so I reduced the number of epochs, image size, and batch size to speed up the process. 
If higher accuracy is required, the model can be improved by increasing the number of epochs, using a larger image size, and applying data augmentation.
