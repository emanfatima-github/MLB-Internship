## README.md

### YOLO Model Improvement – V1 vs V2

What I Changed:
I improved the dataset by fixing incorrect class labels and used data augmentation such as rotation, scaling, flipping, and color changes. 
Then I retrained the model for 4 epochs to create Model V2.

Why I Changed It:
The goal was to improve the model's ability to detect bottles and make it perform better on different images.

Results:

| Metric    |     V1 |     V2 |
| --------- | -----: | -----: |
| Precision | 0.7772 | 0.7756 |
| Recall    | 0.5114 | 0.4886 |
| mAP@50    | 0.5764 | 0.5809 |
| mAP@50-95 | 0.3263 | 0.2766 |

Did the Model Improve?
V2 improved slightly in mAP@50, but the overall performance did not improve. Precision, Recall, and mAP@50-95 decreased. 
This shows that augmentation does not always improve a model, especially with limited training epochs.
