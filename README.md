#  Nail Condition Analyzer 

This project tackles nail health assessment in two stages:

1. Classifying nail diseases from images
2. Segmenting the affected region of the nail to estimate how severe the condition is...

# Disease Classification
The starting point was the approach described in Han et al. 2018, which I used as a baseline and then improved on using aggressive data augmentation techniques (cutout and mixup) to boost accuracy beyond the original results. To further validate the model, I evaluated it on an independent test set of European nail images collected from the web. The final model, an ensemble of EfficientNet-B5 and ResNet-152 — performs strongly at the binary task of separating healthy nails from diseased ones. Pinpointing the exact disease, however, remains a harder problem.

# Severity Segmentation
For the second stage, I trained a YOLOv5x-Segmentation model to localize the specific area of the nail impacted by disease, which serves as a proxy for severity. The model produces reasonably accurate segmentation masks of the affected region.
