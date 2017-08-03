# Nuclei Segmentation and localization

We find and classify the normal epithelial, malignant epithelial and lymphocyte cells in subsections of H&E stained histology images taken at 20x magnification. Example with the malignant cells labelled in red:
<img src="https://github.com/renebidart/cancer_hist/blob/master/Images/labeled_cell.png"  width = "300"/>



---
Some visualizations of the data, as well as some problem areas are in [exploring_data notebook](notebooks/exploring_data.ipynb).

---
Procedure:
The general outline of our method is:
1. **Classifier** - Train a classifier to detect if an image is centered on a normal, malignant or lymphocyte nuclei, or else not on any nuclei. We [create a data set](notebooks/read_make_data.ipynb) by selecting square regions centered at the nuclei centers. We test different fully convolutional and regular CNNs. Augmentation is with flips, rotations and cropping. ([code](src/heat_models.py))
2. **Heat Maps** - Apply the classifier to the image, outputting a 4 dimensional heatmap of the probability of each cell class. ([code](src/gen_heatmaps_fc.py))
3. **Cell Locations** - Based on the heat maps, output locations and classifications for the cells using [non-maximum supression](https://github.com/renebidart/cancer_hist/blob/master/notebooks/non-max%20supression.ipynb)

Some of the testing is included in this [notebook](notebooks/evaluate_models.ipynb)

---
Other  [regression based models](src/reg_models.py) and [unet](src/unet_models.py) based models are also tested.
---
