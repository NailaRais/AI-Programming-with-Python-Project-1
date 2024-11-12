#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from classifier import classifier
import os

def classify_images(images_dir, results_dic, model):
    """ 
    Creates classifier labels with classifier function, compares pet labels 
    to classifier labels, and adds classifier label and comparison results 
    to the results dictionary. Formats classifier labels to lowercase and 
    strips leading/trailing whitespace for consistent comparison.
    
    Parameters:
    - images_dir: (str) Path to the folder of images for classification.
    - results_dic: (dict) Results dictionary with 'key' as image filename 
      and 'value' as a list containing:
        - index 0: pet image label (str)
        - NEW index 1: classifier label (str)
        - NEW index 2: 1/0 (int) - 1 if match between labels, else 0
    - model: (str) CNN model architecture for classification ("resnet", "alexnet", "vgg")
    
    Returns:
    None (modifies results_dic in-place)
    """

    # Ensure the images_dir has a trailing slash to avoid path issues
    if not images_dir.endswith(os.path.sep):
        images_dir += os.path.sep

    # Process all files in the results_dic
    for key in results_dic:
        # 3a. Run classifier function to classify the image and set model_label
        model_label = classifier(os.path.join(images_dir, key), model)

        # 3b. Process the model_label for consistency
        model_label = model_label.lower().strip()

        # Define the true pet image label
        truth = results_dic[key][0]

        # 3c. Check if truth label matches any label in model_label
        # and use extend to add model_label and comparison result (1 or 0)
        if truth in model_label:
            results_dic[key].extend([model_label, 1])
        else:
            results_dic[key].extend([model_label, 0])
