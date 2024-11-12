#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results_hints.py
#                                                                             
# PROGRAMMER: Naila RAIS
# DATE CREATED: 2024-11-11
# REVISED DATE: 
# PURPOSE: This is a *hints* file to help guide students in creating the 
#          function print_results that prints the results statistics from the
#          results statistics dictionary (results_stats_dic). It should also
#          allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results 
#          dictionary (results_dic).  
#         This function inputs:
#            - The results dictionary (results_dic) and results for the function call within main.
#            - The results statistics dictionary (results_stats_dic) and results_stats for the function call within main.
#            - The CNN model architecture (model) within print_results function and in_arg.arch for the function call within main. 
#            - Prints Incorrectly Classified Dogs as print_incorrect_dogs within print_results function and set as either True or False in the main function call (defaults to False)
#            - Prints Incorrectly Classified Breeds as print_incorrect_breed within print_results function and set as either True or False in the main function call (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.
##

def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if requested.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             [0] = pet image label (string)
             [1] = classifier label (string)
             [2] = 1/0 (int) where 1 = match between pet image and classifier labels, 0 = no match
             [3] = 1/0 (int) where 1 = pet image 'is-a' dog, 0 = pet image 'is-NOT-a' dog
             [4] = 1/0 (int) where 1 = Classifier classifies image 'as-a' dog, 0 = 'as-NOT-a' dog
      results_stats_dic - Dictionary containing the results statistics with key names starting with 'pct' for percentages or 'n' for count
      model - The CNN model architecture used by the classifier function (resnet, alexnet, or vgg)
      print_incorrect_dogs - If True, prints incorrectly classified dog images, otherwise does not (default False)
      print_incorrect_breed - If True, prints incorrectly classified dog breeds, otherwise does not (default False)
    Returns:
           None - simply printing results.
    """   
    print("\n\n*** Results Summary for CNN Model Architecture", model.upper(), "***")
    print("{:20}: {:3d}".format('N Images', results_stats_dic['n_images']))
    print("{:20}: {:3d}".format('N Dog Images', results_stats_dic['n_dogs_img']))
    print("{:20}: {:3d}".format('N Not-Dog Images', results_stats_dic['n_notdogs_img']))

    # Print summary statistics (percentages) on Model Run
    print(" ")
    for key in results_stats_dic:
        if key.startswith('pct'):
            print("{:20}: {:.2f}".format(key, results_stats_dic[key]))

    # If print_incorrect_dogs is True and there were misclassified dogs, print these cases
    if (print_incorrect_dogs and 
        (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs']) 
          != results_stats_dic['n_images']):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for key in results_dic:
            if (results_dic[key][3] == 1 and results_dic[key][4] == 0) or \
               (results_dic[key][3] == 0 and results_dic[key][4] == 1):
                print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0],
                                                                  results_dic[key][1]))

    # If print_incorrect_breed is True and there were dogs whose breeds were incorrectly classified, print these cases                    
    if (print_incorrect_breed and 
        (results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed'])):
        print("\nINCORRECT Dog Breed Assignment:")
        for key in results_dic:
            if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0):
                print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0],
                                                                  results_dic[key][1]))
