import pandas as pd
import numpy as np
import re

def parse_log_file(file_path):
    outputs = []
    groundtruths = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    regex_pattern = r"[-+]?\d*\.\d+|\d+"

    for i, line in enumerate(lines):
        if 'Output (sample):' in line:
            numbers = re.findall(regex_pattern, line)
            numbers2 = re.findall(regex_pattern, lines[i+1])
            outputs.append([float(numbers[0]),float(numbers2[0])])
        elif 'Groundtruth (sample):' in line:
            numbers = re.findall(regex_pattern, line)
            numbers2 = re.findall(regex_pattern, lines[i+1])
            groundtruths.append([float(numbers[0]),float(numbers2[0])])
    
    return outputs, groundtruths

def parse_log_file_old(file_path):
    outputs = []
    groundtruths = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    regex_pattern = r"[-+]?\d*\.\d+|\d+"

    for i, line in enumerate(lines):
        if 'Output (sample):' in line:
            numbers = re.findall(regex_pattern, line)
            numbers2 = re.findall(regex_pattern, line)
            outputs.append([float(numbers[0]),float(numbers2[0])])
        elif 'Groundtruth (sample):' in line:
            numbers = re.findall(regex_pattern, line)
            numbers2 = re.findall(regex_pattern, line)
            groundtruths.append([float(numbers[0]),float(numbers2[0])])
    
    return outputs, groundtruths
# def parse_log_file_old(file_path):
#     outputs = []
#     groundtruths = []
    
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
    
#     for i, line in enumerate(lines):
#         if 'Output (sample):' in line:
#             # The next line contains the single output value
#             output_value = float(lines[i + 1].strip().strip('[]'))
#             outputs.append([output_value])
#         elif 'Groundtruth (sample):' in line:
#             # The next line contains the single groundtruth value
#             groundtruth_value = float(lines[i + 1].strip().strip('[]'))
#             groundtruths.append([groundtruth_value])
    
#     return outputs, groundtruths

def calculate_individual_mape(outputs, groundtruths):
    outputs = np.array(outputs)
    groundtruths = np.array(groundtruths)
    
    # Initialize lists to hold MAPE values for first and second values
    mape_first_values = []
    mape_second_values = []
    
    # Iterate through each pair of output and groundtruth
    for output, groundtruth in zip(outputs, groundtruths):
        # Calculate MAPE for the first value, if groundtruth is not zero
        if groundtruth[0] != 0:
            mape_first = np.abs((output[0] - groundtruth[0]) / groundtruth[0]) * 100
            mape_first_values.append(mape_first)
        
        # Calculate MAPE for the second value, if groundtruth is not zero
        if groundtruth[1] != 0:
            mape_second = np.abs((output[1] - groundtruth[1]) / groundtruth[1]) * 100
            mape_second_values.append(mape_second)
    
    # Calculate the average MAPE for first and second values
    average_mape_first = np.mean(mape_first_values) if mape_first_values else float('nan')
    average_mape_second = np.mean(mape_second_values) if mape_second_values else float('nan')
    
    return average_mape_first, average_mape_second

log_file_path = '../../results/test_result_v2_unseen_24-1.log'
log_file_path = '../../train_result.log'

outputs, groundtruths = parse_log_file(log_file_path)

# outputs, groundtruths = parse_log_file_old(log_file_path)

# print(outputs)

# Calculate the average MAPE
average_mape_green, average_mape_blue = calculate_individual_mape(outputs, groundtruths)

print(f"Average Green MAPE: {average_mape_green}%")
print(f"Average Blue MAPE: {average_mape_blue}%")
