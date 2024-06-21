from torchvision.models.vision_transformer import VisionTransformer
from transformers import EfficientFormerImageProcessor, EfficientFormerForImageClassificationWithTeacher
from timm.data import resolve_data_config
from datasets import load_dataset
from transformers import pipeline
from PIL import Image
from datetime import datetime

import ast
import os
import copy
import numpy as np
import torch
import pandas as pd
import time
import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as torchvision_transforms_functional
import torch.nn.functional as torch_nn_functional
import torch.nn.utils.prune as prune
import torch_pruning as tp
import torchvision.transforms.functional as torchvision_transforms_functional
import torch._dynamo
import subprocess
import requests
import json
import threading
import csv
torch._dynamo.config.suppress_errors = True


def load_models(task):
    models = pd.read_csv(f'../models/HF_{task}_models_filtered_stratified_sampled.csv')
    models['datasets'] = models['datasets'].fillna('')
    models.reset_index(inplace=True)

    return models


def get_model_sparsity(model):
    zero_weights = 0
    total_weights = 0
    for _, param in model.named_parameters():
        zero_weights += torch.sum(param == 0).item()
        total_weights += param.nelement()
    sparsity = 100 * zero_weights / total_weights

    return sparsity


def estimate_flops_macs_params(model, image):
    # Estimate FLOPs using thop (it only works for pytorch models)

    # Convert the image to a PyTorch tensor
    image_tensor = torchvision_transforms_functional.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    if hasattr(model, 'model'):
        model = model.model  # Access the underlying PyTorch model

    if hasattr(model, 'config') and hasattr(model.config, 'image_size'):
        model_config = model.config  # Accessing the model configuration
        image_size = model_config.image_size
        input_shape = (3, image_size, image_size)
    else:
        # Resize the image to the input shape of the model
        input_shape = resolve_data_config({}, model=model)['input_size']

    image_tensor = torch_nn_functional.interpolate(
        image_tensor, size=input_shape[1:], mode='bilinear', align_corners=False)
    macs, params = tp.utils.count_ops_and_params(model, image_tensor)

    flops = macs * 2
    sparsity = get_model_sparsity(model)
    total_model_size = sum(p.element_size() * p.numel()
                           for p in model.parameters())  # in bytes

    return flops, macs, params, sparsity, total_model_size


def log_error(task, exp_id, model_name, size, ds, datasets, datasets_size, library, file_size_quartile, popularity_quartile, error_df, error_message):
    print("ERROR")
    print(error_message)
    
    
    # Log error and update the error dataframe
    new_data = pd.DataFrame([[exp_id, model_name, size, datasets, datasets_size, library, file_size_quartile, popularity_quartile, error_message]], columns=[
                            'Experiment', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library', 'File size quartile', 'Popularity quartile', 'Error_Message'])
    error_df = pd.concat([error_df, new_data], ignore_index=True)
    error_df.to_csv(f'error_log_{task}_{ds}_opt.csv', index=False)

    return error_df
    

# Global variable to control the loop
running = True

def get_wattmeter_data():
    filename = f'wattmeter_{task}.csv'
    error_log = f'wattmeter_error_log_{task}.txt'
    fieldnames = ['Wattmetter Timestamp', 'True timestamp', 'ID', 'Name', 'State', 'Action', 'Delay',
                  'Current', 'PowerFactor', 'Phase', 'Energy', 'ReverseEnergy', 'EnergyNR', 'ReverseEnergyNR', 'Load']
    url = "http://147.83.72.195/netio.json"
    
    while running:
        with open(filename, mode='a', newline='') as file:
            try:
                response = requests.get(url)
                
                # Parse JSON data
                data = json.loads(response.text)

                timestamp = data['Agent']['Time']
                print(timestamp)

                # Extract data for output ID 1
                output_1_data = None
                for output in data['Outputs']:
                    if output['ID'] == 1:
                        output_1_data = output
                        break

                if output_1_data:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)

                    if file.tell() == 0:
                        writer.writeheader()

                    output_1_data['Wattmetter Timestamp'] = timestamp
                    output_1_data['True timestamp'] = datetime.fromtimestamp(
                        time.time())
                    writer.writerow(output_1_data)
                else:
                    error_message = f"{datetime.now()} - Output ID 1 data not found in the JSON.\n"
                    with open(error_log, 'a') as error_file:
                        error_file.write(error_message)
            except Exception as e:
                error_message = f"{datetime.now()} - {e}\n"
                with open(error_log, 'a') as error_file:
                    error_file.write(error_message)
            finally: # lets you execute code, regardless of the result of the try-except block
                time.sleep(0.5)  # Pause execution for 0.5 seconds before next call


#########################
# OPTIMIZATION FUNCITONS
#########################

def compile_model(model):
    model = torch.compile(model)
    model.eval()

    return model


def prune_locally_structured(model, pruning_amount):
    # Apply structured pruning along the first dimension of the weight tensor (i.e., the rows of a matrix), based on the channels’ L1 norm.

    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm, torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Embedding, torch.nn.MultiheadAttention,
                               torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
            #  As structured pruning can only be applied to multidimensional tensors, we need to check if the tensor is multidimensional
            if len(module.weight.shape) > 1:
                # Prune tensor by removing channels with the lowest Ln-norm along the specified dimension.
                prune.ln_structured(module, name='weight',
                                    amount=pruning_amount, n=1, dim=0)

                # Prune tensor by removing random channels along the specified dimension.
                # prune.random_structured

                # Prune tensor by removing random (currently unpruned) units.
                # prune.random_unstructured

                prune.remove(module, 'weight')

    model.eval()

    return model


def prune_globally_unstructured(model, pruning_amount):
    # Define the parameters to prune globally
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm, torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Embedding, torch.nn.MultiheadAttention,
                               torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
            parameters_to_prune.append((module, 'weight'))

    # Perform global pruning on all parameters
    prune.global_unstructured(
        parameters_to_prune,
        # This method performs unstructured pruning based on the L-1 norm of the weights.
        pruning_method=prune.L1Unstructured,

        # Prune entire (currently unpruned) channels in a tensor at random.
        # pruning_method = prune.RandomStructured

        # Prune (currently unpruned) units in a tensor at random.
        # pruning_method = prune.RandomUnstructured

        amount=pruning_amount
    )

    # Remove the original tensors from the model and keep only the pruned tensors (masked tensors)
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm, torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d, torch.nn.Embedding, torch.nn.MultiheadAttention,
                               torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
            prune.remove(module, 'weight')

    model.eval()

    return model


def dynamic_quantize_model(model):
    #backend = 'qnnpack' # NO!!!

    # Set the backend engine for quantization, which is used by the quantized operators to perform the computations.
    #torch.backends.quantized.engine = backend # NO!!!

    model = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8)
        
    #print("quantized model", model)
    
    model.eval()

    return model
    
 

# Define the quantization-aware model
class QuantizedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.model = base_model
        
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        print("BEFORE SELF.QUANT")
        x = self.quant(x)
        
        print("BEFORE SELF.MODEL")
        #print("MODEL:", self.model)
        
        #x = self.model(x) # error function 
        
        x = self.model.vit(x) 
        print("BEFORE SELF.DEQUANT")
        x = self.dequant(x)
        
        print("BEFORE self.model.vit.encoder")
        x = self.quant(x)
        x = self.model.classifier(x) 
        x = self.dequant(x)
        
        print("AFTER SELF.DEQUANT")
        return x
        
def static_quantize_model(model): # PYTORCH STATIC QUANTIZATION
    #print("model before quantization", model)

    # Set the model to evaluation mode before q∫uantization
    model.eval()

    # Prepare the model for quantization (qconf == quantization configuration, it specifies how should we quantize the model)
    # It is necessary to set the qconfig before calling prepare
    # Mandatory to perform quantization!!!
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    
    # Modify the model's modules' qconfig based on their type
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            # Set quantization configuration for embeddings
            mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    #model = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])


    # CALIBRATION: Prepare the model for quantization. It applies the configured quantization settings (from qconfig) to the py_model.
    # This inserts observers in the model that will observe activation tensors during calibration.
    model_static_prepared = torch.ao.quantization.prepare(model)
    print("Model prepared for quantization")

    if hasattr(model, 'config') and hasattr(model.config, 'image_size'):
        model_config = model.config # Accessing the model configuration
        image_size = model_config.image_size
        input_shape = (3, image_size, image_size)
    else:
        # Resize the image to the input shape of the model
        input_shape = resolve_data_config({}, model=model)['input_size']
    input_fp32 = torch.randn(1, 3, input_shape[1], input_shape[1])
    print("input_fp32 shape:", input_fp32.shape)

    # Perform inference with the prepared model to dynamically calibrate activations
    # Calibrate the prepared model to determine quantization parameters for activations
    model_static_prepared(input_fp32)
    print("Performed inference for dynamic calibration")

    # Convert the prepared model to a quantized version. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.ao.quantization.convert(model_static_prepared)
    
    print("Model has been quantized")
    print(type(model_int8))
    #print("model after quantization", model_int8)
    
    print("Quantized model saved locally")
    
    #print(sum(p.element_size() * p.numel()
    #                       for p in model.parameters()))
    
    print("input_fp32 type:", input_fp32.dtype)
    
    model_int8.eval()
    with torch.no_grad():
        print('inside torch grad')
        print("EXAMPLE INFERENCE:", model_int8(input_fp32)) 
    return model_int8

    

######################
# IMAGE CLASSIFICATION
######################

def perform_inference_image_classification(model_name, models, loaded_datasets, ds, datasets, task, m, library, result_df, error_df):
    
    if library == "transformers" or library == "transformers.js" or library == "nemo":
        for optimization in optimizations:
            # Repeat the experiment 10 times
            for exp_id in range(10):
                try:
                    print("\n", optimization)
                    model = pipeline(task, model=model_name)

                    if "pruning" in optimization:
                        #  The optimization word ends with a number that indicates the pruning amount
                        pruning_amount = float(optimization.split("_")[2])

                        model = model.model  # Access the underlying PyTorch model

                        if "local_pruning" in optimization:  # Prune the model using pytorch local pruning
                            model = prune_locally_structured(
                                model, pruning_amount=pruning_amount)
                        elif "global_pruning" in optimization:  # Prune the model using pytorch global pruning
                            model = prune_globally_unstructured(
                                model, pruning_amount=pruning_amount)

                    elif optimization == "torch.compile":
                        model = model.model  # Access the underlying PyTorch model
                        model = compile_model(model)    
                        
                    elif optimization == "static_quantization":
                        model = model.model
                        model = static_quantize_model(QuantizedModel(model))
                        
                    elif optimization == "dynamic_quantization":
                        model = model.model
                        model = dynamic_quantize_model(model)

                    # Use 100 images from the test set to evaluate the model or the maximum number of images available
                    for i in range(100):
                        print("\n Image", i)
                        
                        print("start inference pass", datetime.fromtimestamp(time.time()))
                        print("before reading img", datetime.fromtimestamp(time.time()))
                        
                        if "imagenet" in datasets:
                            # Open '../datasets/ILSVRC2010_test/ILSVRC2010_test_common_gt_categories.txt' and read the i-th line
                            with open("../datasets/ILSVRC2010_test/ILSVRC2010_test_common_gt_categories.txt") as f:
                                lines = f.readlines()
                                #  i+1 because the first line is the header
                                label = lines[i+1].split(', ')
                                img_idx = label[0]
                                y_true = int(label[1].strip("'"))
                                y_true_name = label[2]
                                # Remove the quotes from the label name
                                y_true_name = y_true_name[1:-2]

                            # Create a string with 8 chars that begins with 0s and ends with the id of the image
                            id = str(img_idx).zfill(8)
                            image = Image.open(
                                f"../datasets/ILSVRC2010_test/ILSVRC2010_test_{id}.JPEG")
                        else:
                            with open(f"../datasets/{ds}/mapping.txt") as f:
                                lines = f.readlines()
                                label = lines[i].split('	')
                                img_name = label[0]
                                y_true = label[1].strip()  # Remove '\n' using strip()
                                
                            print("y_true", y_true)
                           
                            with open(f"../datasets/{ds}/class_names.txt") as f:
                                lines = f.readlines()
                                y_true_name = lines[int(y_true)].rstrip()  # Remove '\n' from the end of the line
                                print("y_true_name", y_true_name)

                            image = Image.open(
                                f"../datasets/{ds}/{img_name}")
                        
                        print("after reading img", datetime.fromtimestamp(time.time()))

                        # Get image size
                        image_size = image.size

                        # Get the input shape expected by the model
                        input_size = resolve_data_config(
                            {}, model=model)['input_size']

                        # Record the start time of the inference
                        start_time = time.time()
                     
                        if optimization == "no_optimization":
                            # Perform inference on the image
                            predictions = model.predict(image)

                            # Extract the predicted label (label name) with the highest score
                            y_pred = max(predictions, key=lambda x: x['score'])[
                                'label']

                        else:
                            # Convert the image to a PyTorch tensor
                            image_tensor = torchvision_transforms_functional.to_tensor(
                                image)
                            image_tensor = image_tensor.unsqueeze(
                                0)  # Add batch dimension

                            try:
                                image_tensor = torch_nn_functional.interpolate(
                                    image_tensor, size=input_size[1:], mode='bilinear', align_corners=False)

                                with torch.no_grad():
                                    outputs = model(image_tensor)
                            except:
                                # Accessing the model configuration
                                model_config = model.config
                                input_size = model_config.image_size
                                input_size = (3, input_size, input_size)
                                image_tensor = torch_nn_functional.interpolate(
                                    image_tensor, size=input_size[1:], mode='bilinear', align_corners=False)

                                with torch.no_grad():
                                    outputs = model(image_tensor)

                            logits = outputs.logits

                            # Apply softmax to obtain probabilities
                            probabilities = torch_nn_functional.softmax(
                                logits, dim=1)

                            # Get the index of the maximum probability (predicted class)
                            y_pred_id = int(torch.argmax(probabilities, dim=1))

                            # Get the name of the predicted label
                            if "imagenet" in datasets:
                                if "imagenet-21k" in datasets:
                                    with open("../datasets/imagenet21k_wordnet_lemmas.txt") as f:
                                        lines = f.readlines()
                                        y_pred = lines[y_pred_id][:-1]
                                else:
                                    with open("../datasets/imagenet-1k_label_names.txt") as f:
                                        lines = f.readlines()
                                        y_pred = lines[y_pred_id].split(': ')[1]
                                        y_pred = y_pred[:-2]
                            else:
                                y_pred = y_pred_id

                        # Record the end time of the inference
                        end_time = time.time()
                        print("end time inf", datetime.fromtimestamp(end_time))
                
                        # If the prediction is "LABEL_"+number, remove the "LABEL_" prefix and convert the number to a string
                        if "imagenet" in datasets and y_pred.startswith("LABEL_"):
                            # Remove the "LABEL_" prefix
                            y_pred = int(y_pred[6:])

                            if "imagenet-21k" in datasets:
                                with open("../datasets/imagenet21k_wordnet_lemmas.txt") as f:
                                    lines = f.readlines()
                                    y_pred_name = lines[y_pred][:-1]
                            elif "imagenet" in datasets:
                                with open("../datasets/imagenet-1k_label_names.txt") as f:
                                    lines = f.readlines()
                                    y_pred_name = lines[y_pred].split(': ')[1]
                                    y_pred_name = y_pred_name[:-2]
                            else:
                                y_pred_name = int(y_pred)
                        else:
                            y_pred_name = y_pred

                        # Check if the prediction is correct
                        correct_predictions = 1 if y_true_name == y_pred_name else 0
        
                        # Get the FLOPs, MACs, and number of parameters of the model
                        if optimization == "torch.compile" and hasattr(model, '_orig_mod'):
                            model = model._orig_mod
                        
                        print("before flops", datetime.fromtimestamp(time.time()))
                        
                        flops, macs, params, sparsity, total_model_size = estimate_flops_macs_params(
                            model, image)
                            
                        print("after flops", datetime.fromtimestamp(time.time()))
                        print("inference time", end_time - start_time)
                        print("flops, macs, params, total_model_size:", flops, macs, params, total_model_size)

                        # Append the results to the dataframe
                        new_data = pd.DataFrame([[exp_id, optimization, model_name, total_model_size, datasets, models.iloc[m]['datasets_size'], library,
                                                models.iloc[m]['file_size_quartile'], models.iloc[m][
                                                    'popularity_quartile'], i, image_size, input_size,
                                                flops, macs, params, y_true, y_true_name, y_pred_name, correct_predictions, datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time), end_time - start_time, sparsity]],
                                                columns=['Experiment', 'Optimization', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library',
                                                        'File size quartile', 'Popularity quartile', 'Image ID', 'Image size', 'Input size',
                                                        'FLOPs', 'MACs', 'Parameters', 'y_true', 'y_true_name', 'y_pred', 'Correct Prediction', 'Start Time', 'End Time', 'Total Time', 'Global Sparsity'])
                                                        
                        result_df = pd.concat(
                            [result_df, new_data], ignore_index=True)
                        result_df.to_csv(
                            f'results_{task}_{ds}_opt.csv', index=False)
                            
                        print("end inference pass", datetime.fromtimestamp(time.time()))

                except Exception as e:
                    # Handle errors and log them
                    error_message = f"Error for model {model_name}: {str(e)}"
                    error_df = log_error(task, exp_id, model_name, models.iloc[m]['size'], ds, datasets, models.iloc[m]['datasets_size'],
                                        library, models.iloc[m]['file_size_quartile'], models.iloc[m]['popularity_quartile'], error_df, error_message)

    elif library == "timm":
        for optimization in optimizations:
            # Repeat the experiment 10 times
            for exp_id in range(10):
                try:
                    print("\n", optimization)

                    processor = EfficientFormerImageProcessor.from_pretrained(
                        model_name)
                    model = EfficientFormerForImageClassificationWithTeacher.from_pretrained(
                        model_name)

                    if "pruning" in optimization:
                        #  The optimization word ends with a number that indicates the pruning amount
                        pruning_amount = float(optimization.split("_")[2])

                        if "local_pruning" in optimization:
                            model = prune_locally_structured(
                                model, pruning_amount=pruning_amount)
                        elif "global_pruning" in optimization:
                            model = prune_globally_unstructured(
                                model, pruning_amount=pruning_amount)

                    elif optimization == "torch.compile":
                        model = compile_model(model)
                    elif optimization == "dynamic-quantization":
                        model = dynamic_quantize_model(model)

                    # Use the 100 images from the test set to evaluate the model or the maximum number of images available
                    for i in range(100):
                        print("image", i)
                        if "imagenet" in datasets:
                            # Open '../datasets/ILSVRC2010_test/ILSVRC2010_test_common_gt_categories.txt' and read the i-th line
                            with open("../datasets/ILSVRC2010_test/ILSVRC2010_test_common_gt_categories.txt") as f:
                                lines = f.readlines()
                                #  i+1 because the first line is the header
                                label = lines[i+1].split(', ')
                                img_idx = label[0]
                                y_true = int(label[1].strip("'"))
                                y_true_name = label[2]
                                # Remove the quotes from the label name
                                y_true_name = y_true_name[1:-2]

                            # Create a string with 8 chars that begins with 0s and ends with the id of the image
                            id = str(img_idx).zfill(8)
                            image = Image.open(
                                f"../datasets/ILSVRC2010_test/ILSVRC2010_test_{id}.JPEG")
                        else:
                            with open(f"../datasets/{ds}/mapping.txt") as f:
                                lines = f.readlines()
                                label = lines[i].split('	')
                                img_name = label[0]
                                y_true = label[1]
                                y_true_name = y_true.strip()  # Remove '\n' using strip()
                                print(img_name)
                                print(y_true)
                            
                            print("y_true", y_true)
                           
                            with open(f"../datasets/{ds}/class_names.txt") as f:
                                lines = f.readlines()
                                y_true_name = lines[int(y_true)].rstrip()  # Remove '\n' from the end of the line
                                print("y_true_name", y_true_name)

                            image = Image.open(
                                f"../datasets/{ds}/{img_name}")

                        # Get image size
                        image_size = image.size

                        # Preprocess input image
                        inputs = processor(images=image, return_tensors="pt")

                        # Get the input shape expected by the model: after applying the processor
                        input_size = inputs['pixel_values'].shape
                        # As it is a tensor, we need to convert it to a tuple
                        input_size = tuple(input_size)

                        # Record the start time of the inference
                        start_time = time.time()

                        # Inference
                        with torch.no_grad():
                            outputs = model(**inputs)

                        # Record the end time of the inference
                        end_time = time.time()

                        logits = outputs.logits
                        scores = torch.nn.functional.softmax(logits, dim=1)
                        y_pred = torch.argmax(scores, dim=1).item()

                        if "imagenet-21k" in datasets:
                            with open("../datasets/imagenet21k_wordnet_lemmas.txt") as f:
                                lines = f.readlines()
                                y_pred_name = lines[y_pred][:-1]
                        elif "imagenet" in datasets:
                            with open("../datasets/imagenet-1k_label_names.txt") as f:
                                lines = f.readlines()
                                y_pred_name = lines[y_pred].split(': ')[1]
                                y_pred_name = y_pred_name[:-2]
                        else:
                            # Convert the number to a string
                            y_pred_name = int(y_pred)

                        # Check if the prediction is correct
                        correct_predictions = 1 if y_true_name == y_pred_name else 0

                        # Get the FLOPs, MACs, and number of parameters of the model
                        if optimization == "torch.compile" and hasattr(model, '_orig_mod'):
                            model = model._orig_mod

                        flops, macs, params, sparsity, total_model_size = estimate_flops_macs_params(
                            model, image)

                        # Append the results to the dataframe
                        new_data = pd.DataFrame([[exp_id, optimization, model_name, total_model_size, datasets, models.iloc[m]['datasets_size'], library,
                                                models.iloc[m]['file_size_quartile'], models.iloc[m][
                                                    'popularity_quartile'], i, image_size, input_size,
                                                flops, macs, params, y_true, y_true_name, y_pred_name, correct_predictions, datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time), end_time - start_time, sparsity]],
                                                columns=['Experiment', 'Optimization', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library',
                                                        'File size quartile', 'Popularity quartile', 'Image ID', 'Image size', 'Input size',
                                                        'FLOPs', 'MACs', 'Parameters', 'y_true', 'y_true_name', 'y_pred', 'Correct Prediction', 'Start Time', 'End Time', 'Total Time', 'Global Sparsity'])

                        result_df = pd.concat(
                            [result_df, new_data], ignore_index=True)
                        result_df.to_csv(
                            f'results_{task}_{ds}_opt.csv', index=False)

                except Exception as e:
                    # Handle errors and log them
                    error_message = f"Error for model {model_name}: {str(e)}"
                    error_df = log_error(task, exp_id, model_name, models.iloc[m]['size'], ds, datasets, models.iloc[m]['datasets_size'],
                                        library, models.iloc[m]['file_size_quartile'], models.iloc[m]['popularity_quartile'], error_df, error_message)

    elif library == "fastai":
        for optimization in optimizations:
            # Repeat the experiment 10 times
            for exp_id in range(10): 
                try:
                    print("\n", optimization)

                    model = timm.create_model(
                        'vit_tiny_patch16_224.augreg_in21k', pretrained=True)

                    if "pruning" in optimization:
                        #  The optimization word ends with a number that indicates the pruning amount
                        pruning_amount = float(optimization.split("_")[2])

                        if "local_pruning" in optimization:
                            model = prune_locally_structured(
                                model, pruning_amount=pruning_amount)
                        elif "global_pruning" in optimization:
                            model = prune_globally_unstructured(
                                model, pruning_amount=pruning_amount)

                    elif optimization == "torch.compile":
                        model = compile_model(model)
                    elif optimization == "dynamic-quantization":
                        model = dynamic_quantize_model(model)
                    else:
                        model.eval()

                    # get model specific transforms (normalization, resize)
                    data_config = timm.data.resolve_model_data_config(model)
                    transforms = timm.data.create_transform(
                        **data_config, is_training=False)

                    # Use the 100 images from the test set to evaluate the model or the maximum number of images available
                    for i in range(100):
                        print("image", i)
                        if "imagenet" in datasets:
                            # Open '../datasets/ILSVRC2010_test/ILSVRC2010_test_common_gt_categories.txt' and read the i-th line
                            with open("../datasets/ILSVRC2010_test/ILSVRC2010_test_common_gt_categories.txt") as f:
                                lines = f.readlines()
                                #  i+1 because the first line is the header
                                label = lines[i+1].split(', ')
                                img_idx = label[0]
                                y_true = int(label[1].strip("'"))
                                y_true_name = label[2]
                                # Remove the quotes from the label name
                                y_true_name = y_true_name[1:-2]

                            # Create a string with 8 chars that begins with 0s and ends with the id of the image
                            id = str(img_idx).zfill(8)
                            image = Image.open(
                                f"../datasets/ILSVRC2010_test/ILSVRC2010_test_{id}.JPEG")
                        else:
                            with open(f"../datasets/{ds}/mapping.txt") as f:
                                lines = f.readlines()
                                label = lines[i].split('	')
                                img_name = label[0]
                                y_true = label[1].strip()  # Remove '\n' using strip()
                                y_true_name = y_true
                                print(img_name)
                                print(y_true)
                            
                            print("y_true", y_true)
                           
                            with open(f"../datasets/{ds}/class_names.txt") as f:
                                lines = f.readlines()
                                y_true_name = lines[int(y_true)].rstrip()  # Remove '\n' from the end of the line
                                print("y_true_name", y_true_name)
                                
                            image = Image.open(
                                f"../datasets/{ds}/{img_name}")

                        # Get image size
                        image_size = image.size

                        # Get the input shape expected by the model: the shape of the image after applying the transforms
                        input_size = transforms(image).unsqueeze(0).shape
                        # As it is a tensor, we need to convert it to a tuple
                        input_size = tuple(input_size)

                        # Record the start time of the inference
                        start_time = time.time()

                        # Unsqueeze single image into batch of 1 and perform inference
                        output = model(transforms(image).unsqueeze(0))
                        
                        # Record the end time of the inference
                        end_time = time.time()

                        y_pred_prob, y_pred = torch.topk(
                            output.softmax(dim=1) * 100, k=1)
                        y_pred = y_pred.item()

                        if "imagenet-21k" in datasets:
                            with open("../datasets/imagenet21k_wordnet_lemmas.txt") as f:
                                lines = f.readlines()
                                y_pred_name = lines[y_pred][:-1]
                        else:
                            y_pred_name = int(y_pred)

                        # Check if the prediction is correct
                        correct_predictions = 1 if y_true_name == y_pred_name else 0

                        # Get the FLOPs, MACs, and number of parameters of the model
                        if optimization == "torch.compile" and hasattr(model, '_orig_mod'):
                            model = model._orig_mod

                        flops, macs, params, sparsity, total_model_size = estimate_flops_macs_params(
                            model, image)

                        # Append the results to the dataframe
                        new_data = pd.DataFrame([[exp_id, optimization, model_name, total_model_size, datasets, models.iloc[m]['datasets_size'],
                                                library, models.iloc[m]['file_size_quartile'], models.iloc[
                                                    m]['popularity_quartile'], i, image_size, input_size,
                                                flops, macs, params, y_true, y_true_name, y_pred_name, correct_predictions, datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time), end_time - start_time, sparsity]],
                                                columns=['Experiment', 'Optimization', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library',
                                                        'File size quartile', 'Popularity quartile', 'Image ID', 'Image size', 'Input size',
                                                        'FLOPs', 'MACs', 'Parameters', 'y_true', 'y_true_name', 'y_pred', 'Correct Prediction', 'Start Time', 'End Time', 'Total Time', 'Global Sparsity'])

                        result_df = pd.concat(
                            [result_df, new_data], ignore_index=True)
                        result_df.to_csv(
                            f'results_{task}_{ds}_opt.csv', index=False)
                        
                except Exception as e:
                    # Handle errors and log them
                    error_message = f"Error for model {model_name}: {str(e)}"
                    error_df = log_error(task, exp_id, model_name, models.iloc[m]['size'], ds, datasets, models.iloc[m]['datasets_size'],
                                        library, models.iloc[m]['file_size_quartile'], models.iloc[m]['popularity_quartile'], error_df, error_message)
                
    return result_df, error_df


###################
# OBJECT DETECTION
###################
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    box1 and box2 should be in the format [x_min, y_min, x_max, y_max].
    """

    # Intersection coordinates
    # Find the maximum of the minimum x-coordinate of both bounding boxes
    x_min_inter = max(box1[0], box2[0])
    # Find the maximum of the minimum y-coordinate of both bounding boxes
    y_min_inter = max(box1[1], box2[1])
    # Find the minimum of the maximum x-coordinate of both bounding boxes
    x_max_inter = min(box1[2], box2[2])
    # Find the minimum of the maximum y-coordinate of both bounding boxes
    y_max_inter = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x_max_inter - x_min_inter) * \
        max(0, y_max_inter - y_min_inter)

    # Calculate area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def calculate_iou_and_label_accuracy(predicted_boxes, predicted_labels, true_boxes, true_labels, iou_threshold=0.5):
    iou_scores = []
    label_accuracies = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0  #  Every ground truth box that is not matched with a predicted box
    predicted_boxes_matched = []
    total_intersection_area = 0
    total_union_area = 0

    for pred_box, pred_label in zip(predicted_boxes, predicted_labels):
        max_iou = 0
        matched_label = False

        for true_box, true_label in zip(true_boxes, true_labels):
            iou = calculate_iou(pred_box, true_box)
            if iou > max_iou:
                max_iou = iou
                matched_label = pred_label == true_label

        iou_scores.append(max_iou)
        label_accuracies.append(matched_label)

        if max_iou >= iou_threshold:
            if matched_label:
                true_positives += 1

                if pred_box not in predicted_boxes_matched:
                    predicted_boxes_matched.append(pred_box)
            else:
                false_positives += 1
            total_intersection_area += max_iou * \
                (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        else:
            false_positives += 1

    false_negatives = len(true_boxes) - len(predicted_boxes_matched)
    total_union_area = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in true_boxes]) + sum(
        [(box[2] - box[0]) * (box[3] - box[1]) for box in predicted_boxes]) - total_intersection_area

    mean_iou = total_intersection_area / \
        total_union_area if total_union_area > 0 else 0
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    return iou_scores, label_accuracies, mean_iou, precision, recall, f1_score


def perform_inference_object_detection(model_name, models, loaded_datasets, ds, datasets, task, m, library, result_df, error_df):
    if library == "transformers" or library == "transformers.js" or library == "nemo":
        for optimization in optimizations:
            # Repeat the experiment 10 times
            for exp_id in range(10): 
                try:
                    print("\n", optimization)
                    model = pipeline(task, model=model_name)

                    if "pruning" in optimization:
                        #  The optimization word ends with a number that indicates the pruning amount
                        pruning_amount = float(optimization.split("_")[2])

                        model = model.model  # Access the underlying PyTorch model

                        if "local_pruning" in optimization:
                            model = prune_locally_structured(
                                model, pruning_amount=pruning_amount)
                        elif "global_pruning" in optimization:
                            model = prune_globally_unstructured(
                                model, pruning_amount=pruning_amount)

                    elif optimization == "torch.compile":
                        model = model.model  # Access the underlying PyTorch model
                        model = compile_model(model)

                    elif optimization == "dynamic_quantization":
                        model = dynamic_quantize_model(model.model)
                        
                    elif optimization == "static_quantization":
                        model = static_quantize_model(QuantizedModel(model.model))

                    # Use the 100 images from the test set to evaluate the model or the maximum number of images available
                    for i in range(105):  # coco
                    #for i in range(29): # only 29 test images for cppe-5
                        print("IMAGE", i)
                        
                        if i == 80: # (for coco ds) it only has 1 channel (outlier)
                            continue

                        if i == 9 or i == 26 or i == 78 or i == 80 or i == 90: # (for cd45rb ds)
                            continue

                        with open(f"../datasets/{ds}/mapping.txt") as f:
                            lines = f.readlines()
                            label = lines[i].split('	')
                            img_name = label[0]
                            true_annotations = label[3].strip()  # Remove '\n' using strip()
                            true_annotations=ast.literal_eval(true_annotations)
                        
                        if ds == 'coco':
                            true_id = true_annotations['bbox_id']
                        else:
                            true_id = true_annotations['id']
                        
                        true_area = true_annotations['area']
                        true_bbox = true_annotations['bbox']
                        true_category = true_annotations['category']
                        true_labels = []
                        
                        image = Image.open(f"../datasets/{ds}/{img_name}")

                        # Get image size
                        image_size = image.size

                        if ds == 'cppe-5':
                            cpp_5_class_names = [
                                'Coverall', 'Face_Shield', 'Gloves', 'Goggles', 'Mask']
                            # Map true_categories to its class names
                            true_labels = [cpp_5_class_names[int(
                                true_category[i])] for i in range(len(true_category))]
                        elif ds == 'coco':
                            coco_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


                            true_labels = [coco_class_names[int(
                                true_category[i])] for i in range(len(true_category))]
                        elif ds == 'cd45rb':
                            cd45rb_class_names = ['leukocyte']
                            # Map true_categories to its class names
                            true_labels = [cd45rb_class_names[int(
                                true_category[i])] for i in range(len(true_category))]

                        input_size = resolve_data_config({}, model=model)['input_size']

                        # Record the start time of the inference
                        start_time = time.time()

                        if optimization == "no_optimization":
                            # Perform inference on the image
                            predictions = model(image)

                            # Record the end time of the inference
                            end_time = time.time()

                            predicted_labels = [prediction['label']
                                                for prediction in predictions]
                            predicted_scores = [prediction['score']
                                                for prediction in predictions]
                            predicted_boxes = [(prediction['box']['xmin'], prediction['box']['ymin'],
                                                prediction['box']['xmax'], prediction['box']['ymax']) for prediction in predictions]

                            # Generate an array named predicted_categories with the id of the classes (extracted from cppe-5 class names)
                            if ds == 'cppe-5':
                                predicted_categories = [cpp_5_class_names.index(
                                label) for label in predicted_labels]
                            elif ds == 'coco':
                            	predicted_categories = [coco_class_names.index(
                                label) for label in predicted_labels]
                            elif ds == 'cd45rb':
                            	predicted_categories = [cd45rb_class_names.index(
                                label) for label in predicted_labels]
                        else:
                            # Convert the image to a PyTorch tensor
                            image_tensor = torchvision_transforms_functional.to_tensor(
                                image)
                            image_tensor = image_tensor.unsqueeze(
                                0)  # Add batch dimension

                            try:
                                image_tensor = torch_nn_functional.interpolate(
                                    image_tensor, size=input_size[1:], mode='bilinear', align_corners=False)
                            except:
                                model_config = model.model.config
                                image_size = model_config.image_size
                                input_size = (3, image_size, image_size)
                                image_tensor = torch_nn_functional.interpolate(
                                    image_tensor, size=input_size[1:], mode='bilinear', align_corners=False)

                            print("image tensor:",image_tensor.size())
                            outputs = model(image_tensor)

                            # Record the end time of the inference
                            end_time = time.time()

                            logits = outputs.logits
                            pred_boxes = outputs.pred_boxes

                            # Applying softmax to logits to get scores
                            probabilities = torch.softmax(logits, dim=-1)

                            # Getting labels with highest score for each prediction: the index of the highest probability for each set of scores.
                            y_pred_id = torch.argmax(probabilities, dim=-1)

                            # Converting pred_boxes to bounding box coordinates (x_min, y_min, x_max, y_max)
                            # Assuming only one image in the batch
                            bounding_boxes = pred_boxes[0]
                            x_c, y_c, w, h = bounding_boxes[:, 0], bounding_boxes[:,
                                                                                1], bounding_boxes[:, 2], bounding_boxes[:, 3]
                            x_min, y_min = x_c - w / 2, y_c - h / 2
                            x_max, y_max = x_c + w / 2, y_c + h / 2

                            # Convert to absolute pixel coordinates
                            image_width = image_size[0]
                            image_height = image_size[1]
                            x_min_abs = (
                                x_min * image_width).detach().numpy().astype(int).squeeze()
                            y_min_abs = (
                                y_min * image_height).detach().numpy().astype(int).squeeze()
                            x_max_abs = (
                                x_max * image_width).detach().numpy().astype(int).squeeze()
                            y_max_abs = (
                                y_max * image_height).detach().numpy().astype(int).squeeze()

                            # Stack absolute coordinates
                            bounding_boxes = np.stack(
                                (x_min_abs, y_min_abs, x_max_abs, y_max_abs), axis=-1)

                            # Convert to numpy arrays
                            probabilities = probabilities.detach().numpy()
                            y_pred_id = y_pred_id.detach().numpy().squeeze()  # Remove the batch dimension
                            
                            #  Use cppe-5 class names to get the predicted labels. Note: if any prediction is not in the cppe-5 class names, it will be ignored.
                            if ds == 'cppe-5':
                                predicted_labels = []
                                predicted_boxes = []
                                predicted_scores = []
                                predicted_categories = []
                                for j in range(len(y_pred_id)):
                                    label_id = y_pred_id[j]-1 
                                    if label_id < 5:  # cppe-5 has 5 classes
                                        predicted_categories.append(label_id)
                                        predicted_labels.append(
                                            cpp_5_class_names[label_id])

                                        # Create a list of lists with the bounding box coordinates
                                        predicted_boxes.append(
                                            [int(coord) for coord in bounding_boxes[j]])

                                        # Get the score of the predicted label (column of the predicted label)
                                        predicted_scores.append(
                                            probabilities[0, j][label_id])
                            elif ds == 'coco':
                                predicted_labels = []
                                predicted_boxes = []
                                predicted_scores = []
                                predicted_categories = []
                                print("len(y_pred_id)", len(y_pred_id))
                                print("len(coco_class_names)", len(coco_class_names))
                                
                                for j in range(len(y_pred_id)):
                                    label_id = y_pred_id[j]
                                    
                                    if label_id < 80:  # cppe-5 has 80 classes
                                        predicted_categories.append(label_id)
                                        predicted_labels.append(coco_class_names[label_id])

                                        # Create a list of lists with the bounding box coordinates
                                        predicted_boxes.append([int(coord) for coord in bounding_boxes[j]])

                                        # Get the score of the predicted label (column of the predicted label)
                                        predicted_scores.append(probabilities[0, j][label_id])
                                        
                            elif ds == 'cd45rb':
                                predicted_labels = []
                                predicted_boxes = []
                                predicted_scores = []
                                predicted_categories = []
                                print("len(y_pred_id)", len(y_pred_id))
                                print("len(coco_class_names)", len(cd45rb_class_names))
                                
                                for j in range(len(y_pred_id)):
                                    label_id = y_pred_id[j]
                                    
                                    # If the prediction is "LABEL_"+number, remove the "LABEL_" prefix and convert the number to a string
                                    if isinstance(label_id, str) and label_id.startswith("LABEL_"):
                                        # Convert label_id to string, remove the "LABEL_" prefix
                                        label_id = str(label_id)[6:]
                                    
                                    if label_id < 1:  # cppe-5 has 1 class
                                        predicted_categories.append(label_id)
                                        predicted_labels.append(cd45rb_class_names[label_id])

                                        # Create a list of lists with the bounding box coordinates
                                        predicted_boxes.append([int(coord) for coord in bounding_boxes[j]])

                                        # Get the score of the predicted label (column of the predicted label)
                                        predicted_scores.append(probabilities[0, j][label_id])
                        
                        # Get the FLOPs, MACs, and number of parameters of the model
                        flops, macs, params, sparsity, total_model_size = estimate_flops_macs_params(
                            model, image)

                        # iou_scores, label_accuracies = calculate_iou_and_label_accuracy(predicted_boxes, predicted_labels, true_bbox, true_labels)
                        iou_scores, label_accuracies, mean_iou, precision, recall, f1_score = calculate_iou_and_label_accuracy(
                            predicted_boxes, predicted_labels, true_bbox, true_labels)
                            
                        print("inference time", end_time - start_time)
                        print("total_model_size", total_model_size)

                        # Append the results to the dataframe
                        new_data = pd.DataFrame([[exp_id, optimization, model_name, total_model_size, datasets, models.iloc[m]['datasets_size'], library, models.iloc[m]['file_size_quartile'],
                                                models.iloc[m]['popularity_quartile'], i, image_size, input_size, flops, macs, params, true_id, true_area, true_bbox, true_category, true_labels,
                                                predicted_scores, predicted_categories, predicted_labels, predicted_boxes, iou_scores, label_accuracies, mean_iou, precision, recall, f1_score,
                                                datetime.fromtimestamp(start_time), datetime.fromtimestamp(end_time), end_time - start_time, sparsity]],
                                                columns=['Experiment', 'Optimization', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library', 'File size quartile',
                                                        'Popularity quartile', 'Image ID', 'Image size', 'Input size', 'FLOPs', 'MACs', 'Parameters',
                                                        'True Id', 'True Area', 'True Bbox', 'True Category', 'True Label', 'Predicted Score', 'Predicted Category',
                                                        'Predicted Label', 'Predicted Box', 'IoU Scores', 'Label Accuracies', 'mean IoU', 'Precision', 'Recall', 'F1 score', 'Start Time', 'End Time', 'Total Time', 'Global Sparsity'])

                        result_df = pd.concat(
                            [result_df, new_data], ignore_index=True)
                        result_df.to_csv(
                            f'results_{task}_{ds}_opt.csv', index=False)
                        
                except Exception as e:
                    # Handle errors and log them
                    error_message = f"Error for model {model_name}: {str(e)}"
                    error_df = log_error(task, exp_id, model_name, models.iloc[m]['size'], ds, datasets, models.iloc[m]['datasets_size'],
                                     library, models.iloc[m]['file_size_quartile'], models.iloc[m]['popularity_quartile'], error_df, error_message)
    
    return result_df, error_df



#optimizations = ["no_optimization", "dynamic_quantization", "local_pruning_0.25", "local_pruning_0.5", "local_pruning_0.75", "global_pruning_0.25", "global_pruning_0.5", "global_pruning_0.75", "torch.compile"]

#optimizations = ["no_optimization", "static_quantization", "local_pruning_0.05", "local_pruning_0.1", "local_pruning_0.15", "local_pruning_0.2", "global_pruning_0.05", "global_pruning_0.1", "global_pruning_0.15", "global_pruning_0.2"]

optimizations = ["no_optimization", "dynamic_quantization", "local_pruning_0.1", "local_pruning_0.2", "local_pruning_0.3", "global_pruning_0.1", "global_pruning_0.2", "global_pruning_0.3", "torch.compile"]

#optimizations = ["local_pruning_0.1", "local_pruning_0.2", "local_pruning_0.3", "local_pruning_0.4", "local_pruning_0.5", "local_pruning_0.6", "local_pruning_0.7", "local_pruning_0.8", "local_pruning_0.9", "global_pruning_0.1", "global_pruning_0.2", "global_pruning_0.3", "global_pruning_0.4", "global_pruning_0.5", "global_pruning_0.6", "global_pruning_0.7", "global_pruning_0.8", "global_pruning_0.9"]


#task = "image-classification"
task = "object-detection"


def main_task():
#if __name__ == "__main__":
    global running
    print(task)
   
    # Initialize a dictionary to store the loaded dataset and its name
    loaded_datasets = {'dataset': None, 'name': None}

    # Load the models
    models = load_models(task)

    # Sort models by the dataset used for training
    sorted_data = models.sort_values(by=['datasets'])
    sorted_data = sorted_data.sort_values(by='datasets_size', ascending=True)

    if task == 'image-classification':
        #selected_datasets = ["cifar10", "imagenet", "food101"]
        selected_datasets = ["imagenet"]
    else:  #  object-detection
        selected_datasets = ["cd45rb"] # cppe-5, cd45rb, coco
    
    
    gpu_id = os.getenv("GPU_DEVICE_ORDINAL", 0)

    for ds in selected_datasets:
        # Use an auxiliary csv file to store the GPU metrics for the current execution
        gpu_metrics = f"gpu-power_{task}_{ds}.csv"
        command = f"nvidia-smi -i {gpu_id} --query-gpu=timestamp,gpu_name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.max_limit,temperature.gpu --format=csv -lms 100 -f {gpu_metrics}"

        nvidiaProfiler = subprocess.Popen(command.split())
        
        # Create an empty dataframe to store errors
        error_df = pd.DataFrame(columns=['Experiment', 'Model', 'Model Size', 'Datasets',
                                'Datasets Size', 'Library', 'File size quartile', 'Popularity quartile', 'Error_Message'])
        error_df.to_csv(f'error_log_{task}_{ds}_opt.csv', index=False)
        
        # Create an empty dataframe to store information about the models
        if task == "image-classification":
            result_df = pd.DataFrame(columns=['Experiment', 'Optimization', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library',
                                            'File size quartile', 'Popularity quartile', 'Image ID', 'Image size', 'Input size',
                                            'FLOPs', 'MACs', 'Parameters', 'y_true', 'y_true_name', 'y_pred', 'Correct Prediction', 'Start Time', 'End Time', 'Total Time', 'Global Sparsity'])
        else:
            result_df = pd.DataFrame(columns=['Experiment', 'Optimization', 'Model', 'Model Size', 'Datasets', 'Datasets Size', 'Library', 'File size quartile',
                                            'Popularity quartile', 'Image ID', 'Image size', 'Input size', 'FLOPs', 'MACs', 'Parameters',
                                            'True Id', 'True Area', 'True Bbox', 'True Category', 'True Label', 'Predicted Score', 'Predicted Category',
                                            'Predicted Label', 'Predicted Box', 'IoU Scores', 'Label Accuracies', 'mean IoU', 'Precision', 'Recall', 'F1 score', 'Start Time', 'End Time', 'Total Time', 'Global Sparsity'])
        result_df.to_csv(f'results_{task}_{ds}_opt.csv', index=False)
        
        if ds == "imagenet":
            models = sorted_data[sorted_data['datasets'].str.contains(
                ds, case=False)]

            #  Sort by library name
            #models = models.sort_values(by=['library_name'])
        else:
            models = sorted_data[sorted_data['datasets'] == f"['{ds}']"]

        # Print len models in an extra file
        with open(f"models_{task}_{ds}.txt", "w") as file:
            file.write(str(models['modelId'].count()))
            
        # Group by 'File size quartile' and repeat each group x times
        # IMAGENET
        #no_models = ['google/vit-large-patch16-384', 'facebook/deit-tiny-distilled-patch16-224', 'sail/poolformer_s36', 'deepmind/vision-perceiver-conv', 'microsoft/beit-large-patch16-224-pt22k', 'microsoft/beit-large-patch16-224-pt22k-ft22k']
        
        # COCO
        #no_models = ['liaujianjie/detr-resnet-50', 'moveparallel/detr-resnet-50-clone', 'SenseTime/deformable-detr-single-scale', 'Sembiance/detr-resnet-101-fixed', 'facebook/detr-resnet-50-dc5', 'facebook/detr-resnet-101', 'microsoft/conditional-detr-resnet-50', 'hustvl/yolos-base', 'facebook/detr-resnet-50']
        
        #models = models[~models['modelId'].isin(no_models)]
        #models = models.groupby('file_size_quartile').head(5)
        
        print(len(models))
        print(models['modelId'].unique())

        #for m in range(len(models)):
        for m in range(23):
            print("model_id:", m)
            model_name = models.iloc[m]['modelId']
            library = models.iloc[m]['library_name']
            datasets = models.iloc[m]['datasets']

            # Avoid 'facebook/regnet-y-10b-seer-in1k' model as it makes my kernel crash
            if model_name != "facebook/regnet-y-10b-seer-in1k" and model_name != 'microsoft/beit-large-patch16-224-pt22k' and model_name != 'microsoft/beit-large-patch16-224-pt22k-ft22k':
            #if model_name == 'sayakpaul/vit-base-patch16-224-in21k-finetuned-lora-food101' or model_name == 'skylord/swin-finetuned-food101' or model_name=='paolinox/segformer-finetuned-food101' or model_name=='juns/my_awesome_food_model' or model_name=='lu5/swin-tiny-patch4-window7-224-finetuned-eurosat' or model_name=='lu5/swinv2-small-patch4-window8-256-finetuned-eurosat' or model_name=='kwc20140710/my_awesome_food_model' or model_name=='StephenSKelley/my_awesome_food_model':
                print(model_name)

                if task == "image-classification":
                    result_df, error_df = perform_inference_image_classification(
                        model_name, models, loaded_datasets, ds, datasets, task, m, library, result_df, error_df)
                else:  #  object-detection
                    result_df, error_df = perform_inference_object_detection(
                        model_name, models, loaded_datasets, ds, datasets, task, m, library, result_df, error_df)

        nvidiaProfiler.terminate() # Stop nvidia-smi 
    
    # Set running to False when you want to stop the while loop in get_wattmeter_data()
    running = False
             


# Inicia la tarea principal en un hilo
if __name__ == "__main__": 
    task_thread = threading.Thread(target=main_task)
    task_thread.start()
    
    get_wattmeter_data()

