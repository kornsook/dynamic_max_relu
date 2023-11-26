import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
from pathlib import Path
from cleverhans.tf2.attacks import fast_gradient_method, madry_et_al
from cw_attacks import l2_attack
import matplotlib.pyplot as plt
import json
import pickle
from autoattack.autoattack import AutoAttack
from autoattack import utils_tf2
import torch
from sklearn.model_selection import train_test_split
from MaxReLU import MaxReLU
from models import create_dense_model, create_shallow_cnn_model, create_vgg16_model, create_resnet50_model, create_resnet101_model, create_mobilenetv2_model, create_inceptionv3_model
from train_evaluation import train_models,train_test, adversarial_train_models
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your Script Description")

#     parser.add_argument("folder_name", type=str, help="Name of the folder")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--n-runs", type=int, help="Number of runs")
    parser.add_argument("--eps", type=float, help="Perturbation bound", default=0.1)
    parser.add_argument("--type", type=str, choices=["train", "test"], help="Train or test")
    parser.add_argument("--base-dir", type=str, help="Base directory for models and results")
    parser.add_argument("--drelu-loc", type=str, choices=["end", "beginning", "all"], default="end", help="The location of layer that has D-ReLU")
    parser.add_argument("--adv-training", action="store_false", help="Adversarial training ?")
    parser.add_argument("--adv-epochs", type=int, default=50, help="Adversarial training epochs")

    args = parser.parse_args()
    balancers = [0, 1e-7, 0.00001, 0.001, 0.1, 1, 100, 10000]
    folder = f'{args.base_dir}/models/{args.drelu_loc}/{args.model}_{args.dataset}'
    result_folder = f'{args.base_dir}/results/{args.drelu_loc}/{args.model}_{args.dataset}'
    if(args.dataset == "mnist"):
        # Load and preprocess the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        _, x_test, _, y_test = train_test_split(x_test, y_test, test_size = 0.1, random_state=42)
        # Normalize pixel values to be between 0 and 1
        x_train, x_test = np.expand_dims(x_train / 255.0, -1).astype(np.float32), np.expand_dims(x_test / 255.0, -1).astype(np.float32)    
        y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
    elif(args.dataset == "cifar10"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        _, x_test, _, y_test = train_test_split(x_test, y_test, test_size = 0.1, random_state=42)

        # Normalize pixel values to be between 0 and 1
        x_train, x_test = (x_train / 255.0).astype(np.float32), (x_test / 255.0).astype(np.float32)
        y_train, y_test = y_train.astype(np.int32).squeeze(-1), y_test.astype(np.int32).squeeze(-1)    
    
    if(args.model == "dense"):
        model_fnc = create_dense_model
        if(args.drelu_loc == "end"):
            max_index = 4
        elif(args.drelu_loc == "beginning"):
            max_index = 2
    elif(args.model == "shallow_cnn"):
        model_fnc = create_shallow_cnn_model
        if(args.drelu_loc == "end"):
            max_index = 5
        elif(args.drelu_loc == "beginning"):
            max_index = 1
    elif(args.model == "vgg16"):
        model_fnc = create_vgg16_model
        if(args.drelu_loc == "end"):
            max_index = 3
        elif(args.drelu_loc == "beginning"):
            max_index = 1
    elif(args.model == "resnet50"):
        model_fnc = create_resnet50_model
        if(args.drelu_loc == "end"):
            max_index = 3
        elif(args.drelu_loc == "beginning"):
            max_index = 1
    elif(args.model == "resnet101"):
        model_fnc = create_resnet101_model
        if(args.drelu_loc == "end"):
            max_index = 3
        elif(args.drelu_loc == "beginning"):
            max_index = 1
    elif(args.model == "mobilenetv2"):
        model_fnc = create_mobilenetv2_model
        if(args.drelu_loc == "end"):
            max_index = 3
        elif(args.drelu_loc == "beginning"):
            max_index = 1
    elif(args.model == "inceptionv3"):
        model_fnc = create_inceptionv3_model
        if(args.drelu_loc == "end"):
            max_index = 4
        elif(args.drelu_loc == "beginning"):
            max_index = 1
    if(args.type == "train"):
        print("Training...")
        print(args.adv_training)
        if(args.adv_training):
            adversarial_train_models(balancers, args.n_runs, max_index, folder, model_fnc, 
             x_train, y_train, args.eps, adv_epochs = args.adv_epochs, location = args.drelu_loc)
        else:
            train_models(balancers, args.n_runs, max_index, folder, result_folder, model_fnc, 
             x_train, y_train, location = args.drelu_loc)
    elif(args.type == "test"):
        print("Testing...")
        results = train_test(balancers, args.n_runs, 
                    max_index, folder,
                    result_folder,
                    model_fnc,
                    x_train, y_train, 
                    x_test, y_test, args.eps, batch_size=args.batch_size, location = args.drelu_loc)
    print("Done!")