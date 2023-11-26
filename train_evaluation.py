import tensorflow as tf
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

def custom_loss(model, alpha=0.01, index=4):
    def loss(y_true, y_pred):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
        max_loss = alpha * tf.reduce_sum(tf.square(model.layers[index].max_values))  # Penalty on max values
        return ce_loss + max_loss
    return loss

def random_noise(model, image, epsilon=0.1):
    # Generate random noise within the range [-epsilon, epsilon]
    noise = tf.random.uniform(shape=image.shape, minval=-epsilon, maxval=epsilon, dtype=image.dtype)
    
    # Add the noise to the original image
    perturbed_image = image + noise
    perturbed_image = tf.clip_by_value(perturbed_image, clip_value_min=0.0, clip_value_max=1.0)
    
    return perturbed_image

def create_adversarial_examples(model, x_data, y_data, epsilon=0.1, attack = 'fgsm', batch_size=1, norm=np.inf, verbose = True):
    num_samples = len(x_data)
    new_dataset = []
    if(attack == 'cw_l2'):
        y_data = tf.one_hot(y_data, model.output_shape[-1])
    if(attack == 'apgd_ce' or attack == 'apgd_dlr'):
        tmp_model = tf.keras.models.Model(inputs = model.inputs, outputs = model.layers[-2].output)
        tmp_model = utils_tf2.ModelAdapter(tmp_model)
        adversary = AutoAttack(tmp_model, norm='Linf', eps=epsilon, version='rand', is_tf_model=True
                               , verbose=False)
    for i in tqdm(range(0, num_samples, batch_size), disable= not verbose):
        original_image = x_data[i:min(num_samples, i+batch_size)]
        true_label = y_data[i:min(num_samples, i+batch_size)]
        
        if(attack == 'fgsm'):
            perturbed_image = fast_gradient_method.fast_gradient_method(model_fn = model, 
                                                                        x = original_image, 
                                                                        eps = epsilon,
                                                                        norm = norm,
                                                                        clip_min=0,
                                                                        clip_max=1,
                                                                        y=true_label)
        elif(attack == 'pgd'):
            perturbed_image = madry_et_al.madry_et_al(model_fn = model, 
                                                    x = original_image, 
                                                    eps = epsilon,
                                                    eps_iter = epsilon / 10,
                                                    nb_iter = 100,
                                                    norm = norm,
                                                    clip_min=0,
                                                    clip_max=1,
                                                    y=true_label,
                                                    sanity_checks=False)
        elif(attack == 'cw_l2'):
            tmp_model = tf.keras.models.Model(inputs = model.inputs, outputs = model.layers[-2].output)
#             pred = np.argmax(model.predict(original_image), axis = 1)
            perturbed_image = l2_attack.CarliniL2(model = tmp_model, batch_size=len(original_image), confidence = 0,
                                                 targeted = False, learning_rate = 0.01,
                                                 binary_search_steps = 5, max_iterations = 10,
                                                 abort_early = True, initial_const = 0.001,
                                                 boxmin = 0, boxmax = 1, verbose = False).attack(original_image,true_label)
        elif(attack == 'apgd_ce'):
            torch_testX = torch.from_numpy( np.transpose(original_image, (0, 3, 1, 2)) ).float().cuda()
            torch_testY = torch.from_numpy( true_label ).long().cuda()
            adversary.attacks_to_run = ['apgd-ce']
            x_adv, y_adv = adversary.run_standard_evaluation(torch_testX, torch_testY, bs=len(original_image)
                                                             , return_labels=True)
            perturbed_image = np.moveaxis(x_adv.cpu().numpy(), 1, 3).tolist()
        elif(attack == 'apgd_dlr'):
            torch_testX = torch.from_numpy( np.transpose(original_image, (0, 3, 1, 2)) ).float().cuda()
            torch_testY = torch.from_numpy( true_label ).long().cuda()
            adversary.attacks_to_run = ['apgd-dlr']
            x_adv, y_adv = adversary.run_standard_evaluation(torch_testX, torch_testY, bs=len(original_image)
                                                             , return_labels=True)
            perturbed_image = np.moveaxis(x_adv.cpu().numpy(), 1, 3).tolist()
        elif(attack == 'random'):
            perturbed_image = random_noise(model, tf.convert_to_tensor(original_image), epsilon)
        torch.cuda.empty_cache()
        new_dataset.extend(perturbed_image)
    return tf.convert_to_tensor(new_dataset)

def compute_robust_accuracy(model, x_data, y_data, epsilon=0.1, attack = 'fgsm', batch_size = 1, norm=np.inf):
    new_dataset = create_adversarial_examples(model, x_data, y_data, epsilon, attack, batch_size, norm)
    if(attack == 'cw_l2'):
        revised_new_dataset = []
        diff_lst = np.sqrt(np.sum(np.square(x_data - new_dataset), axis = (3,2,1)))
#         print(np.mean(diff_lst))
        for i in range(len(x_data)):
            if(epsilon == 0.1 and diff_lst[i] > 18): #MNIST
                revised_new_dataset.append(x_data[i])
            elif(epsilon == 0.01 and diff_lst[i] > 0.25): #CIFAR10
                revised_new_dataset.append(x_data[i])
            else:
                revised_new_dataset.append(new_dataset[i])
        new_dataset = tf.convert_to_tensor(revised_new_dataset)
#     else:
    _, output = model.evaluate(new_dataset, y_data)
    return output

def plot_accuracy(results, path):
    plt.rcParams.update({'font.size': 14})  # Change the 14 to your desired font size
    plt.plot(results['balancers'],np.mean(results['accuracy'], axis = 0))
    # plt.plot(results['balancers'],np.mean(results['random_accuracy'], axis = 0))
    plt.plot(results['balancers'],np.mean(results['fgsm_accuracy'], axis = 0))
    plt.plot(results['balancers'],np.mean(results['pgd_accuracy'], axis = 0))
    plt.plot(results['balancers'],np.mean(results['apgd_ce_accuracy'], axis = 0))
    plt.plot(results['balancers'],np.mean(results['apgd_dlr_accuracy'], axis = 0))
    plt.plot(results['balancers'],np.mean(results['cw_l2_accuracy'], axis = 0))
    plt.setp(plt.gca().lines, linewidth=2)
    plt.legend(['Clean', 'FGSM', 'PGD', 'APGD_CE', 'APGD_DLR','CW_L2'])
    plt.xscale('log')
    plt.xlabel('Balancer')
    plt.ylabel('Accuracy')
    plt.savefig(path, dpi=450, bbox_inches="tight")
    plt.show()
def plot_perturbation(results, path):
    plt.plot(results['balancers'],np.mean(results['cw_l2_perturbation'], axis = 0))
    plt.rcParams.update({'font.size': 14})  # Change the 14 to your desired font size
    plt.xscale('log')
    plt.setp(plt.gca().lines, linewidth=2)
    plt.xlabel('Balancer')
    plt.ylabel('L2 Perturbation')
    plt.savefig(path, dpi=450, bbox_inches="tight")
    plt.show()
def plot_mean_max(results, path):
    plt.plot(results['balancers'],np.mean(results['mean_max'], axis = 0))
    plt.rcParams.update({'font.size': 14})  # Change the 14 to your desired font size
    plt.xscale('log')
    plt.yscale('log')
    plt.setp(plt.gca().lines, linewidth=2)
    plt.xlabel('Balancer')
    plt.ylabel('Mean Max')
    plt.savefig(path, dpi=450, bbox_inches="tight")
    plt.show()
def train_models(balancers, n_runs, max_index, folder, result_folder, get_model, x_train, y_train, location="end"):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0001)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    for run in range(n_runs):
        tmp_results = {}
        for inx, balancer in enumerate(balancers):
            results_balancer = {}
            path = f"{folder}/balancer{balancer}_run{run}.h5"
            print(f"Run {run}, Balancer {balancer}:")            
            if(not os.path.exists(path)):      
                # Train the model
                model = get_model(x_train.shape[1:], location)
                # Compile the model with the custom loss function
                model.compile(optimizer='adam', loss=custom_loss(model, alpha=balancer, index = max_index), metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=2000, batch_size=128, validation_split=0.2
                          , callbacks=[reduce_lr, early_stop], verbose=1)
                model.save_weights(path)
#             else:
#                 print("Already Exists!")
                
def train_test(balancers, n_runs, max_index, folder, result_folder, get_model, x_train, y_train, x_test, y_test
               , epsilon, batch_size=1, stored_results=None, location="end"):
    info_list = ['accuracy', 'random_accuracy', 'fgsm_accuracy', 'pgd_accuracy'
                 , 'apgd_ce_accuracy', 'apgd_dlr_accuracy'
                 ,'cw_l2_accuracy','mean_max']
    acc_attacks = ['random_accuracy', 'fgsm_accuracy', 'pgd_accuracy', 
                   'apgd_ce_accuracy', 'apgd_dlr_accuracy', 
                   'cw_l2_accuracy']
    acc2attack = {
        'random_accuracy': 'random',
        'fgsm_accuracy': 'fgsm',
        'pgd_accuracy': 'pgd',
        'apgd_ce_accuracy': 'apgd_ce',
        'apgd_dlr_accuracy': 'apgd_dlr',
        'cw_l2_accuracy': 'cw_l2'
    }
    result_folder += f'/nruns={n_runs}_maxindex={max_index}_eps={epsilon}_batchsize={batch_size}'
    accuracy_score_path = result_folder + '/accuracy_scores.pkl'
    
    results = {}
    if(os.path.exists(accuracy_score_path)):
        f = open(accuracy_score_path, "rb")
        results = pickle.load(f)
        if(results['balancers'] != balancers):
            results = {}
    for info in info_list:
        if(info not in results):
            results[info] = []
    results['balancers'] = balancers
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    results_path = Path(result_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    for run in range(n_runs):
        tmp_results = {}
        for info in info_list:
            tmp_results[info] = []
        for inx, balancer in enumerate(balancers):
            results_balancer = {}
            path = f"{folder}/balancer{balancer}_run{run}.h5"
            print(f"Run {run}, Balancer {balancer}:")
            model = get_model(x_train.shape[1:], location = location)
            # Compile the model with the custom loss function
            model.compile(optimizer='adam', loss=custom_loss(model, alpha=balancer, index = max_index), metrics=['accuracy'])
            if(os.path.exists(path)):
                model.load_weights(path)
            else:        
                # Train the model
                model.fit(x_train, y_train, epochs=2000, batch_size=128, validation_split=0.2
                          , callbacks=[reduce_lr, early_stop], verbose=1)
                model.save_weights(path)

            # Evaluate the model on the test set
            test_loss, test_accuracy = model.evaluate(x_test, y_test)
#             print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
            tmp_results['accuracy'].append(test_accuracy)    
            for acc_attack in acc_attacks:
                if(len(results[acc_attack]) <= run):
                    tmp_results[acc_attack].append(compute_robust_accuracy(model, x_test, y_test, epsilon = epsilon, attack = acc2attack[acc_attack],batch_size=batch_size))
                else:
                    tmp_results[acc_attack].append(results[acc_attack][run])
#                 if(acc_attack == 'cw_l2_perturbation'):
#                     print(f"{acc2attack[acc_attack]} perturbation: {tmp_results[acc_attack][-1]}")
#                 else:
#                 print(f"{acc2attack[acc_attack]} acc: {tmp_results[acc_attack][-1]}")
            if(len(results['mean_max']) <= run):
                tmp_results['mean_max'].append(np.mean(model.layers[max_index].max_values))
            else:
                tmp_results['mean_max'].append(results['mean_max'][run])
#             print(f"Mean max: {tmp_results['mean_max'][-1]}")
#             print()
        for info in info_list:
            if(len(results[info]) <= run):
                results[info].append(tmp_results[info])
    with open(accuracy_score_path, "wb") as outfile:
        pickle.dump(results, outfile)
    plot_accuracy(results, result_folder + '/accuracy_plot.png')
#     plot_perturbation(results, result_folder + '/perturbation_plot.png')
    plot_mean_max(results, result_folder + '/mean_max_plot.png')
    return results

def adversarial_training(model, X, y, X_val, y_val, epochs, batch_size, attack, eps):
  n_batches = len(X) // batch_size + (len(X) % batch_size != 0)
  for i in range(epochs):
#     print(f"Epoch {i+1}:", )
    for j in tqdm(range(n_batches)):
#       print(f"\rBatch {j+1}/{n_batches}", end = "")
      X_batch = X[j * batch_size : min(len(X), (j+1) * batch_size)]
      Y_batch = y[j * batch_size : min(len(X), (j+1) * batch_size)]
#       print("Generate adversarial training batch...")
      adv_X_train = create_adversarial_examples(model, X_batch, Y_batch, epsilon = eps, attack=attack, batch_size=batch_size, verbose = False)      
      model.train_on_batch(adv_X_train, Y_batch)
    print("\nGenerate adversarial val set...")
    adv_X_val = create_adversarial_examples(model, X_val, y_val, epsilon = eps, attack=attack, batch_size=batch_size)
    # new_X_val = tf.concat((X_val, adv_X_val), 0)
    # new_y_val = tf.concat((y_val, y_val), 0)
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size = batch_size, verbose = False)
    adv_val_loss, adv_val_acc = model.evaluate(adv_X_val, y_val, batch_size = batch_size, verbose = False)
    print(f"Epoch {i+1}: Val loss: {val_loss}, Val acc: {val_acc}, ", end = "")
    print(f"Rob val loss: {adv_val_loss}, Rob val acc: {adv_val_acc}")
#     if(save_epoch):
#       model.save_weights(f"{initial}_epoch{i+1}.h5")
  return model

def adversarial_train_models(n_runs, max_index, folder, get_model, x_train, y_train, epsilon, adv_epochs = 100, location="end"):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0001)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    folder += '/adv_training'
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    for run in range(n_runs):
        path = f"{folder}/run{run}.h5"
        print(f"Run {run}:")            
        if(not os.path.exists(path)):      
            X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            # Train the model
            model = get_model(X_train.shape[1:], location)
            # Compile the model with the custom loss function
            model.compile(optimizer='adam', loss=custom_loss(model, alpha=0, index = max_index), metrics=['accuracy'])
            model.fit(X_train, Y_train, epochs=2000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[reduce_lr, early_stop], verbose=1)
            model = adversarial_training(model, X_train, Y_train, X_val, Y_val, adv_epochs, 128, "pgd", epsilon)
            model.save_weights(path)
            
def adversarial_train_test(n_runs, max_index, folder, result_folder, get_model, x_train, y_train, x_test, y_test
               , epsilon, batch_size=1, stored_results=None, location="end", adv_epochs = 5):
    info_list = ['accuracy', 'random_accuracy', 'fgsm_accuracy', 'pgd_accuracy'
                 , 'apgd_ce_accuracy', 'apgd_dlr_accuracy'
                 ,'cw_l2_accuracy','mean_max']
    acc_attacks = ['random_accuracy', 'fgsm_accuracy', 'pgd_accuracy', 
                   'apgd_ce_accuracy', 'apgd_dlr_accuracy', 
                   'cw_l2_accuracy']
    acc2attack = {
        'random_accuracy': 'random',
        'fgsm_accuracy': 'fgsm',
        'pgd_accuracy': 'pgd',
        'apgd_ce_accuracy': 'apgd_ce',
        'apgd_dlr_accuracy': 'apgd_dlr',
        'cw_l2_accuracy': 'cw_l2'
    }
    folder += '/adv_training'
    result_folder += f'/adv_training/nruns={n_runs}_maxindex={max_index}_eps={epsilon}_batchsize={batch_size}'
    accuracy_score_path = result_folder + '/accuracy_scores.pkl'
    
    results = {}
    if(os.path.exists(accuracy_score_path)):
        f = open(accuracy_score_path, "rb")
        results = pickle.load(f)
    for info in info_list:
        if(info not in results):
            results[info] = []
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    results_path = Path(result_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    for run in range(n_runs):
        path = f"{folder}/run{run}.h5"
        print(f"Run {run}:")
        model = get_model(x_train.shape[1:], location = location)
        # Compile the model with the custom loss function
        model.compile(optimizer='adam', loss=custom_loss(model, alpha=0, index = max_index), metrics=['accuracy'])
        if(os.path.exists(path)):
            model.load_weights(path)
        else:        
            # Train the model
            X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            model.fit(X_train, Y_train, epochs=2000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[reduce_lr, early_stop], verbose=1)
            model = adversarial_training(model, X_train, Y_train, X_val, Y_val, adv_epochs, 128, "pgd", epsilon)
            model.save_weights(path)

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        tmp_results['accuracy'].append(test_accuracy)    
        for acc_attack in acc_attacks:
            if(len(results[acc_attack]) <= run):
                results[acc_attack].append(compute_robust_accuracy(model, x_test, y_test, epsilon = epsilon, attack = acc2attack[acc_attack],batch_size=batch_size))
        if(len(results['mean_max']) <= run):
            results['mean_max'].append(np.mean(model.layers[max_index].max_values))
    with open(accuracy_score_path, "wb") as outfile:
        pickle.dump(results, outfile)
    for key, item in results.items():
        print(f"{key}: {np.mean(items)}")
    return results