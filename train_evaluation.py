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
            if(epsilon == 0.1 and diff_lst[i] > 8): #MNIST
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
    plt.figure()
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
    plt.figure()
    plt.plot(results['balancers'],np.mean(results['cw_l2_perturbation'], axis = 0))
    plt.rcParams.update({'font.size': 14})  # Change the 14 to your desired font size
    plt.xscale('log')
    plt.setp(plt.gca().lines, linewidth=2)
    plt.xlabel('Balancer')
    plt.ylabel('L2 Perturbation')
    plt.savefig(path, dpi=450, bbox_inches="tight")
    plt.show()
def plot_mean_max(results, path):
    plt.figure()
    plt.plot(results['balancers'],np.mean(results['mean_max'], axis = 0))
    plt.rcParams.update({'font.size': 14})  # Change the 14 to your desired font size
    plt.xscale('log')
    plt.yscale('log')
    plt.setp(plt.gca().lines, linewidth=2)
    plt.xlabel('Balancer')
    plt.ylabel('Mean Max')
    plt.savefig(path, dpi=450, bbox_inches="tight")
    plt.show()
def train_models(balancers, n_runs, max_index, folder, result_folder, get_model, x_train, y_train, location="end", batch_size=128):
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
                model.fit(x_train, y_train, epochs=2000, batch_size=batch_size, validation_split=0.2
                          , callbacks=[reduce_lr, early_stop], verbose=1)
                model.save_weights(path)
#             else:
#                 print("Already Exists!")

def test(balancers, n_runs, max_index, folder, result_folder, get_model, x_train, y_train, x_test, y_test
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
                print(f"No model for balancer {balancer}, run {run}")
                continue
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
    for key, item in results.items():
        if(key != "balancers"):
            print(f"{key}: {np.mean(item, axis=0)}")
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

def adversarial_train_models(n_runs, max_index, folder, get_model, x_train, y_train, epsilon, adv_epochs = 100, location="end", batch_size=128):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.0001)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    for run in range(n_runs):
        path = f"{folder}/run{run}.h5"
        print(f"Run {run}:")
        if(not os.path.exists(path)):
            X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            # Train the model
            model = get_model(X_train.shape[1:], location, activation = "relu")
            # Compile the model with the custom loss function
            # model.compile(optimizer='adam', loss=custom_loss(model, alpha=0, index = max_index), metrics=['accuracy'])
            model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
            model.fit(X_train, Y_train, epochs=2000, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[reduce_lr, early_stop], verbose=1)
            model = adversarial_training(model, X_train, Y_train, X_val, Y_val, adv_epochs, batch_size, "pgd", epsilon)
            model.save_weights(path)

def adversarial_test(n_runs, max_index, folder, result_folder, get_model, x_train, y_train, x_test, y_test
               , epsilon, batch_size=1, stored_results=None, location="end", adv_epochs = 5):
    info_list = ['accuracy', 'random_accuracy', 'fgsm_accuracy', 'pgd_accuracy'
                 , 'apgd_ce_accuracy', 'apgd_dlr_accuracy'
                 ,'cw_l2_accuracy']
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
        model = get_model(x_train.shape[1:], location = location, activation="relu")
        # Compile the model with the custom loss function
        model.compile(optimizer='adam', metrics=['accuracy'])
        print()
        if(os.path.exists(path)):
            model.load_weights(path)
        else:
            # Train the model
            print(f"No model for run {run}")
            continue

        # Evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        results['accuracy'].append(test_accuracy)
        for acc_attack in acc_attacks:
            if(len(results[acc_attack]) <= run):
                results[acc_attack].append(compute_robust_accuracy(model, x_test, y_test, epsilon = epsilon, attack = acc2attack[acc_attack],batch_size=batch_size))
        # if(len(results['mean_max']) <= run):
        #     results['mean_max'].append(np.mean(model.layers[max_index].max_values))
    with open(accuracy_score_path, "wb") as outfile:
        pickle.dump(results, outfile)
    for key, item in results.items():
        print(f"{key}: {np.mean(item)}")
    return results



def trades_loss(model,
                x_natural,
                y,
                beta,
                step_size=0.003,
                epsilon=0.01,
                perturb_steps=100,
                distance='l_inf',
                training = True,
                x_adv = None):
    # Define KL-loss
    criterion_kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
    batch_size = tf.shape(x_natural)[0]

    # model.trainable = False
    # model.trainable_variables = []

    if(x_adv == None):
        # Generate adversarial example
        x_adv = x_natural + 0.001 * tf.random.normal(shape=tf.shape(x_natural))
        if distance == 'l_inf':
            x_adv = create_adversarial_examples(model, x_natural, y, epsilon = epsilon, attack='pgd', batch_size=batch_size, verbose = False)
            # x_adv = madry_et_al.madry_et_al(model_fn = model,
            #                                         x = x_natural,
            #                                         eps = epsilon,
            #                                         eps_iter = epsilon / 10,
            #                                         nb_iter = 100,
            #                                         norm = np.inf,
            #                                         clip_min=0,
            #                                         clip_max=1,
            #                                         y=y,
            #                                         sanity_checks=False)
            # for _ in range(perturb_steps):
            #     x_adv = tf.Variable(x_adv, trainable=True)
            #
            #     with tf.GradientTape() as tape:
            #         logits_adv = model(x_adv)
            #         logits_natural = model(x_natural)
            #
            #         loss_kl = criterion_kl(tf.nn.log_softmax(logits_adv, axis=1),
            #                                tf.nn.softmax(logits_natural, axis=1))
            #
            #     gradients = tape.gradient(loss_kl, x_adv)
            #     x_adv = x_adv + step_size * tf.sign(gradients)
            #     x_adv = tf.clip_by_value(x_adv, x_natural - epsilon, x_natural + epsilon)
            #     x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

        elif distance == 'l_2':
            delta = 0.001 * tf.random.normal(shape=tf.shape(x_natural))
            delta = tf.Variable(delta, trainable=True)

            optimizer_delta = tf.keras.optimizers.SGD(learning_rate=epsilon / (perturb_steps * 2))

            for _ in range(perturb_steps):
                adv = x_natural + delta

                with tf.GradientTape() as tape:
                    logits_adv = model(adv)
                    logits_natural = model(x_natural)

                    loss = (-1) * criterion_kl(tf.nn.log_softmax(logits_adv, axis=1),
                                            tf.nn.softmax(logits_natural, axis=1))

                gradients = tape.gradient(loss, delta)
                grad_norms = tf.norm(gradients, ord=2, axis=(1, 2, 3))
                delta.assign(tf.math.divide_no_nan(gradients, tf.reshape(grad_norms, [-1, 1, 1, 1])))

                if tf.math.reduce_any(tf.math.equal(grad_norms, 0)):
                    delta.assign(tf.random.normal(shape=tf.shape(delta)))

                optimizer_delta.apply_gradients([(gradients, delta)])

                # Projection
                delta.assign_add(x_natural)
                delta.assign(tf.clip_by_value(delta, 0, 1))
                delta.assign(tf.clip_by_norm(delta - x_natural, epsilon, axes=[1, 2, 3]))

            x_adv = x_natural + delta

        else:
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

    # model.trainable = True

    # Calculate robust loss
    logits_natural = model(x_natural,training = training)
    logits_adv = model(x_adv, training=training)

    small_added = 1e-10
    loss_natural = tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y, logits_natural))
    loss_robust = (1.0 / tf.cast(batch_size, dtype=tf.float32)) * criterion_kl(tf.nn.softmax(logits_adv, axis=1) + small_added,
                                                                              tf.nn.softmax(logits_natural, axis=1) + small_added)
    loss = loss_natural + beta * loss_robust
    # print(f"Loss Nat: {loss_natural}, Loss Rob: {loss_robust}")
    return loss, loss_natural, loss_robust

def trades_train_models(n_runs, max_index, folder, get_model, x_train, y_train, epsilon, beta, location="end", batch_size=128):
    class CustomEarlyStopping(EarlyStopping):
        def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='min', restore_best_weights=False):
            super().__init__(monitor=monitor, patience=patience, verbose=verbose, mode=mode, restore_best_weights=restore_best_weights)
            self.best_metric = float('inf') if mode == 'min' else float('-inf')

        def on_epoch_end(self, epoch, logs=None):
            current_metric = logs.get(self.monitor)
            if current_metric is None:
                print(f"Warning: Early stopping conditioned on metric `{self.monitor}` which is not available. Available metrics are: {','.join(logs.keys())}")
                return

            print(f"Current {self.monitor}: {current_metric}, Best {self.monitor}: {self.best_metric}")
            # print(self.monitor_op(0, 1))
            # print(self.monitor_op(1,0))
            if self.monitor_op(current_metric, self.best_metric):
                self.best_metric = current_metric
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
                print(f"Improved! Resetting wait count to 0.")
            else:
                self.wait += 1
                print(f"No improvement. Wait count: {self.wait}")
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights and self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    print(f"\nEarly stopping conditioned on metric `{self.monitor}` improved, stopping training.")
                    return True
                else:
                    print(f"Still waiting. Training continues.")
            return False
    def val_func(model, x_val, y_val, batch_size, epsilon, step_size):
        val_loss = 0.0
        val_loss_nat = 0.0
        val_loss_robust = 0.0
        val_acc = 0.0
        num_batches = (len(x_val) // batch_size) + (len(x_val) % batch_size)

        for step in tqdm(range(0, len(x_val), batch_size)):
            x_batch_val = x_val[step:step + batch_size]
            y_batch_val = y_val[step:step + batch_size]
            x_adv = create_adversarial_examples(model, x_batch_val, y_batch_val, epsilon = epsilon, attack='pgd', batch_size=batch_size, verbose = False)
            with tf.GradientTape() as tape:
                # Assuming trades_loss is your loss function
                loss, loss_nat, loss_robust = trades_loss(tf.keras.models.Model(inputs = model.inputs, outputs = model.layers[-2].output)
                , x_batch_val, y_batch_val, beta, x_adv=x_adv)

            val_loss += loss.numpy()
            val_loss_nat += loss_nat.numpy()
            val_loss_robust += loss_robust.numpy()
            pred = model.predict(x_batch_val, verbose=0).argmax(axis = 1)
            val_acc += np.sum(pred == y_batch_val)
            # print(pred)
            # print(y_val)
            # print(val_acc)
        # Calculate average validation loss
        avg_val_loss = val_loss / num_batches
        avg_val_loss_nat = val_loss_nat / num_batches
        avg_val_loss_robust = val_loss_robust / num_batches
        return avg_val_loss, avg_val_loss_nat, avg_val_loss_robust, val_acc / len(x_val)
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    num_epochs = 2000
    for run in range(n_runs):
        path = f"{folder}/run{run}.h5"
        print(f"Run {run}:")
        if(not os.path.exists(path)):
            X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
            # Set up an optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            # Train the model
            model = get_model(X_train.shape[1:], location, activation="relu")
            # Compile the model with the custom loss function
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # model.summary()
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=0.0001, verbose=1)

            early_stop = CustomEarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr.set_model(model)
            early_stop.set_model(model)
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

                # Training
                for step in tqdm(range(0, len(X_train), batch_size)):
                    x_batch = X_train[step:step + batch_size]
                    y_batch = Y_train[step:step + batch_size]
                    x_adv = create_adversarial_examples(model, x_batch, y_batch, epsilon = epsilon, attack='pgd', batch_size=batch_size, verbose = False)
                    with tf.GradientTape() as tape:
                        loss, _ , _ = trades_loss(tf.keras.models.Model(inputs = model.inputs, outputs = model.layers[-2].output)
                        # loss, _ , _ = trades_loss(model
                        , x_batch, y_batch, beta, x_adv=x_adv)
                        # Calculate robust loss
                        # outs = model(x_batch, training=True)
                        # loss = tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_batch, outs))

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Validation
                val_loss , val_loss_nat, val_loss_robust, val_acc = val_func(model, X_val, Y_val, batch_size, epsilon, epsilon/10.0 * 3)

                # Print validation loss
                print(f"Epoch {epoch + 1}, Training Loss: {loss.numpy()}, Validation Loss: {val_loss}, Validation Natural Loss: {val_loss_nat}, Validation Robust Loss: {val_loss_robust}, Validation Acc: {val_acc}")

                # Update learning rate and check for early stopping
                reduce_lr.on_epoch_end(epoch, logs={'val_loss': val_loss})
                if early_stop.on_epoch_end(epoch, logs={'val_loss': val_loss}):
                    print("Early stopping.")
                    model.set_weights(early_stop.model.get_weights())
                    break
            model.save_weights(path)
