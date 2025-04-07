import gym
from gym import spaces
import numpy as np
# import pygraphviz as pgv
import pandas as pd
import random
import torch
import optuna
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from agent_utils import *
from smart_contract_env import *
from reward_callback import *

#################################### Initialisation ###########################################

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load the dataset
data = pd.read_csv('../data/smart_contract_dataset_with_cfgs.csv')
# Split dataset into training, validation, and test sets
train_data = data[data['dataset'] == 'train']
val_data = data[data['dataset'] == 'val']
test_data = data[data['dataset'] == 'test']
# Initialize environments
train_env = SmartContractEnv(train_data)
val_env = SmartContractEnv(val_data)
test_env = SmartContractEnv(test_data)

# Function to optimize DQN
def optimize_dqn(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    buffer_size = trial.suggest_int('buffer_size', 50000, 1000000)
    learning_starts = trial.suggest_int('learning_starts', 1000, 10000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)

    model = DQN('MlpPolicy', train_env, learning_rate=learning_rate, buffer_size=buffer_size,
                learning_starts=learning_starts, batch_size=batch_size, gamma=gamma, verbose=0, seed=42)
    model.learn(total_timesteps=10000)

    mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=10)
    return mean_reward

# Function to optimize PPO
def optimize_ppo(trial):
    n_steps = trial.suggest_categorical('n_steps', [64, 128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)

    model = PPO('MlpPolicy', train_env, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                gamma=gamma, learning_rate=learning_rate, ent_coef=ent_coef, clip_range=clip_range, verbose=0, seed=42)
    model.learn(total_timesteps=10000)

    mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=10)
    return mean_reward

# Optuna study for DQN
study_dqn = optuna.create_study(direction='maximize')
study_dqn.optimize(optimize_dqn, n_trials=20)

# Optuna study for PPO
study_ppo = optuna.create_study(direction='maximize')
study_ppo.optimize(optimize_ppo, n_trials=20)

# Best hyperparameters
best_params_dqn = study_dqn.best_params
best_params_ppo = study_ppo.best_params

print("Best DQN params:", best_params_dqn)
print("Best PPO params:", best_params_ppo)

# Initialize callbacks to store rewards
dqn_callback = RewardCallback()
ppo_callback = RewardCallback()

# Train the models with best parameters and reward callback
best_dqn_model = DQN('MlpPolicy', train_env, **best_params_dqn, verbose=1, seed=42)  # verbose=1 
best_dqn_model.learn(total_timesteps=15000, callback=dqn_callback)

best_ppo_model = PPO('MlpPolicy', train_env, **best_params_ppo, verbose=1, seed=42)  # **best_params_ppo
best_ppo_model.learn(total_timesteps=10000, callback=ppo_callback)

# Plot learning curves for DQN and PPO
plt.figure(figsize=(15, 12))
plt.subplot(2,1,1)
plt.plot(dqn_callback.cumulative_rewards, label='DQN')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Learning Curve for DQN')

plt.subplot(2,1,2)
plt.plot(ppo_callback.cumulative_rewards, label='PPO')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Learning Curve for PPO')
plt.savefig("../plots/agents_plot1.png")
plt.close()

# Evaluate the best DQN and PPO models on the validation set
accuracy_dqn, recall_dqn, f1_dqn, pred_dqn, label_dqn = evaluate_model(best_dqn_model, val_env)
accuracy_ppo, recall_ppo, f1_ppo, pred_ppo, label_ppo = evaluate_model(best_ppo_model, val_env)

print(f"DQN - Accuracy: {accuracy_dqn}, Recall: {recall_dqn}, F1: {f1_dqn}")
print(f"PPO - Accuracy: {accuracy_ppo}, Recall: {recall_ppo}, F1: {f1_ppo}")

print(pred_dqn), print(label_dqn), print(pred_ppo), print(label_ppo)

# DQN
num_classes = 6

fpr_dict = dict()
tpr_dict = dict()
roc_threshold_dict = dict()
roc_auc_dict = dict()

precision_dict = dict()
recall_dict = dict()
pr_threshold_dict = dict()
pr_auc_dict = dict()

# Binarize multi-label classes using One vs ALL methodology
y_binarize = label_binarize(pred_dqn, classes=[0,1,2,3,4,5])
y_pred_multiclass = label_binarize(label_dqn, classes=[0,1,2,3,4,5])

for label_num in range(num_classes):
    y_true_for_curr_class = y_binarize[:, label_num]
    y_pred_for_curr_class = y_pred_multiclass[:, label_num]

    # calculate fpr,tpr and thresholds across various decision thresholds; pos_label = 1 because one hot encode guarantees it
    
    fpr_dict[label_num], tpr_dict[label_num], roc_threshold_dict[label_num] = roc_curve(
        y_true=y_true_for_curr_class, y_score=y_pred_for_curr_class, pos_label=1
    )
    roc_auc_dict[label_num] = auc(fpr_dict[label_num], tpr_dict[label_num])
    
    precision_dict[label_num], recall_dict[label_num], pr_threshold_dict[label_num] = precision_recall_curve(y_true_for_curr_class, y_pred_for_curr_class)
    pr_auc_dict[label_num] = average_precision_score(y_true_for_curr_class, y_pred_for_curr_class)

    print(f"ROC score for class {label_num} is {roc_auc_dict[label_num]}. ")
    print(f"PR score for class {label_num} is {pr_auc_dict[label_num]}. \nNote we are considering class {label_num} as the positive class and treating other classes as negative class.\n")

print(fpr_dict)
print(tpr_dict)
print(roc_threshold_dict)

# plot ROC  
plt.plot(fpr_dict[0], tpr_dict[0], linestyle='-.',color='orange', label=f'access-control (AUC = {roc_auc_dict[0]:.2f})')
plt.plot(fpr_dict[1], tpr_dict[1], linestyle='-.',color='green', label=f'arithmetic (AUC = {roc_auc_dict[1]:.2f})')
plt.plot(fpr_dict[2], tpr_dict[2], linestyle='-.',color='blue', label=f'other (AUC = {roc_auc_dict[2]:.2f})')
plt.plot(fpr_dict[3], tpr_dict[3], linestyle='-.',color='brown', label=f'reentrancy (AUC = {roc_auc_dict[3]:.2f})')
plt.plot(fpr_dict[4], tpr_dict[4], linestyle='-.',color='red', label=f'safe (AUC = {roc_auc_dict[4]:.2f})')
plt.plot(fpr_dict[5], tpr_dict[5], linestyle='-.',color='darkgray', label=f'unchecked-calls (AUC = {roc_auc_dict[5]:.2f})')

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('DQN Multiclass ROC',dpi=300)

# plot PR for DQN
plt.plot(recall_dict[0], precision_dict[0], linestyle='-.',color='orange', label=f'access-control (area = {pr_auc_dict[0]:.2f})')
plt.plot(recall_dict[1], precision_dict[1], linestyle='-.',color='green', label=f'arithmetic (area = {pr_auc_dict[1]:.2f})')
plt.plot(recall_dict[2], precision_dict[2], linestyle='-.',color='blue', label=f'other (area = {pr_auc_dict[2]:.2f})')
plt.plot(recall_dict[3], precision_dict[3], linestyle='-.',color='brown', label=f'reentrancy (area = {pr_auc_dict[3]:.2f})')
plt.plot(recall_dict[4], precision_dict[4], linestyle='-.',color='red', label=f'safe (area = {pr_auc_dict[4]:.2f})')
plt.plot(recall_dict[5], precision_dict[5], linestyle='-.',color='darkgray', label=f'unchecked-calls (area = {pr_auc_dict[5]:.2f})')

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig('DQN Multiclass PR Curve',dpi=300)

# PPO
num_classes = 6

fpr_dict = dict()
tpr_dict = dict()
roc_threshold_dict = dict()
roc_auc_dict = dict()

precision_dict = dict()
recall_dict = dict()
pr_threshold_dict = dict()
pr_auc_dict = dict()

# Binarize multi-label classes using One vs ALL methodology
y_binarize = label_binarize(pred_ppo, classes=[0,1,2,3,4,5])
y_pred_multiclass = label_binarize(label_ppo, classes=[0,1,2,3,4,5])

for label_num in range(num_classes):
    y_true_for_curr_class = y_binarize[:, label_num]
    y_pred_for_curr_class = y_pred_multiclass[:, label_num]

    # calculate fpr,tpr and thresholds across various decision thresholds; pos_label = 1 because one hot encode guarantees it
    
    fpr_dict[label_num], tpr_dict[label_num], roc_threshold_dict[label_num] = roc_curve(
        y_true=y_true_for_curr_class, y_score=y_pred_for_curr_class, pos_label=1
    )
    roc_auc_dict[label_num] = auc(fpr_dict[label_num], tpr_dict[label_num])
    
    precision_dict[label_num], recall_dict[label_num], pr_threshold_dict[label_num] = precision_recall_curve(y_true_for_curr_class, y_pred_for_curr_class)
    pr_auc_dict[label_num] = average_precision_score(y_true_for_curr_class, y_pred_for_curr_class)

    print(f"ROC score for class {label_num} is {roc_auc_dict[label_num]}. ")
    print(f"PR score for class {label_num} is {pr_auc_dict[label_num]}. \nNote we are considering class {label_num} as the positive class and treating other classes as negative class.\n")

print(fpr_dict)
print(tpr_dict)
print(roc_threshold_dict)

# plot ROC  
plt.plot(fpr_dict[0], tpr_dict[0], linestyle='-.',color='orange', label=f'access-control (AUC = {roc_auc_dict[0]:.2f})')
plt.plot(fpr_dict[1], tpr_dict[1], linestyle='-.',color='green', label=f'arithmetic (AUC = {roc_auc_dict[1]:.2f})')
plt.plot(fpr_dict[2], tpr_dict[2], linestyle='-.',color='blue', label=f'other (AUC = {roc_auc_dict[2]:.2f})')
plt.plot(fpr_dict[3], tpr_dict[3], linestyle='-.',color='brown', label=f'reentrancy (AUC = {roc_auc_dict[3]:.2f})')
plt.plot(fpr_dict[4], tpr_dict[4], linestyle='-.',color='red', label=f'safe (AUC = {roc_auc_dict[4]:.2f})')
plt.plot(fpr_dict[5], tpr_dict[5], linestyle='-.',color='darkgray', label=f'unchecked-calls (AUC = {roc_auc_dict[5]:.2f})')

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('PPO Multiclass ROC',dpi=300)

# plot PR for PPO
plt.plot(recall_dict[0], precision_dict[0], linestyle='-.',color='orange', label=f'access-control (area = {pr_auc_dict[0]:.2f})')
plt.plot(recall_dict[1], precision_dict[1], linestyle='-.',color='green', label=f'arithmetic (area = {pr_auc_dict[1]:.2f})')
plt.plot(recall_dict[2], precision_dict[2], linestyle='-.',color='blue', label=f'other (area = {pr_auc_dict[2]:.2f})')
plt.plot(recall_dict[3], precision_dict[3], linestyle='-.',color='brown', label=f'reentrancy (area = {pr_auc_dict[3]:.2f})')
plt.plot(recall_dict[4], precision_dict[4], linestyle='-.',color='red', label=f'safe (area = {pr_auc_dict[4]:.2f})')
plt.plot(recall_dict[5], precision_dict[5], linestyle='-.',color='darkgray', label=f'unchecked-calls (area = {pr_auc_dict[5]:.2f})')

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig('PPO Multiclass PR Curve',dpi=300)

# Compare DQN and PPO to select the optimal model - PPO in this case
optimal_model = best_dqn_model if f1_dqn > f1_ppo else best_ppo_model
optimal_model

# Evaluate the optimal model on the test set
accuracy_test, recall_test, f1_test, pred_test, label_test = evaluate_model(optimal_model, test_env)

print(f"Test - Accuracy: {accuracy_test}, Recall: {recall_test}, F1: {f1_test}")

print(pred_test), print(label_test)

# test set
num_classes = 6

fpr_dict = dict()
tpr_dict = dict()
roc_threshold_dict = dict()
roc_auc_dict = dict()

precision_dict = dict()
recall_dict = dict()
pr_threshold_dict = dict()
pr_auc_dict = dict()

# Binarize multi-label classes using One vs ALL methodology
y_binarize = label_binarize(pred_test, classes=[0,1,2,3,4,5])
y_pred_multiclass = label_binarize(label_test, classes=[0,1,2,3,4,5])

for label_num in range(num_classes):
    y_true_for_curr_class = y_binarize[:, label_num]
    y_pred_for_curr_class = y_pred_multiclass[:, label_num]

    # calculate fpr,tpr and thresholds across various decision thresholds; pos_label = 1 because one hot encode guarantees it
    
    fpr_dict[label_num], tpr_dict[label_num], roc_threshold_dict[label_num] = roc_curve(
        y_true=y_true_for_curr_class, y_score=y_pred_for_curr_class, pos_label=1
    )
    roc_auc_dict[label_num] = auc(fpr_dict[label_num], tpr_dict[label_num])
    
    precision_dict[label_num], recall_dict[label_num], pr_threshold_dict[label_num] = precision_recall_curve(y_true_for_curr_class, y_pred_for_curr_class)
    pr_auc_dict[label_num] = average_precision_score(y_true_for_curr_class, y_pred_for_curr_class)

    print(f"ROC score for class {label_num} is {roc_auc_dict[label_num]}. ")
    print(f"PR score for class {label_num} is {pr_auc_dict[label_num]}. \nNote we are considering class {label_num} as the positive class and treating other classes as negative class.\n")

print(fpr_dict)
print(tpr_dict)
print(roc_threshold_dict)

# plot ROC  
plt.plot(fpr_dict[0], tpr_dict[0], linestyle='-.',color='orange', label=f'access-control (AUC = {roc_auc_dict[0]:.2f})')
plt.plot(fpr_dict[1], tpr_dict[1], linestyle='-.',color='green', label=f'arithmetic (AUC = {roc_auc_dict[1]:.2f})')
plt.plot(fpr_dict[2], tpr_dict[2], linestyle='-.',color='blue', label=f'other (AUC = {roc_auc_dict[2]:.2f})')
plt.plot(fpr_dict[3], tpr_dict[3], linestyle='-.',color='brown', label=f'reentrancy (AUC = {roc_auc_dict[3]:.2f})')
plt.plot(fpr_dict[4], tpr_dict[4], linestyle='-.',color='red', label=f'safe (AUC = {roc_auc_dict[4]:.2f})')
plt.plot(fpr_dict[5], tpr_dict[5], linestyle='-.',color='darkgray', label=f'unchecked-calls (AUC = {roc_auc_dict[5]:.2f})')

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Test multiclass ROC',dpi=300)

# plot PR
plt.plot(recall_dict[0], precision_dict[0], linestyle='-.',color='orange', label=f'access-control (area = {pr_auc_dict[0]:.2f})')
plt.plot(recall_dict[1], precision_dict[1], linestyle='-.',color='green', label=f'arithmetic (area = {pr_auc_dict[1]:.2f})')
plt.plot(recall_dict[2], precision_dict[2], linestyle='-.',color='blue', label=f'other (area = {pr_auc_dict[2]:.2f})')
plt.plot(recall_dict[3], precision_dict[3], linestyle='-.',color='brown', label=f'reentrancy (area = {pr_auc_dict[3]:.2f})')
plt.plot(recall_dict[4], precision_dict[4], linestyle='-.',color='red', label=f'safe (area = {pr_auc_dict[4]:.2f})')
plt.plot(recall_dict[5], precision_dict[5], linestyle='-.',color='darkgray', label=f'unchecked-calls (area = {pr_auc_dict[5]:.2f})')

plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.savefig('Test Multiclass PR Curve',dpi=300)

lab = ['access-control','arithmetic','other','reentrancy','safe','unchecked-calls']

report = classification_report(label_test, pred_test, target_names=lab)
print(report)


cm = confusion_matrix(label_test, pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lab)
disp.plot()
plt.savefig("../plots/final.png")
plt.close()
