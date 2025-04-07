import gym
from gym import spaces
import numpy as np
# import pygraphviz as pgv
import networkx as nx
import numpy as np
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

# Assuming the extract_features_from_cfg function is already defined
# def extract_features_from_cfg(cfg_path):
#     # Read the CFG .dot file using pygraphviz
#     G = pgv.AGraph(cfg_path)
    
#     # Example: Use simple graph features like the number of nodes and edges
#     num_nodes = len(G.nodes())
#     num_edges = len(G.edges())
    
#     # Combine them into a feature vector
#     features = np.array([num_nodes, num_edges])
#     return features

def extract_features_from_cfg(cfg_path):
    # Read the CFG .dot file using NetworkX and pydot
    G = nx.drawing.nx_pydot.read_dot(cfg_path)
    
    # Use simple graph features like the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Combine them into a feature vector
    features = np.array([num_nodes, num_edges])
    return features

def evaluate_model(model, env):
    all_preds = []
    all_labels = []

    # Reset the environment
    obs = env.reset()

    # Loop through the environment until the episode is done 
    for _ in range(len(env.data)):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
        all_preds.append(action)
        all_labels.append(env.data.iloc[env.current_step]['slither'])     

    # Convert predictions and labels to numpy arrays
    all_preds = np.array([int(label) for label in all_preds])
    all_labels = np.array([int(label) for label in all_labels])

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    return accuracy, recall, f1, all_preds, all_labels

def evaluate_model(model, env):
    all_preds = []
    all_labels = []

    # Reset the environment
    obs = env.reset()

    # Loop through the environment until the episode is done 
    for _ in range(len(env.data)):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
        all_preds.append(action)
        all_labels.append(env.data.iloc[env.current_step]['slither'])     

    # Convert predictions and labels to numpy arrays
    all_preds = np.array([int(label) for label in all_preds])
    all_labels = np.array([int(label) for label in all_labels])

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    return accuracy, recall, f1, all_preds, all_labels

