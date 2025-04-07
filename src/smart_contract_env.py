import gym
from gym import spaces
import numpy as np
# import pygraphviz as pgv
import pandas as pd
import random
import torch
# import optuna
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

# Custom Gym Environment for Smart Contract Vulnerability Detection
class SmartContractEnv(gym.Env):
    def __init__(self, data):
        super(SmartContractEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = spaces.Discrete(6)  # 6 possible actions
        
        # Observation space should match the shape of the feature vector extracted from CFG
        sample_cfg_path = data.iloc[0]['cfg_path']
        sample_features = extract_features_from_cfg(sample_cfg_path)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=sample_features.shape, dtype=np.float32)

        # Initialize random seed
        self.seed_value = None

    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
            
    def reset(self):
        self.current_step = 0
        return self._get_observation()
    
    def _get_observation(self):
        # Get the current row of data
        current_row = self.data.iloc[self.current_step]
        
        # Extract the CFG features for the current contract
        cfg_path = current_row['cfg_path']
        cfg_features = extract_features_from_cfg(cfg_path)
        
        # Return the features as the state
        return np.array(cfg_features, dtype=np.float32)

    def step(self, action):
        # Get the current state
        current_row = self.data.iloc[self.current_step]

        self.current_step += 1

        # Determine if the episode is done
        done = self.current_step >= len(self.data)
        next_state = self._get_observation() if not done else None

        # Calculate the reward based on the action and true labels
        reward = self._calculate_reward(action, current_row['slither'])
        
        return next_state, reward, done, {}

    def _calculate_reward(self, action_label, true_labels):
        reward = 1 if [str(action_label)] == [str(true_labels)] else 0
        return reward
