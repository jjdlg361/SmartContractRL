# Custom callback to store rewards during training
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
from smart_contract_env import *

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.cumulative_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # Accumulate rewards during the episode
        self.current_rewards += self.locals['rewards']
        return True

    def _on_rollout_end(self) -> None:
        # Store the accumulated rewards when the episode ends
        self.episode_rewards.append(self.current_rewards)
        self.cumulative_rewards.append(self.current_rewards if len(self.cumulative_rewards) == 0 else self.current_rewards + self.cumulative_rewards[-1])
        self.current_rewards = 0