# SmartContractRL
The following code was created and validated by Jose Juan de Leon, Cenchuan Zhang, Christos Koulouris, Francesca Medda and Rahul Ravji.

## Updates

This fork introduces **GRPO** (Group Relative Policy Optimization).
GRPO trains a policy without a separate value network by sampling
multiple candidate actions for each observation. A reward model scores
the candidates and the average score in the group is used as a baseline
for the policy gradient. This memory‑efficient approach allows larger
models to be trained on limited hardware while still performing well on
complex reasoning tasks. Hyperparameters for DQN and GRPO are tuned with
Optuna.

