import re
import pandas as pd
import seaborn as sns
from hexbytes import HexBytes
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils import *
from tqdm import tqdm
import os
from subprocess import Popen, PIPE
from concurrent.futures import ProcessPoolExecutor

from graphviz import Source
from graphviz import sources
from IPython.display import Image
import networkx as nx
import matplotlib.pyplot as plt


# Load the dataset
train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', ignore_verifications=True, trust_remote_code=True)
test_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='test', ignore_verifications=True, trust_remote_code=True)
val_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='validation', ignore_verifications=True, trust_remote_code=True)

COLS_TO_REMOVE = ['source_code']

LABELS = {0:'access-control', 1:'arithmetic', 2:'other', 3:'reentrancy', 4:'safe', 5:'unchecked-calls'}

datasets = []
for split in [train_set, test_set, val_set]:
    split_df = pd.DataFrame(split.map(get_lenghts, remove_columns=COLS_TO_REMOVE)).explode('slither')
    split_df['slither_label'] = split_df['slither'].map(LABELS)
    datasets.append(split_df)

concatenated = pd.concat([split.assign(dataset=split_name) for split, split_name in zip(datasets, ['train', 'test', 'val'])])
concatenated.head()
concatenated.info(), concatenated['dataset'].value_counts()
eda = concatenated

 
sns.countplot(x='dataset', data=eda)
plt.savefig("../plots/plot1.png")
plt.close()

eda_train = eda[eda['dataset']=='train']
eda_train['slither_label'].value_counts()

for label in ['unchecked-calls', 'safe', 'reentrancy', 'other', 'arithmetic', 'access-control']:
    a = eda_train[eda_train['slither_label']==label]
    for address in a['address']:
        b = eda_train[eda_train['address']==address].count()>1
    print("Percentage of multi-label contracts in", label, "category is", len(b)/len(a)*100, "%")

a = eda_train.groupby('address').count()>1

print('Percentage of multi-lable contracts in training set is', 
      len(a['slither_label']==True)/79641)

###################### Plots #################################

_, ax = plt.subplots(figsize=(15, 8))
sns.countplot(x='slither_label', data=eda, hue='dataset', ax=ax)
plt.savefig("../plots/plot2.png")
plt.close()

eda = eda.drop('slither_label', axis='columns')
eda = eda[~eda['address'].duplicated(keep='first')].reset_index()
eda['dataset'].value_counts(), eda.head()

_, ax = plt.subplots(figsize=(15, 8))
sns.histplot(data=eda, x="sourcecode_len", kde=True, hue='dataset', ax=ax)
plt.savefig("../plots/plot3.png")
plt.close()

_, ax = plt.subplots(figsize=(15, 8))
sns.histplot(data=eda, x="bytecode_len", kde=True, hue='dataset', ax=ax)
plt.savefig("../plots/plot4.png")
plt.close()

empty_bytecodes = eda[eda['bytecode_len'] == 0]['dataset'].value_counts()

empty_bytecodes / eda['dataset'].value_counts()

data_clean = eda.drop(eda[eda['bytecode_len']==0].index)
data_clean = data_clean.drop(columns=['sourcecode_len','bytecode_len'])
data_clean['dataset'].value_counts()

data_clean.to_csv('../data/data_clean.csv')

##############################################################################
###################### PARALLELIZATION #######################################
##############################################################################

data_clean = pd.read_csv("../data/data_clean.csv")

df = data_clean
df['dataset'].value_counts()

with ProcessPoolExecutor(max_workers=20) as executor:
    results = list(tqdm(executor.map(generate_cfg_row, [row for _, row in df.iterrows()]), total=len(df)))
    
df['cfg_path'] = results
df.to_csv('../data/smart_contract_dataset_with_cfgs.csv', index=False)

df = pd.read_csv('../data/smart_contract_dataset_with_cfgs.csv')

example_contract_address = df.loc[1143, 'address'] # df['address'][0]
cfg_path = df.loc[1143, 'cfg_path']    # df['cfg_path'][0]

if cfg_path:
    with open(cfg_path, 'r') as f:
        dot_graph = f.read()

graph = Source(dot_graph)
output_path = f'../plots/cfg_visualization_{example_contract_address}'
graph.render(output_path, format='png')

# # Read the DOT file into a NetworkX graph (requires pydot, install via pip if needed)
# G = nx.drawing.nx_pydot.read_dot(cfg_path)

# plt.figure(figsize=(15, 8))
# # Use a spring layout as an alternative to dot's hierarchical layout
# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos, with_labels=True, node_size=500, font_size=10)

# output_path = f'../plots/cfg_visualization_{example_contract_address}.png'
# plt.savefig(output_path)
# plt.close()
