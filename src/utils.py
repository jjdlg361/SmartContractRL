import re
import pandas as pd
import seaborn as sns
from hexbytes import HexBytes
import matplotlib.pyplot as plt
from datasets import load_dataset
import os
from subprocess import Popen, PIPE

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group is not None, then we have captured a real comment string.
        if match.group(2) is not None:
            return "" 
        else: # otherwise, we will return the 1st group
            return match.group(1) 
    return regex.sub(_replacer, string)

def get_lenghts(example):
    code = remove_comments(example['source_code'])
    example['sourcecode_len'] = len(code.split())
    example['bytecode_len'] = len(HexBytes(example['bytecode']))
    return example


# Function to generate CFG using EtherSolve
def generate_cfg(bytecode, contract_address):
    output_dot = f'../data/cfg_output/{contract_address}.dot'
    command = [
        'java', '-jar', '../data/EtherSolve.jar', '-r', '--dot', 
        bytecode, '-o', output_dot
    ]
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        return output_dot
    else:
        print(f"Error generating CFG for {contract_address}: {stderr.decode('utf-8')}")
        return None
    

def generate_cfg_row(row):
    return generate_cfg(row['bytecode'], row['address'])