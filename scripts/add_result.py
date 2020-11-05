import numpy as np
import pandas as pd
import os
import yaml

path = os.path.join('output','cifar10','preact18-mixup-aa-cutout-ls-sgd','version_3','hparams.yaml')

with open(path) as f:
    df = pd.json_normalize(yaml.load(f))
df = df.drop(['data_path', 'output_path', 'resume'], axis=1)

df.to_csv('results.csv')
