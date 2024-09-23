import numpy as np
import pandas as pd

from utils import *

my_columns = ['cpu_model','cpu_count','duration','emissions','cpu_energy','ram_energy','energy_consumed','region','country_name','cloud_provider','cloud_region']


# model = "bert"
# df = pd.read_csv(f'{RESULTS_DIR}emissions_{model}.csv')

# # energy_consumed = cpu_energy + gpu_energy + ram_energy

# selected_data = df[my_columns]

# print(selected_data)
# print(selected_data.describe())
# selected_data.describe().to_csv(f'{REPORTS_DIR}summary_{model}.csv', index=True)

for model in models:
    
    print(f'model:{model} --------------------------------')
    df = pd.read_csv(f'{RESULTS_DIR}emissions_{model}.csv')
    df = df[my_columns]
    print(df)
    print(df.describe())
    df.describe().to_csv(f'{REPORTS_DIR}summary_{model}.csv', index=True)