#!/usr/bin/env python
# coding: utf-8

"""

# Create one dataset for each profiler

- get profiler data organized by each serving configuration
- run {script}.py to go trough all experiment runs

- Input:
  - results_*/{profiler data}.csv
- Output
  - results_*/processed/{profiler}_processed.csv



"""

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



#device = "gpu"
#all_dir = f"D:/GAISSA/energy-repo/last_repo/june_{device}/"
#all_dir = f"D:/GAISSA/energy-repo/repo_sept_last_update/cudaep_nov_results/nov_cudaep_03/" # [CHANGE]
all_dir = f"D:/GAISSA/energy-repo/repo_sept_last_update/cpuep_nov_results/nov_cpuep_05/" # [CHANGE]

# python experiments/01_get_info_wattmeter.py  > ../cudaep_nov_results/nov_cudaep_03/01_get_info_wattmeter.log # CHANGE

for i in range(1,11):
    print(i)
    #continue

    results_dir = f'results_{str(i)}/'
    print(results_dir)
    #results_dir = 'results_1/' #change


    # In[3]:


    os.chdir(all_dir)


    # In[4]:


    processed_dir = all_dir + results_dir +"processed/"
    try:
        os.mkdir(processed_dir)
    except Exception as e:
        print(e)
        


    # In[5]:


    ENERGIBRIDGE_SAVE_CSV = 'energibridge_dataset.csv'
    NVIDIA_SAVE_CSV = 'nvidia-smi_dataset.csv'
    WATTMETER_SAVE_CSV = 'wattmeter_dataset.csv'


    # ## wattmeter files

    # In[6]:


    csv_dir = results_dir
    os.listdir(csv_dir)


    # In[7]:


    # Initialize an empty list to hold dataframes
    dfs = []

    for file in os.listdir(csv_dir):
        if file.endswith('.csv') and file.startswith('wattmeter'):
            print(f"file: {file}")
            
            # Parse the filename to get runtime and model
            #filename_parts = file.split('_')
            #runtime = filename_parts[1]
            #model = filename_parts[2].split('.')[0]

            filename_parts = file.split('_')
            print(filename_parts)
            model = filename_parts[1].split('_')[0]
            print(model)
            #print(filename_parts[1].split('_')[1])
            #model = models[0]
            #runtime_model = filename_parts[1].split('_')
            runtime = filename_parts[-1].split('.')[0]
            print(runtime)
            #model = runtime_model[1].split('.')[0]  # Split again to remove '.csv'
            
            # Load the CSV file into a dataframe
            df = pd.read_csv(os.path.join(csv_dir, file))

            # Add 'runtime' and 'model' columns
            df['runtime'] = runtime
            df['model'] = model

            # Append the dataframe to the list
            dfs.append(df)
            #print(df.columns)

    # Concatenate all dataframes in the list
    wattmeter_df = pd.concat(dfs, ignore_index=True)

    # Display the combined dataframe
    wattmeter_df


    # In[8]:


    #wattmeter_df['time'] = pd.to_datetime(wattmeter_df['True timestamp'],)
    wattmeter_df['time'] = pd.to_datetime(wattmeter_df['True timestamp'], format='mixed')
    wattmeter_df.head()


    # In[9]:


    columns = list(wattmeter_df.columns)
    new_order = columns[:2] + ['time','runtime', 'model'] + columns[2:-3]
    wattmeter_df = wattmeter_df[new_order]
    wattmeter_df.head()


    # In[10]:


    wattmeter_df = wattmeter_df.sort_values(by='time', ascending=True)
    wattmeter_df.to_csv(processed_dir+WATTMETER_SAVE_CSV, index=False)
    print(f"Results saved in {processed_dir+WATTMETER_SAVE_CSV}")


    # ## Verify timestamps match

    # In[11]:


    runall_df = pd.read_csv(results_dir + 'runall_timestamps.csv', )


    # In[12]:


    runall_df


    # In[13]:


    print("from runall:")
    print(runall_df['timestamp'].iloc[0])
    print(runall_df['timestamp'].iloc[len(runall_df)-1])

    print("wattmeter")
    print(wattmeter_df['time'].iloc[0])
    print(wattmeter_df['time'].iloc[len(wattmeter_df)-1])


    # In[14]:


    wattmeter_df.head()


    # In[15]:


    #selected_columns = ['time','runtime','model','CPU_ENERGY (J)', 'TOTAL_MEMORY', 'TOTAL_SWAP', 'USED_MEMORY', 'USED_SWAP']
    selected_columns = ['time','runtime','model','Current', 'PowerFactor', 'Phase',
        'Energy',  'EnergyNR',  'Load']
    energi_selected = wattmeter_df[selected_columns].copy()


    # In[16]:


    energi_selected.head()


    # In[17]:


    energi_selected['label'] = energi_selected['runtime'] + '_'+energi_selected['model']


    # In[18]:


    energi_selected['label'].unique()


    # In[19]:


    energi_selected['energy_joules'] = energi_selected['Energy']*3600
    energi_selected['energyNR_joules'] = energi_selected['EnergyNR']*3600
    energi_selected


    # In[20]:


    def add_energy_groupby(df, energy_col,groupby_col,new_col_name):
        new_df = df.copy()
        new_df[new_col_name] = new_df.groupby(groupby_col)[energy_col].transform(lambda x: x.max() - x.min())

        #new_df['energy' ] = new_df['e'].dt.total_seconds()
        return new_df


    # In[21]:


    df_last = add_energy_groupby(energi_selected,'energy_joules','label','energy_joules_config')
    df_last = add_energy_groupby(df_last,'energyNR_joules','label','energyNR_joules_config')
    df_last


    # In[22]:


    def calculate_statistics(df, columns, column_types, label_column):
        new_df = df.copy()
        
        # Iterate over the columns and their types
        for col, col_type in zip(columns, column_types):
            # Calculate the average based on the label_column
            if col_type == 'average':
                new_df['avg_' + col] = new_df.groupby(label_column)[col].transform('mean')
            # Calculate the difference, then average
            elif col_type == 'diff_then_average':
                new_df['diff_' + col] = new_df[col].diff()
                new_df['avg_diff_' + col] = new_df.groupby(label_column)['diff_' + col].transform('mean')
            # Calculate average minus last average
            elif col_type == 'average_minus_last_average':
                new_df['avg_' + col] = new_df.groupby(label_column)[col].transform('mean')
                #new_df['avg_minus_last_' + col] = new_df.groupby(label_column)['avg_' + col].diff()

                
                # Conditionally set the value for the first label
                #first_label_mask = new_df[label_column] != new_df[label_column].shift()
                #new_df.loc[first_label_mask, 'avg_minus_last_' + col] = new_df['avg_' + col]

                #pass
            
        return new_df


    # In[23]:


    #numeric_columns = ['CPU_ENERGY (J)', 'TOTAL_MEMORY', 'TOTAL_SWAP', 'USED_MEMORY', 'USED_SWAP']
    numeric_columns = ['Current', 'PowerFactor', 'Phase', 'Load']

    column_types = ['average', 'average', 'average', 'average',]
    #'average', 'diff_then_average', 'average_minus_last_average'
    df_with_metrics = calculate_statistics(df_last,numeric_columns,column_types,'label')
    df_with_metrics


    # In[24]:


    save_dir = 'wattmeter_processed.csv'
    df_with_metrics.to_csv(processed_dir+save_dir, index=False)


    # In[25]:


    runall_df = pd.read_csv(results_dir + 'runall_timestamps.csv', )
    print("from runall:")
    print(runall_df['timestamp'].iloc[0])
    print(runall_df['timestamp'].iloc[len(runall_df)-1])

    print("wattmeter")
    print(wattmeter_df['time'].iloc[0])
    print(wattmeter_df['time'].iloc[len(wattmeter_df)-1])


    # In[ ]:




