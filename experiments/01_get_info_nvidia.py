#!/usr/bin/env python
# coding: utf-8

# # Create one dataset for each profiler
# 
# - normalize datasets
#  -  
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


device = "gpu"
all_dir = f"D:/GAISSA/energy-repo/last_repo/june_{device}/"
#esults_dir = 'results_1/'

for i in range(1,11):
    print(i)
    #continue

    results_dir = f'results_{str(i)}/'
    print(results_dir)

# In[3]:


    os.chdir(all_dir)


    # In[4]:


    processed_dir = all_dir + results_dir +"processed/"
    try:
        os.mkdir(processed_dir)
    except Exception as e:
        print(e)


    # In[5]:


    #os.listdir(results_dir)


    # In[6]:


    ENERGIBRIDGE_SAVE_CSV = 'energibridge_dataset.csv'
    NVIDIA_SAVE_CSV = 'nvidia-smi_dataset.csv'
    WATTMETER_SAVE_CSV = 'wattmeter_dataset.csv'


    # ## nvidia-smi files

    # In[7]:


    csv_dir = results_dir
    os.listdir(csv_dir)


    # In[8]:


    ## add runtime and model columns

    # Initialize an empty list to hold dataframes
    dfs = []

    for file in os.listdir(csv_dir):
        if file.endswith('.csv') and file.startswith('nvidia'):
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
    nvidia_df = pd.concat(dfs, ignore_index=True)

    # Display the combined dataframe
    nvidia_df


    # In[9]:


    nvidia_df.columns


    # In[10]:


    nvidia_df['time'] = pd.to_datetime(nvidia_df['timestamp'],)
    nvidia_df.head()


    # In[11]:


    ## organize columns
    columns = list(nvidia_df.columns)
    new_order = columns[:2] + ['time','runtime', 'model'] + columns[2:-3]
    nvidia_df = nvidia_df[new_order]


    # In[12]:


    nvidia_df.head()


    # In[13]:


    nvidia_df.columns


    # In[14]:


    columns_to_int = [' utilization.gpu [%]', ' utilization.memory [%]',
        ' memory.total [MiB]', ' memory.used [MiB]', ]

    for column in columns_to_int:
        nvidia_df[column] = nvidia_df[column].str.replace(r'\D', '', regex=True).astype(int)


    # In[15]:


    columns_to_float = [' power.draw [W]', ' power.max_limit [W]']

    for column in columns_to_float:
        #df[column] = df[column].str.replace(r'[^\d.]+', '', regex=True).astype(float)
        #df[column] = df[column].str.replace(r'\D', '', regex=True).astype(float)
        nvidia_df[column] = nvidia_df[column].str.replace(r'[^\d.]+', '', regex=True).astype(float)
    #power.draw [W], power.max_limit [W]


    # In[16]:


    # verify no % columns and dont have strings on it 
    nvidia_df.head()


    # In[17]:


    processed_dir


    # In[18]:


    nvidia_df = nvidia_df.sort_values(by='time', ascending=True)
    nvidia_df.to_csv(processed_dir+NVIDIA_SAVE_CSV, index=False)
    print(f"Results saved in {processed_dir+NVIDIA_SAVE_CSV}")


    # In[19]:


    nvidia_df.columns


    # In[21]:


    #selected_columns = ['time','runtime','model','CPU_ENERGY (J)', 'TOTAL_MEMORY', 'TOTAL_SWAP', 'USED_MEMORY', 'USED_SWAP']
    selected_columns = ['time','runtime','model',' utilization.gpu [%]',' utilization.memory [%]',' memory.total [MiB]',' memory.used [MiB]',' power.draw [W]',' power.max_limit [W]',' temperature.gpu']
    energi_selected = nvidia_df[selected_columns].copy()


    # In[22]:


    energi_selected.head()


    # In[23]:


    energi_selected['label'] = energi_selected['runtime'] + '_'+energi_selected['model']


    # In[24]:


    energi_selected['label'].unique()


    # In[26]:


    energi_selected.columns


    # In[27]:


    energi_selected['used_memory_pct_mib'] = energi_selected[' memory.used [MiB]'] / energi_selected[' memory.total [MiB]'] *100
    energi_selected


    # In[ ]:





    # In[29]:


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


    # In[30]:


    #numeric_columns = ['CPU_ENERGY (J)', 'TOTAL_MEMORY', 'TOTAL_SWAP', 'USED_MEMORY', 'USED_SWAP']
    numeric_columns = [' utilization.gpu [%]',' utilization.memory [%]',' memory.used [MiB]',' power.draw [W]',' temperature.gpu','used_memory_pct_mib']

    column_types = ['average', 'average', 'average', 'average','average', 'average']
    #'average', 'diff_then_average', 'average_minus_last_average'
    df_with_metrics = calculate_statistics(energi_selected,numeric_columns,column_types,'label')


    # In[31]:


    df_with_metrics


    # In[32]:


    ## get duration from first instance of label = runtime_model until first of other

    def add_durations_groupby(df, time_col,groupby_col,):
        new_df = df.copy()
        new_df['config_duration' ] = new_df.groupby(groupby_col)[time_col].transform(lambda x: pd.to_datetime(x).max() - pd.to_datetime(x).min())
        #new_df['sum_time_config'] = pd.to_datetime(new['timestamp'],)
        #new_df['config_duration' ] = new_df['config_duration' ].total_seconds()
        new_df['config_duration' ] = new_df['config_duration'].dt.total_seconds()
        return new_df




    # In[33]:


    df_with_duration = add_durations_groupby(df_with_metrics,'time','label')
    #avg_ power.draw [W]


    # In[34]:


    df_with_duration.iloc[0]['config_duration']


    # In[35]:


    df_with_duration


    # In[36]:


    ## add energy from load and time-t
    df_with_duration['energy'] =  df_with_duration['avg_ power.draw [W]']*df_with_duration['config_duration']


    # In[37]:


    df_with_duration


    # In[42]:


    df_with_duration.columns


    # In[45]:


    df_with_duration.rename(columns={'avg_ temperature.gpu': 'avg_temperature_gpu_config'}, inplace=True)
    df_with_duration.rename(columns={'avg_ utilization.gpu [%]': 'avg_utilization_gpu_config'}, inplace=True)
    df_with_duration.rename(columns={'avg_ utilization.memory [%]': 'avg_utilization_memory_config'}, inplace=True)
    df_with_duration.rename(columns={'avg_ memory.used [MiB]': 'avg_memory_used_config'}, inplace=True)
    df_with_duration.rename(columns={'avg_ power.draw [W]': 'avg_power_draw_config'}, inplace=True)


    df_with_duration


    # In[46]:


    # Sort the DataFrame by the 'timestamp' column
    save_dir = 'nvidia_processed.csv'
    df_with_duration.to_csv(processed_dir+save_dir, index=False)


    # ## Verify timestamps match

    # In[39]:


    runall_df = pd.read_csv(results_dir + 'runall_timestamps.csv', )


    # In[40]:


    runall_df


    # In[41]:


    print("from runall:")
    print(runall_df['timestamp'].iloc[0])
    print(runall_df['timestamp'].iloc[len(runall_df)-1])

    print("nvidia")
    print(nvidia_df['time'].iloc[0])
    print(nvidia_df['time'].iloc[len(nvidia_df)-1])


    # In[ ]:




