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


# In[2]:


device = "gpu"
all_dir = f"D:/GAISSA/energy-repo/last_repo/june_{device}/"


# In[3]:


os.chdir(all_dir)

    
for i in range(1,11):
    print(i)
    #continue

    results_dir = f'results_{str(i)}/'
    print(results_dir)
    #continue



    processed_dir = all_dir + results_dir +"processed/"
    try:
        os.mkdir(processed_dir)
    except Exception as e:
        print(e)
        


    # In[6]:


    ENERGIBRIDGE_SAVE_CSV = 'energibridge_dataset.csv'
    NVIDIA_SAVE_CSV = 'nvidia-smi_dataset.csv'
    WATTMETER_SAVE_CSV = 'wattmeter_dataset.csv'


    # ## energibridge files

    # In[7]:


    # Adding runtime and model columns

    import pandas as pd
    import os

    # Directory containing the CSV files
    csv_dir = results_dir

    # Initialize an empty list to hold dataframes
    dfs = []

    # Loop through the files in the directory
    for file in os.listdir(csv_dir):
        if file.endswith('.csv') and file.startswith('energy'):
            print(f"file: {file}")

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
            if runtime == 'idle':
                model = 'idle'
            
            # Add 'runtime' and 'model' columns
            df['runtime'] = runtime
            df['model'] = model
            

            # Append the dataframe to the list
            dfs.append(df)
            #print(df.columns)

    # Concatenate all dataframes in the list
    combined_df = pd.concat(dfs, ignore_index=True)

    print(len(combined_df))
    # Display the combined dataframe
    combined_df


    # In[8]:


    combined_df.columns


    # In[9]:


    combined_df['time'] = pd.to_datetime(combined_df['Time'],unit='ms')


    # In[10]:


    columns = list(combined_df.columns)
    new_order = columns[:2] + ['time','runtime', 'model'] + columns[2:-3]
    combined_df = combined_df[new_order]
    combined_df.head()


    # In[11]:


    combined_df


    # In[12]:


    # check the time in runall_timestamps.csv and compare with other profilers results' csv
    combined_df['time'] += pd.Timedelta(hours=2) 
    combined_df.head()


    # In[13]:


    combined_df = combined_df.sort_values(by='time', ascending=True)


    # In[14]:


    combined_df.head()


    # In[15]:


    # Sort the DataFrame by the 'timestamp' column
    combined_df.to_csv(processed_dir+ENERGIBRIDGE_SAVE_CSV, index=False)
    print(f"Results saved in {processed_dir+ENERGIBRIDGE_SAVE_CSV}")


    # In[16]:


    for c in combined_df.columns:
        print(c)


    # In[18]:


    #combined_df[:10]


    # In[19]:


    # Create a list of column names to include in the mean calculation
    cpu_columns = [col for col in combined_df.columns if col.startswith('CPU_USAGE_')]

    # Calculate the mean across these columns for each row
    combined_df['AVG_CPU_USAGE_SAMP'] = combined_df[cpu_columns].mean(axis=1)
    combined_df


    # In[20]:


    combined_df['AVG_CPU_USAGE_SAMP'].max()


    # In[21]:


    combined_df[cpu_columns]


    # In[22]:


    combined_df


    # In[ ]:





    # In[23]:


    selected_columns = ['time','runtime','model','CPU_ENERGY (J)', 'TOTAL_MEMORY', 'TOTAL_SWAP', 'USED_MEMORY', 'USED_SWAP','AVG_CPU_USAGE_SAMP']
    energi_selected = combined_df[selected_columns].copy()


    # In[24]:


    energi_selected.head()


    # In[25]:


    energi_selected['label'] = energi_selected['runtime'] + '_'+energi_selected['model']


    # In[26]:


    energi_selected['label'].unique()


    # In[27]:


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


    # In[28]:


    numeric_columns = [ 'TOTAL_MEMORY', 'TOTAL_SWAP', 'USED_MEMORY', 'USED_SWAP','AVG_CPU_USAGE_SAMP']
    column_types = ['average', 'average', 'average', 'average','average']
    #'average', 'diff_then_average', 'average_minus_last_average'
    df_with_metrics = calculate_statistics(energi_selected,numeric_columns,column_types,'label')


    # In[29]:


    df_with_metrics


    # In[30]:


    def add_difference_from_previous_label(df, col1, label_column):
        new_df = df.copy()
        prev_label = None
        prev_value = 0

        current_value1 = 0
        prev_value1 = 0

        aux_value=0
        aux_value2=0
        
        for index, row in new_df.iterrows():
            current_label = row[label_column]
            current_value = row[col1]
            if current_label != prev_label:
                
                if current_value is None:
                    current_value = 0
                if prev_value is None:
                    prev_value = 0
                new_df.at[index, 'minus_'+col1] = current_value - prev_value
                print(current_value," oo ",prev_value)
                
                aux_value2=current_value
                
                aux_value = prev_value
                
                prev_label = current_label
                prev_value = current_value
                print(current_value," oo ",prev_value)
                
                #new_df.at[index, 'difference_from_prev_label'] = current_value - prev_value
                
                #print("-")
            else:
                #if current_value is None:
                #    current_value = 0
                if aux_value is None or aux_value:
                    pass
                
                new_df.at[index, 'minus_'+col1] = aux_value2 - aux_value
                
                #print(aux_value2," -- ",aux_value)
                #new_df.at[index, 'difference_from_prev_label'] = current_value - prev_value
        return new_df

    #df_with_metrics = add_difference_from_previous_label(df_with_metrics,'avg_USED_MEMORY','label')

    #df_with_metrics = add_difference_from_previous_label(df_with_metrics,'avg_USED_SWAP','label')


    # In[31]:


    df_with_metrics


    # In[32]:


    def add_energy_groupby(df, energy_col,groupby_col,):
        new_df = df.copy()
        new_df['energy' ] = new_df.groupby(groupby_col)[energy_col].transform(lambda x: x.max() - x.min())

        #new_df['energy' ] = new_df['e'].dt.total_seconds()
        return new_df


    # In[33]:


    # get total energy of configuration
    df_last = add_energy_groupby(df_with_metrics,'CPU_ENERGY (J)','label')
    df_last


    # In[37]:


    df_last['avg_used_memory_pct_config'] = df_last['avg_USED_MEMORY'] / df_last['TOTAL_MEMORY'] *100
    df_last


    # In[34]:


    df_last.rename(columns={'avg_AVG_CPU_USAGE_SAMP': 'avg_cpu_usage_config'}, inplace=True)
    df_last


    # In[36]:


    df_last['avg_cpu_usage_config'].max()


    # In[38]:


    df_last


    # In[39]:


    # Sort the DataFrame by the 'timestamp' column
    save_dir = processed_dir + 'energi_processed.csv'
    df_last.to_csv(save_dir, index=False)
    print(f"Results saved in {save_dir}")


    # ## Verify timestamps match

    # In[40]:


    runall_df = pd.read_csv(results_dir + 'runall_timestamps.csv', )


    # In[41]:


    runall_df


    # In[42]:


    print("from runall:")
    print(runall_df['timestamp'].iloc[0])
    print(runall_df['timestamp'].iloc[len(runall_df)-1])
    print("energi")
    print(combined_df['time'].iloc[0])
    print(combined_df['time'].iloc[len(combined_df)-1])


    # In[ ]:




