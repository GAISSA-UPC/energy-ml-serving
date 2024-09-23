#!/usr/bin/env python
# coding: utf-8



# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[3]:

device = 'gpu'
new_directory = f"D:/GAISSA/energy-repo/last_repo/june_{device}/" # change
os.chdir(new_directory)


# In[4]:
for i in range(1,11):
    print(i)
    #continue

    results_dir = f'results_{str(i)}/'
    print(results_dir)
    #results_dir = 'results_1/' #change


    ENERGIBRIDGE_SAVE_CSV = 'energibridge_dataset.csv'
    NVIDIA_SAVE_CSV = 'nvidia-smi_dataset.csv'
    WATTMETER_SAVE_CSV = 'wattmeter_dataset.csv'


    # In[5]:


    processed_dir = new_directory + results_dir +"processed/"

    try:
        os.mkdir(processed_dir)
    except Exception as e:
        print(e)
        


    # In[6]:


    #energibridge_df = pd.read_csv(results_dir + ENERGIBRIDGE_SAVE_CSV, )
    #nvidia_df = pd.read_csv(results_dir + NVIDIA_SAVE_CSV, )
    #wattmeter_df = pd.read_csv(results_dir + WATTMETER_SAVE_CSV, )
    #dfs = [energibridge_df,nvidia_df,wattmeter_df]


    # In[7]:


    runall_df = pd.read_csv(results_dir + 'runall_timestamps.csv', )
    load_times = pd.read_csv(results_dir + 'load_times.csv', )


    # In[8]:


    runall_df


    # In[9]:


    runall_df.iloc[0]


    # In[11]:


    #print(load_times)


    # In[12]:


    load_times


    # In[13]:


    # Define a function to transform the dataframe
    def transform_dataframe(df):
        # Convert start_time and end_time columns to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        # Melt the DataFrame to have separate rows for start and end times
        melted_df = df.melt(id_vars=['engine', 'model_name'], value_vars=['start_time', 'end_time'], var_name='start_end')
        
        # Rename the 'value' column to 'time'
        melted_df.rename(columns={'value': 'time'}, inplace=True)
        
        # Convert start_end column to indicate 1 for start_time and 0 for end_time
        melted_df['start_end'] = melted_df['start_end'].apply(lambda x: 1 if x == 'start_time' else 0)
        
        # Sort the DataFrame by time
        melted_df.sort_values(by='time', inplace=True)
        
        # Reset index
        melted_df.reset_index(drop=True, inplace=True)
        
        return melted_df.copy()

    # from two columns to two rows with different time
    # Apply the transformation
    transformed_df = transform_dataframe(load_times)

    # Display the transformed DataFrame
    print(transformed_df)


    # In[14]:


    load_times = transformed_df
    load_times['file'] = 'load_times'
    load_times.head()


    # In[15]:


    runall_df['file'] = 'runall'
    runall_df


    # In[16]:


    load_times.head()


    # In[17]:


    # Concatenate and sort
    #df1 = runall_df
    #df2 = load_times
    # Convert 'timestamp' and 'time' columns to datetime
    runall_df['time'] = pd.to_datetime(runall_df['timestamp'])
    load_times['time'] = pd.to_datetime(load_times['time'])

    general_df = pd.concat([runall_df, load_times,], ignore_index=True)
    general_df = general_df.sort_values(by='time')

    general_df


    # In[18]:


    # Add index as a column for sorting purposes
    general_df['index_col'] = general_df.index

    general_df = general_df.sort_values(by=['time', 'index_col'])
    general_df


    # In[19]:


    # Optionally, you can drop the 'index_col' if not needed afterward
    #df_sorted.drop(columns=['index_col'], inplace=True)


    # In[20]:


    # Drop the redundant 'timestamp' column
    general_df.drop(columns='timestamp', inplace=True)


    # In[21]:


    general_df = general_df[['time', 'runtime', 'model', 'engine', 'model_name', 'start_end', 'file']]

    general_df


    # In[22]:


    # Add new column "label"
    df = general_df.copy()

    # Add new column "label"
    label_values = ['idle'] + [f"{runtime} {model}" for runtime, model in zip(df["runtime"][1:-2], df["model"][1:-2])] + ['name3'] * 2
    df['label'] = label_values

    df


    # In[23]:


    # Reset index
    general_df.reset_index(drop=True, inplace=True)

    general_df


    # In[24]:


    # Create DataFrame
    df = general_df.copy()

    # Iterate through each row
    for idx, row in df.iterrows():
        # If it's the first row, set "idle" as the label
        if idx == 0:
            df.at[idx, 'label'] = 'idle'
        # If it's the last two rows, set "name3" as the label
        elif idx >= len(df) - 1:
            print(idx)
            df.at[idx, 'label'] = 'finish'
        # For other rows, set the label as the concatenation of runtime and model
        else:
            if not isinstance(row['runtime'],str):
                df.at[idx, 'label'] = f"{row['engine']}_{row['model_name']}"
            else:
                df.at[idx, 'label'] = f"{row['runtime']}_{row['model']}"
            #df['label'] = np.where(df['runtime'].isna() | df['model'].isna(), 'nan_value', '')
            #df.at[idx, 'label'] = f"{row['runtime']} {row['model']}"

    df


    # In[25]:


    # Ensure the 'time' column is a datetime type
    df['time'] = pd.to_datetime(df['time'])

    # Calculate the difference in seconds between the current and previous row
    df['duration'] = df['time'].diff().dt.total_seconds()


    # In[26]:


    df


    # In[27]:


    def isNaN(num):
        return num != num

    # Add new column 'label2' based on conditions
    def determine_label(row):
        if row['start_end'] == 0:
            return row['engine'] + "_" + row['model_name']+ "_load"
        if row['file'] == 'runall' and  not isNaN(row['runtime']):
            return row['runtime'] + "_" + row['model']+ "_inference"
        else:
            return None



    # Apply the function to each row
    df['label_time'] = df.apply(determine_label, axis=1)

    df


    # In[28]:


    df.to_csv(processed_dir + 'time_marks.csv', index=False)


    # In[29]:


    print(general_df['time'].iloc[0])
    print(general_df['time'].iloc[-1])


    # In[ ]:




