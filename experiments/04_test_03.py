#!/usr/bin/env python
# coding: utf-8

# # Hypothesis testing
# 
# - One factor, more than two treatments
# 
# - factor: config -> EP_runtime
# - treatments: [GPUEP, CPUEP] * [torch, onnx, ov, torchscript]

# ## Set env

# In[35]:


import os
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
import seaborn as sns
#from cliffs_delta import cliffs_delta
from scipy import stats

#from src.data.analysis import test_assumptions, boxplot, barplot, eta_squared, is_pareto_efficient, print_improvement
#from src.environment import FIGURES_DIR, METRICS_DIR
#from src.data.preprocessing import GJOULES_TO_KJOULES, HOURS_TO_MILISECONDS



sns.set_theme(palette="colorblind", color_codes=True)

#get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.autolayout"] = "true"


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools  # for cycling through colors if necessary
import glob
import re
import os

from scipy.stats import shapiro
from scipy.stats import levene


# In[37]:


#num_models = 5 #CHANGE
#device = 'gpu'
#path = f"D:/GAISSA/energy-repo/last_repo/june_{device}/"  # Adjust this path to your CSV files location #CHANGE
initial_save_dir = f"D:/GAISSA/energy-repo/last_repo/results_tests_03/" 
general_dir = f"D:/GAISSA/energy-repo/last_repo/" 

try:
    os.mkdir(initial_save_dir )
except Exception as e:
    print("could not create directory "+ initial_save_dir )
    print(e)

# In[38]:


SAVE_TABLES = True
SAVE_FIGS = True

variable = 'duration'
all_variables =['global_energy',
                "avg_cpu_usage_config",'avg_used_memory_pct_config',
                'avg_utilization_gpu_config','avg_utilization_memory_config','avg_used_memory_pct_mib',
                'avg_Load', 'duration'
                ]

nvidia_variables =[ # change
                'avg_utilization_gpu_config','avg_utilization_memory_config','avg_used_memory_pct_mib',
                ]


for variable in nvidia_variables:
    try:
        print(f"DV: ------------------ {variable} -----------------")

        save_dir = initial_save_dir +  variable +"/"
        try:
            os.mkdir(save_dir )
        except Exception as e:
            print("could not create directory "+ save_dir )
            print(e)

        # In[39]:


        # configs =['torch_pythia1-4b', 'torch_tinyllama',
        #        'torch_codeparrot-small', 'torch_pythia-410m', 'onnx_pythia1-4b',
        #        'onnx_tinyllama', 'onnx_codeparrot-small', 'onnx_pythia-410m',
        #        'torchscript_pythia1-4b', 'torchscript_tinyllama',
        #        'torchscript_codeparrot-small', 'torchscript_pythia-410m']


        # In[40]:





        # In[15]:


        # save_dir = path+"tests/"


        # try:
        #     os.mkdir(save_dir)
        # except Exception as e:
        #     print("could not create directory "+ save_dir )
        #     print(e)


        # In[41]:


        def remove_condition_rows(condition_to_remove,df):
            # example: condition = merged_df['label']  == 'idle_idle'
            df1 = df.copy()
            return df1[~condition_to_remove]


        # In[42]:


        def save_latex_table(df,save_dir):
            df = df.copy()
            df.columns = [col.replace('_', ' ') for col in df.columns]
            
            latex_table = df.to_latex(index=False)

            # Define the filename
            filename = save_dir
            
            # Open the file in write mode
            with open(filename, 'w') as file:
                print(f"saving in {filename}")
                file.write(latex_table) if SAVE_TABLES else print(f"SAVE_TABLES:{SAVE_TABLES}")
                
            # Print LaTeX table
            print(latex_table)


        # ## DV from CPU and GPU

        # ### Energy

        # In[18]:
        global_df = None

        if variable == 'global_energy':
            cpu_global_energy = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_cpu/tables/final_energy_data.csv", index_col=None, header=0)
            gpu_global_energy = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_gpu_test/tables/final_energy_data.csv", index_col=None, header=0)


            # In[19]:


            print(len(cpu_global_energy['label'].unique()))
            print(cpu_global_energy['label'].unique())

            print(len(gpu_global_energy['label'].unique()))
            print(gpu_global_energy['label'].unique())


            # In[20]:


            condition = cpu_global_energy['label'].str.endswith('idle')
            cpu_global_energy = remove_condition_rows( condition, cpu_global_energy)

            condition = gpu_global_energy['label'].str.endswith('idle')
            gpu_global_energy = remove_condition_rows( condition, gpu_global_energy)

            print(len(cpu_global_energy['label'].unique()))
            print(len(gpu_global_energy['label'].unique()))


            # In[21]:


            cpu_global_energy.head()


            # In[22]:


            gpu_global_energy.head()


            # In[23]:


            cpu_global_energy['config'] = "cpuep_"+cpu_global_energy['label']
            cpu_global_energy


            # In[24]:


            gpu_global_energy['config'] = "gpuep_"+gpu_global_energy['label']
            gpu_global_energy


            # In[25]:


            global_df = pd.concat([gpu_global_energy,cpu_global_energy], axis=0, ignore_index=True)
            global_df


        # ### Energibridge: dependent variables

        # In[65]:

        elif variable in ['avg_cpu_usage_config','avg_used_memory_pct_config']:
            cpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_cpu/tables/final_energibridge_data.csv", index_col=None, header=0)
            gpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_gpu_test/tables/final_energibridge_data.csv", index_col=None, header=0)
            print(len(cpu_ep_data))
            print(len(gpu_ep_data))


            # In[66]:


            print(len(cpu_ep_data['label'].unique()))
            print(cpu_ep_data['label'].unique())

            print(len(gpu_ep_data['label'].unique()))
            print(gpu_ep_data['label'].unique())


            # In[67]:


            condition = cpu_ep_data['label'].str.endswith('idle')
            cpu_ep_data = remove_condition_rows( condition, cpu_ep_data)

            condition = gpu_ep_data['label'].str.endswith('idle')
            gpu_ep_data = remove_condition_rows( condition, gpu_ep_data)

            print(len(cpu_ep_data['label'].unique()))
            print(len(gpu_ep_data['label'].unique()))


            # In[68]:


            print(cpu_ep_data.columns)
            print(gpu_ep_data.columns)


            # In[69]:


            cpu_ep_data


            # In[70]:


            dependent_variables = ['avg_cpu_usage_config','avg_used_memory_pct_config']


            # In[71]:


            cpu_ep_data = cpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()
            cpu_ep_data


            # In[72]:


            gpu_ep_data = gpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()
            gpu_ep_data


            # In[73]:


            cpu_ep_data['config'] = "cpuep_"+cpu_ep_data['label']
            cpu_ep_data


            # In[74]:


            gpu_ep_data['config'] = "gpuep_"+gpu_ep_data['label']
            gpu_ep_data


            # In[75]:


            global_df = pd.concat([cpu_ep_data,gpu_ep_data], axis=0, ignore_index=True)
            print(global_df['config'].unique())
            global_df


        # ### nvidia dependent variables

        # In[94]:

        elif variable in ['avg_utilization_gpu_config','avg_utilization_memory_config','avg_used_memory_pct_mib','avg_power_draw_config']:

            cpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_cpu/tables/final_nvidia_data.csv", index_col=None, header=0)
            gpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_gpu_test/tables/final_nvidia_data.csv", index_col=None, header=0)
            print(len(cpu_ep_data))
            print(len(gpu_ep_data))


            # In[95]:


            print(len(cpu_ep_data['label'].unique()))
            print(cpu_ep_data['label'].unique())

            print(len(gpu_ep_data['label'].unique()))
            print(gpu_ep_data['label'].unique())


            # In[96]:


            condition = cpu_ep_data['label'].str.endswith('idle')
            cpu_ep_data = remove_condition_rows( condition, cpu_ep_data)

            condition = gpu_ep_data['label'].str.endswith('idle')
            gpu_ep_data = remove_condition_rows( condition, gpu_ep_data)

            print(len(cpu_ep_data['label'].unique()))
            print(len(gpu_ep_data['label'].unique()))


            # In[97]:


            print(cpu_ep_data.columns)
            print(gpu_ep_data.columns)


            # In[98]:


            dependent_variables = ['avg_utilization_gpu_config','avg_utilization_memory_config','avg_used_memory_pct_mib','avg_power_draw_config']


            # In[99]:


            cpu_ep_data = cpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()
            gpu_ep_data = gpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()

            cpu_ep_data


            # In[100]:


            cpu_ep_data['config'] = "cpuep_"+cpu_ep_data['label']
            gpu_ep_data['config'] = "gpuep_"+gpu_ep_data['label']

            gpu_ep_data


            # In[101]:


            #global_df = pd.concat([cpu_ep_data,gpu_ep_data], axis=0, ignore_index=True)
            global_df = gpu_ep_data
            
            print(global_df['config'].unique())
            global_df


        # ### wattmetter dependent variables

        # In[120]:

        elif variable in ['avg_Load'] : 
            cpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_cpu/tables/final_wattmeter_data.csv", index_col=None, header=0)
            gpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_gpu_test/tables/final_wattmeter_data.csv", index_col=None, header=0)
            print(len(cpu_ep_data))
            print(len(gpu_ep_data))


            # In[121]:


            print(len(cpu_ep_data['label'].unique()))
            print(cpu_ep_data['label'].unique())

            print(len(gpu_ep_data['label'].unique()))
            print(gpu_ep_data['label'].unique())


            # In[122]:


            condition = cpu_ep_data['label'].str.endswith('idle')
            cpu_ep_data = remove_condition_rows( condition, cpu_ep_data)

            condition = gpu_ep_data['label'].str.endswith('idle')
            gpu_ep_data = remove_condition_rows( condition, gpu_ep_data)

            print(len(cpu_ep_data['label'].unique()))
            print(len(gpu_ep_data['label'].unique()))


            # In[123]:


            print(cpu_ep_data.columns)
            print(gpu_ep_data.columns)


            # In[124]:


            dependent_variables = ['avg_Load']


            # In[125]:


            cpu_ep_data = cpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()
            gpu_ep_data = gpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()

            cpu_ep_data


            # In[126]:


            cpu_ep_data['config'] = "cpuep_"+cpu_ep_data['label']
            gpu_ep_data['config'] = "gpuep_"+gpu_ep_data['label']

            gpu_ep_data


            # In[127]:


            global_df = pd.concat([cpu_ep_data,gpu_ep_data], axis=0, ignore_index=True)
            print(global_df['config'].unique())
            global_df

        elif variable in ['duration']:
            cpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_cpu/tables/final_inference_time_data.csv", index_col=None, header=0)
            gpu_ep_data = pd.read_csv("D:/GAISSA/energy-repo/last_repo/june_gpu_test/tables/final_inference_time_data.csv", index_col=None, header=0)
            print(len(cpu_ep_data))
            print(len(gpu_ep_data))

            print(len(cpu_ep_data['label'].unique()))
            print(cpu_ep_data['label'].unique())

            print(len(gpu_ep_data['label'].unique()))
            print(gpu_ep_data['label'].unique())

            condition = cpu_ep_data['label'].str.endswith('idle')
            cpu_ep_data = remove_condition_rows( condition, cpu_ep_data)

            condition = gpu_ep_data['label'].str.endswith('idle')
            gpu_ep_data = remove_condition_rows( condition, gpu_ep_data)

            print(len(cpu_ep_data['label'].unique()))
            print(len(gpu_ep_data['label'].unique()))

            print(cpu_ep_data.columns)
            print(gpu_ep_data.columns)

            dependent_variables = ['duration']

            cpu_ep_data = cpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()
            gpu_ep_data = gpu_ep_data.groupby(['label', 'experiment'])[dependent_variables].mean().reset_index()

            cpu_ep_data

            cpu_ep_data['config'] = "cpuep_"+cpu_ep_data['label']
            gpu_ep_data['config'] = "gpuep_"+gpu_ep_data['label']

            gpu_ep_data

            global_df = pd.concat([cpu_ep_data,gpu_ep_data], axis=0, ignore_index=True)
            print(global_df['config'].unique())
            global_df


        # ### ?only one

        # ## Normality test
        # 
        # ; (2) assess if the
        # measurements are normally distributed and have equal variances
        # across the different treatments of each RQ. Utilize the Shapiro-Wtest to check for the normality of the data. To check the homogeneity
        # of variances, we use a Levene test for equality of variances, followed
        # by a box plot to make a visual assessmen

        # In[43]:


        # input: df with DV, aggregated ?
        global_df

        # In[75]:

        # [CHANGE] inputs
        models = ['codeparrot-small','pythia-410m', 'tinyllama', 'pythia1-4b',  'phi2']
        for model in models:
            print(f"Model: ------------------ {model} -----------------")
            
        #model = models[1]  # codeparrot-small,pythia-410m, tinyllama, pythia1-4b,  phi2
            dependent_variable = variable
            independent_variable = 'config'

            df = global_df
            df = df[df['label'].str.contains(model)]
            configs = df['config'].unique().tolist()


            # In[76]:


            configs


            # In[77]:


            # Initialize an empty DataFrame to collect test results
            results_df = pd.DataFrame(columns=['config', 'W-statistic', 'p-value'])

            # Iterate over each configuration
            for c in configs:
                # Filter data for current configuration
                subset = df[df[independent_variable].str.startswith(c, na=False)]
                
                # Perform the Shapiro-Wilk test
                w_statistic, p_value = shapiro(subset[dependent_variable])
                
                # Prepare a dictionary to add as a new row
                new_row = pd.DataFrame({'config': [c], 'W-statistic': [w_statistic], 'p-value': [p_value]})
                
                # Concatenate the new row to the results DataFrame
                results_df = pd.concat([results_df, new_row], ignore_index=True)

            print("Data normality: Shapiro-Wilk test")
            results_df['significance'] = results_df['p-value']<0.05
            #print(results_df)
            # Display the results DataFrame
            save_latex_table(results_df,save_dir+f"normality_{model}_{dependent_variable}.tex")
            results_df


            # In[78]:


            save_dir


            # ### Levene: Equal variances?

            # In[79]:


            configs


            # In[80]:


            df_aggregated = df
            sets_levene = []
            for c in configs:
                sets_levene.append(df_aggregated[df_aggregated[independent_variable].str.startswith(c, na=False)][dependent_variable])


            # In[81]:


            #variable = 'energy'
            results_df = pd.DataFrame(columns=['W-statistic', 'p-value'])
            w_statistic, p_value = levene(*sets_levene)
                
            # Prepare a dictionary to add as a new row
            new_row = pd.DataFrame({'W-statistic': [w_statistic], 'p-value': [p_value]})

            # Concatenate the new row to the results DataFrame
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            save_latex_table(results_df,save_dir + f'levene_{model}_{dependent_variable}.tex')
            # Display the results DataFrame
            print(results_df)



            # In[82]:


            if(results_df['W-statistic'][0] > results_df['p-value'][0] ):
                print("Equal variances? NO, reject H0, variances are not equal")
            else:
                print("Equal variances? We cannot reject H0, variances might be equal")


            # In[ ]:





            # ## Hypothesis testing
            # 
            # . (3) assess the statistical
            # significance (i.e., p-value) of the finding
            # 
            # we perform a statistical comparison between the energy consumption and GPU usage between the training environments. We use the Kruskal-Wallis test for this comparison since the data is not normally distributed. Again, we use Dunn’s test as the post-hoc test

            # In[83]:


            df


            result_welch_anova = pg.welch_anova(df, dv=dependent_variable, between=independent_variable)
            F, p_value = result_welch_anova['F'], result_welch_anova['p-unc']
            print(F[0])
            print(p_value[0])

            results_welch = pd.DataFrame({
                'Statistic': [F[0]],
                'P-value': [p_value[0]]
            })

            
            save_latex_table(results_welch,save_dir+f"welch_{model}_{dependent_variable}.tex")
            
            results_welch

            # In[84]:


            result_rq = pg.kruskal(df, dv=dependent_variable, between=independent_variable)
            result_rq


            # In[85]:


            def my_kruskal_wallis(*groups, alpha=0.05):
                # Flatten the group lists and create a grouping array
                data = np.concatenate(groups)
                group_arr = np.concatenate([[i + 1] * len(group) for i, group in enumerate(groups)])  # +1 to ensure group labels are not zero
                
                # Perform the Kruskal-Wallis test
                stat, p_value = stats.kruskal(*groups)
                print(f"Kruskal-Wallis H statistic: {stat}")
                print(f"P-value: {p_value}")
                
                if p_value < alpha:
                    print("Reject the null hypothesis: significant differences between the groups.")
                    print("Proceeding with Dunn's post-hoc test...")

                    return stat, p_value
                else:
                    print("Fail to reject the null hypothesis: no significant differences between the groups.")


            # In[86]:


            def dunn_post_test(*groups, alpha=0.05):
                # Flatten the group lists and create a grouping array
                data = np.concatenate(groups)
                group_arr = np.concatenate([[i + 1] * len(group) for i, group in enumerate(groups)])  # +1 to ensure group labels are not zero
                
                # Perform the Kruskal-Wallis test
                stat, p_value = stats.kruskal(*groups)
                print(f"Kruskal-Wallis H statistic: {stat}")
                print(f"P-value: {p_value}")
                
                if p_value < alpha:
                    print("Reject the null hypothesis: significant differences between the groups.")
                    print("Proceeding with Dunn's post-hoc test...")
                    
                    # Prepare DataFrame for Dunn's test
                    df = pd.DataFrame({'values': data, 'groups': group_arr})
                    
                    # Perform Dunn's test using scikit-posthocs
                    dunn_results = sp.posthoc_dunn(df, val_col='values', group_col='groups', p_adjust='bonferroni')
                    
                    print("Dunn's post-hoc test pairwise p-values:")
                    print(dunn_results)
                    print(dunn_results < 0.05)
                    return dunn_results < 0.05
                else:
                    print("Fail to reject the null hypothesis: no significant differences between the groups.")
                    return None


            # In[87]:


            def kruskal_wallis_with_dunn(*groups, alpha=0.05):
                # Flatten the group lists and create a grouping array
                data = np.concatenate(groups)
                group_arr = np.concatenate([[i + 1] * len(group) for i, group in enumerate(groups)])  # +1 to ensure group labels are not zero
                
                # Perform the Kruskal-Wallis test
                stat, p_value = stats.kruskal(*groups)
                print(f"Kruskal-Wallis H statistic: {stat}")
                print(f"P-value: {p_value}")
                
                if p_value < alpha:
                    print("Reject the null hypothesis: significant differences between the groups.")
                    print("Proceeding with Dunn's post-hoc test...")
                    
                    # Prepare DataFrame for Dunn's test
                    df = pd.DataFrame({'values': data, 'groups': group_arr})
                    
                    # Perform Dunn's test using scikit-posthocs
                    dunn_results = sp.posthoc_dunn(df, val_col='values', group_col='groups', p_adjust='bonferroni')
                    
                    print("Dunn's post-hoc test pairwise p-values:")
                    print(dunn_results)
                    print(dunn_results < 0.05)
                    
                else:
                    print("Fail to reject the null hypothesis: no significant differences between the groups.")


            # In[88]:


            stat, p_value = my_kruskal_wallis(*sets_levene)

            results_kruskal = pd.DataFrame({
                'Statistic': [stat],
                'P-value': [p_value]
            })

            save_latex_table(results_kruskal,save_dir+f"kruskal_{model}_{dependent_variable}.tex")
            results_kruskal


            # In[ ]:





            # In[89]:


            #kruskal_wallis_with_dunn(*sets_levene)
            """
            The function returns the Kruskal-Wallis H statistic and the p-value:

                H statistic: A larger statistic or value suggests more evidence against the null hypothesis.
                P-value: If the p-value is less than the significance level (commonly set at 0.05), you reject the null hypothesis, indicating that there is a statistically significant difference between the groups.
            """


            # ## Post-hoc test: Dunn's test
            # ![image.png](attachment:74aecc7d-61de-49a2-be27-cbbeb8988acb.png)
            # 

            # In[90]:


            p_values = sp.posthoc_dunn(df, val_col=dependent_variable, group_col=independent_variable, p_adjust="holm")

            # Check which assumptions reject the null hypothesis.
            dunn_results = p_values < 0.05
            dunn_results


            # In[91]:


            #dunn_results.to_csv()


            # In[92]:


            # Check which comparisons reject the null hypothesis
            significant_comparisons = p_values < 0.05

            # Create a DataFrame to display results with 'X' for significant comparisons
            result_display = significant_comparisons.applymap(lambda x: 'X' if x else '')

            print("Significance Grid (X indicates a significant difference):")
            result_display.to_csv(save_dir+f"dunn_{model}_{dependent_variable}.csv") if SAVE_TABLES else print(f"SAVE_TABLES:{SAVE_TABLES}")
            result_display


            
            # ## Howell

            # Performing the Games-Howell test
            results = pg.pairwise_gameshowell(data=df, dv=dependent_variable, between=independent_variable)


            # Extract the p-values and reshape into a matrix form
            p_matrix = results.pivot(index='A', columns='B', values='pval')
            print(p_matrix)

            # Visualizing the matrix
            plt.figure(figsize=(8,6))
            sns.heatmap(p_matrix, annot=True, cmap='coolwarm', )
            plt.title('Games-Howell Post-Hoc Test P-Values')
            plt.savefig(save_dir+f"gameshowell_{model}_{dependent_variable}.png") if SAVE_FIGS else print(f"SAVE_FIGS:{SAVE_FIGS}")
            #plt.show()

            significant_comparisons = p_matrix < 0.05

            # Create a DataFrame to display results with 'X' for significant comparisons
            result_display = significant_comparisons.applymap(lambda x: 'X' if x else '')

            print("Significance Grid (X indicates a significant difference):")
            result_display.to_csv(save_dir+f"gameshowell_{model}_{dependent_variable}.csv") if SAVE_TABLES else print(f"SAVE_TABLES:{SAVE_TABLES}")
            result_display



            # In[ ]:

    except Exception as e:
        print(f"Error: {variable} {model}: {e}")



