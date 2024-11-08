"""
nohup python extraction_sml_01.py > extraction_models_01.log 2>&1 &

nohup python extraction_sml_01.py > extraction_models_02.log 2>&1 &
"""

from huggingface_hub import HfApi
import pandas as pd


# Initialize API
api = HfApi()

# Fetch models metadata
limit = None # Change
csv_file_name = "decoder_models_info_02.csv"

#models = api.list_models(task="text-generation", sort="downloads", limit=500)

def get_decoder_models(models):
        # Filter for decoder-only architectures
    decoder_models = []
    i = 0
    num_models = 170000

    for model in models:
        print(f"{i}-->")
        if i > num_models:
            print("breaking")
            break

        model_id = getattr(model, "modelId", "Unknown")
        print(f'model_id: {model_id}')

        # Fetch detailed model information
        model_info = api.model_info(model.modelId)
        #print(f'config: {model_info.config}')
        #print(f'd1: {model_info.downloads_all_time}')
        #print(f'd2: {model_info.downloads}')

        if model_info.config ==None:
            continue
        
        if model_info.card_data == None:
            continue
        
        card_data = getattr(model_info, "card_data", "Unknown")
        # print("________________")
        # print(type(card_data))
        # print(card_data.__dict__)   
        # print(dir(card_data))        
             
        # print("---",getattr(card_data, "datasets", "Unknown"))
        # print("---",getattr(card_data, "eval_results", "Unknown"))
        # print("---",getattr(card_data, "metrics", "Unknown"))
        
        
        # Collect model metadata with the desired column names
        model_metadata = {
            #"model": model.modelId,  # Model name or ID
            "model": getattr(model, "modelId", "Unknown"),

            "id": getattr(model, "modelId", "Unknown"),
            "created_at": getattr(model, "created_at", "Unknown"),
            "downloads_all_time": getattr(model_info, "downloads_all_time", "Unknown"),
            "downloads": getattr(model_info, "downloads", "Unknown"),
            "likes": getattr(model_info, "likes", "Unknown"),
            "safetensors": getattr(model_info, "safetensors", "Unknown"),
            "task": getattr(model_info, "pipeline_tag", "Unknown"),
            "datasets": getattr(card_data, "datasets", "Unknown"),
            "eval_results": getattr(card_data, "eval_results", "Unknown"),
            "metrics": getattr(card_data, "metrics", "Unknown"),
            
            "library_name": getattr(model, "library_name", "Unknown"),
            "transformers_info": getattr(model_info, "transformers_info", "Unknown"),
            "architecture": model_info.config.get("architectures", "Unknown"),
            "tags": getattr(model, "tags", "Unknown"),
            "card_data": card_data,
            
        }
        #print(f'model_metadata: {model_metadata}')
        print("testing:")
        print(type(model_metadata['card_data']))
        decoder_models.append(model_metadata)
        i += 1

    
    return decoder_models

def check_df_consistency(decoder_models):
    #just for log reasons
    #df_consistency = None

    # Check for inconsistencies
    for index, item in enumerate(decoder_models):
        if not isinstance(item, dict):
            print(f"Item at index {index} is not a dictionary: {item}")
            #df_consistency = False
            #return df_consistency
        else:
            # Check for problematic keys or data
            for key, value in item.items():
                if value is None:
                    print(f"Key '{key}' in item at index {index} has a value of None.")
            
    
    




def main():

    

    models = api.list_models(task="text-generation", pipeline_tag="text-generation", limit=limit)

    decoder_models = get_decoder_models(models=models)

    check_df_consistency(decoder_models)

    # Attempt to create the DataFrame in a simplified way
    try:
        df = pd.DataFrame(decoder_models)
        print("DataFrame created successfully")
        #print(df.head())  # Display only the first few rows to avoid rendering issues
        df.to_csv(csv_file_name, index=False)
        print(f"CSV file saved: {csv_file_name}.csv")
        print(f"sum_total_models: {len(df)}")


    except Exception as e:
        print(f"Error creating DataFrame: {e}")

if __name__ == "__main__":
    main()
