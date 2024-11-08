from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
import pandas as pd
import os

max_workers = os.cpu_count()

api_token = os.getenv("API_KEY")
headers = {"authorization": f"Bearer {api_token}"}

# Initialize API
api = HfApi()

# Function to fetch model info
def fetch_model_info(model):
    model_id = getattr(model, "modelId", "Unknown")
    try:
        model_info = api.model_info(model_id,token=api_token)
        if model_info.config is None:
            return None

        # Collect model metadata
        model_metadata = {
            "model": model_id,
            "id": model_id,
            "created_at": getattr(model, "created_at", "Unknown"),
            "downloads_all_time": getattr(model_info, "downloads_all_time", "Unknown"),
            "downloads": getattr(model_info, "downloads", "Unknown"),
            "safetensors": getattr(model_info, "safetensors", "Unknown"),
            "task": getattr(model_info, "pipeline_tag", "Unknown"),
            "library_name": getattr(model, "library_name", "Unknown"),
            "transformers_info": getattr(model_info, "transformers_info", "Unknown"),
            "architecture": model_info.config.get("architectures", "Unknown"),
            "tags": getattr(model, "tags", "Unknown"),
        }
        return model_metadata
    except Exception as e:
        print(f"Error fetching model info for {model_id}: {e}")
        return None

def get_decoder_models_parallel(models):
    decoder_models = []
    # max *5 but 5 is too many requests
    with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust the number of workers as needed
        results = executor.map(fetch_model_info, models)
        for result in results:
            if result is not None:
                decoder_models.append(result)
    return decoder_models

def main():
    limit=None
    
    
    models = api.list_models(task="text-generation", pipeline_tag="text-generation", limit=limit, token=api_token)
    decoder_models = get_decoder_models_parallel(models)

    # Check for inconsistencies (optional)
    for index, item in enumerate(decoder_models):
        if not isinstance(item, dict):
            print(f"Item at index {index} is not a dictionary: {item}")
        else:
            for key, value in item.items():
                if value is None:
                    print(f"Key '{key}' in item at index {index} has a value of None.")

    # Attempt to create the DataFrame
    try:
        df = pd.DataFrame(decoder_models)
        print("DataFrame created successfully")
        df.to_csv("decoder_models_info_01.csv", index=False)
        print(f"CSV file saved: decoder_models_info_01.csv")
        print(f"Total models: {len(df)}")
    except Exception as e:
        print(f"Error creating DataFrame: {e}")

if __name__ == "__main__":
    main()
