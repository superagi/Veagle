from huggingface_hub import snapshot_download
import json

# Specify the file path
file_path = "models/config.json"

# Open the file and load the JSON data
with open(file_path, "r") as file:
    models_config_data = json.load(file)


def download_models():
    # Snapshot will check if the model is available, if not it will download the model.
    for model_type in models_config_data:
        if models_config_data[model_type]['download_in_repo']==True:
            snapshot_download(repo_id = models_config_data[model_type]['hf_path'], repo_type = "model", local_dir = "models")
        else:
            snapshot_download(repo_id = models_config_data[model_type]['hf_path'], repo_type = "model")
            
if __name__ == "__main__":
    download_models()