from huggingface_hub import hf_hub_download

# Define the repository and the specific file you want
repo_id = "pat2echo/fall-detection-posture-classification"
filename = "valid/fall_detection_10.mp4" # The file is inside the 'valid' folder

# Download the file
# local_dir is optional; if you omit it, the file is saved to the HF cache.
# Using local_dir saves it directly into your project folder.
downloaded_file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    local_dir="./videos" # This will save it to a 'videos' folder in your project
)

print(f"File downloaded successfully to: {downloaded_file_path}")