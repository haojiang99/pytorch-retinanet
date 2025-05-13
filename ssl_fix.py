import ssl
import torch.utils.model_zoo as model_zoo
import torch.hub as hub

# Save the original function
original_load_url = model_zoo.load_url
original_download_url_to_file = hub.download_url_to_file

# Create a new function with SSL verification disabled
def load_url_no_verify(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    # Create an unverified SSL context
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        result = original_load_url(url, model_dir, map_location, progress, check_hash, file_name)
    finally:
        # Restore the original SSL context
        ssl._create_default_https_context = old_context
    return result

# Replace the original function with our new one
model_zoo.load_url = load_url_no_verify

# Also fix the hub download function
def download_url_to_file_no_verify(url, dst, hash_prefix=None, progress=True):
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        result = original_download_url_to_file(url, dst, hash_prefix, progress)
    finally:
        ssl._create_default_https_context = old_context
    return result

hub.download_url_to_file = download_url_to_file_no_verify
