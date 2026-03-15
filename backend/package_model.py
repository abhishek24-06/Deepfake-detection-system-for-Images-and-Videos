import zipfile
import os
import torch

folder = r"c:\Users\vaishnavi\OneDrive\Desktop\Miniproj-2B\deepfake_full_model"
dest = r"c:\Users\vaishnavi\OneDrive\Desktop\Miniproj-2B\backend\deepfake_model.pth"

# PyTorch zip format requires all files to be in a single top-level directory inside the zip.
# We will invoke the root directory "archive".
with zipfile.ZipFile(dest, 'w', zipfile.ZIP_STORED) as zf:
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, folder)
            archive_path = os.path.join("archive", rel_path)
            # Make sure to format to POSIX paths
            archive_path = archive_path.replace("\\", "/")
            zf.write(file_path, archive_path)

print(f"Packaged {folder} into {dest}")

try:
    weights = torch.load(dest, map_location='cpu', weights_only=True)
    print("Successfully loaded model weights with PyTorch!")
except Exception as e:
    print(f"Failed to load: {e}")
