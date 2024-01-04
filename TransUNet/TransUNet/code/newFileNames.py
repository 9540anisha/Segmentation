import os
from zipfile import ZipFile

# Path to the directory containing the npz files
directory_path = '/content/drive/MyDrive/Colab_notebooks/TransUNet/TransUNet/data/Synapse/npz_only'

# Path to the new folder for the zip files
output_folder = '/content/drive/MyDrive/Colab_notebooks/TransUNet/TransUNet/data/Synapse/train_npz'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get a list of all .npz files in the input folder
npz_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.npz')])

# Create zip files with corresponding pairs
for i in range(0, len(npz_files), 2):
    image_file = npz_files[i]
    label_file = npz_files[i + 1]

    # Extract subject ID and session from the file names
    subject_session = image_file.split('_')[1] + '_' + image_file.split('_')[2]

    # Create the zip file name using the original image file name
    zip_file_name = os.path.join(output_folder, image_file.replace('_T1w.npz', '_T1w.zip'))

    # Create a zip file and add the image and label files to it
    with ZipFile(zip_file_name, 'w') as zip_file:
        zip_file.write(os.path.join(directory_path, image_file), 'image.npz')
        zip_file.write(os.path.join(directory_path, label_file), 'label.npz')

print("Zip files created successfully.")
