import os
import h5py
import nibabel as nib
import numpy as np
import argparse

# Define a recursive function to process subdirectories
def process_subdirectories(input_dir, npz_output_root, h5_output_root):
    for item in os.listdir(input_dir):
        print(item)
        item_path = os.path.join(input_dir, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Recursively process subdirectories
            process_subdirectories(item_path, npz_output_root, h5_output_root)
        elif item.endswith('.nii'):
            # Load the NIfTI file
            nifti_image = nib.load(item_path)
            # Get the file name without extension
            file_name_without_extension = os.path.splitext(item)[0]

            # Extract the "anat" part of the path
            print(item_path)
            anat_part = None
            parts = item_path.split(os.sep)
            for i, part in enumerate(parts):
                if part == "anat":
                    anat_part = os.path.join(*parts[i:])

            if anat_part:
                # Create an output directory for this "anat" subdirectory
                anat_output_dir_npz = os.path.join(npz_output_root, os.path.dirname(anat_part))
                anat_output_dir_h5 = os.path.join(h5_output_root, os.path.dirname(anat_part))
                os.makedirs(anat_output_dir_npz, exist_ok=True)
                os.makedirs(anat_output_dir_h5, exist_ok=True)

                # Convert the slices to NumPy format, clip, and normalize
                axial_slices = []
                for axial_slice_number in range(nifti_image.shape[2]):
                    axial_slice = nifti_image.get_fdata()[:, :, axial_slice_number]
                    # Clip pixel values within [-125, 275]
                    axial_slice = np.clip(axial_slice, -125, 275)
                    # Normalize to [0, 1]
                    axial_slice = (axial_slice - (-125)) / (275 - (-125))
                    axial_slices.append(axial_slice)

                # Convert the list of axial slices to a NumPy array
                axial_slices_array = np.stack(axial_slices, axis=2)

                # Save the NumPy array as a .npz file for training cases
                npz_file_path = os.path.join(anat_output_dir_npz, f'{file_name_without_extension}.npz')
                np.savez(npz_file_path, axial_slices=axial_slices_array)

                # Create an H5 file in the output directory for testing cases
                h5_file_path = os.path.join(anat_output_dir_h5, f'{file_name_without_extension}.npy.h5')
                with h5py.File(h5_file_path, 'w') as h5_file:
                    # Store the 3D volume data in the H5 file as a dataset named 'volume_data'
                    h5_file.create_dataset('volume_data', data=nifti_image.get_fdata())

# Define command-line arguments
parser = argparse.ArgumentParser(description='NIfTI to NumPy and H5 Converter')
parser.add_argument('--input_dir', type=str, required=True,
                    help='Input directory containing NIfTI files')
parser.add_argument('--npz_output_dir', type=str, required=True,
                    help='Root output directory for .npz files')
parser.add_argument('--h5_output_dir', type=str, required=True,
                    help='Root output directory for .npy.h5 files')
args = parser.parse_args()

# Ensure the output root directories exist
os.makedirs(args.npz_output_dir, exist_ok=True)
os.makedirs(args.h5_output_dir, exist_ok=True)

# Call the recursive function to process subdirectories
process_subdirectories(args.input_dir, args.npz_output_dir, args.h5_output_dir)