# Segmentation
NOTE: Code is taken from the following repos to be run and tested using the ATLAS dataset. Alterations have been made to the code to work with the different filetype supported by ATLAS and to gauge the accuracy/dice/precision of the models by testing various parameters.

Segmentation for Atlas Data
- Dataset: https://www.icpsr.umich.edu/web/ADDEP/studies/36684
Cited: https://github.com/Beckschen/TransUNet
Cited: https://github.com/Beckschen/3D-TransUNet
Cited: https://github.com/BMIRDS/3dMRISegmentation

Introduction

Stroke, characterized by the disruption of blood flow to the brain, is a leading cause of disability and mortality globally. Swift and precise diagnosis is imperative for effective treatment. 
Automated segmentation, with the promise of increasing efficiency of diagnosis, has emerged as a vital area of research. Stroke lesions are areas of damaged brain tissue resulting from insufficient blood supply and identifying these is a tedious and repetitive task that is usually done by hand by radiologists. (Now talk about what automated segmentation is).In the realm of medical imaging, particularly 3D brain scans, automated segmentation has emerged as a vital area of research. 

The integration of Machine Learning (ML) techniques has revolutionized medical image analysis, offering the potential for automated and precise segmentation. This engineering project delves into this intersection of medical imaging, stroke pathology, and machine learning, with a specific focus on comparing two state-of-the-art architectures: 3D-UNet and TransUNet.

The classic UNet architecture, known by  its U-shaped structure, utilizes an encoder-decoder format with skip connections resulting in precise localization of structures and is commonly used in various medical imaging tasks.  We utilize a UNet adapted for 3D data.

Concurrently, the introduction of transformers has revolutionized the field of medical imaging but
was originally designed for natural language processing (like in ChatGPT). Transformers
capture the long-range dependencies and contextual information across an input to “decide”
where to place its attention. In the TransUNet architecture this is done through self-attention
mechanisms. The TransUNet leverages the strengths of CNNs in feature extraction and then
integrates transformer encoders to complete a more effective segmentation.

The goal of this project is to enhance the accuracy of these architectures and compare their
performance in segmentation of stroke lesions within the 3D brain scans.

Dataset:

To train and test the models the Anatomical Tracings of Lesions after Stroke (ATLAS) dataset was used. This is a publicly available dataset with a total of 955 T1-weighted MRIs (Magnetic Resonance Imaging) with manually segmented diverse lesions and metadata of patient brains.



