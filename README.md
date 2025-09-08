# MICCAI-FLARE-2025-Task-1

MICCAI FLARE 2025 Task 1: Pan-cancer segmentation in CT scans 

**Official Website** :https://www.codabench.org/competitions/7149/#/pages-tab

# A Lightweight and Effective nnU-Net Framework for Whole-Body Pan-Cancer Segmentation

## Environments and Requirements

First, ensure you have **PyTorch > 2.1.2** installed with CUDA support. We conducted our experiments using **nnUNet v2.6.2**. Set up your environment by running:

```bash
conda create -n FLARE25_hias
conda activate FLARE25_hias
pip install -e .
```

## Dataset

The link to download the data :https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-Pancancer

Organize your labeled data in `nnUNet_raw` in the following structure:

```
Dataset523_FLARE25_Task1/
├── imagesTr/
│   ├── Adrenal_Ki67_Seg_001_0000.nii.gz
│   ├── ...
├── labelsTr/
│   ├── Adrenal_Ki67_Seg_001.nii.gz
│   ├── ...
└── dataset.json
```

## Preprocessing

#### 1.  Extract Fingerprints and Plan the Experiment:

```bash
nnUNetv2_extract_fingerprint -d 523
nnUNetv2_plan_experiment -d 523
```

#### 2. Modify the Plans:

Edit the `plans file` in your `nnUNet_preprocessed` directory. Refer to our [plans.json](https://github.com/zpy2223/MICCAI-FLARE-2025-Task-1/blob/main/nnUNet_results/Dataset523_train_all/nnUNetTrainer_Epoch5000_Lr1e3__nnUNetPlans__3d_fullres_S4D2W32/plans.json) for guidance. We modified the "patch_size" and "spacing" under  "3d_fullres" and create a new configuration "3d_fullres_S4D2W32".

#### 3. Preprocess the Data:

```bash
nnUNetv2_preprocess -d 523 -c 3d_fullres -np 8
```

## Training

Train the network using the following command:

```bash
nnUNetv2_train 523 3d_fullres_S4D2W32 all -tr nnUNetTrainer_Epoch5000_Lr1e3
```

## Inference

To perform inference, run:

```bash
nnUNetv2_predict -i ./inputs -o ./outputs -c 3d_fullres_S4D2W32 -f all -d 523 -tr nnUNetTrainer_Epoch5000_Lr1e3
```

## Evaluation

```bash
nnUNetv2_evaluate_folder GT  your_prediction -djfile dataset.json -pfile  plans.json
```

## Results

| Methods | Public Validation |               | Online Validation |        | Testing |        |
| ------- | ----------------- | ------------- | ----------------- | ------ | ------- | ------ |
|         | DSC(%)            | NSD(%)        | DSC(%)            | NSD(%) | DSC(%)  | NSD(%) |
| Ours    | 53.48 ± 38.06     | 43.84 ± 35.29 | -                 | -      | -       | -      |
