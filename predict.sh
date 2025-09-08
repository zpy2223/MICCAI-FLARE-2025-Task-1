export nnUNet_results='./nnUNet_results'
nnUNetv2_predict -i ./inputs -o ./outputs -c 3d_fullres_S4D2W32 -f all -d 523 -tr nnUNetTrainer_Epoch5000_Lr1e3