# DeCoLearn: Deformation-Compensated Learning (IEEE Transactions on Medical Imaging, 2022)

This is the official repository of [Deformation-Compensated Learning for Image Reconstruction without Ground Truth](https://ieeexplore.ieee.org/document/9743932).

![](./file/cover_img.gif)

## DeCoLearn Extension Work by Chicago Park

| Research Report | Research Presentation |
|:-----------:|:------------------:|
|    [Final_Report.pdf](https://github.com/ChicagoPark/DeCoLearn/files/10252730/Final_Report.pdf)    |       [presentation.pptx](https://github.com/ChicagoPark/DeCoLearn/files/10252731/presentation.pptx)        |





## Abstract
Deep neural networks for medical image reconstruction are traditionally trained using high-quality ground-truth images as training targets. Recent work on Noise2Noise (N2N) has shown the potential of using multiple noisy measurements of the same object as an alternative to having a ground-truth. However, existing N2N-based methods are not suitable for learning from the measurements of an object undergoing nonrigid deformation. This paper addresses this issue by proposing the deformation-compensated learning (DeCoLearn) method for training deep reconstruction networks by compensating for object deformations. A key component of DeCoLearn is a deep registration module, which is jointly trained with the deep reconstruction network without any ground-truth supervision. We validate DeCoLearn on both simulated and experimentally collected magnetic resonance imaging (MRI) data and show that it significantly improves imaging quality.

## Code

### Download Datasets

- Download the brain mri dataset [here](https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view) and put it into the `decolearn/dataset` folder.

### Detail Information about Dataset

- Used dataset is the parallel imaging dataset given and used by MoDL: Model Based Deep Learning Architecture for Inverse Problems.

- This dataset consist of parallel magnetic resonance imaging (MRI) brain data of five human subjects. Four of which are used during training of the model and fifth subject is used during testing. Above link contain fully sampled preprocessed data in numpy format for both training and testing. We also provide coil-sensitity-maps (CSM) pre-computed using E-SPIRIT algorithm. Total file size is 3 GB and contains following arrays:

Data  	      |      Explanation
:---------------: | :-------------:
`trnOrg`  | This is complex arrary of 256x232x360 containing 90 slices from each of the 4 training subjects. Each slice is of spatial dimension 256x232. This is the original fully sampled data.
`trnCSM`  | This is a complex array of 256x232x12x360 representing coil sensitity maps (csm). Here 12 represent number of coils.
`trnMask`  | This is the random undersampling mask to do 6-fold acceleration. We use different mask for different slices.
`tstOrg`  | Similar arrays for testing purpose. (164 testing images)
`tstCSM`  | Similar arrays for testing purpose. (164 testing images)
`tstMask`  | Similar arrays for testing purpose. (164 testing images)

----
```diff
+ Code, in this repository, provides the program to process the dataset.hdf5 to generate datasets for single-coil and multi-coil datasets. In addition, non-linearly deformed datasets are also created.

- Original Dataset Explanation: https://github.com/hkaggarwal/modl
```
----

### Additionally Created Dataset 1: `alignment H5 file`
Data  	      |      Explanation
:---------------: | :-------------:
`moved_x`  | non-linearly deformed ground truth

### Additionally Created Dataset 2: `MRI H5 file`
Data  	      |      Explanation
:---------------: | :-------------:
`fixed_y`  | undersampled measurement of fixed data(`[256, 232, 2]`)
`fixed_mask`  | fixed undersampling mask (`[256, 232, 2]`)
`fixed_y_tran`  | fixed input MRI image (`[1, 2, 256, 232]`)
`moved_y`  | undersampled measurement of moved data(`[256, 232, 2]`)
`moved_mask`  | moved undersampling mask (`[256, 232, 2]`)
`moved_y_tran`  | moved input MRI image (`[1, 2, 256, 232]`)
`mul_fixed_y`  | undersampled measurement of multi-coil fixed data(`[12, 256, 232, 2]`)
`mul_moved_y`  | undersampled measurement of multi-coil moved data(`[12, 256, 232, 2]`)
`mul_fixed_y_tran`  | multi-coil fixed input MRI image (`[1, 2, 256, 232]`)
`mul_moved_y_tran`  | multi-coil moved input MRI image (`[1, 2, 256, 232]`)



### Setup Environment
Setup the environment
```
conda env create -n decolearn_env --file decolearn.yml
```
To activate this environment, use
```
conda activate decolearn_env
```
To deactivate an active environment, use
```
conda deactivate
```

### Run
Enter the decolearn folder
```
cd ./decolearn
```

Use the following command to run DeCoLearn
```
python main.py --gpu_index=0 --is_optimize_regis=true
```


Use the following command to run A2A (Unregistered)
```
python main.py --gpu_index=0 --is_optimize_regis=false
```

The training and testing will be conducted sequentially.

Please specify the GPU index (i.e., --gpu_index) based on your resources. No multi-gpu support so far.

### Results
Outputs can be found in `decolearn/experimental_results/`

#### Visual Examples
![](./file/results.gif)

![](./file/results.png)

#### Quantitative Evaluation

|      | Zero-Filled | A2A (Unregistered) | DeCoLearn |
|:----:|:-----------:|:------------------:|:---------:|
| PSNR |    27.85    |       26.48        | **31.87** |
| SSIM |    0.694    |       0.708        | **0.861** |

## Citation

```
@article{gan2021deformation,
  title={Deformation-Compensated Learning for Image Reconstruction without Ground Truth},
  author={Gan, Weijie and Sun, Yu and Eldeniz, Cihat and Liu, Jiaming and An, Hongyu and Kamilov, Ulugbek S},
  journal={IEEE Transactions on Medical Imaging},
  year={2022}
}
```

It is worth mentioning that the brain mri data used in this repo is provided by [here](https://github.com/hkaggarwal/modl). Please consider also cite their paper.

## Supplementary Materials

In this repo, we provide [a supplementary document](./file/supplemental_documents.pdf) showing (a) an illustration of simulated sampling masks, (b) validation with additional levels of deformation, (c) validation with additional sub-sampling rates, (d) an illustration of the influence of the trade-off parameter in Equ. (10) of the paper, and (e) validation on MRI measurements simulated using complex-value ground-truth images.

