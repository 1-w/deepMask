<h1 align="center">
  <b>Accurate Brain Segmentation in Malformations of Cortical Development</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.7-ff69b4.svg" /></a>
      <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.7-2BAF2B.svg" /></a>
      <a href= "https://github.com/NOEL-MNI/deepMask/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
      <a href="https://doi.org/10.5281/zenodo.4521706">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4521706.svg" alt="DOI"></a>


</p>

PyTorch Implementation using V-net variant of Fully Convolutional Neural Networks

Authors: [Ravnoor Gill](https://github.com/ravnoor), [Benoit Caldairou](https://github.com/bcaldairou), [Neda Bernasconi](https://noel.bic.mni.mcgill.ca/~noel/people/neda-bernasconi/) and [Andrea Bernasconi](https://noel.bic.mni.mcgill.ca/~noel/people/andrea-bernasconi/)

------------------------

![](assets/diagram.png)

Implementation based on:<br>
Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). [V-net: Fully convolutional neural networks for volumetric medical image segmentation](https://arxiv.org/abs/1606.04797). In 2016 Fourth International Conference on 3D vision (3DV) (pp. 565-571). IEEE.


### Please cite:
```TeX
@misc{Gill2021,
  author = {Gill RS, et al},
  title = {Accurate and Reliable Brain Extraction in Cortical Malformations},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NOEL-MNI/deepMask}},
  doi = {10.5281/zenodo.4521716}
}
```

## Pre-requisites
###TODO: Update version requirements
```console
1. Python >= 3.5
2. Pytorch >= 1.0.0
3. Nibabel >= 1.14.0
```

## Installation

```console
conda create -n deepMask python=3.8
conda activate deepMask
pip install -r app/requirements.txt
```


## Usage
### TODO: Training routine
### Inference using Docker
```console
docker run -it -v /tmp:/tmp docker.pkg.github.com/noel-mni/deepmask/app:latest /app/inference.py \
                                            $PATIENT_ID \
                                            /tmp/T1.nii.gz /tmp/FLAIR.nii.gz \
                                            /tmp
```

## License
<a href= "https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
```console
Copyright 2021 Neuroimaging of Epilepsy Laboratory, McGill University
```
