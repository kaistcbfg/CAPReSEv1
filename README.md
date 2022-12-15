# CAPReSE version 1

## Introduction
The regulatory effect of non-coding large-scale structural variations (SVs) on proto-oncogene activation remains unclear. In this study, we investigated SV-mediated gene dysregulation by profiling 3D cancer genome maps from 40 colorectal cancer (CRC) patients. To systematically identify such de novo chromatin contacts, we developed a new machine-learning method named ‘Chromatin Anomaly Pattern Recognition and Size Estimation,’ (CAPReSE) comprising a deep neural network (DNN)-based feature extractor combined with an XGBoost classifier. CAPReSE utilized a unique chromatin contact signature that shows enriched contact frequencies at the break-ends of SVs and a gradual decrease in contact frequencies along the rearranged genomic regions. The input tumor Hi-C contact map was normalized against a pan-normal Hi-C contact map, and a series of image processing algorithms were applied to identify the unique chromatin contact signatures. The SVs supported by both WGS and Hi-C data were used as a ground truth set for the final classifier.

*Note: current version is optimized for the 40 CRC patient cohort used in the citation paper. Upgraded version avaialble for general cases will be released as version 2.*

## Publication and Citation
Under review in *Cell Reports*.

## License
Copyright (c) YEAR: 2020 COPYRIGHT HOLDER: KAIST (Corporation registration number: 114471-0000668).
Will be registered at the Korea Copyright Commission in accordance with Article 53 of the Copyright Act. 
Developed by Kyukwang Kim & Inkyung Jung, KAIST Dept. of Biological Sciences.
For commercial use of the software, please contact the authors.

## Sample Data and Input Format
Example input image and model binary files can be downloaded with:
```bash
wget http://junglab.kaist.ac.kr/Dataset/SNUCRC_16-178T_chr8_pannormdiv.png #Example input image

wget http://junglab.kaist.ac.kr/Dataset/mnist_cnn.pt #Pretrained CNN weights
wget http://junglab.kaist.ac.kr/Dataset/xg_model_cis.pkl #XGBoost weight for cis- SV
wget http://junglab.kaist.ac.kr/Dataset/xg_model_trans.pkl #XGBoost weight for trans- SV
``` 
Tumor Hi-C contact map was divided by the control (pan-normal) Hi-C contact map (pseudocount added). Depth of the contact maps were pre-scaled by adjusting their mean and standard devidations to average values. Input image was scaled to n-fold and converted to np.uint8 array for OpenCV processing. Please check our publication for details.

For the overall Hi-C data processing procedures, please refer to the following papers.

> Kyukwang Kim and Inkyung Jung,  
> covNorm: an R package for coverage based  normalization of Hi-C and capture Hi-C data,  
> *Computational and Structural Biotechnology Journal*, Volume 19, pages 3149-3159(2021).  
> doi: https://doi.org/10.1016/j.csbj.2021.05.041 

> Kim, K., *et al*,
> 3DIV update for 2021: a comprehensive resource of 3D genome and 3D cancer genome  
> *Nucleic Acids Research*, Volume 49, Issue D1, pages D38–D46(2020).  
> doi: https://doi.org/10.1093/nar/gkaa1078

*Note*

-Please modify/fill the path in the code for the deployment.

-We recommend to use new version of the CAPReSE (will be released soon) for practical purpose.

-Comment WGS coordinate list to obtain Hi-C only SV detection result.

