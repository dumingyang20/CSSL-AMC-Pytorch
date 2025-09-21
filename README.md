# A Contrastive Learner for Automatic Modulation Classification

## Abstract

This is the official implementation for the paper: 
["A Contrastive Learner for Automatic Modulation Classification" 
(IEEE Trans. Wireless Commun., vol. 24, no. 4, 2025).](https://ieeexplore.ieee.org/document/10857965)

## Data Preparation
- Segment by SNR: The RadioML 2018.01A dataset is split into individual files, 
each containing all modulation categories for a single SNR value.

- Generate Low-SNR Data: Synthetic low-SNR (5 dB) samples are created by adding random Gaussian noise to high-SNR (30 dB) data.

- Form Training Pairs: For contrastive learning, 
pairs can be created either from synthetically generated data or by directly pairing samples from existing SNR values (e.g., 30 dB and 4 dB).


## Run

- Run ```run.py``` to generate a pretrained model through self-supervised learning on both noisy and clean data. 
- Then, execute ```finetune.py``` to perform downstream classification, 
which learns a non-linear mapping from semantic features to category labels.

## Citation
```
  @ARTICLE{10857965,
  author={Du, Mingyang and Pan, Jifei and Bi, Daping},
  journal={IEEE Transactions on Wireless Communications}, 
  title={A Contrastive Learner for Automatic Modulation Classification}, 
  year={2025},
  volume={24},
  number={4},
  pages={3575-3589},
  keywords={Noise measurement;Signal to noise ratio;Feature extraction;Modulation;Data models;Contrastive learning;Training;Time-frequency analysis;Robustness;Accuracy;Automatic modulation classification;contrastive learning;noise corruption;self-supervised},
  doi={10.1109/TWC.2025.3532438}}
```

## Contact

If you have any question about our work or code, please email [dumingyang17@nudt.edu.cn]().
