# PSRMTE
This repository is used to store all codes of our paper "PSRMTE: Paper Submission Recommendation Using Mixtures of Transformer Encoders", which is published in the journal "Expert Systems and Applications".

The paper is available at: https://www.sciencedirect.com/science/article/abs/pii/S0957417422005024#!


Please use the *.ipynb files in these folders to execute.
 
Download and place this data folder into the directory where you clone our codes (https://drive.google.com/drive/folders/1vIsyD962Msm4aXYbUrOGNwj4aLvx-fGL?usp=sharing). The dataset is file.jsonl

# References
https://github.com/google-research/bert

https://github.com/zihangdai/xlnet

https://github.com/google-research/electra


# Further Usage
For any usage related to all codes and data used from our repository, please cite our following paper:

```
@article{NGUYEN2022117096,
title = {PSRMTE: Paper submission recommendation using mixtures of transformer},
journal = {Expert Systems with Applications},
pages = {117096},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.117096},
url = {https://www.sciencedirect.com/science/article/pii/S0957417422005024},
author = {Dac Huu Nguyen and Son Thanh Huynh and Cuong Viet Dinh and Phong Tan Huynh and Binh Thanh Nguyen},
keywords = {Recommendation system, Deep learning, Transformer encoders},
abstract = {Nowadays, there has been a rapidly increasing number of scientific submissions in multiple research domains. A large number of journals have various acceptance rates, impact factors, and rankings in different publishers. It becomes time-consuming for many researchers to select the most suitable journal to submit their work with the highest acceptance rate. A paper submission recommendation system is more critical for the research community and publishers as it gives scientists another support to complete their submission conveniently. This paper investigates the submission recommendation system for two main research topics: computer science and applied mathematics. Unlike the previous works (Wang et al., 2018; Son et al., 2020) that extract TF-IDF and statistical features as well as utilize numerous machine learning algorithms (logistics regression and multiple perceptrons) for building the recommendation engine, we present an efficient paper submission recommendation algorithm by using different bidirectional transformer encoders and the Mixture of Transformer Encoders technique. We compare the performance between our methodology and other approaches by one dataset from Wang et al. (2018) with 14012 papers in computer science and another dataset collected by us with 223,782 articles in 178 Springer applied mathematics journals in terms of top K accuracy (K=1,3,5,10). The experimental results show that our proposed method extensively outperforms other state-of-the-art techniques with a significant margin in all top K accuracy for both two datasets. We publish all datasets collected and our implementation codes for further references.11https://github.com/BinhMisfit/PSRMTE.}
}
```

For any questions, please contact our corresponding author: Dr. Binh T. Nguyen at ngtbinh@hcmus.edu.vn. 
