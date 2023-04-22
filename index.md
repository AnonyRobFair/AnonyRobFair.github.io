---
layout: default
---

## Towards Distributionally Robust Fairness-aware Recommendation

## 1 Abstract
Due to the progressive advancement of trustworthy machine learning algorithms, fairness in recommender systems is attracting increasing attention and is often considered from the perspective of users. 
Conventional fairness-aware recommendation models make the assumption that user preferences remain the same between the training set and the testing set.
However, this assumption is disagreed with reality, where user preference can shift in the testing set due to the natural spatial or temporal heterogeneity.
It is concerning that conventional fairness-aware models may be unaware of such distribution shifts, leading to a sharp decline in the model performance.
To address the distribution shift problem, we propose a robust fairness-aware recommendation framework based on Distributionally Robust Optimization (DRO) technique.
In specific, we assign learnable weights for each sample to approximate the distributions that leads to the worst-case model performance, and then optimize the fairness-aware recommendation model to improve the worst-case performance in terms of both fairness and recommendation accuracy.
By iteratively updating the weights and the model parameter, our framework can be robust to unseen testing sets.
To ease the learning difficulty of DRO, we use a hard clustering technique to reduce the number of learnable sample weights.
To optimize our framework in a full differentiable manner, we soften the above clustering strategy.
Empirically, we conduct extensive experiments based on four real-world datasets to verify the effectiveness of our proposed framework.
For benefiting the research community, we have released our project at this page.

## 2 Contributions

In summary, this paper makes the following contributions:
1. We propose a novel and robust framework for fairness-aware recommendations. To the best of our knowledge, this is the first time in the recommendation domain. 
2. To achieve our proposed framework, we first propose a dual DRO framework to improve the worst-case performance in terms of fairness and recommendation quality. The generalization bound is also analyzed with the help of Rademacher Complexity. We then further enhance this framework by clustering training samples using both hard and softened strategies.
3. We empirically conduct extensive experiments based on four real-world datasets to demonstrate the effectiveness of our framework. To promote this research direction, we have released our project in this page.

## 3 Dataset Overview

| Dataset        | ML-1M | BC | AZ-SO | CIAO |
| -------------- | ------ | ------ | ------------- | -------- |
| # Users | 6,038  | 6,810 | 	22,685	| 7,373   |
| # Items   | 3,952  | 9,135  | 12,300     | 106,796   |
| # Interactions  | 518,482 | 114,426  | 185,718     | 204,425  |
| Sparsity         | 97.83% | 99.82%    | 99.93%     | 99.97%  |
| User attribute | Gender	| Country |	 Activity |	Activity  |
| Domain          | Movies | Books | E-commerce       | DVDs  |



## 4 Quick Start

### Step 1: Download the project

Our project `RobFair.zip` is available at [Google Drive](). Download and unzip our project. It contains both codes and datasets.
### Step 2: Create the running environment

Create `Python 3.9` enviroment and install the packages that the project requires.
- numpy==1.23.2
- scikit_learn==1.1.2 
- torch==1.12.1
- pybind11==2.10.0
- tkinter==0.1.0

You can install the packages with the following command. `pybind11` is necessary in our code to accelerate the sampling process.

```
    pip install -r requirements.txt
```

### Step 3: Run the project
Run our frameworks with the following command:
```
    cd ./code
    python main.py --dataset ml-1m --model mf --methods simplex --lr 0.1
```
where 
- `--dataset`: dataset chosen from `['ml-1m', 'bookcrossing', 'sports', 'ciao]`;
- `--model`: base model chosen from `['mf', 'lgn']`;
- `--methods`: methods chosen from `['none', 'value', 'non_parity', 'gdro','reweight','simplex','cluster','deep_cluster']`;

### Step 4: Check the performance

For the recommendation performance, we use NDCG@10, Recall@10 as evaluation metrics. For the fairness performance, we use GRF(R) and GRF(N) as evaluation metrics. Readers can refer to the paper for more detailed definition of these metrics. We report the overall results in Table 3 in the original paper.


## 5 Hyper-parameters search range

We tune hyper-parameters according to the following table.

| Hyper-parameter     | Explanation | Range |
| ------------------- | ---------------------------------------------------- | ------------------- |
| lr | learning rate of recommender model | \{0.00001, 0.0001, 0.001, 0.01, 0.1\} |
| recdim | embedding size | \{16, 32, 64\} |
| batch_size | batch size of each mini-batch  | \{512, 1024, 2048\} |
| weight_fair | the coefficient of fairness constraint |  \{0.0001, 0.001, 0.01, 0.1, 1, 10, 100\} |
| weight_sc | the coefficient of softened clustering | \{0.0001, 0.001, 0.01, 0.1, 1, 10, 100\} |
| weight_entropy | the coefficient of entropy-based regularizer | \{0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1\} |
| k_clusters | the number of clusters |  \{10, 50, 100, 150, 200, 250\} |
| layer | the layers of LightGCN base model | \{1, 2, 3, 4, 5\} |
