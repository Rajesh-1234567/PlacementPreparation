https://docs.google.com/document/d/10XqvDZsXXqlpcgXS3kzkllNsZaPD11B_tr0mhN1DOcQ/edit?tab=t.0

ML -   https://www.geeksforgeeks.org/machine-learning/machine-learning/
DL/CV - https://www.geeksforgeeks.org/computer-vision/computer-vision/

----------------------------------------------------------------------------------------------------------------------------------------
1- What is the difference between Parametric and Non Parametric Algorithms?
----------------------------------------------------------------------------------------------------------------------------------------
### **Parametric Algorithms**:
- **Fixed structure**: Assumes a specific data distribution with a fixed number of parameters.
- **Simpler & faster**: Easier to train, but less flexible.
- **Examples**: Linear Regression, Logistic Regression, Naive Bayes.

### **Non-Parametric Algorithms**:
- **Flexible structure**: No fixed assumptions; adapts to the data.
- **Complex & slower**: Captures intricate patterns but may overfit.
- **Examples**: k-NN, Decision Trees, Random Forests.

### **Key Difference**:  
- **Parametric**: Fixed, simpler, less flexible.  
- **Non-Parametric**: Grows with data, more complex, highly flexible.
----------------------------------------------------------------------------------------------------------------------------------------
2- Difference between convex and non-convex cost function; what does it mean when a cost function is non-convex?
----------------------------------------------------------------------------------------------------------------------------------------
### **Difference Between Convex and Non-Convex Cost Functions**

| **Aspect**           | **Convex Cost Function**                          | **Non-Convex Cost Function**                     |
|-----------------------|--------------------------------------------------|-------------------------------------------------|
| **Shape**            | Bowl-shaped, a single global minimum.            | Complex, may have multiple local minima and maxima. |
| **Optimization**      | Easier to optimize using gradient descent; guaranteed to find the global minimum. | Optimization is challenging; may get stuck in local minima. |
| **Properties**        | Satisfies convexity properties: \[f(\lambda x_1 + (1-\lambda)x_2) \leq \lambda f(x_1) + (1-\lambda)f(x_2)\]. | Does not satisfy convexity properties; has irregular or multi-modal shapes. |
| **Examples**          | Loss functions like Mean Squared Error (MSE) for linear regression. | Loss functions for deep learning models, like neural networks. |

---

### **What Does it Mean When a Cost Function is Non-Convex?**
A **non-convex cost function** means:
1. The cost function has multiple peaks (local maxima) and valleys (local minima).
2. Optimization algorithms (e.g., gradient descent) might:
   - Get trapped in a local minimum instead of reaching the global minimum.
   - Require additional techniques like momentum, learning rate scheduling, or advanced optimizers (e.g., Adam).
3. Common in complex models like deep learning, where the parameter space is highly non-linear and multidimensional.

**Implication**: Finding the "best" solution in a non-convex cost function is harder and may require careful tuning or heuristic approaches.
----------------------------------------------------------------------------------------------------------------------------------------
3- How do you decide when to go for deep learning for a project?
----------------------------------------------------------------------------------------------------------------------------------------
### **Use Deep Learning When:**
1. **Large Data**: Ample labeled data is available.
2. **Complex Patterns**: Problem involves non-linear or intricate relationships.
3. **Unstructured Data**: Working with images, text, audio, or video.
4. **High Accuracy**: Requires state-of-the-art performance.
5. **Resources**: Sufficient computational power (GPUs/TPUs).
6. **Transfer Learning**: Pre-trained models can be used.

### **Avoid Deep Learning When:**
- **Small Data**: Classical models perform better on limited data.
- **Simple Problems**: Overkill for straightforward tasks.
- **Limited Resources**: Compute-heavy deep learning isn't practical.
  
----------------------------------------------------------------------------------------------------------------------------------------
4- Give an example of when False positive is more crucial than false negative and vice versa.
----------------------------------------------------------------------------------------------------------------------------------------
### **False Positive (FP) More Crucial than False Negative (FN)**

**Example: Email Spam Filtering**
- **False Positive**: A legitimate email is incorrectly classified as spam.
- **False Negative**: A spam email is not detected and ends up in the inbox.

**Why FP is more crucial**: In this case, a **false positive** (legitimate email marked as spam) is more disruptive because it can cause important emails to be missed, requiring manual checking of the spam folder. However, a **false negative** (spam email in inbox) might be annoying but does not result in serious consequences.

---

### **False Negative (FN) More Crucial than False Positive (FP)**

**Example: Medical Disease Diagnosis (e.g., Cancer Detection)**
- **False Positive**: A healthy person is incorrectly diagnosed with cancer.
- **False Negative**: A person with cancer is incorrectly diagnosed as healthy.

**Why FN is more crucial**: In this case, a **false negative** (cancer missed) is much more dangerous because it means the person may not receive the necessary treatment, leading to potentially life-threatening consequences. A **false positive** (healthy person wrongly diagnosed) might lead to unnecessary tests or treatments, but it’s less critical than missing a diagnosis.

### **Summary:**
- **FP more crucial**: Situations where the cost of an incorrect positive is high, e.g., spam filtering.
- **FN more crucial**: Situations where the cost of missing a critical case is high, e.g., medical diagnosis.

----------------------------------------------------------------------------------------------------------------------------------------
5- Why is “Naive” Bayes naive?
----------------------------------------------------------------------------------------------------------------------------------------
The "naive" in **Naive Bayes** refers to the assumption of **conditional independence** between the features. 

### **Why is it called "naive"?**

- **Conditional Independence Assumption**: Naive Bayes assumes that all features in the dataset are **independent of each other** given the class label, which is often **not true** in real-world data.
  
- **"Naive" Assumption**: In many practical situations, features (like height and weight in predicting health status) are correlated, but Naive Bayes treats them as if they are independent, which is a simplified and often unrealistic assumption.

### **Example**:
- Suppose you're predicting whether an email is spam or not, based on the words "buy," "free," and "limited offer."
- Naive Bayes assumes that the occurrence of each word is independent of the others, given the class (spam or not). However, in reality, the presence of the word "buy" may be highly correlated with the word "limited offer" in a spam email.

Despite this "naive" assumption, Naive Bayes can perform surprisingly well in many practical applications, especially when the independence assumption is approximately true or when correlations between features are weak.

----------------------------------------------------------------------------------------------------------------------------------------
6- Give an example where the median is a better measure than the mean.
----------------------------------------------------------------------------------------------------------------------------------------
The **median** is a better measure than the **mean** when the dataset contains outliers or is skewed, as it is less affected by extreme values.

### Example: Household Income

Imagine a neighborhood with the following yearly household incomes (in thousands of dollars):

`[30, 35, 40, 45, 50, 55, 60, 1000]`

1. **Mean (Average):**
   - Sum of incomes = \(30 + 35 + 40 + 45 + 50 + 55 + 60 + 1000 = 1315\)
   - Mean = \( \frac{1315}{8} = 164.375 \) (thousand dollars)

   The mean is heavily influenced by the outlier (1000), suggesting that the "average" income is much higher than the majority of the households.

2. **Median (Middle Value):**
   - After sorting: `[30, 35, 40, 45, 50, 55, 60, 1000]`
   - Median = \( \frac{45 + 50}{2} = 47.5 \) (thousand dollars)

   The median represents the middle income in the dataset and provides a better sense of the central tendency for the majority of households.

### Why Median is Better:
In this example, the **mean (164.375)** misrepresents the typical income due to the extreme outlier (1000). The **median (47.5)** is a more accurate measure of the central tendency for the majority of the population.
----------------------------------------------------------------------------------------------------------------------------------------
7- What do you mean by the unreasonable effectiveness of data?
----------------------------------------------------------------------------------------------------------------------------------------
The **"unreasonable effectiveness of data"** refers to the idea that in many fields, especially AI and machine learning, having access to large and high-quality datasets often leads to highly effective solutions, sometimes outperforming even sophisticated algorithms. It emphasizes that data quantity and diversity can compensate for simpler models, enabling them to generalize better and handle complex tasks.

For example, large datasets like ImageNet in computer vision or Common Crawl for NLP allow models to learn rich patterns and perform tasks like object recognition or language understanding with remarkable accuracy. The concept also highlights that as the size of data increases, models often exhibit emergent capabilities, solving problems they weren’t explicitly trained for, like reasoning or transfer learning.

This effectiveness stems from real-world data's ability to capture nuances and variability, ensuring that models trained on it are exposed to diverse scenarios. However, the approach has limitations, such as data bias, high labeling costs, and diminishing returns as datasets grow beyond a certain point. Despite this, the concept underscores the critical role of data in driving progress in AI, often outweighing algorithmic complexity.
----------------------------------------------------------------------------------------------------------------------------------------
8- Why KNN is known as a lazy learning technique?
----------------------------------------------------------------------------------------------------------------------------------------
### **Differences Between Lazy and Eager Learning:**

| **Lazy Learning**                      | **Eager Learning**                         |
|----------------------------------------|--------------------------------------------|
| Delays learning until query time.      | Learns a general model during training.    |
| Stores the training data for future reference. | Creates a model or hypothesis based on training data. |
| The prediction is made during the test phase by comparing the query with stored data. | The prediction is made after the model is trained. |
| High memory usage because all the training data is stored. | Lower memory usage as only the model is stored. |
| Can be computationally expensive during prediction (e.g., distance calculations). | Can be faster at prediction time since the model is already built. |

### **Why KNN is Known as Lazy Learning:**
**K-Nearest Neighbors (KNN)** is a classic example of lazy learning. Here's why:
1. **No explicit training phase:** KNN does not "train" a model. Instead, it stores all the training data and waits for a query to classify or predict.
2. **Prediction at query time:** When a new input is provided (a test sample), KNN calculates the **distance** between the test point and all the training data points. Based on the **nearest neighbors**, it makes a prediction (classification or regression).
3. **No generalization:** KNN does not generalize from the training data to create a model. It simply "remembers" the data and makes decisions based on it when required.

Thus, KNN is considered lazy because it postpones the computation until a prediction is needed and relies heavily on the training data without precomputing a model.
----------------------------------------------------------------------------------------------------------------------------------------
9- What do you mean by semi supervised learning?
----------------------------------------------------------------------------------------------------------------------------------------
**Semi-supervised learning** is a machine learning approach that combines a small amount of labeled data with a large amount of unlabeled data to train a model. It reduces the need for extensive labeled data while improving model performance by leveraging patterns found in the unlabeled data. 

For example, in **image classification**, labeling thousands of images can be expensive and time-consuming. Using semi-supervised learning, a model might initially train on a small set of labeled images, then use a larger pool of unlabeled images to refine its understanding. The model predicts labels for the unlabeled data and incorporates them into the training process. This helps improve the model's accuracy without requiring a vast amount of manually labeled data.

In **speech recognition**, labeled transcriptions are costly to produce, but large volumes of unlabeled audio data are available. Semi-supervised learning allows the model to use both labeled and unlabeled audio to better recognize speech patterns.
----------------------------------------------------------------------------------------------------------------------------------------
10- What is an OOB error and how is it useful?
----------------------------------------------------------------------------------------------------------------------------------------
**OOB (Out-of-Bag) Error** is a concept used in ensemble learning, particularly with **Random Forests** and **Bagging** techniques. It refers to the error rate estimated using data points that were not selected in the bootstrap sampling process (i.e., the out-of-bag samples).

### How OOB Error Works:
1. **Bootstrap Sampling**: In techniques like Random Forest, each decision tree is trained on a random subset of the training data, selected with replacement (bootstrap sampling). This means some data points are repeated in the training set, while others are left out.
   
2. **Out-of-Bag Samples**: The data points that are left out of a particular tree's training set are known as **out-of-bag** samples.

3. **Error Calculation**: For each tree, the model is tested on its own out-of-bag samples. The OOB error is calculated by averaging the prediction errors for all the out-of-bag samples across all trees.

### Usefulness of OOB Error:
- **Model Evaluation**: OOB error provides an **internal validation** method to estimate the model's performance without needing a separate validation or test set. This is especially useful when you have limited data, as it makes use of the data that is not used for training.
  
- **Bias-Variance Trade-off**: It helps in assessing the **bias-variance trade-off** of the model by providing a reliable error estimate without overfitting to the training data.
  
- **Efficiency**: Since OOB error uses the data that is not part of each tree’s training set, it saves the need for a separate cross-validation or test set, making it a more efficient use of data.

In summary, **OOB error** is a valuable metric for evaluating ensemble models, particularly in Random Forests, as it allows you to assess performance using unused data, thus improving efficiency and reducing the need for separate validation sets.
----------------------------------------------------------------------------------------------------------------------------------------
11- In what scenario decision tree should be preferred over random forest?
----------------------------------------------------------------------------------------------------------------------------------------
A decision tree might be preferred over a random forest in the following scenarios:

### 1. **Interpretability and Simplicity**
   - **Why:** A single decision tree is much easier to interpret, visualize, and explain to stakeholders than a random forest, which combines multiple decision trees.
   - **Example Use Case:** If your primary goal is to provide a straightforward explanation of the model's predictions, such as in business or healthcare applications where interpretability is crucial.

---

### 2. **Low Resource and Computational Constraints**
   - **Why:** Training and using a single decision tree require less computational power and memory compared to a random forest.
   - **Example Use Case:** When working on a low-resource device or when quick predictions are needed in a resource-constrained environment.

---

### 3. **Small Datasets**
   - **Why:** For smaller datasets, a decision tree can effectively capture patterns without the risk of overfitting to the extent that might require ensemble methods.
   - **Example Use Case:** When the dataset size is small and straightforward relationships exist between features and the target variable.

---

### 4. **Avoiding Overfitting Concerns**
   - **Why:** A well-tuned decision tree with proper constraints (like maximum depth or minimum samples per leaf) may suffice without requiring the complexity of a random forest.
   - **Example Use Case:** In problems where the feature space is limited, and the risk of overfitting can be controlled easily.

---

### 5. **Faster Training and Prediction**
   - **Why:** Training a single decision tree is much faster than training multiple trees in a random forest.
   - **Example Use Case:** When time-sensitive decisions need to be made during training or prediction.

---

### 6. **Exploratory Data Analysis (EDA)**
   - **Why:** A decision tree can quickly show feature importance and splits, which is helpful during EDA to understand relationships in the data.
   - **Example Use Case:** When trying to identify the most influential features for the target variable during initial model exploration.

---

### Key Consideration:
While decision trees are useful in the above scenarios, they are prone to overfitting and can produce high variance. Random forests are generally preferred for their robustness and ability to generalize well, but the added complexity isn't always necessary depending on the use case.
----------------------------------------------------------------------------------------------------------------------------------------
12- Why Logistic Regression is called regression?
----------------------------------------------------------------------------------------------------------------------------------------
Logistic Regression is called "regression" because it originates from linear regression, modeling the relationship between features and the probability of an outcome. It uses regression techniques to predict continuous probabilities (not directly class labels) and applies the sigmoid function to map these probabilities. The term comes from its historical development as an extension of regression for classification problems.

| Aspect              | Linear Regression                  | Logistic Regression                                |
| ------------------- | ---------------------------------- | -------------------------------------------------- |
| **Output Type**     | Continuous value (−∞ to +∞)        | Probability (0 to 1)                               |
| **Target Variable** | Continuous                         | Categorical (binary/multiclass)                    |
| **Equation Form**   | $y = \beta_0 + \beta_1x + \dots$   | $p = \frac{1}{1+e^{-z}}$                           |
| **Assumptions**     | Linearity between input and output | Linearity between input and **log-odds** of output |
| **Error Measure**   | Mean Squared Error (MSE)           | Log Loss / Cross-Entropy                           |
| **Use Case**        | Predicting prices, scores, trends  | Predicting classes (spam, churn, disease)          |

----------------------------------------------------------------------------------------------------------------------------------------
13- What is Online Machine Learning? How is it different from Offline machine learning? List some of it’s applications
----------------------------------------------------------------------------------------------------------------------------------------
### **What is Online Machine Learning?**  
Online Machine Learning trains models incrementally, updating them as new data arrives, without needing to process the entire dataset at once. It is efficient, adaptable to real-time changes, and suitable for streaming data.

---

### **Difference Between Online and Offline Machine Learning**  
| **Aspect**         | **Online ML**                      | **Offline ML**                  |
|--------------------|------------------------------------|---------------------------------|
| **Data Processing** | Incremental (one instance/batch). | Full dataset at once (batch).  |
| **Model Updates**   | Continuous, real-time.            | Requires retraining.           |
| **Adaptability**    | Adapts to changes (concept drift).| Less adaptable.                |
| **Resource Use**    | Memory and computationally efficient. | Resource-intensive.           |

---

### **Applications**  
- Real-time recommendations (Netflix, Spotify).  
- Fraud detection (credit cards).  
- Stock market predictions.  
- Predictive maintenance.  
- Spam filtering.  
- IoT data analysis.  
- Dynamic pricing (ride-sharing apps).  
----------------------------------------------------------------------------------------------------------------------------------------
14- What is No Free Lunch Theorem?
----------------------------------------------------------------------------------------------------------------------------------------
The **No Free Lunch Theorem** asserts that no algorithm outperforms all others across every possible problem. Each algorithm's success depends on the problem's structure and assumptions. For machine learning, this means:  

1. **Algorithm Suitability:** An algorithm may excel in one scenario (e.g., neural networks for complex patterns) but fail in another (e.g., linear regression for simple trends).  
2. **Trade-offs:** Models must balance bias, variance, and complexity based on the data.  
3. **Hyperparameter Dependence:** No universal settings exist; tuning is specific to the task.  

The theorem emphasizes the need for problem-specific choices in model selection and optimization.
----------------------------------------------------------------------------------------------------------------------------------------
15- Imagine you are woking with a laptop of 2GB RAM, how would you process a dataset of 10GB?
----------------------------------------------------------------------------------------------------------------------------------------
To process a 10GB dataset on a laptop with only 2GB of RAM, you can use the following approaches:

1. **Incremental Learning:** Update models in small increments as new data comes in without needing to load the entire dataset into memory. 
   - **Libraries:** Scikit-learn, TensorFlow, Keras.

2. **Chunking:** Load data in smaller chunks instead of the whole dataset at once.
   - **Libraries:** Pandas, Dask.

3. **Out-of-Core Processing:** Work with large datasets stored on disk rather than loading them into memory all at once.
   - **Libraries:** Vaex, Dask, PySpark.

4. **Efficient Storage Formats:** Use compressed or more efficient data formats to reduce memory usage.
   - **Libraries:** HDF5, Apache Arrow.

These methods allow you to efficiently work with a 10GB dataset without exceeding the available RAM.
----------------------------------------------------------------------------------------------------------------------------------------
16- What are the main differences between Structured and Unstructured Data?
----------------------------------------------------------------------------------------------------------------------------------------
1. **Structured data** is highly organized and stored in predefined formats like tables, while **unstructured data** lacks a specific format and includes text, images, or videos.  
2. **Structured data** is stored in relational databases, whereas **unstructured data** is stored in data lakes or NoSQL systems.  
3. **Structured data** follows a defined schema, making it easy to query, while **unstructured data** requires specialized tools for analysis.  
4. Examples of **structured data** include customer databases and financial transactions, while **unstructured data** includes emails, social media posts, and multimedia.  
5. **Structured data** is less flexible, needing schema changes for updates, while **unstructured data** adapts to various formats but is harder to analyze.  
----------------------------------------------------------------------------------------------------------------------------------------
17- What are the main points of difference between Bagging and Boosting?
----------------------------------------------------------------------------------------------------------------------------------------
Here are the main points of difference between **Bagging** and **Boosting**:

### 1. **Objective**:
   - **Bagging**: Aims to reduce variance by creating multiple independent models and combining their predictions (e.g., voting or averaging).  
   - **Boosting**: Aims to reduce bias by sequentially building models, where each new model corrects the errors of the previous one.  

---

### 2. **Model Independence**:
   - **Bagging**: Models are built independently of each other.  
   - **Boosting**: Models are built sequentially, with each model relying on the performance of the previous ones.  

---

### 3. **Data Sampling**:
   - **Bagging**: Uses random sampling with replacement (bootstrap sampling) to create diverse subsets of the training data.  
   - **Boosting**: Focuses on the misclassified samples by assigning higher weights to them in subsequent iterations.  

---

### 4. **Combining Models**:
   - **Bagging**: Combines predictions through averaging (for regression) or majority voting (for classification).  
   - **Boosting**: Combines models by assigning weights to them based on their performance, giving more importance to stronger models.  

---

### 5. **Overfitting**:
   - **Bagging**: Less prone to overfitting, as it focuses on reducing variance.  
   - **Boosting**: More prone to overfitting, especially if the base models are too complex or iterations are too high.  

---

### 6. **Performance**:
   - **Bagging**: Works best with high-variance models (e.g., decision trees) by stabilizing predictions.  
   - **Boosting**: Works well with weak learners (e.g., shallow trees), turning them into strong learners.  

---

### 7. **Examples**:
   - **Bagging**: Random Forest, BaggingClassifier.  
   - **Boosting**: AdaBoost, Gradient Boosting, XGBoost, CatBoost, LightGBM.  

By balancing these differences, you can choose the appropriate method based on the problem's nature, model's bias/variance trade-off, and computational constraints.
----------------------------------------------------------------------------------------------------------------------------------------
18- What are the assumptions of linear regression?
----------------------------------------------------------------------------------------------------------------------------------------
Linear regression relies on several key assumptions to produce reliable and valid results. These assumptions are:

### 1. **Linearity**  
   - The relationship between the independent variables and the dependent variable is linear.  

### 2. **Independence of Errors**  
   - The residuals (errors) are independent of each other.  
   - This is particularly important for time-series data, where residuals should not be correlated across time (checked using the Durbin-Watson test).  

### 3. **Homoscedasticity**  
   - The residuals have constant variance across all levels of the independent variables.  
   - If the variance changes (heteroscedasticity), predictions might become unreliable.  

### 4. **No Multicollinearity**  
   - The independent variables should not be highly correlated with each other.  
   - Multicollinearity can make it difficult to estimate the coefficients accurately.  

### 5. **Normality of Errors**  
   - The residuals are normally distributed.  
   - This is especially important for hypothesis testing and constructing confidence intervals.  

### 6. **No Measurement Error in Independent Variables**  
   - The independent variables are measured without error.  

### 7. **Correct Model Specification**  
   - The model is correctly specified, with no omitted variables or unnecessary terms.  

By checking these assumptions through diagnostic tests and visualizations (e.g., residual plots, variance inflation factor, Q-Q plots), you can validate the appropriateness of linear regression for your data.
----------------------------------------------------------------------------------------------------------------------------------------
19- How do you measure the accuracy of a Clustering Algorithm?
----------------------------------------------------------------------------------------------------------------------------------------
We rely on internal metrics that evaluate cluster quality:
Silhouette Coefficient (−1 to +1): Higher means better defined, well-separated clusters.
Davies–Bouldin Index (lower = better): Measures average similarity between clusters.
Dunn Index (higher = better): Ratio of inter-cluster to intra-cluster distance.
Elbow Method (for k-means): Helps decide optimal number of clusters based on within-cluster variance.
----------------------------------------------------------------------------------------------------------------------------------------
20- What is Matrix Factorization and where is it used in Machine Learning?
----------------------------------------------------------------------------------------------------------------------------------------
Nice one 👍 Matrix Factorization is a **core technique in Machine Learning**, especially in **recommendation systems**. Let’s break it down simply:

---

## 🔹 What is Matrix Factorization?

Matrix Factorization means breaking a **large matrix** into the product of **smaller matrices** that capture hidden patterns (latent features).

Mathematically:

$$
R \approx U \times V^T
$$

* $R$: Original matrix (e.g., user-item ratings matrix).
* $U$: User feature matrix (latent representation of users).
* $V$: Item feature matrix (latent representation of items). Mainly used in recomendation system.

👉 Instead of directly storing user–item interactions, we represent users and items in a **lower-dimensional latent space**.

---
----------------------------------------------------------------------------------------------------------------------------------------
21- What is an Imbalanced Dataset and how can one deal with this problem?
----------------------------------------------------------------------------------------------------------------------------------------

## 🔹 Imbalanced Dataset

When one class has **many more samples** than the other (e.g., 99% normal, 1% fraud).
👉 Problem: Model becomes biased toward the majority class.

---

## 🔹 How to Handle It

1. **Data-Level (Resampling)**

   * Oversample minority (e.g., **SMOTE**)
   * Undersample majority

2. **Algorithm-Level**

   * Use **class weights / cost-sensitive learning**
   * Apply ensemble methods (Balanced RF, SMOTEBoost)

3. **Evaluation Metrics**

   * Don’t use Accuracy → use **Precision, Recall, F1, ROC-AUC, PR-AUC**

---

✅ Example: Fraud detection → use **SMOTE + class weights** and measure **Recall/F1** instead of Accuracy.


---

## 🔹 SMOTE (Synthetic Minority Over-sampling Technique)

* **What it is:** A technique to handle **imbalanced datasets** by generating **synthetic samples** for the minority class instead of just duplicating existing ones.

---

## 🔹 How it Works

1. For each minority class sample, find its **k nearest neighbors** (default k=5).
2. Randomly pick one neighbor.
3. Create a new synthetic sample **along the line** between the two points.

This increases minority class data → makes dataset more balanced.

---

## 🔹 Example

Fraud dataset: 990 normal, 10 fraud.

* After SMOTE: 990 normal, 990 synthetic+real fraud → balanced dataset.

---
----------------------------------------------------------------------------------------------------------------------------------------
22- How do you measure the accuracy of a recommendation engine?
----------------------------------------------------------------------------------------------------------------------------------------
Great question 👌 Measuring the accuracy of a **recommendation engine** depends on the type of recommendation system (rating prediction vs top-N recommendation).

---

## 🔹 1. If Predicting Ratings (e.g., Netflix predicting 4.5⭐)

We treat it like a regression problem:

* **RMSE (Root Mean Squared Error)** → penalizes large errors.
* **MAE (Mean Absolute Error)** → average absolute difference between predicted and actual ratings.

---

## 🔹 2. If Recommending Top-N Items (e.g., Top 10 movies for a user)

We treat it like a ranking/classification problem:

* **Precision\@K:** How many recommended items in top-K are relevant?
* **Recall\@K:** How many of the relevant items were captured in top-K?
* **F1\@K:** Harmonic mean of Precision\@K and Recall\@K.
* **MAP (Mean Average Precision):** Averages precision over ranked lists.
* **NDCG (Normalized Discounted Cumulative Gain):** Rewards correct ranking order (higher score if relevant items appear at top).
* **Hit Rate:** Did at least one relevant item appear in recommendations?

---

----------------------------------------------------------------------------------------------------------------------------------------
23- What are some ways to make your model more robust to outliers?
----------------------------------------------------------------------------------------------------------------------------------------
utliers can really mess up models if not handled properly. Here are **ways to make models more robust to outliers**:

---

## 🔹 1. Data Preprocessing

* **Remove outliers:** Use statistical tests (e.g., z-score > 3, IQR method).
* **Transform data:** Apply **log, square root, or Box-Cox** transformations to reduce skew.
* **Cap/floor (Winsorization):** Replace extreme values with thresholds.

---

## 🔹 2. Use Robust Models

* **Tree-based models** (Decision Trees, Random Forest, XGBoost) → less sensitive to outliers.
* **Regularization models** (Lasso/Ridge) → reduce effect of extreme coefficients.
* **Robust Regression** (Huber loss, RANSAC) → less influenced by outliers compared to ordinary least squares.

---

## 🔹 3. Use Robust Metrics

* Instead of **MSE (very sensitive)**, use:

  * **MAE (Mean Absolute Error)**
  * **Huber Loss** (mix of MAE + MSE)
  * **Quantile Loss**

---

----------------------------------------------------------------------------------------------------------------------------------------
24- How can you measure the performance of a dimensionality reduction algorithm on your dataset?
----------------------------------------------------------------------------------------------------------------------------------------
Sure 👍 Here’s the **short version**:

---

## 🔹 Measuring Dimensionality Reduction Performance

1. **Explained Variance / Reconstruction Error** → How much info is preserved (PCA).
2. **Neighborhood Preservation** → Metrics like **Trustworthiness** (similar points stay close).
3. **Downstream Task Performance** → Accuracy/F1 for classification, Silhouette/ARI for clustering.
4. **Visualization Quality** → How well clusters separate in 2D/3D plots (t-SNE, UMAP).

---

----------------------------------------------------------------------------------------------------------------------------------------
25- What is Data Leakage? List some ways using which you can overcome this problem.
----------------------------------------------------------------------------------------------------------------------------------------
---

## 🔹 Data Leakage

When information from the **test set or future data** sneaks into training, making the model look unrealistically good but fail in production.

---

## 🔹 How to Prevent It

1. Split data **before preprocessing**.
2. Use **pipelines** (fit only on train, apply on test).
3. Remove features unavailable at prediction time.
4. Use **time-based splits** for time-series.
5. Validate with **cross-validation + domain knowledge**.

---

✅ In short: **Keep train/test fully isolated, avoid future info, and use pipelines.**

----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
