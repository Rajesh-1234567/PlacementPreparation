https://docs.google.com/document/d/10XqvDZsXXqlpcgXS3kzkllNsZaPD11B_tr0mhN1DOcQ/edit?tab=t.0
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

----------------------------------------------------------------------------------------------------------------------------------------
8- Why KNN is known as a lazy learning technique?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
9- What do you mean by semi supervised learning?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
10- What is an OOB error and how is it useful?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
11- In what scenario decision tree should be preferred over random forest?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
12- Why Logistic Regression is called regression?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
13- What is Online Machine Learning? How is it different from Offline machine learning? List some of it’s applications
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
14- What is No Free Lunch Theorem?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
15- Imagine you are woking with a laptop of 2GB RAM, how would you process a dataset of 10GB?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
16- What are the main differences between Structured and Unstructured Data?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
17- What are the main points of difference between Bagging and Boosting?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
18- What are the assumptions of linear regression?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
19- How do you measure the accuracy of a Clustering Algorithm?
----------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------------------------
20- What is Matrix Factorization and where is it used in Machine Learning?
----------------------------------------------------------------------------------------------------------------------------------------
