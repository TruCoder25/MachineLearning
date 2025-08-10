# 🧠 Machine Learning

This repository is a comprehensive showcase of my machine learning work, featuring **end-to-end projects** that start from raw, messy datasets and progress through **data cleaning, feature engineering, algorithm selection** (with clear reasoning for each choice), **model training**, and **evaluation using industry-standard metrics**.

---

## 📌 Prerequisites to Learn ML

### 1️⃣ Learn Python
- Have a solid understanding of Python basics.
- **Recommended YouTube Channels:** [BroCode](https://www.youtube.com/@BroCodez) and [Programming with Mosh](https://www.youtube.com/@programmingwithmosh).

### 2️⃣ Learn Essential Data Tools
- Libraries: **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**.
- These help with data manipulation, analysis, and visualization.

### 3️⃣ Understand the Basic ML Workflow
1. **Import Requirements** – Load datasets (e.g., CSV files) and libraries.
2. **Data Cleaning** – Handle missing values (`fillna`) or drop irrelevant rows/columns.
3. **Choose the Best Algorithm** – Based on problem type and dataset characteristics. *(This is where the “Why ML?” question is answered.)*
4. **Train & Predict** – Fit the model to the training data and make predictions.
5. **Evaluate the Model** – Use metrics like **R² Score**, **Confusion Matrix**, **Accuracy**, **Precision**, etc.

💡 *For a beginner-friendly introduction to these steps, watch ["Machine Learning Basics"](https://www.youtube.com/watch?v=7eh4d6sabA0) by Programming with Mosh.*
## 🤖 What is Machine Learning? Why Do We Need It?

**Machine Learning (ML)** is a subset of **Artificial Intelligence (AI)** that focuses on enabling machines to learn patterns from data and make predictions or decisions **without being explicitly programmed** for every possible scenario.  
It combines statistical methods, data analysis, and computational algorithms to understand data, extract insights, and make informed predictions or forecasts.

---

### 🔹 Why We Need ML
- **Prediction** – Forecast future outcomes based on historical data (e.g., predicting stock prices, weather, or sales).
- **Automation** – Replace manual decision-making with scalable, data-driven systems.
- **Pattern Discovery** – Detect complex relationships and trends that are hard for humans to spot.
- **Adaptability** – Improve accuracy and performance as more data becomes available.

In the **ML workflow** (as mentioned in the prerequisites), the answer to “Why ML?” comes at **Step 3 – Choosing the Best Algorithm**:  
We use ML when the problem requires learning from examples and generalizing to new, unseen data to make accurate predictions or decisions.

### 📚 Before Entering into Algorithms  

Machine Learning is generally categorized into three main types of learning:  

1. **Supervised Learning**  
   - The model learns from labeled data — meaning each training example has both input features (`X`) and the correct output label (`y`).  
   - Example: Predicting house prices when you already know past prices for similar houses.  

2. **Unsupervised Learning**  
   - The model learns from **unlabeled data**, finding patterns, structures, or groupings without explicit answers.  
   - Example: Customer segmentation based on purchasing behavior, without knowing group labels beforehand.  

3. **Reinforcement Learning** *(optional advanced concept)*  
   - The model learns by interacting with an environment, receiving rewards or penalties for actions, and optimizing its strategy over time.  
   - Example: Training a robot to walk or an AI to play chess.

## 🔍 Types of Supervised and Unsupervised Learning  

---

### 📌 Types of Supervised Learning  
Supervised learning is based on **labeled data** and can be divided into:

1. **Regression**  
   - **Goal:** Predict continuous numeric values.  
   - **Example:** Predicting house prices, temperature forecasting.  
   - **Algorithms:** Linear Regression, Ridge/Lasso Regression, Decision Tree Regressor, Random Forest Regressor, SVR.

2. **Classification**  
   - **Goal:** Predict discrete categories or classes.  
   - **Example:** Classifying emails as spam or not spam, diagnosing a disease as positive or negative.  
   - **Algorithms:** Logistic Regression, Naive Bayes, KNN, Decision Tree Classifier, Random Forest Classifier, SVM.

---

### 📌 Types of Unsupervised Learning  
Unsupervised learning is based on **unlabeled data** and can be divided into:

1. **Clustering**  
   - **Goal:** Group similar data points together based on patterns in features.  
   - **Example:** Customer segmentation, grouping similar news articles.  
   - **Algorithms:** K-Means, DBSCAN, Hierarchical Clustering.

2. **Dimensionality Reduction**  
   - **Goal:** Reduce the number of features while preserving essential information.  
   - **Example:** Data visualization, speeding up training for large datasets.  
   - **Algorithms:** PCA, t-SNE, SVD.

--- 

## 🚀 Here Come the Algorithms  

Before diving into the **practical implementations**, let’s look at the main algorithms used in **Supervised Learning** and **Unsupervised Learning**.

---

### 📌 Supervised Learning Algorithms  
Supervised learning works with **labeled data** and is further divided into two main types:  

#### **1. Regression Algorithms** *(predict continuous values)*  
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regression  
- Random Forest Regression  
- Support Vector Regression (SVR)  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- AdaBoost Regressor  

#### **2. Classification Algorithms** *(predict discrete categories)*  
- Logistic Regression  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Gradient Boosting Classifier  
- XGBoost Classifier  
- AdaBoost Classifier  

---

### 📌 Unsupervised Learning Algorithms  
Unsupervised learning works with **unlabeled data**, focusing on pattern discovery and grouping.

#### **1. Clustering Algorithms**  
- K-Means Clustering  
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)  
- Hierarchical Clustering  

#### **2. Cluster Evaluation Metrics**  
- Silhouette Score
## 📈 Linear Regression  

**Definition:**  
Linear Regression is a supervised learning algorithm used for **predicting continuous values** by finding the **best-fit line** through the data points.

**Main Idea:**  
- The **best-fit line** is the one where the difference between the **actual data points** and the **predicted points** is as small as possible.  
- This difference is called the **error** or **residual**.  
- In Linear Regression, we aim to **minimize the sum of squared errors** (Least Squares Method).  

**Example:**  
Predicting house prices based on size, number of rooms, and location.

**Key Formula:**  
\[
y = mX + c
\]  
Where:  
- \(y\) = predicted value  
- \(m\) = slope (coefficient)  
- \(X\) = input feature  
- \(c\) = intercept (bias term)  
### 📌 Cost Function in Linear Regression  

To find the **best-fit line**, the algorithm adjusts the slope (\(m\)) and intercept (\(c\)) repeatedly over many iterations.  
Instead of guessing randomly, we use a **Cost Function** to measure how far our predictions are from the actual values.

**Definition:**  
The **Cost Function** tells us *how wrong* our model’s predictions are. In Linear Regression, we typically use the **Mean Squared Error (MSE)**:

### 📌 Cost Function in Linear Regression  

To determine the **best-fit line**, we need to measure how far our predicted values are from the actual values.  
This is done using the **Cost Function**, also called the **Mean Squared Error (MSE) Cost Function**.

**Formula:**  

\[
J(θ) = 1/2m *sum_{i=1}^{m} (h(θ)(x^{(i)}) - y^{(i)})^2
\]

Where:  
- (J(θ)) = cost (error) for parameters (θ) (slope and intercept)  
- (m) = number of training examples  
- (h_{θ}(x^{(i)})) = predicted value for the (i)-th input  
- (y^{(i)}) = actual value for the (i)-th input  

**Goal:**  
- Minimize (J(θ)) so that predictions are as close as possible to the actual values.  
- Optimization algorithms like **Gradient Descent** are used to update parameters (θ) and achieve the minimum cost.

It is challenging to minimize the cost function because it often requires many iterations.  
In the process of optimization, the parameters move step-by-step from a high point on the graph (global maxima or local maxima) toward the lowest point (global minima).  
This gradual process leads us to the next important concept — **Convergence**.

**Why Do We Need Convergence?**  
When training a model using Gradient Descent, the parameters (θ) are updated over **many iterations**.  
If we keep updating forever, the process will waste time and computational resources without improving the model.  
We need a stopping point — a condition where we say:  
> “The cost function is no longer decreasing enough to make a meaningful difference.”

This stopping point is called **Convergence**.  
By detecting convergence, we:
- Save computation time.  
- Avoid overfitting by not over-optimizing.  
- Ensure the model has reached its **best possible parameters** for the given data.

### 📌 Convergence in Gradient Descent  

After defining the **Cost Function**, we need a way to **minimize it**.  
We use an optimization technique called **Gradient Descent**, where the parameters (θ) are updated step-by-step to move from a **higher cost** (global maxima or local maxima) towards the **lowest possible cost** (global minima).

This process of updating parameters until the cost stops changing significantly is called **Convergence**.

**Parameter Update Rule (Gradient Descent Formula):**  

\[
θj := θj - α \frac{\partial J(\θ)}{\partial \θj}
\]

Where:  
- \(\theta_j\) = parameter (like slope \(m\) or intercept \(c\)) being updated  
- \(\alpha\) = learning rate (controls the step size in each iteration)  
- \(\frac{\partial J(\theta)}{\partial \theta_j}\) = partial derivative of the cost function with respect to \(\theta_j\) (gradient)  

**Convergence Condition:**  
We stop updating parameters when:

\[
| J(\theta)_{\text{previous}} - J(\theta)_{\text{current}} | < \epsilon
\]

Where:  
- α = it is the learning rate where it is should be the small to get the small small steps and it leads to get the global minima easily

**Goal:**  
- Start from an initial guess of parameters.  
- Take steps in the direction where the cost decreases the fastest (negative gradient).  
- Stop when further updates no longer improve the model significantly → this point is **convergence**.

### 📊 Performance Metrics for Linear Regression  

To evaluate how well our Linear Regression model fits the data, we use the following metrics:  

---

#### 1️⃣ Coefficient of Determination (\(R^2\))  

**Definition:**  
\(R^2\) measures the proportion of the variance in the dependent variable (\(y\)) that is predictable from the independent variables (\(X\)).  

**Formula:**  
\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\]  

Where:  
- \(y_i\) = actual value  
- \(\hat{y}_i\) = predicted value  
- \(\bar{y}\) = mean of actual values  

**Interpretation:**  
- \(R^2 = 1\) → Perfect fit.  
- \(R^2 = 0\) → Model does no better than predicting the mean.  
- Negative \(R^2\) → Model is worse than predicting the mean.

---

#### 2️⃣ Adjusted \(R^2\)  

**Definition:**  
Adjusted \(R^2\) modifies \(R^2\) to account for the number of predictors in the model. It prevents overestimation of performance when adding more features.  

**Formula:**  
\[
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k - 1} \right)
\]  

Where:  
- \(n\) = number of data points  
- \(k\) = number of independent variables (features)  

**Why Use It?**  
- \(R^2\) always increases when more predictors are added, even if they’re irrelevant.  
- Adjusted \(R^2\) increases only if the new predictor improves the model more than expected by chance.

**Example – Why We Use Adjusted \(R^2\):**  

Imagine we are predicting the **price of a room**.  
- Initially, we use **room size** as the only feature → we get a certain \(R^2\) score.  
- Then, we add **location** as a feature → since location strongly affects price, \(R^2\) increases meaningfully.  
- Next, we add **gender of the occupant** as a feature → even though it has **no actual relationship** to room price, \(R^2\) will still increase slightly (because \(R^2\) never decreases when adding more features).  

This is misleading because irrelevant features should not improve the model.  
**Adjusted \(R^2\)** solves this problem by penalizing the addition of features that don’t improve the model’s predictive power.

---

## 📌 Ridge Regression-(L2 Regularization)

### 🔹 Why We Use Ridge Regression  
- **Overfitting Prevention:** In ordinary linear regression, the model can fit too closely to training data, especially when there are many features. Ridge reduces coefficient sizes, making the model simpler and more generalizable.  
- **Handles Multicollinearity:** When features are correlated, normal regression coefficients can become unstable. Ridge stabilizes them.  
- **Better Predictions with Many Features:** Works well even when the number of features is greater than the number of data points.


### 🔹 When Does Overfitting Occur?  
Overfitting happens when a model **learns the noise** in the training data instead of just the underlying patterns.  
This usually occurs when:  
- The model is **too complex** (too many features, high-degree polynomials).  
- There is **too little training data** compared to the number of features.  
- The model trains for **too many iterations** and fits every small fluctuation in the data.  

In overfitting, the model performs **very well on training data** but **poorly on unseen/test data** because it fails to generalize.

---

**Definition:**  
Ridge Regression is a type of **regularized linear regression** that adds a penalty term to the cost function to prevent overfitting.  
It is particularly useful when we have **multicollinearity** (high correlation between independent variables) or when the model has too many features.  

---

### 🔹 Cost Function for Ridge Regression  
The Ridge cost function modifies the **Linear Regression cost** by adding an \(L2\) penalty term:  

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} \theta_j^2
\]  

Where:  
- \(m\) = number of training examples  
- \(n\) = number of features  
- \(\theta_j\) = model coefficients (parameters)  
- \(\lambda\) = regularization parameter (controls penalty strength)  
- Larger \(\lambda\) → stronger penalty → smaller coefficients → less complex model  

---

### 🔹 Key Points:  
- **Reduces overfitting** by shrinking large coefficients.  
- Works well when features are correlated.  
- Does **not** eliminate coefficients completely (unlike Lasso Regression).  
- Best used when all features are potentially useful but need to be reduced in influence.  

---
### 🔹 To Avoid Overfitting  
- The slope of the regression line should not be **too steep**, otherwise the model will likely overfit.  
- To control this, Ridge Regression adds a unique parameter called **lambda**
- **Lambda** is a hyperparameter that determines how strongly we penalize large coefficients.  
- A higher lambda value reduces the steepness of the slope more aggressively, helping the model generalize better to unseen data.


**Example Use Case:**  
Predicting house prices where multiple features like square footage, number of bedrooms, and lot size may be correlated. Ridge helps stabilize the model.


## 📌 Lasso Regression – (L1 Regularization)

### 🔹 Why We Use Lasso Regression  
- **Feature Selection:** Lasso can shrink some coefficients to **exactly zero**, effectively removing those features from the model. This helps in simplifying models and selecting only the most important predictors.  
- **Overfitting Prevention:** By reducing the impact of less important features, Lasso helps prevent overfitting and improves model generalization.  
- **Interpretability:** Models become easier to interpret when irrelevant features are eliminated.  

---

### 🔹 When Does Overfitting Occur?  
Overfitting happens when a model **memorizes noise** in the training dataset instead of learning the underlying relationships.  
This typically happens when:  
- The model is **too complex** (too many features, too much flexibility).  
- The dataset is **small** compared to the number of features.  
- The model is trained for **too many epochs/iterations** and adapts to random fluctuations.  

The result is a model that performs **extremely well on training data** but **fails on new, unseen data**.

---

**Definition:**  
Lasso Regression is a form of **regularized linear regression** that adds an \(L1\) penalty to the cost function.  
Unlike Ridge, Lasso can **completely eliminate some coefficients**, performing both regularization and feature selection.  

---

### 🔹 Cost Function for Lasso Regression  
The Lasso cost function modifies the **Linear Regression cost** by adding an \(L1\) penalty term:  

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} |\theta_j|
\]  

Where:  
- \(m\) = number of training examples  
- \(n\) = number of features  
- \(\theta_j\) = model coefficients (parameters)  
- \(\lambda\) = regularization parameter controlling penalty strength  
- Larger \(\lambda\) → more coefficients become zero → simpler model  

---

### 🔹 Key Points:  
- **Performs feature selection** by setting some coefficients to zero.  
- Helps in reducing model complexity and avoiding overfitting.  
- Best used when we suspect that **many features are irrelevant**.  
- May perform worse than Ridge if **all features are important** because it can eliminate useful ones.  

---

### 🔹 To Avoid Overfitting  
- Overfitting often occurs when too many irrelevant features influence predictions.  
- Lasso's \(L1\) penalty removes these unnecessary features automatically.  
- The **lambda** hyperparameter controls the amount of shrinkage—larger values mean more aggressive feature elimination.  

---

**Example Use Case:**  
Predicting house prices where many property features are available, but only a few (like location, area, and number of rooms) are truly important. Lasso automatically removes less important features like wall color or garden size.

### 🔹 Ridge vs Lasso – Key Difference  
- **Ridge Regression (L2 Regularization):** Adds the **square** of the coefficients to the cost function, multiplied by the regularization parameter lambda. Primarily used to **prevent overfitting** by shrinking coefficients, but it does not eliminate features entirely.  
- **Lasso Regression (L1 Regularization):** Adds the **absolute value** (modulus) of the coefficients to the cost function, multiplied by lambda. Used to **prevent overfitting** and also perform **feature selection** by reducing some coefficients to exactly zero.  

