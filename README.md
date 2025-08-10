# 🧠 Machine Learning

This repository is a comprehensive showcase of my machine learning work, featuring **end-to-end projects** that start from raw, messy datasets and progress through **data cleaning, feature engineering, algorithm selection** (with clear reasoning for each choice), **model training**, and **evaluation using industry-standard metrics**.
---

**Note:** The mathematical formulas in this document are based on standard definitions from academic and open-source references.  
For a deeper understanding, please refer to detailed explanations in the linked resources or official documentation.


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
- (y) = predicted value  
- (m) = slope (coefficient)  
- (X) = input feature  
- (c) = intercept (bias term)  
### 📌 Cost Function in Linear Regression  

To find the **best-fit line**, the algorithm adjusts the slope (\(m\)) and intercept (\(c\)) repeatedly over many iterations.  
Instead of guessing randomly, we use a **Cost Function** to measure how far our predictions are from the actual values.

**Definition:**  
The **Cost Function** tells us *how wrong* our model’s predictions are. In Linear Regression, we typically use the **Mean Squared Error (MSE)**:

### 📌 Cost Function in Linear Regression  

To determine the **best-fit line**, we need to measure how far our predicted values are from the actual values.  
This is done using the **Cost Function**, also called the **Mean Squared Error (MSE) Cost Function**.

**Formula:**  
J(θ) = (1 / 2m) * Σᵢ₌₁ᵐ [ hθ(xᵢ) - yᵢ ]²

Where:  
- (J(θ)) = cost (error) for parameters (θ) (slope and intercept)  
- (m) = number of training examples  
- hθ(xᵢ) = predicted value for the (i)-th input  
- (yᵢ) = actual value for the (i)-th input  

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

θⱼ := θⱼ - α * ( ∂J(θ) / ∂θⱼ )

Where:
- θⱼ = parameter (like slope `m` or intercept `c`) being updated
- α = learning rate (controls the step size in each iteration)
- ∂J(θ) / ∂θⱼ = partial derivative of the cost function with respect to θⱼ (gradient)

Convergence Condition:
We stop updating parameters when:
| J(θ)_previous - J(θ)_current | < ε


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
R² measures the proportion of the variance in the dependent variable (y) that is predictable from the independent variables (X).

Formula:
R² = 1 - [ Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)² ]

Where:
- yᵢ = actual value
- ŷᵢ = predicted value
- ȳ = mean of actual values

Interpretation:
- R² = 1 → Perfect fit.
- R² = 0 → Model does no better than predicting the mean.
- Negative R² → Model is worse than predicting the mean.

---

**Adjusted R²:**
Definition:
Adjusted R² modifies R² to account for the number of predictors in the model. It prevents overestimation of performance when adding more features.

Formula:
Adjusted R² = 1 - [ (1 - R²) * (n - 1) / (n - k - 1) ]

Where:
- n = number of data points
- k = number of independent variables (features)

**Why Use It?**
- R² always increases when more predictors are added, even if they’re irrelevant.
- Adjusted R² increases only if the new predictor improves the model more than expected by chance.

**Example – Why We Use Adjusted R²:**
Suppose we are predicting room prices. Initially, we use 'size of the room' as a feature. If we add 'location', the price prediction improves — good! But if we then add an irrelevant feature like 'gender of the homeowner', R² will still go up slightly, even though it has nothing to do with the price. Adjusted R² helps detect such useless features by penalizing unnecessary complexity.


Imagine we are predicting the **price of a room**.  
- Initially, we use **room size** as the only feature → we get a certain \(R^2\) score.  
- Then, we add **location** as a feature → since location strongly affects price, \(R^2\) increases meaningfully.  
- Next, we add **gender of the occupant** as a feature → even though it has **no actual relationship** to room price, \(R^2\) will still increase slightly (because \(R^2\) never decreases when adding more features).  

This is misleading because irrelevant features should not improve the model.  
**Adjusted \(R^2\)** solves this problem by penalizing the addition of features that don’t improve the model’s predictive power.

## 📌 Assumptions of Linear Regression  

When using Linear Regression, certain assumptions must be met for the model to produce reliable and accurate results.  

### 1. **Linearity**  
- The relationship between the independent variables (features) and the dependent variable (target) should be **linear**.  
- If the relationship is non-linear, the model will not fit well, and predictions may be inaccurate.  

### 2. **Independence of Errors**  
- The residuals (errors) should be independent of each other.  
- In time series data, this means there should be no autocorrelation between errors.  

### 3. **Homoscedasticity**  
- The variance of residuals should be **constant** across all levels of the independent variables.  
- If the spread of residuals increases or decreases with the predicted values, the model violates this assumption.  

### 4. **Normality of Residuals**  
- The residuals should be **approximately normally distributed**.  
- This is important for hypothesis testing and calculating confidence intervals.  

### 5. **No Multicollinearity**  
- Independent variables should not be highly correlated with each other.  
- High multicollinearity can make coefficient estimates unstable and inflate standard errors.  

### 6. **No Measurement Errors**  
- The variables should be measured


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

J(θ) = (1 / 2m) * Σ ( hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾ )² + λ * Σ ( θⱼ² )

Where:
- m = number of training examples
- n = number of features
- θⱼ = model coefficients (parameters)
- λ = regularization parameter (controls penalty strength)
- Larger λ → stronger penalty → smaller coefficients → less complex model


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

J(θ) = (1 / 2m) * Σ ( hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾ )² + λ * Σ |θⱼ|

Where:
- m = number of training examples
- n = number of features
- θⱼ = model coefficients (parameters)
- λ = regularization parameter controlling penalty strength
- Larger λ → more coefficients become zero → simpler model


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

## 📌 Logistic Regression

### 🔹 Why We Use Logistic Regression Instead of Linear Regression  
- **Linear Regression** works well for continuous outputs but fails for classification problems because:
  1. Predictions can go beyond the range [0, 1], which doesn’t make sense for probabilities.
  2. The relationship between the features and the probability is **non-linear**, but Linear Regression assumes linearity.
  3. Linear Regression cost function for classification problems is **non-convex**, which can trap optimization algorithms in local minima.

- **Logistic Regression** solves these issues by:
  - Mapping outputs to a **probability range (0 to 1)** using the **Sigmoid Function**.
  - Producing a **convex cost function**, making optimization easier and more reliable.

---

### 🔹 The Sigmoid Function (Logistic Function)  
The Sigmoid function transforms any real number into a range between 0 and 1:

hθ(x) = 1 / (1 + e^(-θᵀx))

Where:
- hθ(x) = predicted probability that y = 1
- θ = model parameters
- x = input features
- e = Euler’s number (~2.718)


The output can be interpreted as:
- Close to **1** → strong likelihood of belonging to class 1  
- Close to **0** → strong likelihood of belonging to class 0  

---

### 🔹 Logistic Regression Cost Function – Why Not Use the Squared Error?  
If we used the **Mean Squared Error (MSE)** for classification:

J(θ) = (1 / 2m) * Σ ( hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾ )²

This produces a non-convex cost function for classification, making gradient descent unreliable.


### 🔹 Logistic Regression Cost Function (Log Loss)  
Instead, we use **Log Loss**, which is convex:

J(θ) = - (1 / m) * Σ [ y⁽ⁱ⁾ * log(hθ(x⁽ⁱ⁾)) + (1 - y⁽ⁱ⁾) * log(1 - hθ(x⁽ⁱ⁾)) ]

Where:
- If y = 1 → only the first term matters.
- If y = 0 → only the second term matters.
- Log Loss penalizes confident but wrong predictions heavily.

---

### 🔹 Performance Evaluation – Confusion Matrix  
The **Confusion Matrix** is a table that helps evaluate the performance of a classification model by comparing predicted vs. actual values.

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| **Actual Positive** | True Positive (TP)   | False Negative (FN)  |
| **Actual Negative** | False Positive (FP)  | True Negative (TN)   |

#### Meaning:
- **TP (True Positive):** Model correctly predicts positive class.  
- **TN (True Negative):** Model correctly predicts negative class.  
- **FP (False Positive):** Model predicts positive when it’s actually negative (Type I Error).  
- **FN (False Negative):** Model predicts negative when it’s actually positive (Type II Error).

---

### 🔹 Metrics Derived from Confusion Matrix:
1. **Accuracy** – Overall correctness of the model:  
Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. **Precision** – Of all predicted positives, how many are correct:  
Precision = TP / (TP + FP)

3. **Recall (Sensitivity)** – Of all actual positives, how many did we correctly predict:  
Recall = TP / (TP + FN)

4. **F1 Score** – Harmonic mean of Precision and Recall:  
F1 = 2 * (Precision * Recall) / (Precision + Recall)

## 📌 Naive Bayes

### 🔹 First, What’s Bayes’ Theorem?
Before talking about Naive Bayes, we need to know Bayes’ Theorem.  
It’s basically a formula that helps us figure out the probability of something happening, based on some other thing we already know.

**Simple Example:**  
If it’s cloudy outside, there’s a higher chance it might rain.  
Bayes’ Theorem is a way of calculating that chance using math.

**Formula:**
P(A|B) = [ P(B|A) * P(A) ] / P(B)  

Where:  
- **P(A|B)** → Probability of A happening given that B is true.  
- **P(B|A)** → Probability of B happening given that A is true.  
- **P(A)** → Probability of A happening (overall).  
- **P(B)** → Probability of B happening (overall).  

---

### 🔹 Why “Naive” Bayes?
The “Naive” part comes from the assumption it makes — it assumes that **all the features are independent** from each other.  
In real life, this is not always true, but surprisingly, this simple assumption works really well in many cases.

---

### 🔹 Why Do We Use Naive Bayes?
- It’s **super fast** to train compared to other algorithms.  
- Works great for **text classification** problems like spam detection, sentiment analysis, etc.  
- Even with small amounts of training data, it can still give good results.  

---

**Example Use Case:**  
If we want to classify whether an email is spam or not, we can look at the words inside the email.  
Naive Bayes will check the probability of “spam” given the words, and probability of “not spam” given the words, and choose whichever is higher.

## 📌 K-Nearest Neighbors (KNN)

### 🔹 What is KNN?
KNN stands for **K-Nearest Neighbors**.  
It’s one of the simplest algorithms out there — it doesn’t try to “learn” any complicated rules.  
Instead, when you give it something new to predict, it just looks at the **K closest data points** from the training data and decides based on them.

Think of it like asking your neighbors for advice — if most of your neighbors think something is true, you’ll probably go with that answer.

---

### 🔹 Why Do We Use KNN?
- **Easy to understand** – no heavy math during training.  
- **Versatile** – works for both classification (labels) and regression (numbers).  
- **No training phase** – it’s called a “lazy learner” because it waits until it actually needs to make a prediction before doing the work.  

---

### 🔹 KNN for Classification
When predicting a **category** (like spam/not spam, pass/fail, disease/no disease):  
1. Pick a value for **K** (number of neighbors to check).  
2. Find the K closest points to the new data point.  
3. See which category most of those points belong to.  
4. Assign that category to the new data point.

#### 📏 Distance Measures in Classification
Since KNN relies on **closeness**, we need a way to measure distance between points.  
The most common ones are:

1. **Euclidean Distance** – The straight-line distance between two points.  
   Formula:  
   d = sqrt( (x1 - y1)² + (x2 - y2)² + ... + (xn - yn)² )  
   Good when your features are continuous (like height, weight, temperature).

2. **Manhattan Distance** – The distance you’d travel if you could only move along a grid (like streets in a city).  
   Formula:  
   d = |x1 - y1| + |x2 - y2| + ... + |xn - yn|  
   Useful when movement is more “step-like” or when features are not continuous.


**Example:**  
If 3 out of your 5 nearest neighbors say “cat” and 2 say “dog,” the prediction will be “cat.”

---

### 🔹 KNN for Regression
When predicting a **number** (like house price, temperature, or score):  
1. Find the K closest points to the new data point.  
2. Take the **average** (or weighted average) of their values.  
3. Use that as the prediction.

**Example:**  
If your 3 nearest neighbors have house prices $200k, $220k, and $210k, the prediction will be about $210k.

---

### 🔹 Things to Keep in Mind
- Choosing **K** is important — too small and it can be noisy, too big and it might miss important details.  
- KNN can be **slow for large datasets** because it needs to look at all points every time.  
- Works best when data is scaled (because distance matters a lot).  

---

**Example Use Cases:**  
- **Classification:** Handwritten digit recognition, spam filtering, recommendation systems.  
- **Regression:** Predicting house prices, weather forecasting.

**Disadvantage**
- It works bad when there is a outlier and imbalanced Data
