# Neural Network Diabetes Prediction
## Computational Intelligence Assignment 2
Students are tasked with finding a dataset from [Kaggle](https://www.kaggle.com/) and implement a Neural Network System based on the dataset chosen.

***

### Chosen dataset

The dataset chosen for this assignment was the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
The neural network will utilize this dataset to predict whether a patient is diabetic based on:
* Pregnancies – Number of pregnancies the
patient has had
* Glucose – Plasma glucose concentration
from 2 hours in oral glucose tolerance test
* Glucose – Plasma glucose concentration
 BloodPressure – Diastolic blood pressure
(mm Hg)
* Glucose – Plasma glucose concentration
 SkinThickness – Thickness of the patient’s
triceps skin folds (mm)
* Glucose – Plasma glucose concentration
 Insulin – 2 hours serum insulin (mu U/ml)
* Glucose – Plasma glucose concentration
 BMI – Body Mass Index measured from
weight(kg) / height(m)2
* Glucose – Plasma glucose concentration
 Age – Age of the patient in years
* Glucose – Plasma glucose concentration
 DiabetesPedigreeFunction – Risk of diabetes
based on family history

---

### Implementation

The assignment was completed by implemented Multi-Layer Perceptron (MLP) Neural Network using Python and the following libraries:
* Pandas
* Matplotlib
* NumPy
* Scikit-learn

#### Test Cases for Optimization
1. Activation Function and Solver combination
2. Number of Hidden Layers and Neurons
3. Number of Training Iteration

---

### Results
The final setting for the MLP Neural Network are as such:
* Activation function - Identity Function
* Solver - Limited-memory BFGS solver
* Hidden layers and Neurons - 4 Hidden Layer with 6 Neurons in each layer

After running 200 randomized test cases, the results of the prediction system are as follow:
#### Mean Accuracy = 77.45%
| |Actual Non-Diabetic| Actual Diabetic|
---|---|---
|Predicted Non-Diabetic|17715|2379|
|Predicted Diabetic|4567|6139|

#### Possible Reasoning for Low Accuracy
Dataset chosen has biasness towards non-diabetic patients which consisted of 65% of the dataset.
