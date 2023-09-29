# Neural Network Diabetes Prediction
Students of Computational Intelligence (CSC3034) at Sunway University were tasked with finding a dataset from [Kaggle](https://www.kaggle.com/) and implement a Neural Network System based on the dataset chosen.

***

### Chosen dataset

The dataset chosen for this assignment was the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
The neural network will utilize this dataset to predict whether a patient is diabetic based on:
* Pregnancies – Number of pregnancies the
patient has had
* Glucose – Plasma glucose concentration
from 2 hours in oral glucose tolerance test
* BloodPressure – Diastolic blood pressure
(mm Hg)
* SkinThickness – Thickness of the patient’s
triceps skin folds (mm)
* Insulin – 2 hours serum insulin (mu U/ml)
* BMI – Body Mass Index measured from
weight(kg) / height(m)2
* Age – Age of the patient in years
* DiabetesPedigreeFunction – Risk of diabetes
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

Read the full report [here](https://github.com/Nonentity5565/Neural-Network-Diabetes-Prediction/blob/main/report.pdf)

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
