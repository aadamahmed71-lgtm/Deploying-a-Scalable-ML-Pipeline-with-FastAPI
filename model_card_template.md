# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier from scikit-learn's `RandomForestClassifier`, trained with 100 estimators and a maximum depth of 15. It was developed as part of a Udacity project to demonstrate building, testing, and deploying a machine learning pipeline with a REST API. The model was trained by Aadam Ahmed using scikit-learn version 1.5.1 and Python 3.10.

## Intended Use
This model is intended to predict whether an individual's annual income exceeds $50,000 based on U.S. Census demographic and employment data. It is intended for educational purposes, to demonstrate an end-to-end ML pipeline with model training, testing, and deployment via a FastAPI REST API. It is not intended for use in real-world decisions about individuals, such as lending, hiring, or other consequential decisions.

## Training Data
The model was trained on the publicly available Census Income dataset (also known as "Adult" dataset), sourced from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The dataset contains approximately 32,561 rows, each representing an individual, with 14 features (both categorical and continuous) such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours worked per week, and native country. The training set consists of 80% of the data (26,048 rows), selected via a random train-test split with a fixed random seed of 42. Categorical features were one-hot encoded, and the target label ("salary": <=50K or >50K) was binarized using scikit-learn's LabelBinarizer.

## Evaluation Data
The evaluation (test) data consists of the remaining 20% of the dataset (6,513 rows), held out from the training data using the same random train-test split (random_state=42). The same trained OneHotEncoder and LabelBinarizer from the training data were applied to the test data to ensure consistent preprocessing.

## Metrics
The model was evaluated using precision, recall, and F1 score (fbeta with beta=1). On the held-out test set, the model achieved:

- **Precision:** 0.7918
- **Recall:** 0.5786
- **F1 Score:** 0.6686

Performance was also computed on slices of the data for each categorical feature (workclass, education, marital-status, occupation, relationship, race, sex, and native-country), broken out by each unique value within those features. These per-slice metrics are available in `slice_output.txt`. Performance varies notably across slices; for example, some education levels with fewer high-income examples in the test set show lower recall, reflecting class imbalance within those slices rather than a flaw in the overall model.

## Ethical Considerations
This dataset includes sensitive demographic attributes such as race, sex, and native country. Because the model uses these attributes as inputs, its predictions can reflect and potentially amplify historical biases present in the underlying census data (e.g., systemic disparities in income by race or sex). The per-slice performance metrics reveal that model performance is not uniform across all demographic groups, which is an important consideration before using this model in any real-world context. This model should not be used to make consequential decisions about real individuals, such as lending, hiring, or insurance decisions, without significant additional fairness auditing.

## Caveats and Recommendations
The dataset originates from 1994 U.S. Census data and does not reflect current economic conditions, income distributions, or demographic patterns. The $50,000 income threshold used as the classification boundary is also outdated in real-world terms. Users should not deploy this model for any purpose beyond demonstrating an ML pipeline. Future improvements could include hyperparameter tuning, addressing class imbalance across demographic slices, and evaluating fairness metrics (such as demographic parity or equalized odds) across sensitive attributes before any real-world consideration.
