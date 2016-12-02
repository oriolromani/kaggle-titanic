import pandas
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex


def processing_data(data):

    data.drop(['Embarked'], axis=1, inplace=True)

    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    average_age_titanic = data["Age"].mean()
    std_age_titanic = data["Age"].std()
    count_nan_age_titanic = data["Age"].isnull().sum()

    random_age = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,
                                   size=count_nan_age_titanic)

    # fill NaN values in Age column with random values generated
    data["Age"][np.isnan(data["Age"])] = random_age
    data['Age'] = data['Age'].astype(int)

    # Family
    data['Family'] = data["Parch"] + data["SibSp"]
    data['Family'].loc[data['Family'] > 0] = 1
    data['Family'].loc[data['Family'] == 0] = 0
    data = data.drop(['SibSp', 'Parch'], axis=1)

    data['Person'] = data[['Age', 'Sex']].apply(get_person, axis=1)
    data.drop(['Sex'], axis=1, inplace=True)

    person_dummies_titanic = pandas.get_dummies(data['Person'])
    person_dummies_titanic.columns = ['Child', 'Female', 'Male']
    person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

    data = data.join(person_dummies_titanic)
    data.drop(['Person'], axis=1, inplace=True)

    return data

titanic = pandas.read_csv("train.csv")
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

titanic = processing_data(titanic)

# Initialize algorithms
logistic_regression = LogisticRegression(random_state=1)
svc = SVC()
knn_classifier = KNeighborsClassifier()
random_forest = RandomForestClassifier(n_estimators=100)

algs = [svc, knn_classifier, logistic_regression, random_forest]
results = []

for alg in algs:
    # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
    scores = cross_val_score(alg, titanic, titanic["Survived"], cv=3)
    # Take the mean of the scores (because we have one for each fold)
    print(alg, "RESULT %s: " % scores.mean())
    results.append(scores.mean())

# select algorithm with best accuracy
alg = algs[np.argmax(results, axis=0)]

titanic_test = pandas.read_csv("test.csv")
titanic_test = titanic_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# change nan values in Age colum
titanic_test = processing_data(titanic_test)

# Train the algorithm using all the training data
alg.fit(titanic, titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("titanic.csv", index=False)
