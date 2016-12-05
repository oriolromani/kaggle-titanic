import pandas
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex


def get_title(name):
    # Use a regular expression to search for a title.
    # Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def processing_data(data):

    data.drop(['Embarked'], axis=1, inplace=True)

    data['Fare'] = data['Fare'].astype(int)

    average_age_titanic = data["Age"].mean()
    std_age_titanic = data["Age"].std()
    count_nan_age_titanic = data["Age"].isnull().sum()

    random_age = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,
                                   size=count_nan_age_titanic)
    data['Age'].dropna().astype(int)
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

    pclass_dummies_titanic = pandas.get_dummies(data['Pclass'])
    pclass_dummies_titanic.columns = ['Class_1', 'Class_2', 'Class_3']
    pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

    data.drop(['Pclass'], axis=1, inplace=True)
    data = data.join(pclass_dummies_titanic)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    titles = data["Name"].apply(get_title)
    for k, v in title_mapping.items():
        titles[titles == k] = v

    # Add in the title column and drop Name
    data["Title"] = titles
    data.drop(["Name"], axis=1, inplace=True)

    return data


def main():
    # Read training data
    titanic = pandas.read_csv("train.csv")
    titanic = titanic.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)

    titanic = processing_data(titanic)

    # Read test data
    titanic_test = pandas.read_csv("test.csv")
    titanic_test = titanic_test.drop(['Ticket', 'Cabin'], axis=1)
    titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

    titanic_test = processing_data(titanic_test)

    # Define test and train

    X_train = titanic.drop("Survived", axis=1)
    Y_train = titanic["Survived"]
    X_test = titanic_test.drop("PassengerId", axis=1).copy()

    svc = SVC()
    random_forest = RandomForestClassifier(n_estimators=100)
    logistic_regression = LogisticRegression()
    knn = KNeighborsClassifier()
    gaussian = GaussianNB()

    algs = [svc, random_forest, logistic_regression, knn, gaussian]
    scores = []

    for alg in algs:
        alg.fit(X_train, Y_train)
        score = alg.score(X_train, Y_train)
        print(alg, score)
        scores.append(score)

    alg = algs[np.argmax(scores)]
    print(alg)
    Y_pred = alg.predict(X_test)

    # Create a new data frame with only the columns Kaggle wants from the dataset.
    submission = pandas.DataFrame({
            "PassengerId": titanic_test["PassengerId"],
            "Survived": Y_pred
        })
    submission.to_csv("titanic.csv", index=False)

if __name__ == "__main__":
    main()
