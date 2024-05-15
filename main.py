from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('datasets/titanic.csv')
    df = df.dropna(subset=['Age'])

    df['Male'] = df['Sex'] == 'male'

    X = df[['Fare', 'SibSp', 'Pclass', 'Male', 'Age']].values
    y = df['Survived'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # model = RandomForestClassifier(random_state=0)
    model = LogisticRegression(random_state=0)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("acurácia: ", accuracy_score(y_test, y_pred))
    print("precisão: ", precision_score(y_test, y_pred))
