import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data


def create_model(data):
    # Dependent and Independent Features

    x = data.drop(['diagnosis'], axis=1)
    print(x)
    y = data['diagnosis']

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train Model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Test Model
    y_pred = model.predict(x_test)
    print(f"Accuracy of our model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report \n {classification_report(y_test, y_pred)}")
    return model, scaler


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



if __name__ == '__main__':
    main()