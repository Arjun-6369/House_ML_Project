from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv('house_prices.csv')

# Features and target
X = data[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = data['Price']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    age = int(request.form['age'])

    new_house = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age]
    })

    price = model.predict(new_house)[0]

    if price < 0:
        result = 'Invalid input (outside training range)'
    else:
        result = f'₹ {price:,.2f}'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)