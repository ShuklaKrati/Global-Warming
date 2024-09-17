import os
secret_key = os.urandom(24)
print(secret_key)

from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_mail import Mail, Message
import pickle
import pandas as pd
import re
import numpy as np

app = Flask(__name__)
app.secret_key =b'\xd4\xb8\xa4\xf8B\x9f\n\x1f\x10\xfe\xfe\xb6\x8c\xf7\xf7{\xe3\xd2\xd5$\x91\xb7}$'

# Load the ARIMAX model
with open('arimax_model.pkl', 'rb') as pkl_file:
    arimax_model = pickle.load(pkl_file)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Colab link route
@app.route('/colab')
def colab():
    return render_template('Colab Link.html')

# Team route
@app.route('/team')
def team():
    return render_template('Team.html')


# Set up Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'drishtichakarvarty@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'jnaf njvo zhys hslv'  # Replace with your email password
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# Update the contact route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Get form data
        first_name = request.form.get('first_name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        subject = request.form.get('subject')
        message = request.form.get('message')

        # Handle form validation
        if not first_name or not email or not message:
            flash('Please fill in all required fields', 'error')
            return redirect(url_for('contact'))

        # Send email
        msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=[app.config['MAIL_USERNAME'], email])
        msg.body = f"From: {first_name} ({email}, {phone})\n\nMessage:\n{message}"
        mail.send(msg)


        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('contact'))

    return render_template('contact.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from the form
        co2 = float(request.form['co2'])
        urban = float(request.form['urban'])

        # Create DataFrame for exogenous variables
        exog_vars = pd.DataFrame([[co2, urban]], columns=['co2', 'urban'])

        # Make prediction using ARIMAX model
        prediction = arimax_model.forecast(steps=1, exog=exog_vars)

        # Debugging: Print the prediction output
        print("Prediction output:", prediction)

        # Access the predicted temperature
        if isinstance(prediction, np.ndarray):
            predicted_temp = prediction[0]
        elif isinstance(prediction, pd.Series) or isinstance(prediction, pd.DataFrame):
            predicted_temp = prediction.iloc[0]  # Adjust for Series or DataFrame
        else:
            raise ValueError("Unexpected prediction type")

        # Average Earth surface temperature
        average_temp = 15.0  # Current average temperature in °C

        # Determine message and color based on prediction
        if predicted_temp > average_temp:
            message = "Warning: Predicted temperature is above the average temperature!"
            color = "red"
            tips = "To mitigate global warming, consider reducing CO2 emissions by using renewable energy sources, improving energy efficiency, and promoting public transportation. Additionally, initiatives to control population growth can contribute positively."
        elif predicted_temp < average_temp:
            message = "Note: Predicted temperature is below the average temperature."
            color = "blue"
            tips = "While lower temperatures are beneficial, it’s essential to maintain a sustainable environment. Promoting reforestation and reducing pollution will help stabilize temperatures."
        else:
            message = "Predicted temperature is equal to the average temperature."
            color = "green"
            tips = "Maintaining the current temperature is crucial. Continuing efforts in reducing carbon footprints, conserving energy, and protecting natural ecosystems are vital."

        # Return the prediction to the HTML template
        return render_template('model.html', 
                               prediction_text=f'Predicted Temperature: {predicted_temp:.2f}°C',
                               message=message,
                               color=color,
                               tips=tips)
    
    # If GET request, just render the model input form
    return render_template('model.html')

if __name__ == '__main__':
    app.run(debug=True)
