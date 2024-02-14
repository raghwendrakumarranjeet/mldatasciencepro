# Import necessary modules
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create a Flask application
application = Flask(__name__)
app = application  # Assigning the Flask app to 'app'

## Route for the home page
@app.route('/')
def index():
    # Render the index.html template for the home page
    return render_template('index.html')

# Route for predicting data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If the request method is GET, render the home.html template
        return render_template('home.html')
    else:
        # If the request method is POST, process the form data and make predictions
        # Extract form data and create a CustomData object
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        # Convert the CustomData object to a DataFrame
        pred_df = data.get_data_as_dataframe()
        print(pred_df)  # Print the DataFrame (for debugging)

        # Create an instance of PredictPipeline and make predictions
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Render the home.html template with the prediction results
        return render_template('home.html', result=results[0])

# Run the Flask application
if __name__ == "__main__":
    # Run the application on host "0.0.0.0" with debug mode enabled
    app.run(host="0.0.0.0", debug=True)
