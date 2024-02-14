# Import necessary modules
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# Define PredictPipeline class
class PredictPipeline:
    def __init__(self):
        pass  # No initialization steps defined

    def predict(self, features):
        try:
            # Define paths for model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")  #sys.argv[2]
            
            # Load trained model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform input features using preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Make predictions using the model
            preds = model.predict(data_scaled)
            
            # Return predictions
            return preds
        
        except Exception as e:
            # If an exception occurs, raise a custom exception with information
            raise CustomException(e, sys)

# Define CustomData class
class CustomData:
    def __init__(self, 
                 gender: str, 
                 race_ethnicity: str, 
                 parental_level_of_education, 
                 lunch: str, 
                 test_preparation_course: str, 
                 reading_score: int, 
                 writing_score: int):
        # Initialize object with input data attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            # Construct dictionary with input data attributes
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            # Create DataFrame from the dictionary
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            # If an exception occurs, raise a custom exception with information
            raise CustomException(e, sys)
