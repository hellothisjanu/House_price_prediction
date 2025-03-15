### Instructions to Run the House Price Prediction Flask Project

# Clone the Repository 
   ```
   git clone https://github.com/hellothisjanu/House_price_prediction.git
   cd House-price-prediction-using-flask-main
   ```
# Install Dependencies
   ```
   pip install -r requirement.txt
   ```

# Train and Save the Model
   ```
   python house.py
   ```
   This will generate a `model.pkl` file.

# Run the Flask Application
   ```
   python app.py
   ```

# Access the Web App
   Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

# Enter House Details
   - Input features such as bedrooms, bathrooms, floors, and year built.
   - Click the Predict button.

# View Prediction Result
   - The predicted house price will be displayed.

# Debugging Issues
   - If you encounter errors, ensure all dependencies are installed.
   - Try reinstalling `scikit-learn`:
     ```
     pip install --upgrade scikit-learn
     ```

# Stop the Server
    - To stop the Flask app, press Ctrl + C in the terminal.

# This guide provides a step-by-step approach 