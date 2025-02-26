from data_import import read_file
import joblib 
# Load the data
df = read_file('insurance.csv')

# Drop the target column ('charges') and select the first row
X_test = df.drop(columns=['charges']).iloc[[0]]

# Load the saved model
model_path = './models/rf_pipeline.pkl'
loaded_model = joblib.load(model_path)

# Make prediction
prediction = loaded_model.predict(X_test)

# Display the result
print("Predicted Charges for the first row:", prediction[0])
print("Real Charges for the first row:", df['charges'][0])