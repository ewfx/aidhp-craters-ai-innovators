import os
from flask import Blueprint, Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
recommendation_bp = Blueprint('recommendation', __name__)

# Selecting features
features = [
    'age', 'income', 'account_balance', 'existing_loans', 
    'marital_status', 'employment_status', 'education_level', 
    'debt_to_income_ratio', 'home_ownership', 'employment_duration', 
    'loan_purpose', 'state', 'dependents', 'annual_expenses'
]
numerical_features = ['age', 'income', 'account_balance', 'existing_loans', 
                      'debt_to_income_ratio', 'employment_duration', 'dependents', 'annual_expenses']

# Default values for missing keys
default_values = {
    "age": 0,
    "income": 0,
    "account_balance": 0,
    "existing_loans": 0,
    "marital_status": "Unknown",
    "employment_status": "Unknown",
    "education_level": "Unknown",
    "debt_to_income_ratio": 0,
    "home_ownership": "Unknown",
    "employment_duration": 0,
    "loan_purpose": "Unknown",
    "state": "Unknown",
    "dependents": 0,
    "annual_expenses": 0
}

# Load the model, scaler, and label encoders at the beginning
model = joblib.load("models/home_loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Initialize Hugging Face embeddings and Chroma vector store
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = Chroma(
    collection_name="product_catalogue",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Preprocess customer data for prediction
def preprocess_customer_data(customer_data):
    # Ensure all required keys are present in the customer data
    for key in features:
        if key not in customer_data:
            customer_data[key] = default_values[key]

    # Create a DataFrame for the customer data
    df = pd.DataFrame([customer_data], columns=features)

    # Encode categorical variables
    for col in label_encoders:
        if df[col][0] not in label_encoders[col].classes_:
            # Add "Unknown" to the encoder's classes if not already present
            label_encoders[col].classes_ = np.append(label_encoders[col].classes_, "Unknown")
        df[col] = label_encoders[col].transform([df[col][0]])  # Transform single value

    # Scale numerical data
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df

# Predict home loan interest using the preprocessed data
def predict_home_loan(customer_data):
    # Preprocess the customer data
    preprocessed_data = preprocess_customer_data(customer_data)

    # Predict
    prediction = model.predict(preprocessed_data)
    return "Likely to apply" if prediction[0] == 1 else "Unlikely to apply"

# Generate personalized recommendations using LLM with RAG
def generate_llm_recommendations(profile_data, home_loan_likelihood):
    likelihood_text = home_loan_likelihood

    # Retrieve relevant context from the vector store
    if likelihood_text == 'Likely to apply':
        query="Home Lending Products"
    else:
        query="Investment Products"
    relevant_docs = vector_store.similarity_search(query, k=1)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are a financial assistant analyzing a customer's banking profile.
    
    Customer Profile:
    - Age: {profile_data['age']}
    - Income: {profile_data['income']}
    - Annual Expenses: {profile_data['annual_expenses']}
    - Account Balance: {profile_data['account_balance']}
    - Dependents: {profile_data['dependents']}
    - Employment Status: {profile_data['employment_status']}
    - Existing Loan Status: {profile_data['existing_loans']}
    - Property Ownership: {profile_data['home_ownership']}
    - Marital Status: {profile_data['marital_status']}
    - Location: {profile_data['state']}
    
    Based on the data, the ML model predicts that the customer is **{likelihood_text}** for a home loan.
    
    Context from knowledge base:
    {context}
    
    If likely, suggest **3 suitable home loan options** based on the profile and context.
    If unlikely, suggest **3 suitable investment product options** based on the profile and context.
    Keep the answer short, crisp, and clear with necessary details.
    """
    #print(prompt)

    model_name = "llama-3.3-70b-versatile"
    llm = init_chat_model(model_name, model_provider="groq")
    response = llm.invoke(prompt)

    llm_recommendations = response.content.split("\n")
    llm_recommendations = [rec.strip() for rec in llm_recommendations if rec.strip()]
    print(llm_recommendations)
    return llm_recommendations

# API route to predict interest in home loan 
@recommendation_bp.route('/api/predict_home_loan', methods=['POST'])
def predict_home_loan_api():
    try:
        data = request.json
        # Predict home loan interest
        home_loan_likelihood = predict_home_loan(data)

        # Get personalized LLM recommendations
        llm_recommendations = generate_llm_recommendations(data, home_loan_likelihood)

        return jsonify({"home_loan_likelihood": home_loan_likelihood, "recommendations": llm_recommendations})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
app.register_blueprint(recommendation_bp)

if __name__ == '__main__':
    app.run(debug=True)