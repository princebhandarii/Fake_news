import joblib

svm_model = joblib.load("svm_model.pkl")
pa_model = joblib.load("pa_model.pkl")
nb_model = joblib.load("nb_model.pkl")
rf_model = joblib.load("rf_model.pkl")  
vectorizer = joblib.load("tfidf_vectorizer.pkl")

real_text = """
The Reserve Bank of India kept its benchmark interest rate 
unchanged at its latest policy meeting, citing stable inflation
and steady economic growth. Officials stated that future rate
decisions will depend on incoming data and global market conditions.
"""

text_vec = vectorizer.transform([real_text])

svm_pred = svm_model.predict(text_vec)[0]
pa_pred = pa_model.predict(text_vec)[0]
nb_pred = nb_model.predict(text_vec)[0]
rf_pred = rf_model.predict(text_vec)[0]  

print("SVM Prediction:", svm_pred)
print("Passive Aggressive Prediction:", pa_pred)
print("Naive Bayes Prediction:", nb_pred)
print("Random Forest Prediction:", rf_pred)   # <-- added