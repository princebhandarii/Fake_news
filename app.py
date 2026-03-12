import streamlit as st
import joblib

# Load models
svm_model = joblib.load("svm_model.pkl")
pa_model = joblib.load("pa_model.pkl")
nb_model = joblib.load("nb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📰 Fake News Detection System")
st.info(
    "⚠️ This model was trained on historical political news datasets (ISOT & LIAR). "
    "Predictions on very recent news, non-political topics, or region-specific stories may be less reliable. "
    "Future versions will include live news API training."
)
st.write("Enter a news article below to classify it as Real or Fake.")

user_input = st.text_area("News Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vec = vectorizer.transform([user_input])

        svm_pred = svm_model.predict(text_vec)[0]
        pa_pred = pa_model.predict(text_vec)[0]
        nb_pred = nb_model.predict(text_vec)[0]
        rf_pred = rf_model.predict(text_vec)[0]

        # Majority voting manually
        predictions = [svm_pred, pa_pred, nb_pred, rf_pred]
        final_pred = max(set(predictions), key=predictions.count)

        label = "Real News ✅" if final_pred == 1 else "Fake News ❌"

        st.subheader("Final Prediction:")
        st.success(label)

        st.write("### Model Predictions:")
        st.write("SVM:", svm_pred)
        st.write("Passive Aggressive:", pa_pred)
        st.write("Naive Bayes:", nb_pred)
        st.write("Random Forest:", rf_pred)