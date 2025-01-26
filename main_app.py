import streamlit as st # type: ignore
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Set page title and favicon
st.set_page_config(page_title="SMS/E-mail Spam Classifier", page_icon="..\Sms-email spam classification_NEW_pkd\page_icon.png")

# Load model and vectorizer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Add title and description
st.title("📩 SMS/E-mail Spam Classifier")
st.markdown("""
    **Click "Predict" to check if it's spam or not.**
    Our model is trained to differentiate between spam and non-spam messages (*can do mistakes).
""")

# Add input area with styling
input_sms = st.text_area(" Enter the message:", height=300, max_chars=500)

# Add Predict button with better styling
if st.button('Predict', key="predict"):
    # 1. Preprocessing
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display results with better styling
    if result == 1:
        st.header("🚨 **Spam Message!** 🚨", anchor="spam")
        st.markdown("""
        This message is likely to be spam, please be cautious before responding or clicking any links.
        """)
    else:
        st.header("✅ **Not Spam**", anchor="not_spam")
        st.markdown("""
        This message appears to be real and legitimate. No actions needed.
        """)

# Optional: Add some branding or footer text
st.markdown("""
    ---  
    Developed by **Piyush Kumar Dey** | [LinkedIn](https://linkedin.com/in/piyush-kumar-dey-291b19342) | [GitHub](https://github.com/Piyushkumar030)
""")

# Add custom CSS for styling
st.markdown("""
    <style>
        .stTextArea textarea {
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: dark blue;
            
        }
        .stButton button {
            background-color: #08086BDA;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #1CA591E4;
            color: black;
        }
        .stHeader {
            color: #660303FF;
            font-weight: bold;
        }
        .stMarkdown {
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)




# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import string
# from nltk.stem.porter import PorterStemmer

# def transform_text(text):
#     text=text.lower()
#     text = nltk.word_tokenize(text)

#     y=[]
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text=y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text=y[:]
#     y.clear()

#     for i in text:
#          y.append(ps.stem(i))
    

#     return " ".join(y)

