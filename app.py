import streamlit as st
import pickle
import string
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from email import policy
from email.parser import BytesParser

# ----------------- Initialize -----------------
ps = PorterStemmer()

# Ensure required NLTK data is available
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download all necessary packages
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)  # NEW FIX
nltk.download('stopwords', download_dir=nltk_data_dir)

# ----------------- Text Preprocessing -----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# ----------------- Suspicious URL / Phishing Detector -----------------
def detect_phishing(text):
    phishing_keywords = [
        "verify", "account", "login", "password", "update",
        "urgent", "click", "bank", "credit card", "ssn",
        "security alert", "limited time", "confirm"
    ]

    # Look for suspicious links
    url_pattern = r"(http|https):\/\/[^\s]+"
    urls = re.findall(url_pattern, text.lower())

    # Check for phishing keywords
    keyword_flag = any(word in text.lower() for word in phishing_keywords)

    if urls or keyword_flag:
        return 1  # Phishing detected
    return 0

# ----------------- Load Model -----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ----------------- Streamlit UI Config -----------------
st.set_page_config(page_title="Spam & Phishing Email Classifier", layout="wide")
st.title("📩 Email/SMS Spam & Phishing Classifier Dashboard")

st.sidebar.header("About This App")
st.sidebar.info("""
- Classifies messages as **Spam**, **Not Spam**, or **Phishing**.
- Upload files or test individual messages.
- Supported formats: `.csv`, `.txt`, `.eml`
""")

# ----------------- Tabs for Navigation -----------------
tabs = st.tabs(["📝 Classifier", "📁 File Upload", "📊 Demo Section"])

# ----------------- Tab 1: Single Message Classifier -----------------
with tabs[0]:
    st.subheader("💬 Test Your Message")
    input_sms = st.text_area("Enter the message here")

    if st.button('Predict Message'):
        if input_sms.strip() == "":
            st.warning("Please enter a message to classify.")
        else:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            # Check for phishing
            phishing_flag = detect_phishing(input_sms)

            if phishing_flag == 1:
                st.error("⚠️ Phishing message Detected!")
            elif result == 1:
                st.error("🚨 Spam Message Detected!")
            else:
                st.success("This message is clean (Not Spam or Phishing).")

# ----------------- Tab 2: File Upload & Analysis -----------------
with tabs[1]:
    st.subheader("📁 Upload a File (.csv, .txt, .eml)")
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'txt', 'eml'])

    def parse_eml(file):
        """Parse .eml file and return combined subject + body."""
        msg = BytesParser(policy=policy.default).parse(file)
        subject = msg['subject'] if msg['subject'] else ''
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            body = msg.get_content()
        return subject + " " + body

    if uploaded_file is not None:
        # Read file depending on type
        if uploaded_file.name.endswith('.csv'):
            df_file = pd.read_csv(uploaded_file)
            if 'Message' not in df_file.columns:
                df_file.rename(columns={df_file.columns[0]: 'Message'}, inplace=True)
        elif uploaded_file.name.endswith('.txt'):
            df_file = pd.read_csv(uploaded_file, delimiter="\n", names=["Message"])
        elif uploaded_file.name.endswith('.eml'):
            try:
                df_file = pd.DataFrame({"Message": [parse_eml(uploaded_file)]})
            except Exception as e:
                st.error(f"Error reading .eml file: {e}")
                df_file = pd.DataFrame(columns=["Message"])
        else:
            st.error("Unsupported file format!")
            df_file = pd.DataFrame(columns=["Message"])

        if not df_file.empty:
            st.success(f"Uploaded file with {len(df_file)} messages.")

            # Preprocess, Predict Spam, and Phishing
            df_file['transformed_text'] = df_file['Message'].apply(transform_text)
            vector_input = tfidf.transform(df_file['transformed_text'])
            df_file['Spam_Prediction'] = model.predict(vector_input)
            df_file['Phishing_Prediction'] = df_file['Message'].apply(detect_phishing)

            # Final Label
            def classify(row):
                if row['Phishing_Prediction'] == 1:
                    return "Phishing"
                elif row['Spam_Prediction'] == 1:
                    return "Spam"
                else:
                    return "Not Spam"

            df_file['Final_Label'] = df_file.apply(classify, axis=1)

            # Display predictions
            st.dataframe(df_file[['Message', 'Final_Label']])

            # Download CSV
            csv = df_file.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='email_predictions.csv',
                mime='text/csv'
            )

            # ----------------- Visualization -----------------
            st.subheader("📊 File Analysis")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='Final_Label', data=df_file, palette=['green', 'red', 'orange'], ax=ax)
            ax.set_title("Spam vs Phishing vs Clean Emails", fontsize=14)
            ax.set_ylabel("Count")
            ax.set_xlabel("Message Type")
            st.pyplot(fig)

            # WordClouds
            st.subheader("🌐 WordClouds")
            phishing_text = " ".join(df_file[df_file['Final_Label'] == "Phishing"]['Message'])
            spam_text = " ".join(df_file[df_file['Final_Label'] == "Spam"]['Message'])
            ham_text = " ".join(df_file[df_file['Final_Label'] == "Not Spam"]['Message'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**⚠️ Phishing WordCloud**")
                if phishing_text.strip():
                    phishing_wc = WordCloud(width=400, height=300, background_color='white', colormap='Oranges').generate(phishing_text)
                    plt.imshow(phishing_wc, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.info("No phishing messages to generate WordCloud.")

            with col2:
                st.markdown("**🚨 Spam WordCloud**")
                if spam_text.strip():
                    spam_wc = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(spam_text)
                    plt.imshow(spam_wc, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.info("No spam messages to generate WordCloud.")

            with col3:
                st.markdown("**✅ Not Spam WordCloud**")
                if ham_text.strip():
                    ham_wc = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(ham_text)
                    plt.imshow(ham_wc, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.info("No clean messages to generate WordCloud.")

# ----------------- Tab 3: Demo Section -----------------
with tabs[2]:
    st.subheader("📊 Demo Spam & Not Spam Messages")
    demo_data = {
        "Message": [
            "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
            "Hey, are we meeting today?",
            "URGENT! Your account has been suspended. Verify immediately!",
            "Don't forget to submit your assignment.",
            "Free entry in 2 a weekly competition! Text WIN to 12345.",
            "Lunch at 1 PM?"
        ],
        "Label": ["Spam", "Not Spam", "Spam", "Not Spam", "Spam", "Not Spam"]
    }
    df_demo = pd.DataFrame(demo_data)
    st.dataframe(df_demo)

    # ------------------ Metrics ------------------
    st.subheader("📈 Demo Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", len(df_demo))
    col2.metric("Spam Messages", len(df_demo[df_demo['Label'] == "Spam"]))
    col3.metric("Not Spam Messages", len(df_demo[df_demo['Label'] == "Not Spam"]))

    st.markdown("---")

    # ------------------ Visualizations ------------------
    st.subheader("📊 Visualizations")

    # Countplot of Spam vs Not Spam
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(x="Label", data=df_demo, hue="Label", palette={"Spam": "red", "Not Spam": "green"}, legend=False, ax=ax1)
    ax1.set_title("Spam vs Not Spam Messages", fontsize=14)
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Message Type")
    st.pyplot(fig1)

    # WordClouds
    st.subheader("🌐 WordClouds of Messages")
    spam_text = " ".join(df_demo[df_demo['Label'] == "Spam"]['Message'])
    ham_text = " ".join(df_demo[df_demo['Label'] == "Not Spam"]['Message'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🚨 Spam WordCloud**")
        spam_wc = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(spam_text)
        plt.imshow(spam_wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("**✅ Not Spam WordCloud**")
        ham_wc = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(ham_text)
        plt.imshow(ham_wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
        plt.clf()

    # ------------------ Additional Insights ------------------
    st.subheader("📊 Message Length Distribution")
    df_demo['num_words'] = df_demo['Message'].apply(lambda x: len(x.split()))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.histplot(data=df_demo, x='num_words', hue='Label', multiple='stack', palette={'Spam': 'red','Not Spam':'green'}, bins=10)
    ax2.set_title("Word Count Distribution by Message Type")
    ax2.set_xlabel("Number of Words")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

st.markdown("---")
st.info("Created by **Tanvika Bhumipal Padole** | ML & Cybersecurity Enthusiast")
