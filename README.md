Emotion Detector using TF-IDF and Naive Bayes

A machine learning-based emotion detection system that classifies text into emotions such as joy, sadness, anger, fear, and surprise using natural language processing techniques.
 Project Overview:
This project focuses on building a text-based emotion classification system using supervised machine learning. The goal is to analyze a given piece of text (e.g., a tweet or a sentence) and predict the underlying emotion expressed by the user. This model is particularly useful in areas like customer sentiment analysis, social media monitoring, and mental health applications.

The system uses *TF-IDF (Term Frequency-Inverse Document Frequency)* for feature extraction and *Multinomial Naive Bayes*, a probabilistic classifier, to predict the emotion. It is trained on a real-world labeled dataset consisting of text samples mapped to emotional categories.

This project demonstrates how natural language processing and machine learning can work together to understand and interpret human emotions from written text.
Features:

- Detects emotions from raw text input
- Preprocesses text data (tokenization, lowercasing, stopword removal)
- Converts text to numeric vectors using TF-IDF
- Trains a Naive Bayes model on labeled emotion data
- Supports classification of multiple emotions (happy, sad, angry, etc.)
- Evaluates performance using accuracy and confusion matrix
Requirements:

Make sure the following libraries are installed:

- Python 3.7+
- pandas
- scikit-learn
- numpy
- nltk (for stopword removal and tokenization)

Install dependencies using:

bash
pip install -r requirements.txt
Installation:

1. Clone the Repository

bash
git clone https://github.com/yourusername/emotion-detector.git


2. Navigate to the Project Folder

bash
cd emotion-detector


3. Install Required Libraries

bash
pip install -r requirements.txt

4. Add the Dataset

Make sure the emotion-labeled dataset (e.g., emotion_dataset.csv) is added to the working directory. Update the path in the script if needed.

 Usage:

1. Open emotion_classifier.py in your code editor.
2. Run the script:

bash
python emotion_classifier.py


3. Enter any sentence when prompted.
4. The model will output the predicted emotion.
## 📷 Output Example

Input:


Text: I’m so excited about my new job!


Output:


Predicted Emotion: joy


 Troubleshooting:

* *FileNotFoundError:* Ensure the dataset is available in the directory.
* *ModuleNotFoundError:* Use pip install -r requirements.txt to fix missing modules.
* *Wrong predictions:* Check if the model was trained properly and the dataset is balanced.

 Future Enhancements:

* Expand dataset to support more nuanced or mixed emotions.
* Add support for real-time input from social media APIs (e.g., Twitter).
* Integrate into a chatbot or voice assistant.
* Build a web interface using Flask or Streamlit for live emotion detection.
* Use deep learning (e.g., LSTM, BERT) for better accuracy on complex sentences.



