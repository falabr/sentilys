Sentilys  
Sentilys is a web-based sentiment analysis application designed to assist researchers and developers in analyzing text sentiment patterns from user reviews, particularly from Google Play Store applications. It leverages advanced algorithms and trained models to transform raw data into actionable insights through Natural Language Processing (NLP) and Machine Learning techniques.

Features
* Sentiment Classification: Classifies user reviews into positive or negative sentiments using a pre-trained SVM model.
* Text Preprocessing: Includes extensive text normalization, tokenization, and stemming using Sastrawi Stemmer for Indonesian language support.
* Data Visualization: Generates pie charts to display sentiment distributions and word clouds to visualize frequently used words in reviews.
* Data Export: Download the results in CSV or Excel formats for further analysis.
Installation
To set up and run the Sentilys application locally, follow these steps:

Prerequisites
1. Python 3.x installed on your machine.
2. Virtual environment (optional but recommended).
3. Required libraries listed in the requirements.txt file.

Steps
1. Clone the repository:
- git clone https://github.com/yourusername/sentilys.git
- cd sentilys
2. (Optional) Create and activate a virtual environment:
- python -m venv venv
- source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the dependencies:
- pip install -r requirements.txt
4. Ensure you have the trained SVM model and CountVectorizer stored in the model directory:
- svm_model.pkl
- count_vectorizer.pkl
5. Run the Flask application:
- python app.py
6. Open your web browser and go to http://127.0.0.1:5000/ to access the application.

Usage
1. On the main page, enter the Application ID of the Google Play Store app whose reviews you want to analyze.
2. Set the number of comments to scrape for analysis.
3. Optionally filter reviews by score.
4. Click Submit to start the sentiment analysis.
5. View the analysis results on the results page, including sentiment distribution and word clouds.
6. Download the analyzed results in CSV or Excel formats.

Contributions are welcome! Feel free to submit a pull request or open an issue for any bugs, features, or improvements.
