from flask import Flask, render_template, request, send_file
from google_play_scraper import Sort, reviews
import pandas as pd
import re
from joblib import load
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64
import uuid

app = Flask(__name__)

# Load model SVM and CountVectorizer
model_path = "model/svm_model.pkl"
vectorizer_path = "model/count_vectorizer.pkl"
model = load(model_path)
vectorizer = load(vectorizer_path)

# Initialize Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Custom stopwords for word cloud
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(["yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan", "adalah", "sebagai", "saya", "kamu", "dia", "mereka", "kita", "kami", "aplikasi",
                         "harus", "buat", "guna", "nya", "namun", "sudah", "lagi", "bisa", "tidak", "baik", "malah", "ada", "jadi"])

# Text preprocessing function
def preprocess_text(text):
    # Stage 1: Normalization (remove punctuation)
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Stage 2: Word normalization
    text_normalized = text_clean
    text_normalized = re.sub(r'\bgk\b|\bg\b|\bgak\b|\bngak\b|\bnggak\b|\bgx\b|\btdk\b|\bga\b', 'tidak', text_normalized)
    text_normalized = re.sub(r'\bgabisa\b', 'tidak bisa', text_normalized)
    text_normalized = re.sub(r'\btp\b|\btapi\b', 'namun', text_normalized)
    text_normalized = re.sub(r'\bok\b|\boke\b|\bokee\b|\bokk\b', 'baik', text_normalized)
    text_normalized = re.sub(r'\bsdh\b|\budh\b|\budah\b', 'sudah', text_normalized)
    text_normalized = re.sub(r'\bapk\b', 'aplikasi', text_normalized)
    text_normalized = re.sub(r'\beror\b', 'error', text_normalized)
    text_normalized = re.sub(r'\bbs\b|\bbsa\b', 'bisa', text_normalized)
    text_normalized = re.sub(r'\bdpt\b|\bdpat\b|\bdapet\b', 'dapat', text_normalized)
    text_normalized = re.sub(r'\byg\b', 'yang', text_normalized)
    text_normalized = re.sub(r'\bjgn\b|\bjngn\b|\bjangn\b', 'jangan', text_normalized)
    text_normalized = re.sub(r'\bblm\b|\bblum\b|\blom\b', 'belum', text_normalized)
    text_normalized = re.sub(r'\bkrn\b|\bkrna\b', 'karena', text_normalized)
    text_normalized = re.sub(r'\bjos\b', 'sangat baik', text_normalized)
    text_normalized = re.sub(r'\butk\b|\buntk\b', 'untuk', text_normalized)
    text_normalized = re.sub(r'\bsmntr\b|\bsmntara\b', 'sementara', text_normalized)
    text_normalized = re.sub(r'\bckp\b|\bckup\b', 'cukup', text_normalized)
    text_normalized = re.sub(r'\bkalo\b|\bklau\b', 'kalau', text_normalized)
    
    # Stage 3: Lowercasing
    text_lower = text_normalized.lower()
    
    # Stage 4: Tokenization
    tokens = text_lower.split()
    
    # Stage 5: Stemming using Sastrawi
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens after stemming
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return {
        'cleaned_text': text_clean,
        'normalized_text': text_normalized,
        'lower_text': text_lower,
        'tokens': tokens,
        'stemmed_tokens': stemmed_tokens,
        'preprocessed_text': preprocessed_text
    }

@app.route('/')
def index():
    return render_template('index.html')

# Function to generate pie chart
def generate_pie_chart(labels, sizes):
    fig, ax = plt.subplots(figsize=(4, 4))  # Adjust the size of the pie chart
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#BEDC74', '#EE4E4E'])
    ax.axis('equal')
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')  # Remove excess whitespace
    img.seek(0)
    plt.close(fig)  # Close the figure to free memory
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Function to generate word cloud
def generate_word_cloud(text):
    if not text.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/scrape', methods=['POST'])
def scrape():
    app_id = request.form.get('app_id')
    comment_limit = int(request.form.get('limit'))
    filter_score = request.form.get('filter_score')

    filter_score_with = None if filter_score == 'None' else int(filter_score)

    scraped_reviews, _ = reviews(app_id, lang='id', country='id', sort=Sort.MOST_RELEVANT, count=comment_limit, filter_score_with=filter_score_with)
    
    if scraped_reviews and all(key in scraped_reviews[0] for key in ['userName', 'content', 'score', 'at']):
        comments_df = pd.DataFrame(scraped_reviews)[['userName', 'content', 'score', 'at']]
        comments_df['content'] = comments_df['content'].apply(str)
    else:
        return "Data yang diambil tidak memiliki kolom yang sesuai atau aplikasi tidak ditemukan."

    preprocessing_results = comments_df['content'].apply(preprocess_text)

    comments_df['cleaned_content'] = preprocessing_results.apply(lambda x: x['cleaned_text'])
    comments_df['normalized_content'] = preprocessing_results.apply(lambda x: x['normalized_text'])
    comments_df['lower_content'] = preprocessing_results.apply(lambda x: x['lower_text'])
    comments_df['tokens'] = preprocessing_results.apply(lambda x: x['tokens'])
    comments_df['stemmed_content'] = preprocessing_results.apply(lambda x: x['preprocessed_text'])

    comments_df['stemmed_content'] = comments_df['stemmed_content'].fillna('')

    comments_transformed = vectorizer.transform(comments_df['stemmed_content'])

    comments_df['label'] = model.predict(comments_transformed)

    unique_id = uuid.uuid4()
    csv_filename = f'classified_comments_{unique_id}.csv'
    excel_filename = f'classified_comments_{unique_id}.xlsx'

    comments_df.to_csv(csv_filename, index=False)
    comments_df.to_excel(excel_filename, index=False)

    table_data = comments_df[['userName', 'content', 'score', 'stemmed_content', 'label']].to_dict(orient='records')

    sentiment_counts = comments_df['label'].value_counts()
    labels = ['positif', 'negatif']
    sizes = [sentiment_counts.get('positif', 0), sentiment_counts.get('negatif', 0)]

    pie_chart = generate_pie_chart(labels, sizes)

    positive_reviews = ' '.join(comments_df[comments_df['label'] == 'positif']['stemmed_content'])
    negative_reviews = ' '.join(comments_df[comments_df['label'] == 'negatif']['stemmed_content'])
    wordcloud_positive = generate_word_cloud(positive_reviews)
    wordcloud_negative = generate_word_cloud(negative_reviews)

    return render_template('result.html', 
                           file=csv_filename,
                           file_xlsx=excel_filename, 
                           table_data=table_data, 
                           pie_chart=pie_chart,
                           positive_count=sentiment_counts.get('positif', 0),
                           negative_count=sentiment_counts.get('negatif', 0), 
                           wordcloud_positive=wordcloud_positive if wordcloud_positive else 'No data for positive word cloud',
                           wordcloud_negative=wordcloud_negative if wordcloud_negative else 'No data for negative word cloud')

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

@app.route('/download_excel/<filename>')
def download_excel(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
