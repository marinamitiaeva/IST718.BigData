from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def train_lda_model(texts, num_topics=10):
    """Train LDA topic modeling"""
    vectorizer = CountVectorizer(max_features=5000)
    text_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_matrix)

    return lda, vectorizer

if __name__ == "__main__":
    from preprocess import load_data
    df = load_data('../data/sample_reddit_data.csv')
    lda_model, vectorizer = train_lda_model(df['cleaned_text'])
    print("LDA Model Trained")

from transformers import pipeline

def predict_toxicity(texts):
    """Use Toxic-BERT model to predict toxicity"""
    classifier = pipeline("text-classification", model="unitary/toxic-bert")
    scores = [classifier(text)[0]['score'] for text in texts]
    return scores

if __name__ == "__main__":
    from preprocess import load_data
    df = load_data('../data/sample_reddit_data.csv')
    df['toxicity_score'] = predict_toxicity(df['cleaned_text'])
    df.to_csv('../results/toxicity_scores.csv', index=False)