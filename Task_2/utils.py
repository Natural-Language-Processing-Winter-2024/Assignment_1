from transformers import pipeline

# Load the emotion classifier
classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion')


def emotion_scores(sample):
    emotion = classifier(sample)
    return emotion[0]
