import re

import pandas as pd

df = pd.read_csv("data/anew.csv")

tests = [
    "I loved this movie, it was fantastic!",
    "I hated this movie, it was terrible.",
    "This movie is okay.",
    "I am so happy today!",
    "I am feeling sad and depressed.",
    "I am so angry right now!",
    "I am so anxious and nervous about this.",
]

def analyze_sentiment(list_of_texts, df):
    answer = {}

    for text in list_of_texts:

        values = []
        arousal_mean = 0
        valence_mean = 0

        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        for word in words:

            if word in df['term'].values:

                valence = df.loc[df['term'] == word, 'pleasure'].values[0]
                arousal = df.loc[df['term'] == word, 'arousal'].values[0]
                arousal_mean += arousal
                valence_mean += valence

        if len(words)>0:

            arousal_mean /= len(words)
            valence_mean /= len(words)

        values.append((valence_mean, arousal_mean))
        answer[text] = values
        
    return answer

answer = analyze_sentiment(tests, df)

for text, values in answer.items():
    print(f"Text: {text}")
    for valence_mean, arousal_mean in values:
        print(f"Valence Mean: {valence_mean}, Arousal Mean: {arousal_mean}")
    print()



print(min(df['pleasure']), max(df['pleasure']))
print(min(df['arousal']), max(df['arousal']))
print(min(df['dominance']), max(df['dominance']))

#what word is the most positive and negative in the dataset?
most_positive_word = df.loc[df['pleasure'].idxmax()]['term']

most_negative_word = df.loc[df['pleasure'].idxmin()]['term']

print(f"The most positive word in the dataset is: {most_positive_word}")

print(f"The most negative word in the dataset is: {most_negative_word}")

most_arousing_word = df.loc[df['arousal'].idxmax()]['term']

most_calm_word = df.loc[df['arousal'].idxmin()]['term']

print(f"The most arousing word in the dataset is: {most_arousing_word}")

print(f"The most calm word in the dataset is: {most_calm_word}")

most_dominant_word = df.loc[df['dominance'].idxmax()]['term']

most_submissive_word = df.loc[df['dominance'].idxmin()]['term']

print(f"The most dominant word in the dataset is: {most_dominant_word}")

print(f"The most submissive word in the dataset is: {most_submissive_word}")


    
