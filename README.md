# CS4248 Final Report

## Introduction

In the era of disinformation and propaganda, our project aims to develop a natural language processing model that can distinguish between trustworthy and satirical news sources. While satire can be entertaining and provide social commentary, it is essential to recognize it as such to avoid misinformation. We explore various machine learning techniques and their applications in satirical news detection.

## Related Work

Several studies have explored the use of natural language processing for satirical news detection. Here are some key findings and techniques that motivated our approach:

- **RNNs**: Recurrent Neural Networks (RNNs) are commonly used for sequence data, including text classification tasks. RNNs process input data sequentially, making them effective in capturing linguistic cues and understanding context. RNNs have been successfully applied to satirical news detection and other NLP tasks.
- **LSTMs**: Long Short-Term Memory (LSTM) networks are advanced RNNs designed to handle longer-range dependencies in text. They introduce memory cells that retain information over long sequences, improving robustness and reducing overfitting compared to traditional RNNs.
- **Transformers (BERT)**: BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer-based model pre-trained on a large text corpus. It employs a multi-layer bidirectional transformer encoder to learn contextual word embeddings. BERT captures rich linguistic information, including syntactic and semantic relationships, making it a powerful tool for satirical news detection.

## Corpus Analysis & Method

### Exploratory Data Analysis

Our training and test corpus consists of news articles labeled as trusted, satire, hoax, or propaganda. We focus on classifying articles as reliable or unreliable by grouping the latter three categories. Here are some key insights from our exploratory data analysis:

- **Imbalanced Dataset**: The dataset is heavily imbalanced, with only 20\% of articles classified as reliable. This imbalance needs to be addressed to prevent the model from biased classification.
- **Variable Article Lengths**: News articles have varying lengths, ranging from a few characters to hundreds of thousands. This variation requires preprocessing techniques that can handle different input sizes.

### Pre-processing Techniques

To address the challenges identified in the exploratory data analysis, we applied the following pre-processing techniques:

- **Oversampling**: We randomly oversampled the dataset to rectify the imbalance and prevent model bias towards the satire classification.
- **BERT Tokenizer**: The BERT Tokenizer was used for tokenization and encoding, handling positional encoding and padding for inputs of different lengths.
- **TF-IDF**: The TF-IDF algorithm converts documents into vectors based on keyword frequency and importance, assuming that certain words may indicate a news article's reliability.
- **GloVe**: The GloVe algorithm provides vector representation by constructing a co-occurrence matrix of the corpus. We used pre-trained word vectors due to our small dataset size, ensuring better starting relationships and handling out-of-vocabulary words.

## Experiments

### Baseline Models

We trained traditional machine learning models as baselines before exploring neural networks. Here are the key results:

| Model | Preprocessing | F1-Score | Accuracy |
| --- | --- | --- | --- |
| Naive Bayes | TF-IDF | 0.683 | 0.879 |
| Naive Bayes | GloVe | 0.602 | 0.849 |
| Logistic Regression | TF-IDF | 0.836 | 0.924 |
| Logistic Regression | GloVe | 0.663 | 0.670 |

Our best-performing baseline model is Logistic Regression with TF-IDF pre-processing, achieving an F1-score of 0.836 and an accuracy of 0.94.

### Neural Networks

We experimented with various neural network architectures, including RNNs, LSTMs, and BERT. Here are the key findings:

- **RNN**: Recurrent Neural Network architecture, effective for sequential data like text.
- **LSTM**: Long Short-Term Memory network, an advanced RNN variant that handles longer-range dependencies with memory cells.
- **BERT**: State-of-the-art transformer-based model, pre-trained on a large text corpus, capturing rich linguistic information.

#### BERT Model

We fine-tuned the pre-trained BERT base model ("bert-base-uncased") due to computational limitations. Here are the key details:

- **Data Sampling**: Down-sampled data due to memory allocation issues, with a 90-10 training-validation split.
- **Hyperparameters**: Small learning rate, weight decay, 3 epochs, and binary cross-entropy loss function.
- **Batch Size**: Negligible effect on performance, with similar results for 4 and 6 batches.
- **Performance**: Achieved F1-score and accuracy of over 0.97 in training and validation but only 0.75 F1-score on the test dataset.

#### Smaller BERT Models

We tested smaller BERT models with fewer attention layers to investigate potential overfitting. However, the performance was similar, indicating that the number of attention layers was not the cause of lower test scores.

## Results

Our final model is the BERT Transformer, and we analyzed its performance on the test dataset segmented by topic, sentiment, and length. Here are the key findings:

### Domain and Subtopic Analysis

The test dataset was classified into four domains and twelve subtopics, with an even spread of reliable and satirical articles.

- **Domain-specific F1 Scores**: The model performed best on 'Soft News' (F1-score: 0.81) and worst on 'Health/Science' (F1-score: 0.71). However, the difference in performance across domains was not significant.
- **Subtopic-specific F1 Scores**: The model achieved the highest F1-scores for sports, tech, finance, and entertainment news, and the lowest for elections and debates. Again, the variation in performance across subtopics was not significant.

### Sentiment Analysis

We analyzed the sentiment of news articles using the Afinn Python library, marking articles with sentiment scores between -3 and 3 as neutral.

- **Neutral Articles**: The model performed best on neutral articles (F1-score: 0.88), followed by negative articles (F1-score: 0.75), and positive articles (F1-score: 0.68).

### Performance by Statement Length

We divided the test dataset into quartiles based on article length (character count).

- **Quartile-specific F1 Scores**: The model performed exceptionally well on shorter articles (F1-score: 0.93 in Q1) and poorly on longer articles (F1-score: 0.14 in Q4).
- **Impact of Maximum Sequence Length**: The dramatic performance difference may be due to the maximum sequence length of 512, which may not be sufficient for context understanding in very long articles.

## Discussion

Our project explored the use of traditional NLP techniques and neural networks for satirical news detection. Here are some insights and conclusions:

- Neural networks do not always outperform traditional NLP techniques, especially for simple tasks or small datasets. In our project, traditional NLP techniques achieved higher F1-scores and accuracies.
- Neural networks, such as RNNs and LSTMs, are well-suited for handling longer inputs and capturing context and structure in text.
- Smaller BERT models with fewer attention layers did not improve test scores, indicating that the number of attention layers was not the cause of lower test performance.
- Our final BERT model performed well on shorter articles but struggled with longer ones, possibly due to the maximum sequence length limitation.

## Conclusion

In conclusion, our project demonstrates the effectiveness of natural language processing techniques in satirical news detection. While neural networks offer advanced capabilities, traditional NLP techniques can also achieve impressive results, especially with smaller datasets. Our final BERT model, despite its limitations, showcases the potential of transformer-based models in this domain. Further research and larger datasets can help improve the performance and applicability of these models in real-world scenarios.