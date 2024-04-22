# Don't Take It Too Seriously: Machine Learning to Detect Satire

## Overview

In an era characterized by rampant misinformation, our goal is to equip individuals with a tool that can discern the credibility of textual information. Through this project, we develop and compare various natural language processing models to distinguish between trustworthy news articles and satirical content. We explore traditional NLP models, Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Bidirectional Encoder Representations (BERT) architecture. We analyze the performance of our model and attempt to discuss why it performs well and its weak points. We conclude on the curse of dimensionality and how transformers could benefit from the blessing of dimensionality.

## Table of Contents

- Introduction
- Related Work
- Corpus Analysis \& Method
- Experiments
  - Baseline Models
  - Neural Networks
  - RNN
  - LSTM
  - BERT
  - BERT Variants
- Final Model and Analysis
- Discussion
- Conclusion

## Introduction

Satire isn’t necessarily a bad thing. It’s often a great source of laughs and can provide social commentary on the world. But for it to be effective, one must recognize satire as satire. If not, you can find yourself believing falsehoods and have an incorrect perception of the world.

In an age where disinformation is prevalent, it is more vital than ever to be able to distinguish truth from fiction. And not everyone has that ability. For example, young children might lack the capacity to comprehend when a source is satirical or genuine. And even experienced individuals still get it wrong sometimes. Pennycook et al. (2015) found people were less likely to accurately distinguish between satirical and genuine headlines when presented out of context. And in media posts and short clips on TikTok, it’s very easy to take things out of context.

Using Natural Language Processing techniques, we aim to construct a machine-learning model to classify news articles as either reliable or satirical. We explore multiple types of pre-processing techniques and machine-learning architectures to develop a well-performing model; we will then analyze and discuss our model's performance. To train the model, we use the dataset from Rashkin et. al (2017).

This project takes a step back and compares traditional NLP techniques versus more modern neural network models, as compared to other research which largely focuses on either solely traditional NLP techniques or solely or neural networks. This allows us to conclude with a foundational data science problem of choosing the right complexity of a function to model the complexity of the problem.

## Related Work

In this section, we will discuss some of the related work done in the field of NLP that has motivated us to pursue certain models for satirical news detection.

Identifying satirical news often involves understanding the context and nuances of language used in the articles. RNNs, LSTMs, and transformers are particularly effective in capturing these linguistic cues.

### RNNs

RNNs (Recurrent Neural Networks) are a type of neural network that is commonly used for sequence data, such as text. RNNs process input data sequentially, with each step of the network dependent on the previous step. This makes them well-suited for tasks such as text classification, where the order of words in a sentence is important. In addition, RNNs have been successfully applied to satirical news detection (Dutta \& Chakraborty, 2019). Hence we have chosen to investigate RNNs.

### LSTMs

LSTMs (Long Short-Term Memory) are an advanced type of RNN designed to handle longer-range dependencies in text. They introduce a memory cell that retains information over long sequences, making them more robust to noise and less prone to overfitting compared to traditional RNNs. Rashkin et al. (2017) successfully trained an LSTM to detect satirical articles and fact-check news articles. Therefore, this is also an avenue we want to try.

### Transformers (BERT)

BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art transformer-based model pre-trained on a large text corpus. It has been widely used in NLP tasks, including text classification, question answering, and natural language inference. BERT employs a multi-layer bidirectional transformer encoder to learn contextual word embeddings. The model is trained using a self-supervised objective, predicting masked words in sentences. BERT captures rich linguistic information, including syntactic and semantic relationships, making it a powerful tool for satirical news detection and other NLP tasks, as demonstrated in Devlin et al. (2018).

## Corpus Analysis & Method

### Exploratory Data Analysis

The training and test corpus consists of a series of news articles that are labeled as trusted news, satire, hoax, and propaganda. The dataset we use comes from Rashkin et. al (2017), which constructed these corpora by sampling articles from 8 different reliable and unreliable online news sites. For our problem statement, we will classify the 'trusted' news as reliable and group the 'satire', 'hoax', and 'propaganda' categories together under the label 'unreliable'.

In total, there are almost 49,000 data points in the training dataset (to be precise, 48,854 data points). The breakup of this distribution is given in Figure 1. The dataset is heavily imbalanced, with the majority of the articles being classified as propaganda. Only 20\% of the news items are classified as reliable. This needs to be addressed for our model to not automatically lean towards classifying a news article as unreliable.


The news articles are also of varying length. Figure 2 shows the box plot of the number of characters in each news article. Input text lengths range from 9 to 735,607. Note that the data point with 735,607 characters is the sole point not included in Figure 2 for the sake of scale. Nevertheless, even while excluding this entry, we see that there are several outliers. The interquartile range for character length is 3399, which is quite large. We cannot afford to filter out data points if they have a large character length, since news articles will always be of variable sizes. Therefore, our model's preprocessing techniques should account for varying sizes as well.


### Pre-processing Techniques

Before feeding data into our models, we pre-processed the data with the following techniques:

- **Oversampling**: We rectify the imbalance in the dataset categories by randomly oversampling the dataset. This will prevent the model from being biased towards the satire classification. We chose oversampling over undersampling since our dataset is already quite small (less than 50,000 entries) and we do not need to reduce its size even further.
- **BERT Tokenizer**: The BERT Tokenizer was used to tokenize and encode words in the news article. The BERT Tokenizer will insert a [CLS] sentence-level classification and [SEP] separator token. It also includes the positional encoding of words, so the model had extra information in its input that might be useful in determining whether a news article is reliable or not. It also directly handles padding so that inputs of different lengths can be converted into same-length vectors.
- **TF-IDF**: The TF-IDF (Term Frequency- Inverse Document Frequency) algorithm converts documents into vectors based on the frequency and importance of the statements’ keywords. We applied this technique with the assumption that certain words (for example, opinion words like ‘think’) might be a good judge of a news article’s reliability. In general, this technique was mainly used on our baseline models. It was applied after lemmatisation and tokenisation.
- **GloVe**: The GloVe algorithm obtains vector representation of words by constructing a co-occurrence matrix of the corpus. We have used the pre-trained word vectors while utilizing GloVe. This is largely due to the small size of our dataset. First, GloVe will provide a good starting point for learning the relationships between various words, so the resultant vectors will be more accurate than if we started from scratch. Second, using pre-trained vectors ensures that GloVe can later handle words that aren’t present in the training corpus.

## Experiments

### Baseline Models

Before trying advanced models, we used traditional ML models to get a baseline for our model's performance. We trained various Naive Bayes and Logistic Regression models, with both TF-IDF and GloVe preprocessing techniques. During this process, the grid search mechanism was used to tune the hyperparameters for the models. Figure 3 shows our most prominent results.

| Model | Preprocessing | Performance |
| --- | --- | --- |
| Naive Bayes | TF-IDF | F1-score: 0.683 Accuracy: 0.879 |
| Naive Bayes | GloVE | F1-score: 0.602 Accuracy: 0.849 |
| Logistic Regression | TF-IDF | F1-score: 0.836 Accuracy: 0.924 |
| Logistic Regression | GloVE | F1-score: 0.663 Accuracy: 0.670 |

As the table shows, our best-performing model is the Logistic Regression model with TF-IDF pre-processing, which has an F1 score of 0.836 and an accuracy of 0.94. Our baseline models are already very accurate and high-performing.

From this, we move on to trying different types of neural network architecture.

### Neural Networks

Neural networks are multi-layered models inspired by the neurons in our brains. Each layer of a neural network consists of perceptrons, which take a weighted sum of the output of the previous layer and run the result through an activation function.

Here, we implemented a basic four-layer neural network as a baseline for other neural network models. The neural network uses the GloVe vectors as input. Each linear layer is followed by a ReLU activation function, except the last layer which has a Sigmoid activation function, as this is a binary classification problem. The choice of neural network size is arbitrary, as testing many different layer sizes and layer counts yielded similar F1-scores on the test set.

After training for 400 epochs, this neural network achieves a validation F1-score of 0.882 and an accuracy of 0.909 and provides a baseline performance measurement before moving on to more complex models. The test F1-score is 0.833 with an accuracy of 0.806.

### RNN

Recurrent Neural Networks (RNNs) are a fundamental type of neural network architecture designed to handle sequential data, such as text. Unlike traditional feedforward neural networks, RNNs have feedback connections, allowing them to maintain an internal state and process data with temporal dependencies. This characteristic makes RNNs well-suited for tasks where the order of the input data matters, such as text classification, machine translation, and language modeling.

In the context of satirical news detection, RNNs can be leveraged to analyze news articles and classify them as reliable or satirical. Each word in the article is fed into the RNN sequentially, and the network learns to capture the context and relationships between words. By considering the sequential nature of text, RNNs can identify patterns and linguistic cues that may indicate the satirical nature of an article.

Our model utilizes the bidirectional wrapper to apply the SimpleRNN layer in both forward and backward directions, capturing contextual information. The Dense layer with a sigmoid activation function performs binary classification. The model is compiled with the Adam optimizer and binary cross-entropy loss function, which is suitable for binary classification tasks.

Our RNN model was able to attain an F1-score of about 0.867 against the validation set and a final F1-score of 0.74 against the training set.

It is important to note that RNNs have some limitations, such as the vanishing gradient problem, where gradients can diminish or explode during backpropagation, making it challenging to learn long-range dependencies. Advanced variants of RNNs such as Long Short-Term Memory (LSTM) networks have been developed to address these limitations and improve the modeling of long-range dependencies in text.

### LSTM

Long Short-Term Memory (LSTM) networks are an advanced type of RNN designed to address the limitations of traditional RNNs, particularly the vanishing gradient problem. LSTMs introduce a memory cell and a set of gates that regulate the flow of information, allowing them to capture long-range dependencies in sequential data more effectively. This makes LSTMs well-suited for tasks that require remembering or modeling long-term patterns, such as text classification.

We used a method similar to RNN model construction, simply replacing the RNN layer with an LSTM layer. Our LSTM model was able to attain an F1-score of about 0.93 against the validation set, giving us an improved F1-score against the test set of 0.84.

### BERT

The pre-trained model used for BERT is the base model, "bert-base-uncased". It has 12 layers in the encoder stack, 768 hidden units in the feedforward network, 12 attention heads, and 110 million parameters. Smaller models like the "bert-medium" and "bert-tiny" were also tested, but the performance was slightly worse, and hence we chose to stick with the base model. The main reason why we chose "bert-base" over the "bert-large" model is mainly due to computational limitations, as we ran into memory allocation errors on Kaggle.

The data is down-sampled for the BERT base model, as we are unable to use oversampling due to memory allocation issues. After shuffling data, a 90-10 training-validation split was observed. The model was trained, shuffled, and split 90-10 into training and validation sets. The model will be trained on the training set and will be tested on the validation set after every epoch.

The pre-trained BERT model is fine-tuned with a small learning rate and weight decay and ran for 3 epochs. Only a small number of epochs is needed since BERT models are already pre-trained and can quickly adjust the weights. A larger number of epochs would lead to over-fitting. The loss function used is Binary Crossentropy. We experimented with different batch sizes, running the model over both 4 and 6 batches. However, the effect was negligible since training, validation, and test results are nearly identical.

Our best version has accuracy and F1-score of over 0.97 in training and validation after 3 epochs. However, it achieves an F1-score of 0.75 when evaluated on the test dataset. This is quite shocking as there is a large discrepancy between the validation results and the test results.

#### BERT Variants

We experimented with changing the number of attention layers in the BERT model. We hypothesized that the default 12 attention layer model might be too powerful for our small dataset, and therefore a lower number of attention layers might prevent overfitting. Therefore, we constructed models using different variants of BERT with 2,4,8, and 12 attention layers. However, the performance for all the variants is nearly identical, indicating that changing the number of attention layers has very little effect. The default BERT model narrowly gave the best performance, with an F1-score of 0.75.

#### Micro-instance analysis

Further analysis of the outputs of the model shows that certain keywords will flip the prediction from unreliable to reliable and vice-versa. For example, certain countries/states are often deemed more reliable than others, as demonstrated in Figure 6.

"Court approves death sentences for leaders of Bangladesh's 1975 coup." This data point from the test corpora was labeled unreliable by the model; however, it is actually reliable. We experimented with replacing Bangladesh with different countries and cities and realized that this was enough to flip the model's prediction. Figure 7 shows the substituting country/city and the resulting prediction of the sentence. With a short article, each word carries more significance, and replacing a keyword could change the prediction, compared to a longer article with more context supporting the prediction. Interestingly, the model seems to generally associate countries from the "Global North" as reliable and those from the "Global South" as unreliable (though there are some exceptions such as Australia). This could reflect a potential bias in the training dataset.

| Prediction | Country |
| --- | --- |
| Reliable | New York, United States, England, Germany, France, London, Singapore, California |
| Unreliable | Australia, Texas, Mexico, India, China, Cuba, Russia, North Korea, Brazil |

### Final Model and Analysis

After trying different models, we settled on the LSTM as our final model, as it has the best F1-score of 0.84.

To analyze its performance further, the test dataset was segmented by three primary criteria: the topic of the news articles, the overall sentiment of the news articles, and the length of the news articles. This approach reveals the data points and conditions where the model excels or underperforms.

#### Domain and subtopic analysis

The test set is classified into 4 distinct domains and 12 subtopics. The 4 domains are "Politics/Civics", "Soft News", "Health/Science", and "Business/Technology". The subtopics include “Entertainment”, “Local News”, and “Election/Debates”. The test set has an even spread of domains and subtopics; the domain-based groups contain 60, 59, 60, and 61 news articles, respectively, and each subtopic has 20 news articles classified under it. Furthermore, there is an equal division of satirical and reliable news articles within the domains and subtopics.

Figure 8 shows the domain-specific F1 scores produced by our LSTM model. As seen, the model works best for ‘Politics/Civics’ articles, with an F1-score of 0.85, and worst for ‘Soft News’ articles, with an F1-score of 0.82. However, the range in F1-scores for all four categories is only 0.04, which is extremely small. There is no domain seen where the model works significantly better or significantly worse.

We see similar results when we break the dataset down by subtopics. Figure 9 illustrates that the F1-score is highest for the "Environment" subtopic with an F1-Score of 0.91 and lowest for the "Hard Science" subtopic with an F1-Score of 0.77. While there is more variation than in the domain segmentation, there is still no subtopic where the model performs exceptionally well or badly.

#### Sentiment analysis

To analyze the sentiment of each news article, the Afinn Python library was utilized. Afinn uses a custom word list to assign sentiment scores to overall sentences, with a higher score indicating a more positive sentiment of the document (Nielsen, 2019). In our analysis, we marked all documents with a sentiment score between -3 and 3 as neutral. The results of our analysis are shown in Figure 10. We hypothesized that statements with larger absolute sentiment scores would be better classified since their intent might be clearer. This is correct for our model, where articles with Afinn scores closer to zero (i.e., more neutral statements) have higher F1-scores than the more positive or negative articles.

| Afinn Score | F1-score |
| --- | --- |
| <-3 | 0.83 |
| -3-3 | 0.89 |
| >3 | 0.75 |

#### By article length

To analyze how article length affects the model's performance, we broke down the dataset into four equal segments, based on the length of the articles (measured by the number of characters in the document). Each segment, therefore, represents a quartile of the dataset when measured by article length. Figure 11 shows there is significant variation in the performance of the model according to article length. The model is extremely accurate on smaller articles (with an F1-score of 0.93) and performs extremely poorly on larger articles (with an F1-score of 0.14). Article length is a significant contributing factor that affects how accurately the model can classify news articles.

| Quartile | F1-score |
| --- | --- |
| Q1 | 0.92 |
| Q2 | 0.92 |
| Q3 | 0.73 |
| Q4 | 0.28 |

## Discussion

1. Do neural networks always perform better than traditional NLP techniques? If not, at what point do neural networks start performing better?

   Neural networks do not always outperform traditional NLP techniques. Neural networks were conceptualized in 1943 but only gained traction as the amount of data available, compute power, and increasingly parallelizable vector calculations became more prevalent and affordable. Neural networks generally begin to show their prowess when there is sufficient data to generalize on or when the task is increasingly complex. In our project, we found that traditional NLP techniques, such as logistic regression with TF-IDF preprocessing, achieved an F1-score of 0.83. Our best-performing LSTM model just narrowly beat its performance with an F1-score of 0.84. For our satire detection use case, traditional NLP techniques are performing at a comparable level to neural networks.

   This could be due to the limitations of our dataset. As mentioned in the Exploratory Data Analysis, our dataset has less than 50,000 data points. These data points come from just 8 different news sites that are all very America-centric. Since our dataset is not so varied, simpler models like Logistic Regression may, therefore, be able to pick up on the same patterns as the more complicated models like LSTM. However, if we expand the dataset in the future and add news articles from other sources, this trend might change.

2. Why does LSTM outperform BERT?

   Our LSTM model performs better than our BERT models, with our final LSTM model having an F1-score of 0.84 and our best BERT model having an F1-score of 0.75. One possible reason for this is because LSTM is better suited to classifying lengthier news articles than BERT. Figure 12 compares the F1-scores of the two models for different article lengths (the dataset being segmented into four quartiles based on their article lengths). This is similar to the approach we took when generating Figure 11.

   We observe that in the first two quartiles (Q1 and Q2), the models perform almost equally well. However, their performance differs as article length increases. LSTM performs twice as well for the quartile with the largest article lengths as compared to BERT.

   | Quartile | BERT(F1) | LSTM(F1) |
   | --- | --- | --- |
   | Q1 | 0.93 | 0.92 |
   | Q2 | 0.89 | 0.92 |
   | Q3 | 0.64 | 0.73 |
   | Q4 | 0.14 | 0.28 |

   These dramatic differences in performance could be due to BERT's maximum sequence length of 512. The minimum length of words is 619 in this test dataset, which is already greater than the maximum sequence length for BERT. The range of character lengths where the model performs exceptionally poorly is from 4485 to 21851. Therefore, the maximum length of 512 might not be sufficient to gain context of the entire article when the article is very long. On the other hand, LSTM does not have this maximum sequence length limitation, which might be why it can classify longer articles better for our use case.

## Conclusion

The performance of different machine learning models on the task of distinguishing satirical news from trustworthy sources highlights the impact of the curse and blessing of dimensionality. Our experimental results reveal surprising yet insightful findings. While there is significant hype surrounding complex transformer models like GPT-4 and Bard, our best-performing models were simpler ones: logistic regression, a simple neural network, and LSTM.

The curse of dimensionality comes into play when considering the size of the dataset and the complexity of the problem. In scenarios with scarcer datasets and less complex problems, simpler traditional machine learning models, such as logistic regression and simple neural networks, tend to excel. These models have fewer parameters and are less prone to overfitting, making them well-suited for situations where data is limited.

On the other hand, we also observed the blessing of dimensionality. As we experimented with BERT, a powerful transformer model, we noticed that simplifying the architecture by reducing the number of attention layers did not significantly improve its performance. BERT's F1-score remained around 0.8, which was lower than the scores achieved by our simpler models. This indicates that for more complex models like transformers, a certain level of model complexity and architectural design are necessary to handle high-dimensional feature spaces effectively.

Therefore, our conclusion is that the choice of the best model depends on the specific needs of the problem and the available resources. For smaller datasets or simpler problems, simpler models, such as traditional NLP techniques, are preferable. They are less prone to overfitting, easier to interpret, and more computationally efficient. However, as the dataset size increases and the problem becomes more complex, the blessing of dimensionality comes into play, and more complex models like transformers can leverage larger amounts of data and higher-dimensional feature spaces to achieve superior performance.

In summary, the curse of dimensionality underscores the importance of selecting simpler models for smaller datasets and less complex problems, while the blessing of dimensionality highlights the advantages of more complex models when ample data and higher-dimensional feature spaces are available. The key takeaway is to strike a balance between model complexity and the specific requirements of the task at hand, considering both the needs of the problem and the available resources.

What future works could look at is to find the tipping point where large language models start pulling ahead with the blessing of dimensionality.