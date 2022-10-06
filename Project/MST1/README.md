# Text Clustering using K-Means Clustering Algorithm, Bag of Words and Tf-IDF Vectorizer
## Applied Tf-Idf Vectorizer and K-Means Algorithm to segment the articles fetched using requests library from Machine Learning Geek
---
- Submitted By : By Piyush Joshi
- Roll Number : DS5B-2121
- Submitted To : Prof. Upendra Singh
- Subject : Text and Image Analytics
- Batch : MSc 3rd Sem (Data Science and Analytics)
- College : School of Data Science and Forecasting, DAVV, Indore

---
---
## Problem Statement
To scrap textual articles from Machine Learning Geek and create an NLP model to cluster those articles using K-Means Clustering
## Libraries and Packages Used
- pandas
- numpy
- re
- sklearn
- BeautifulSoup
- requests
- nltk
- wordcloud
- matplotlib
## Procedure
To create an NLP Model using scrapped articles, we need to follow the following steps:
- Step 1 : Data Acquisition - Acquire textual articles from https://machinelearninggeek.com/ using Python’s request library and create a
dataset
- Step 2 : Text Preprocessing - Preprocessing will involve steps such as
  - tokenization
  - removing HTML Tags
  - removing URLs
  - removing stopwords
  - stemming
  - lemmatization
- Step 3 : Feature Engineering - This step will involve creating vectors from text by vectorization techniques such as Bag of Words TF-IDF(Term Frequency Inverse Docuement Frequency)
- Step 4 : Modelling - This step will involve modelling using K-Means Clustering
- Step 5 : Evaluation - Evaluating the model using Silhouette score Davies Bouldin score
#### Once the model is ready we can use it to cluster the scrapped articles
