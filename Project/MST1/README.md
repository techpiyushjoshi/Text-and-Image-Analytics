# Text Clustering using K-Means Clustering Algorithm, Bag of Words and Tf-IDF Vectorizer
## Applied Tf-Idf Vectorizer and K-Means Algorithm to segment the articles fetched from API using requests library
- Submitted By : By Piyush Joshi
- Roll Number : DS5B-2121
- Submitted To : Prof. Upendra Singh
- Subject : Text and Image Analytics
- Batch : MSc 3rd Sem (Data Science and Analytics)
- College : School of Data Science and Forecasting, DAVV, Indore

# Text and Image Analytics - Text Clustering

# **Table of Contents**

# Introduction to Text Clustering

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled.png)

***Text Clustering*** is a process of grouping most similar articles, tweets, reviews, and documents together. Clustering is also known as the **data segmentation method.** It divides the data points into a number of specific batches or groups, such that the data points in the same groups have similar properties and data points in different groups have different properties in some sense. Every group is known as a **cluster.** It is an **unsupervised learning** technique which implies it discovers hidden patterns or data groupings within the data without the need for human intervention.

# Applications of Clustering

Clustering can be utilized in: 

- Outlier detection problems such as fraud detection
- Clustering or organizing documents
- Text summarization.
- Customer Segmentation
- Recommender System
- Visualization

# Types of Clustering

There are various clustering techniques based on differential evolution such as:

- **K-Means** based on the distance between points
- **Affinity propagation** based on graph distance
- **DBSCAN** based on the distance between the nearest points
- **Spectral clustering** based on graph distance
- **Gaussian Mixtures** based on Mahalanobis distance to centers
- **Hierarchical clustering** based on hierarchies

# Working of Clustering techniques

Fundamentally, all clustering methods use the same approach i.e. 

> First we calculate similarities, and then we use it to cluster the data points into groups or batches. Some techniques use distances while others use statistical distributions to compute similarities.
> 

These techniques require text to be converted into some type of vectors by techniques such as 

- Bag of Words(BoW)
- Term Frequency-Inverse Document Frequency (TF-IDF)
- Word2Vec
- Doc2Vec
- Sent2Vec, etc.

Let’s explore some of the clustering techniques in more detail

## K-Means Clustering

***K-means*** is one of the simplest and most widely used clustering algorithms. It is a type of **partitioning clustering method** that partitions the dataset into random segments. K-means is a faster and more robust algorithm that generates spherical clusters. 

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%201.png)

The k-means method requires the **number of clusters** as input at the beginning and it does not guarantee convergence to the global solution, rather its result may depend upon the initial cluster center. The k-means method is not suitable for finding **non-convex clusters** and **nominal attributes.** 

The K-Means Clustering **groups similar data points** together and **discovers underlying patterns**. To achieve this objective, K-means looks for a fixed number **(k)** of clusters in a dataset. The K-means algorithm identifies the k number of **centroids**, and then allocates every data point to the nearest cluster. 

The **‘means’** in the K-means refers to averaging the data; that is, finding the centroid.

# Use Case : Applying K-Means Clustering on articles from [Machinelearninggeek.com](http://Machinelearninggeek.com)

## Overview

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%209.png)

**Document clustering** (or **text clustering**) is the application of cluster analysis to textual documents. It has applications in automatic document organization, topic extraction and fast information retrieval or filtering.

## Problem Statement

To scrap textual articles from [Machine Learning Geek](https://machinelearninggeek.com/)  and create an NLP model to cluster those articles using K-Means Clustering

## Libraries and Packages Used

- `pandas`
- `numpy`
- `re`
- `sklearn`
- `BeautifulSoup`
- `requests`
- `nltk`
- `wordcloud`
- `matplotlib`

## Procedure

To create an NLP Model using scrapped articles, we need to follow the following steps:

- **Step 1 : Data Acquisition** - Acquire textual articles from [https://machinelearninggeek.com/](https://machinelearninggeek.com/) using Python’s `request` library and create a dataset
- **Step 2 : Text Preprocessing** - Preprocessing will involve steps such as
    - `tokenization`
    - `removing HTML Tags`
    - `removing URLs`
    - `removing stopwords`
    - `stemming`
    - `lemmatization`
- **Step 3 : Feature Engineering** - This step will involve creating vectors from text by vectorization techniques such as
    - `Bag of Words`
    - `TF-IDF(Term Frequency Inverse Docuement Frequency)`
- **Step 4 : Modelling** - This step will involve modelling using `K-Means Clustering`
- **Step 5 : Evaluation** - Evaluating the model using
    - `Silhouette score`
    - `Davies Bouldin score`

Once the model is ready we can use it to cluster the scrapped articles
