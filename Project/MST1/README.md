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

# Various Clustering Techniques

## K-Means Clustering

***K-means*** is one of the simplest and most widely used clustering algorithms. It is a type of **partitioning clustering method** that partitions the dataset into random segments. K-means is a faster and more robust algorithm that generates spherical clusters. 

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%201.png)

The k-means method requires the **number of clusters** as input at the beginning and it does not guarantee convergence to the global solution, rather its result may depend upon the initial cluster center. The k-means method is not suitable for finding **non-convex clusters** and **nominal attributes.** 

The K-Means Clustering **groups similar data points** together and **discovers underlying patterns**. To achieve this objective, K-means looks for a fixed number **(k)** of clusters in a dataset. The K-means algorithm identifies the k number of **centroids**, and then allocates every data point to the nearest cluster. 

The **‘means’** in the K-means refers to averaging the data; that is, finding the centroid.

## DBSCAN

***Density-Based Spatial Clustering of Applications with Noise (DBSCAN)*** is a base algorithm for **density-based clustering.** It can discover clusters of different shapes and sizes from a large amount of data, which is containing noise and outliers.

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%202.png)

The DBSCAN algorithm uses two parameters:

- **minPts:** The minimum number of points (a threshold) clustered together for a region to be considered dense.
- **eps (ε):** A distance measure that will be used to locate the points in the neighbourhood of any point.

These parameters can be understood better if we explore two concepts called **Density Reachability** and **Density Connectivity.**

- **Reachability** in terms of density establishes a point to be reachable from another if it lies within a particular distance (eps) from it.
- **Connectivity**, on the other hand, involves a transitivity based chaining-approach to determine whether points are located in a particular cluster. For example, p and q points could be connected if p->r->s->t->q, where a->b means b is in the neighborhood of a.

![https://miro.medium.com/max/627/1*yT96veo7Zb5QeswV7Vr7YQ.png](https://miro.medium.com/max/627/1*yT96veo7Zb5QeswV7Vr7YQ.png)

There are **three** types of points after the DBSCAN clustering is complete:

- **Core** — This is a point that has at least *m* points within distance *n* from itself.
- **Border** — This is a point that has at least one Core point at a distance *n*.
- **Noise** — This is a point that is neither a Core nor a Border. And it has less than *m* points within distance *n* from itself.

## Affinity Propagation

**Affinity Propagation** was first published in 2007 by **Brendan Frey** and **Delbert Dueck**. In contrast to other traditional clustering methods, Affinity Propagation **does not require** you to specify the number of clusters. In layman’s terms, in Affinity Propagation, each data point sends messages to all other points informing its targets of each target’s relative attractiveness to the sender. Each target then responds to all senders with a reply informing each sender of its availability to associate with the sender, given the attractiveness of the messages that it has received from all other senders. Senders reply to the targets with messages informing each target of the target’s revised relative attractiveness to the sender, given the availability messages it has received from all targets. 

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%203.png)

The message-passing procedure proceeds until a consensus is reached. Once the sender is associated with one of its targets, that target becomes the point’s exemplar. All points with the same exemplar are placed in the same cluster.

Affinity Propagation creates clusters by sending messages between data points until convergence. Unlike clustering algorithms such as **k-means** or **k-medoids**, affinity propagation does not require the number of clusters to be determined or estimated before running the algorithm, for this purpose the two important parameters are 

- **Preference**, which controls how many exemplars (or prototypes) are used
- **Damping factor** which damps the responsibility and availability of messages to avoid numerical oscillations when updating messages.

A dataset is described using a small number of exemplars, **‘exemplars’** are members of the input set that are representative of clusters. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at that point the final exemplars are chosen, and hence we obtain the final clustering. 

## Hierarchical Clustering

Hierarchical clustering is another unsupervised machine learning algorithm, which is used to group the unlabeled datasets into a cluster and also known as **hierarchical cluster analysis** or **HCA**.

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%204.png)

In this algorithm, we develop the hierarchy of clusters in the form of a tree, and this tree-shaped structure is known as the **dendrogram**.

Sometimes the results of K-means clustering and hierarchical clustering may look similar, but they both differ depending on how they work. As there is no requirement to predetermine the number of clusters as we did in the K-Means algorithm.

The hierarchical clustering technique has two approaches:

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%205.png)

- **Agglomerative:** Agglomerative is a **bottom-up** approach, in which the algorithm starts with taking all data points as single clusters and merging them until one cluster is left. These algorithms treat each document as a singleton cluster at the outset and then successively merge (or *agglomerate*) pairs of clusters until all clusters have been merged into a single cluster that contains all documents.
- **Divisive:** Divisive algorithm is the reverse of the agglomerative algorithm as it is a **top-down approach.** Top-down clustering requires a method for splitting a cluster. It proceeds by splitting clusters recursively until individual documents are reached.

## Spectral Clustering

***Spectral Clustering*** is a growing clustering algorithm which has performed better than many traditional clustering algorithms in many cases. It treats each data point as a **graph-node** and thus transforms the clustering problem into a **graph-partitioning problem.**

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%206.png)

**Properties:**

1. **Assumption-Less:** This clustering technique, unlike other traditional techniques do not assume the data to follow some property. Thus this makes this technique to answer a more-generic class of clustering problems.
2. **Ease of implementation and Speed:** This algorithm is easier to implement than other clustering algorithms and is also very fast as it mainly consists of mathematical computations.
3. **Not-Scalable:** Since it involves the building of matrices and computation of eigenvalues and eigenvectors it is time-consuming for dense datasets.
    
    ![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%207.png)
    

## Gaussian Mixture Models (GMMs) Clustering

***Gaussian Mixture Models (GMMs)*** assume that there are a certain number of **Gaussian distributions**, and each of these distributions represent a cluster. Hence, a Gaussian Mixture Model tends to group the data points belonging to a single distribution together.

![Untitled](Text%20and%20Image%20Analytics%20-%20Text%20Clustering%20e67051e227be4b7e87127b6a4a546659/Untitled%208.png)

Gaussian Mixture Models are **probabilistic models** and use the **soft clustering approach** for distributing the points in different clusters. Each GMM would identify the probability of each data point belonging to a Gaussian distributions.

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
