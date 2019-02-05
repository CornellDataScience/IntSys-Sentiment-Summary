# Sentiment Summarization of Amazon Reviews

Team Members: Jane Zhang, Matthew Bayer, Peter Husisian, Arnav Ghosh, Wes Gurnee

Evaluating potential purchases online is often made easier by reviews provided by other customers. However, sifting through hundreds of reviews to create a holistic view of the sentiment the product evokes can be time-consuming. Consequently, we propose a model that can provide a comprehensive summary of all the reviews on a product. Our task is a specific instantiation of the problem of summarizing the sentiments and opinions of a large group of people into a few key bullet points. That is, given a set of n reviews about a particular product, what are the k most common comments/complaints that are made about this product.

Specifically, our project is divided up into 3 components:

  1. Content Representation/Encoding: Figure out how to represent each review to machine, encode reviews into content vectors.

  2. Summary Creation/Output: Use clustering to group similar reviews together and then decode final clusters into natural language.

  3. Training/Evaluation: Decide loss metrics to evaluate performance of neural architecture and obtain human evaluation.

Related fields: natural language processing, sentiment analysis, abstractive summarization, deep learning, natural language generation

