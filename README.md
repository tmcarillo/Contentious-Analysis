# NLP_Annotation_Project : Contentiousness Analysis of Reddit Posts

## Objective
In our Annotation Project, we aimed to classify Reddit posts into three levels of contentiousness: **Low**, **Medium**, and **High**. Contentiousness is defined as "likely to cause disagreement or argument." Our goal was to determine whether Reddit posts were likely to cause disagreement by applying a machine learning model to classify the posts based on their language and content.

## Approach
We decided to use an **Ordinal Regression Classifier** for our analysis, as the three classes—Low, Medium, and High—are naturally ordered. This makes ordinal regression a suitable choice, as it takes into account the inherent order in the labels.

## Model Performance

### 1. Base Classifier (Before Feature Engineering)
```python
accuracy = 0.430
confidence_interval = [0.333, 0.527]
```

### 2. Improved Classifier (After Feature Engineering)
```python
accuracy = 0.430
confidence_interval = [0.333, 0.527]
```

## After adding five additional features, our model's accuracy improved. These features were:

- **swear_featurize**: Detects if swear words were used.

- **opinionated_featurize**: Detects if opinionated words were used.

- **unigram_featurize**: Utilizes unigrams (individual words).

- **afinn_featurize**: Determines how positive or negative the text is on a scale from -5 to 5.

- **pos_tagging_featurize**: Tokenizes the text and labels each word with a part of speech (POS) tag.
 
We observed that some additional features, such as counting different punctuations or the number of "negative" words, did not improve performance. In fact, they either decreased accuracy or left it unchanged, so they were excluded from the final feature set.

## **Feature Impact**
- **Pos_tagging_featurize:** This feature had the most significant positive impact. It tokenizes the text and assigns each word a POS tag, which provides richer context for the words. This allowed the model to better understand the relationships between words and their contribution to the text's meaning, ultimately improving accuracy.

- Accuracy with **pos_tagging_featurize**: 0.480

- **Swear and Opinionated Features:** Surprisingly, the **swear_featurize** and **opinionated_featurize** features did not improve accuracy as expected. These features did not consistently indicate higher contentiousness, suggesting that swear words and opinionated language can be used in a variety of contexts, including non-controversial ones. This realization helped refine our understanding of how contentiousness is not solely determined by the presence of swear words or strong opinions.

## **Evaluation Metrics**

We calculated the following metrics for each class:

### Low Class:
```python
precision = 0.75  # (9/12)
recall = 0.30  # (9/30)
f1_score = 0.4286  # 2 * (precision * recall) / (precision + recall)
```
### Medium Class:
```python
precision = 0.4783  # (11/23)
recall = 0.3929  # (11/28)
f1_score = 0.4783  # 2 * (precision * recall) / (precision + recall)
```
### High Class:
```python
precision = 0.5231  # (34/65)
recall = 0.8095  # (34/42)
f1_score = 0.6355  # 2 * (precision * recall) / (precision + recall)
```
We also visualized the confusion matrix to compare the actual vs. predicted class labels. This helped identify where our model struggled, particularly with identifying Low contentiousness posts.

## **Observations**

### **Low Class:**
- Precision was high, but the recall was very low, meaning the model accurately predicted a few **Low** posts, but many were mistaken for **Medium** or **High** posts.
- The **Low** class was particularly challenging for the model to identify. Despite a relatively high precision, its recall was low due to the imbalance in the dataset (30 **Low** posts vs. 12 actual **Low** posts). This imbalance resulted in the **Low** posts being more difficult to predict accurately.
  
### **Medium Class:**
- The **Medium** class showed low recall and F1 score. Interestingly, it did not predict any **Low** class posts, which likely led to the slightly higher F1 score for **Medium**. This indicates that **Medium** class posts often overlapped with **High** class posts in terms of language, making them harder to classify accurately.

### **High Class:**
- The **High** class performed the best in terms of recall, with most of the predicted **High** posts being accurate. Despite having the lowest precision, the overall high recall contributed to a relatively high F1 score.
- The success with **High** class posts highlights the model's ability to handle more polarized or reactionary language, which was present in these posts.

## **Challenges**
1. **Label Imbalance:** The imbalance in the number of posts across the three classes (Low, Medium, High) affected the model’s performance. A balanced dataset would likely improve model performance and help it better differentiate between the categories.
2. **Labeling Uncertainty:** During annotation (AP2), we often found it challenging to differentiate between Medium and High contentiousness, leading to a tendency to label uncertain posts as Medium. A possible solution could involve adding more categories (e.g., **Somewhat Low** or **Somewhat High**) to refine the classifications.
3. **Feature Limitations:** While the **pos_tagging_featurize** feature improved accuracy, some expected features (like swear words and opinionated language) did not have the desired impact. This suggests that features like these are not always strong indicators of contentiousness, depending on the context.

## **Recommendations for Future Work**
- **Balanced Dataset**: Future iterations of the model would benefit from a more balanced dataset. Ensuring that each class is equally represented will likely improve model performance.
- **Additional Features**: Experimenting with different features, especially those related to contextual meaning (e.g., sentiment over time, word embeddings), could further improve the model's understanding of contentiousness.
- **Expanded Subreddit Selection**:  Expanding the data source to include a more diverse set of subreddits would likely improve the model’s generalizability. The current six subreddits have overlapping content styles and may limit the variety of posts used for training.

## **Conclusion**
Classifying the contentiousness of Reddit posts is a challenging task due to the inherent subjectivity in interpreting language. However, our model provides a solid first step in automating this process. With improvements in data labeling, feature selection, and dataset diversity, this model can serve as a valuable tool for understanding contentious discussions in online forums.
