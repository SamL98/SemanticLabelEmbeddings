# Semantic Label Embeddings

Create word-like embeddings for the Pascal VOC2012 labels.

## Motivation

Label embeddings should transform the onehot labels of the dataset into information-rich vectors where two embeddings are close to one another if they coexist in many ground truth masks.

It is expected that these coexistences will also provide some semantic information.
