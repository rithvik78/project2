# Feature Selection 

- Forward Selection
- Backward Elimination

Feature selection is the process of selecting a subset of features from a dataset to be used for training and testing a model. This is important because using a subset of features can improve the performance of a model and make it easier to interpret. 

In the backward elimination algorithm, the initial set of features includes all features in the dataset. At each iteration, the algorithm removes the feature that results in the greatest improvement in accuracy, until no further improvement can be made.

In the forward selection algorithm, the initial set of features is empty. At each iteration, the algorithm adds the feature that results in the greatest improvement in accuracy, until no further improvement can be made.

Both algorithms use cross-validation to evaluate the accuracy of the model with different subsets of features. The final set of features selected by the algorithms is the one that results in the highest accuracy.


To run the file 

```sh
python3 main.py
```

and then input the name of file on which Feature Selection is to be done     
then select option 1 or 2 to choose between Forward Selection or Backward Elimination

---


CS170 Project 2