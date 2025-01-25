# Interactive Decision Tree Classifier Tool

This web application allows users to interactively train and evaluate a Decision Tree Classifier on a synthetic dataset. The app provides an easy-to-use interface to experiment with various hyperparameters of the model, visualize the decision boundaries, and evaluate model performance using metrics like accuracy, confusion matrix, feature importance, and cross-validation scores.

## Features

- **Dataset**: The app uses a synthetic dataset created using `make_moons`, a common dataset for classification tasks, with some added noise to make the problem non-trivial.
  
- **Interactive Controls**: The app offers several hyperparameters of the `DecisionTreeClassifier` that can be adjusted via the sidebar:
  - `Criterion`: Choose between Gini impurity and entropy.
  - `Splitter`: Choose between the best split or a random split.
  - `Max Depth`: Set the maximum depth of the tree.
  - `Min Samples Split`: Minimum number of samples required to split an internal node.
  - `Min Samples Leaf`: Minimum number of samples required to be at a leaf node.
  - `Max Features`: The number of features to consider when looking for the best split.
  - `Max Leaf Nodes`: Maximum number of leaf nodes in the tree.
  - `Min Impurity Decrease`: A threshold for the impurity decrease required to split a node.

- **Visualization**:
  - **Scatter Plot**: Displays the data points in the feature space, colored by their class.
  - **Decision Boundary**: Visualizes how the decision tree splits the data in feature space.
  - **Confusion Matrix**: Displays the confusion matrix to evaluate the model's performance on the test set.
  - **Feature Importance**: A bar chart showing the importance of each feature in making predictions.

- **Model Evaluation**:
  - Displays the model's accuracy on the test set.
  - Visualizes the confusion matrix to understand the model's classification performance.
  - Performs cross-validation to assess the model's performance on unseen data.
  
- **Model Export**: Allows users to download a PNG image of the trained decision tree for documentation purposes.

- **Parameter Optimization**: Uses Grid Search to find the best combination of hyperparameters to improve model performance. The grid search results in the best parameters and the highest cross-validation score.

## How to Use

1. **Adjust the hyperparameters**: Use the sidebar to modify the parameters of the Decision Tree Classifier.
2. **Train the model**: Click the "Run Algorithm" button to train the model with the selected settings.
3. **View the results**:
   - Visualize the decision boundary.
   - View the model's accuracy, confusion matrix, feature importance, and cross-validation score.
   - Download the trained decision tree as a PNG image.
4. **Optimize parameters**: Use the Grid Search feature to get recommendations for the best hyperparameters.

## Requirements

- Python 3.x
- Streamlit
- scikit-learn
- Matplotlib
- Seaborn
- graphviz

To install the required dependencies, run the following command:

```bash
pip install streamlit scikit-learn matplotlib seaborn graphviz 
```
Running the Application
To run the app, use the following command in the terminal:
```bash
streamlit run app.py
```
## Conclusion

This tool is designed for users to explore and understand the inner workings of a Decision Tree Classifier, from training to evaluation and optimization. By adjusting hyperparameters and visualizing the model's performance, users can gain insights into how decision trees classify data and how model parameters affect the results.
