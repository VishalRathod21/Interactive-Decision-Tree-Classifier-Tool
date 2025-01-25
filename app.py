import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
import seaborn as sns
import graphviz

st.title("Interactive Decision Tree Classifier Tool")
st.subheader("Visualize and Evaluate Decision Tree Classifier Performance")

# Function to draw meshgrid for decision boundary visualization
def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier Settings")

criterion = st.sidebar.selectbox(
    'Criterion',
    ('gini', 'entropy')
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ('best', 'random')
)

max_depth = int(st.sidebar.number_input('Max Depth'))

min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

max_features = st.sidebar.slider('Max Features', 1, 2, 2,key=1236)

max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')

# Set the title for the graph
ax.set_title("Scatter Plot of Data Points", fontsize=16)

orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    orig.empty()

    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    clf = DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth,random_state=42,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree  " + str(round(accuracy_score(y_test, y_pred), 2)))

    tree = export_graphviz(clf,feature_names=["Col1","Col2"])

    st.graphviz_chart(tree)

    # Add divider line
    st.markdown("---")

    # Feature Importance Visualization
    st.subheader("Feature Importance Visualization")
    st.write("Displays the importance of each feature in the model, showing which features contribute most to predictions.")
    feature_importances = clf.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(["Col1", "Col2"], feature_importances, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance Visualization')
    st.pyplot(fig)

    # Add divider line
    st.markdown("---")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write("Shows a matrix to visualize how well the model classifies data, highlighting correct and incorrect predictions.")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Add divider line
    st.markdown("---")

    # Cross-Validation Score
    st.subheader("Cross-Validation Score")
    st.write("Calculates the model's accuracy using cross-validation to assess its performance on unseen data.")
    cv_scores = cross_val_score(clf, X, y, cv=5)
    st.write(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")

    # Add divider line
    st.markdown("---")

    # Export Decision Tree as PNG and Download
    st.subheader("Export Decision Tree as PNG")
    st.write("Allows exporting the trained decision tree as a PNG file for documentation or reporting purposes.")
    dot_data = export_graphviz(clf, out_file=None, feature_names=["Col1", "Col2"], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph_path = "decision_tree"
    graph.render(graph_path, format='png', cleanup=True)
    with open(f"{graph_path}.png", "rb") as file:
        btn = st.download_button(
            label="Download Decision Tree as PNG",
            data=file,
            file_name="decision_tree.png",
            mime="image/png"
        )

    # Add divider line
    st.markdown("---")

    # Parameter Optimization Recommendation
    st.subheader("Parameter Optimization Recommendation")
    st.write("Provides suggestions for the best hyperparameters using Grid Search to improve model accuracy.")
    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    st.write("Recommended Parameters:", grid_search.best_params_)
    st.write("Best Cross-Validation Score:", grid_search.best_score_)
