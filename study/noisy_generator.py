import pandas as pd
from sklearn.datasets import make_classification
from loguru import logger
import plotly.express as px
from sklearn.model_selection import train_test_split

class ImbalancedNoisyDataGenerator:
    def __init__(self, 
                 class_ratio=0.5, 
                 n_samples=1000, 
                 n_features=2, 
                 distance=1.0, 
                 flip_y=0,
                 test_size = 0.2
                ):
        """
        Initialize the ImbalancedDataGenerator.

        Parameters:
        - class_ratio (float): The ratio of the minority class samples to the total samples.
        - n_samples (int): The total number of samples to be generated.
        - n_features (int): The number of features (dimensions) in the dataset.
        - distance (float): The distance between the centers of the two classes. A higher value indicates more separability.
        - random_state (int or RandomState, optional): Seed for random number generation.
        """
        self.class_ratio = class_ratio
        self.n_samples = n_samples
        self.n_features = n_features
        self.distance = distance
        self.random_state = 123
        self.flip_y = flip_y
        self.test_size = test_size

        self.n_minority_samples = int(self.class_ratio * self.n_samples)
        self.n_majority_samples = self.n_samples - self.n_minority_samples

        self.X = None
        self.y = None

    def generate_data(self):
        """
        Generate imbalanced data using sklearn's make_classification method.

        Returns:
        - X (numpy array): Array of shape (n_samples, n_features) containing feature values.
        - y (numpy array): Array of shape (n_samples,) containing class labels.
        """
        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=int(self.n_features / 2),
            n_redundant=int(self.n_features / 2),
            n_clusters_per_class=1,
            weights=[self.class_ratio, 1 - self.class_ratio],
            flip_y=self.flip_y,
            class_sep=self.distance,
            random_state=self.random_state
        )

        # Introduce random noise to the feature values
        noise_std = np.random.uniform(0.01, 0.1)
        self.X += np.random.normal(loc=0, scale=noise_std, size=self.X.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        return X_train, X_test, y_train, y_test

    def plot_2d(self,feature1: int, feature2: int):

        if self.X is None or self.y is None:
            raise ValueError("Data has not been generated yet. Call generate_data() method first.")

        # Select the first two features for plotting
        x1 = self.X[:, feature1]
        x2 = self.X[:, feature2]

        # Create a DataFrame for plotting
        df = pd.DataFrame({'Feature 1': x1, 'Feature 2': x2, 'Class': self.y})

        # Map class labels to more descriptive names
        class_labels = {0: 'Minority Class', 1: 'Majority Class'}
        df['Class'] = df['Class'].map(class_labels)

        # Create the scatter plot using Plotly
        fig = px.scatter(
            df,
            x='Feature 1',
            y='Feature 2',
            color='Class',
            title='2D Scatter Plot of Imbalanced Data',
            labels={'Feature 1': f'Feature {feature1}', 'Feature 2': f'Feature {feature2}'}
        )
        return fig


#if __name__=="__main__":

 #   data_generator = ImbalancedNoisyDataGenerator(class_ratio=0.1, n_samples=1000, n_features=10, distance=2, flip_y=0.01)
  #  smote_balancer = SMOTE(sampling_strategy='auto', random_state=42)
   # data_balancer = DataBalancer(balancer=smote_balancer)


    #fig = data_generator.plot_2d(0,1)
    #fig.show()
    
    
    