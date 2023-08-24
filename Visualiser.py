import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



class Visualiser:

    def __init__(self, **params):
        self.params = params


    def plot_2d_scatter(self, samples, feature1: int, feature2: int):

        X = samples[0]
        y = samples[1]
        # Select the first two features for plotting
        x1 = X[:, feature1]
        x2 = X[:, feature2]

        # Create a DataFrame for plotting
        df = pd.DataFrame({'Feature 1': x1, 'Feature 2': x2, 'Class': y})

        # Map class labels to more descriptive names
        class_labels = {0: 'Majority Class', 1: 'Minority Class'}
        df['Class'] = df['Class'].map(class_labels)

        # Create the scatter plot using Plotly
        fig = px.scatter(
            df,
            x='Feature 1',
            y='Feature 2',
            color='Class',
            marginal_x= "histogram", 
            marginal_y= "histogram",
            title='2D Scatter Plot of Imbalanced Data',
            labels={'Feature 1': f'Feature {feature1}', 'Feature 2': f'Feature {feature2}'}
        )

        fig.show()
        #return fig
    

    def plot_2d_scatter_multiple_datasets(datasets, feature1=0, feature2=1, title="2D Scatter Plot", color_sequence=None):
        """
        Create a 2D scatter plot for multiple datasets.

        Args:
            datasets (list of tuple): List of tuples, where each tuple contains a dataset (X, y) and a label.
            feature1 (int): Index of the first feature to plot (default is 0).
            feature2 (int): Index of the second feature to plot (default is 1).
            title (str): Title of the plot (default is "2D Scatter Plot").
            color_sequence (list): List of colors for dataset labels (default is None,
                                which uses Plotly's default color sequence).

        Returns:
            None (displays the plot).
        """

        # Create the scatter plot using Plotly
        if color_sequence is None:
            color_sequence = px.colors.qualitative.Plotly

        fig = go.Figure()

        for i, (X, y, label) in enumerate(datasets):
            # Select the two features for plotting
            x1 = X[:, feature1]
            x2 = X[:, feature2]

            # Map class labels to more descriptive names
            class_labels = {val: f'{label} - Class {val}' for val in np.unique(y)}

            df = pd.DataFrame({'Feature 1': x1, 'Feature 2': x2, 'Class': y})
            df['Class'] = df['Class'].map(class_labels)

            for class_val in np.unique(y):
                data = df[df['Class'] == f'{label} - Class {class_val}']
                fig.add_trace(go.Scatter(
                    x=data['Feature 1'],
                    y=data['Feature 2'],
                    mode='markers',
                    name=f'{label} - Class {class_val}',
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        color=color_sequence[i % len(color_sequence)],
                    )
                ))

        fig.update_layout(
            title=title,
            xaxis_title=f'Feature {feature1}',
            yaxis_title=f'Feature {feature2}',
            legend=dict(x=0.85, y=1.0),
        )

        fig.show()



    def plot_3d_scatter(self, samples, feature_x: int, feature_y: int, feature_z: int):

        X = samples[0]
        y = samples[1]

        x1 = X[:, feature_x]
        x2 = X[:, feature_y]
        x3 = X[:, feature_z]
        #print(x[:20])
        #print(y[:20])
        #print(z[:20])

        # Create a DataFrame for plotting
        df = pd.DataFrame({'Feature 1': x1, 'Feature 2': x2, 'Feature 3': x3, 'Class': y})

        # Map class labels to more descriptive names
        class_labels = {1: 'Minority Class', 0: 'Majority Class'}
        df['Class'] = df['Class'].map(class_labels)

        fig = px.scatter_3d(
            df,
            x = 'Feature 1',
            y = 'Feature 2',
            z = 'Feature 3',
            color='Class',
            size = 5* np.ones(len(X)),
            size_max = 15,
            opacity = 0.9,
            width = 1400,
            height = 1000,
            #margin = dict(b=50),
            title='3D Scatter Plot of Imbalanced Data',
            labels={'Feature 1': f'Feature {feature_x}', 'Feature 2': f'Feature {feature_y}', 'Feature 3': f'Feature {feature_z}'}
        )

        fig.show()



    def plot_3d_scatter_multi_class(self, class_data_dict, class_labels):
        """
        Plot samples from multiple classes in a 3D scatter plot.

        Args:
            class_data_dict (dict): A dictionary where keys are class labels and values are 3D data arrays.
            class_labels (list): A list of class labels in the order they should appear in the legend.

        Example:
            class_data_dict = {
                'Class A': np.random.rand(30, 3),
                'Class B': np.random.rand(30, 3),
                'Class C': np.random.rand(30, 3)
            }
            class_labels = ['Class A', 'Class B', 'Class C']
        """

        traces = []

        for label, data in zip(class_labels, class_data_dict.values()):
            trace = go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='markers',
                name=label,
                marker=dict(
                    size=5,
                    opacity=0.7,
                )
            )
            traces.append(trace)

        layout = go.Layout(
            title='3D Scatter Plot for Multiple Classes',
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
            ),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()










if __name__ == "__main__":

    import scipy.stats as st
    from Data_Generator import Multi_Modal_Dist_Generator
    

    """
    Large Normal
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    n = 3
    #set the parameter dictionary for the MV normal. sigma is the standard deviation
    mu_c0_1 = np.zeros(shape = (n))
    mu_c0_2 = 3 * np.ones(shape = (n))
    mu_c1 = [2, -2, 2, -2, 2, -2, 2, -2, 2, -2]

    sigma_c0_1 = np.ones(shape=(n,n)) + np.diag(np.ones(shape = (n)))
    sigma_c0_2 = np.zeros(shape=(n,n)) + np.diag(np.ones(shape = (n)))
    sigma_c1 = np.array([[3, 0, 1, 0, 1, 0, 0, 0, -1, -1],
                         [0, 3, 1, 0, 0, 0, 1, 0, 0, -1],
                         [1, 1, 3, 0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 0, 3, 0, 0, 1, 0, 0, 0],
                         [1, 0, 1, 0, 3, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
                         [0, 1, 0, 1, 0, 1, 3, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 3, 0, 1],
                         [-1, 0, 1, 0, 1, 0, 1, 0, 3, 1],
                         [-1, -1, 0, 0, 0, 0, 0, 1, 1, 3]])

    #print(np.allclose(sigma_c1, sigma_c1.T))
    #print(np.linalg.eigvals(sigma_c1))

    mu_c1 = mu_c1[:n]
    sigma_c1 = sigma_c1[ :n, :n]
    print(sigma_c1)

    distributions = [st.multivariate_normal]

    #set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
    dist_parameter_dicts = [{'modes_c0': 2,
                            'modes_c1': 1,
                            'mixing_weights_c0': [0.3, 0.7],
                            'mixing_weights_c1': [0.3, 0.7],
                            'params_c0': {'mean': [mu_c0_1, mu_c0_2], 'cov': [sigma_c0_1, sigma_c0_2]},
                            'params_c1': {'mean': [mu_c1], 'cov': [sigma_c1]}
                            },
    ]

    size = [9000, 1000]
    """

    """
    Visualise
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    dist_gen_spec = Multi_Modal_Dist_Generator(distributions, dist_parameter_dicts, size)


    dist_gen_spec.create_data()
    spec_samples = (dist_gen_spec.X, dist_gen_spec.y)
    #spec_samples = dist_gen_spec.prepare_data(0.2)

    visualiser = Visualiser()

    visualiser.plot_2d_scatter(spec_samples, 0, n-1)
    visualiser.plot_3d_scatter(spec_samples, 0, 1, 2)
    """


    """
    Plotly Experiments
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    df = px.data.iris()
    print(df)
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", symbol="species")
    
    fig.show()