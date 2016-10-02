import sklearn_pandas
from sklearn2pmml import sklearn2pmml
# from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain


def export_pipeline_to_pmml(df, pipeline, pmml_fp):
    ''' Exports a sklearn pipeline object to a pmml file.

        Inputs
        ========
            df: a pandas dataframe with dimensions
                {n_records, attributes + 1}; where the last column
                contains the target values
            pipeline: A pandas pipeline object
            pmml_fp: filepath to write out the PMML
    '''
    classifier = pipeline.steps[-1][1]
    preprocessing_steps = [x[1] for x in pipeline.steps[:-1]]
    if not preprocessing_steps:
        preprocessing_steps = None

    preprocessing_mapper = sklearn_pandas.DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], preprocessing_steps),
    ("Species", None)
    ])


    # iris_df = pandas.concat((pandas.DataFrame(features, columns=feature_columns), pandas.DataFrame(target, columns=target_columns])), axis=1)
    # iris = iris_mapper.fit_transform(iris_df)
    data_array = preprocessing_mapper.fit_transform(df)

    X = df.ix[:,:-1]
    y = df.ix[:, -1]

    classifier.fit(X, y)

    sklearn2pmml(classifier, preprocessing_mapper, pmml_fp, with_repr = True)