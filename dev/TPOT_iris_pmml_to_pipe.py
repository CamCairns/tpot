from pmml_export import export_pipeline_to_pmml
import pandas

iris = load_iris()

iris_df = pandas.concat((pandas.DataFrame(iris.data[:, :], columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]),
                            pandas.DataFrame(iris.target, columns=["Species"])), axis=1)

export_pipeline_to_pmml(iris_df, tpot._fitted_pipeline, "cam.pmml")
