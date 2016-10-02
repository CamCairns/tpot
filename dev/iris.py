import datetime
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()

# iris.data[0:5], iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

tpot = TPOTClassifier(population_size=30, generations=3, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

utc_now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
tpot.export('tpot_iris_pipeline_{}.py'.format(utc_now))


from pmml_export import export_pipeline_to_pmml
import pandas

iris_df = pandas.concat((pandas.DataFrame(iris.data[:, :], columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]),
                            pandas.DataFrame(iris.target, columns=["Species"])), axis=1)
export_pipeline_to_pmml(iris_df, tpot._fitted_pipeline, "cam_{0}.pmml".format(utc_now))
