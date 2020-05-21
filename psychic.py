import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#   SETUP
PREDICT = "G3"
PREDICTORS = ["G1", "G2", "studytime", "absences", "failures", "health"]


class FileToArray:

    def __init__(self, name, sep, retain):
        data = pandas.read_csv(name, sep=sep)
        self.data = data[retain]

    def get_data(self, **kwargs):
        constraint = kwargs.get('constraint', None)
        if not constraint:
            return self.data
        else:
            return self.data[constraint]


class Psychic:

    def __init__(self, csv_file_name, sep, predict, predictor):
        self.LINEAR_REGRESSION = linear_model.LinearRegression()
        self.RAW_DATA = FileToArray(csv_file_name, sep, numpy.concatenate([predictor, [predict]]))
        self.PREDICT_DATA_ARRAY = numpy.array(self.RAW_DATA.get_data(constraint=predict))
        self.PREDICTOR_DATA_ARRAY = numpy.array(self.RAW_DATA.get_data(constraint=predictor))

    def predict(self, **kwargs):
        predictor = kwargs.get('predictor', None)
        training_sample_pct = kwargs.get('training_sample_pct', 0.1)

        if not predictor:
            predictorDataset_train, predictorDataset_test, predictDataset_train, predictDataset_test = sklearn.model_selection.train_test_split(
                self.PREDICTOR_DATA_ARRAY, self.PREDICT_DATA_ARRAY, test_size=training_sample_pct)
            self.LINEAR_REGRESSION.fit(predictorDataset_train, predictDataset_train)
            TEST_DATA_PREDICTIONS = self.LINEAR_REGRESSION.predict(predictorDataset_test)

            for index in range(len(TEST_DATA_PREDICTIONS)):
                print(
                    f"We predict a score of {TEST_DATA_PREDICTIONS[index]} when the actual score is {predictDataset_test[index]}. The prediction was made based on the data: {predictorDataset_test[index]}")

        else:
            self.LINEAR_REGRESSION.fit(self.PREDICTOR_DATA_ARRAY, self.PREDICT_DATA_ARRAY)
            print(self.LINEAR_REGRESSION.predict(predictor))


Tom_The_Psychic = Psychic("student-mat.csv", ";", "G3", ["G1", "G2", "studytime", "absences", "failures", "health"])
Tom_The_Psychic.predict()