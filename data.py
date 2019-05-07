import pandas as pd
import datetime
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import ast
from sklearn import preprocessing
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import auc
import matplotlib.pyplot as plt

activity_classes = {
    "fitness": ["Running", "Football"],
    "entertainment": ["Video games", "In computer", "Watching TV", "Movie"],
    "transportation": ["In vehicle", "In bus", "On bus stop", "Train"],
    "recreation": ["At home", "Picnic", "Walk", "Cooking", "Pause", "Shop", "Shopping& wearing", "Walking&party"],
    "sleep": ["Sleep"],
    "eat": ["Eat"],
    "work": ["Meeting", "Work"],
    "miscellaneous": ["Phone was out of the pocket (forgot)", "Took off glasses"]
}

activities = {
    "fitness": [1, 0, 0, 0, 0, 0, 0, 0],
    "entertainment": [0, 1, 0, 0, 0, 0, 0, 0],
    "transportation": [0, 0, 1, 0, 0, 0, 0, 0],
    "recreation": [0, 0, 0, 1, 0, 0, 0, 0],
    "sleep": [0, 0, 0, 0, 1, 0, 0, 0],
    "eat": [0, 0, 0, 0, 0, 1, 0, 0],
    "work": [0, 0, 0, 0, 0, 0, 1, 0],
    "miscellaneous": [0, 0, 0, 0, 0, 0, 0, 1]
}

MAX_DIFFERENCE = datetime.timedelta(0,0,500000)
ZERO_TIME = datetime.timedelta(0,0)


class Activity():
    def __init__(self, name=None, start=None, end=None):
        if name is not None:
            for n, values in activity_classes.items():
                if name in values:
                    self.name = n
            # print(self.name)
        else:
            self.name = None
        if start is not None:
            self.start = get_time(start)
        else:
            self.start = None
        if end is not None:
            self.end = get_time(end)
        else:
            self.end = None

    def set_name(self, name):
        for n, values in activity_classes.items():
            if name in values:
                self.name = n
        self.fake_name = name

    def set_start_end(self, start, end):
        self.start = get_time(start)
        self.end = get_time(end)

    def get_class_number(self):
        return activities[self.name]

    def in_between(self, date_time):
        return self.start <= date_time <= self.end


def make_data():
    with open("human-activity-smart-devices/glasses.csv") as g:
        glasses_reader = csv.reader(g)
        with open("human-activity-smart-devices/smartwatch.csv") as w:
            watch_reader = csv.reader(w)
            lines = []
            glasses_rows = []
            glasses_columns = []
            watch_rows = []
            watch_columns = []
            for i, row in enumerate(glasses_reader):
                if i == 0:
                    glasses_columns = row
                else:
                    glasses_rows.append(row)
            for i, row in enumerate(watch_reader):
                if i == 0:
                    watch_columns = row
                else:
                    watch_rows.append(row)

            iter_glasses = iter(glasses_rows)
            iter_watches = iter(watch_rows)
            row_glass = next(iter_glasses)
            row_watch = next(iter_watches)

            write = False
            stop_glass = False
            stop_watch = False
            counter = 0
            while True:
                if row_watch[1] == "heart_rate":
                    if stop_watch and stop_glass:
                        break

                    delta = get_time(row_glass[1]) - get_time(row_watch[2])
                    if abs(delta) < MAX_DIFFERENCE:

                        if delta < ZERO_TIME:
                            lines.append([get_time(row_glass[1])]+ row_glass[3:] + [ast.literal_eval(row_watch[3])[0]])
                        else:
                            lines.append([get_time(row_watch[2])] + row_glass[3:] + [ast.literal_eval(row_watch[3])[0]])
                        counter += 1
                        row_watch = next(iter_watches)
                    elif delta < ZERO_TIME:
                        if write:
                            lines.append([get_time(row_watch[2])]+ ["-" for i in range(10)] + [ast.literal_eval(row_watch[3])[0]])
                            counter += 1
                            write = False
                        else:
                            write = True

                        try:
                            if not stop_glass:
                                row_glass = next(iter_glasses)
                        except StopIteration:
                            stop_glass = True
                    elif delta > ZERO_TIME:
                        if write:
                            lines.append([get_time(row_glass[1])]+ row_glass[3:]+["-"])
                            counter += 1
                            write = False
                        else:
                            write = True

                        try:
                            if not stop_watch:
                                row_watch = next(iter_watches)
                        except StopIteration:
                            stop_watch = True
                    else:

                        raise KeyboardInterrupt()
                else:
                    if stop_watch and stop_glass:
                        break
                    try:
                        if not stop_watch:
                            row_watch = next(iter_watches)
                    except StopIteration:
                        stop_watch = True
                    try:
                        if not stop_glass:
                            row_glass = next(iter_glasses)
                    except StopIteration:
                        stop_glass = True
            columns = ['datetime'] + glasses_columns[3:] + ['heart_beat']

            """"
            for row_glass in glasses_rows:
                for row_watch in watch_rows:
                    if row_watch[1] == "heart_rate":
                        delta = get_time(row_glass[1]) - get_time(row_watch[2])
                        if abs(delta) < MAX_DIFFERENCE:
                            lines.append(row_glass[3:]+ [ast.literal_eval(row_watch[3])[0]])
                        elif delta > MAX_DIFFERENCE:
                            lines.append(row_glass)
                        elif delta < ZERO_TIME:
                            lines.append(row_glass)
                        else:
                            print("SHOULDN'T BE HERE")
                            raise KeyboardInterrupt()
                            """
            return columns, lines


def read_report():
    with open("human-activity-smart-devices/report.csv") as f:
        csv_reader = csv.reader(f)
        lines = []
        for i, row in enumerate(csv_reader):
            lines.append(row)
        list_activities = []
        for line in lines[1:]:
            list_activities.append(Activity(line[1], line[3], line[4]))

    return list_activities


def get_time(string):
    split1 = string.split(" ")
    ymd = split1[0].split("-")
    if len(ymd) < 2:
        ymd = split1[0].split("/")
    hms = split1[1].split(":")
    if len(hms) > 2:
        sm = hms[2].split(".")
    else:
        sm = ["0", "0"]
    return datetime.datetime(int(ymd[0]), int(ymd[1]), int(ymd[2]), int(hms[0]), int(hms[1]), int(sm[0]), int(float("0." + sm[1])*1000000))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def experiment(data, columns, y_data):
    df = pd.DataFrame(data, columns=columns)
    ydf = pd.DataFrame(y_data)
    x_train, x_test = df[:4698], df[4698:]
    y_train, y_test = ydf[:4698], ydf[4698:]
    scaler = preprocessing.StandardScaler()
    names = x_train.columns
    scaled_df = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(scaled_df, columns=names)

    names = x_test.columns
    scaled_df = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(scaled_df, columns=names)
    x_test = np.expand_dims(x_test, axis=2)
    x_train = np.expand_dims(x_train, axis=2)


    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=2, padding='same', activation='relu', input_shape=(10,1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=3)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    y_pred_keras = model.predict(x_test)
    y_test = y_test.values

    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')

    list_of_keys = list(activities.keys())
    print(list_of_keys)
    for i in range(8):
        print(y_test[:, i])
        print(y_pred_keras.shape)
        fpr, tpr, threshold = roc_curve(y_test[:, i], y_pred_keras[:, i])  # YOUR CODE HERE construct ROC Curve
        plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(list_of_keys[i], auc(fpr, tpr)))


    # fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    # auc_keras = auc(fpr_keras, tpr_keras)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

    new_y = []
    for example in y_test:
        for i, e in enumerate(example):
            if e == 1:
                new_y.append(i)
                break
    new_y_pred = []
    for example in y_pred_keras:
        for i, e in enumerate(example):
            if e == 1:
                new_y_pred.append(i)
                break
    print(confusion_matrix(new_y, new_y_pred, activities.keys()))
    # plot_confusion_matrix(y_test, y_pred_keras, activities.keys())





if __name__ == "__main__":
    # a = get_time("2017/06/29 08:00:00.406")
    # b = get_time("2017/06/29 08:00:00.426")
    #
    # print(b-a)
    # print(a-b)

    acts = read_report()
    columns, data = make_data()

    complete_lines = []
    for line in data:
        if "-" not in line:
            complete_lines.append(line)
    final_data = []


    for line in complete_lines:
        for activity in acts:
            if activity.in_between(line[0]):
                final_data.append(line + [activity.get_class_number()])
                break

    final_data = sorted(final_data, key=lambda x: x[0])
    mid_data = []
    for f in final_data:
        mid_data.append(f[1:])
    float_data = []
    y_data = []
    for f in mid_data:
        float_data.append([float(x) for x in f[:-1]])
        y_data.append(f[-1])
    experiment(float_data, columns[1:], y_data)



    # print(report.shape)
    # print(glasses.shape)
    # print(watch.shape)
