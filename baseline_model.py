import numpy as np
import time
import psutil
import pandas as pd
import imageio
import os
import pickle

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras import backend as K
import matplotlib
import matplotlib.pyplot as plt

##### Beginn des gesamten Codes
beginn = time.time()

##### Beginn der execution-time
start_time = time.time()

##### Einlesen der Daten img_align_celeba\\
whiteSpaceRegex = "\\s";
file = open("C:\\Users\\user\\Desktop\\CelebA\\list_identity_celeba_alt.txt", "r")
file.readline()
path = "C:\\Users\\user\\Desktop\\CelebA\\Neu_Original\\";

# Gibt eine Liste "arr" zurück, die die Namen der Einträge aus "path" enthält -> also eine Liste der Namen der Bilder
allItems = os.listdir(path);

arr = []
for names in allItems:
    if names.endswith(".jpg"):
        arr.append(names)

# Erstellt einen Array "celeb_images" mit der Shape "Anzahl Bilder"(len(arr)) und 2352. 2352 beschreibt die Breite des Arrays.
celeb_images = np.zeros((len(arr), 2352))

# Erstellt eine leere Liste "celeb_labels"
celeb_labels = []

# für jede Nummer ("idx") und filename in dem durchnummerierten Array("arr"), der die Bilder enthält...
for idx, filenames in enumerate(arr):
    # ...speichere den Pfad + dem filename in "image_file"
    image_file = path + filenames

    # lies das bild ein und speichere es in "image"
    image = imageio.imread(image_file, as_gray = False, pilmode = "RGB")

    # verändere den shape des "image" in "28x28" und speichere es in img
    img = np.resize(image, (3, 28, 28))

    # speichere das img als array ab.
    img = np.asarray(img, dtype=np.float32)

    # bringe den array "img" in einen eindimensionalen array und speichere ihn in der "idx"ten Zeile in den Spalten 0-2352
    celeb_images[idx, 0:2352] = img.flatten()

    # lies die Zeile mit dem filename aus dem Dokument mit den identitys und speichere es in string
    string = file.readline()

    # splitte die eingelesene Zeile anhand eines Leerzeichens auf, da vor dem label/Namen noch der Bild-Dateiname steht und speichere die beiden infos in einer liste
    splitStr = string.split()

    # Speichere nicht den Bild-Dateinnamen, der in "splitStr[0]" ist, sondern nimm nur den Namen ("splitStr[1]") und füge ihn der Liste celeb_labels hinzu
    celeb_labels.append(splitStr[1])

# Konvertiere die Liste celeb_images in einen array, deren Werte als float32 gespeichert werden.
celeb_images = np.asarray(celeb_images, dtype=np.float32)
# Konvertiere die Lste celeb.labels in einen array, deren Werte als string gespeichert werden
celeb_labels = np.asarray(celeb_labels, dtype=np.str)

# Normalisieren der Bilder
celeb_images /= 255

#with open("ground_data.pickle","wb") as f:
#    pickle.dump([celeb_images, celeb_labels],f)

time_dateneinlesen = time.time() - start_time
print("loading_time:", time_dateneinlesen)

####################################################################
##### Teilen der Daten in Train-Test-Sets
#### Aufgrund der Menge der Daten wird sich nicht für einen 70/30 oder 80/20 Split entschieden, sondern einen 90/10 Split
### Um sowohlim Test- als auch im Train-set alle Klassen enthalten zu haben, wird ein statified ShuffleSplit genutzt
start_time = time.time()

# speichere die Bilddaten in einen DataFrame df_celebs
df_celebs = pd.DataFrame(celeb_images)

# ergänze die entsprechenden labels zu den Bildern in der Spalte "labels"
df_celebs['labels'] = pd.DataFrame(celeb_labels)

# zähle wie häufig jedes label im Datensatz vorkommt und speichere es in einem neuen DataFrame unique_celebs
unique_celebs = pd.DataFrame(np.unique(df_celebs["labels"], return_counts=True))

# transponiere den DataFrame unique_celebs
unique_celebs = unique_celebs.transpose()

#### Exploration der Anzahl examples je label, um ggf. die Sample Anzahl anhand der vorhandenen Examples je Label zu reduzieren
# häufigste label
max_example = unique_celebs.max()
# seltenste label
min_example = unique_celebs.min()
# Mittelwert der Anzahl labels
mean_example = unique_celebs[1].mean()
# Median der Anzahl labels
median_example = unique_celebs[1].median()
# Modus der Anzahl labels
modal_example = unique_celebs[1].mode()

# wähle aus unique_celebs die labels aus, die seltener als 7 mal in den Grunddaten vorkommen
few_celebs = unique_celebs[(unique_celebs[0]=="Charlott_Cordes") | (unique_celebs[0]=="Christopher_Lee") | (unique_celebs[0]=="Elizabeth_McGovern")]

# speichere in einem neuen Dataframe df_final alle Reihen des DataFrames df_celebs, deren label nicht in dem DataFrame few_celebs vorkommt
df_final = df_celebs.loc[df_celebs["labels"].isin(few_celebs[0])]

# konvertiere den DataFrame df_final wieder in 2 arrays - 1 array mit den images und 1 array mit den entsprechenden labels
celeb_images = np.array(df_final.loc[:, df_final.columns!= "labels"])
celeb_labels = np.array(df_final["labels"])

time_cleaning = time.time() - start_time
print("cleaning_time:", time_cleaning)
##################

start_time = time.time()
shufflesplit = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in shufflesplit.split(celeb_images, celeb_labels):
    train_images, test_images = celeb_images[train_index], celeb_images[test_index]
    train_labels, test_labels = celeb_labels[train_index], celeb_labels[test_index]

time_splitting = time.time() - start_time
print("splitting_time:", time_splitting)

##### Bilddaten umformen
### Festlegen der Bildinformationen
channels = 3
height = 28
width = 28

### Umformung der Bilder
train_images = train_images.reshape(train_images.shape[0], channels, height, width)
test_images = test_images.reshape(test_images.shape[0], channels, height, width)

# ## Umformung der Labels, damit jedes label eine Kategorie darstellt. Dazu ist One-Hot-Encoding bzw.
# Dichotomisierung notwendig
train_labels = pd.get_dummies(train_labels)
test_labels = pd.get_dummies(test_labels)

class_labels = train_labels.columns.values

train_labels = train_labels.to_numpy()
test_labels = test_labels.to_numpy()

##### Modelling
### Festlegen der Anzahl Klassen
number_of_classes = train_labels.shape[1]

#with open("small_data.pickle","wb") as f:
#    pickle.dump([number_of_classes, channels, height, width, train_images, train_labels, test_images, test_labels, class_labels],f)


# Beginn des neuronalen Netzes
np.random.seed(45)

# Beginn des neuronalen Netzes
cnn = models.Sequential()

# Füge einen Covolutional-Layer mit 64 Filtern und einem 5x5 Kernel hinzu. Die Aktivierungsfunktion dieses Layers ist die ReLU-Funktion.
cnn.add(Conv2D(filters=64,
               kernel_size=(5, 5),
               input_shape=(channels, height, width),
               activation="relu",
               data_format="channels_first",
               padding="same"))

# Füge eine Max-Pooling Schicht hinzu mit einem Pooling-Kernel der größe 2x2
cnn.add(MaxPooling2D(pool_size=(2, 2),
                    data_format="channels_first",
                     padding="same"))

# Füge eine Dropout Schicht hinzu, um Overfitting zu vermeiden (user.phil.hhu.de/~petersen/SoSe17_Teamprojekt/AR/neuronalenetze.html)
cnn.add(Dropout(0.5))

# Füge eine Flatten-Schicht hinzu, um den Output der vorherigen Schicht in einen Ein-Dimensionalen-Vektor zu konvertieren
cnn.add(Flatten(data_format="channels_first"))

# Füge einen Dense-Layer hinzu, mit 128 Neuronen und der ReLU-Aktiverungsfunktion
cnn.add(Dense(128, activation="relu"))

# Füge eine erneute Dropout-Schicht hinzu
cnn.add(Dropout(0.5))

# Füge einen weiteren Dense-Layer hinzu, mit der Anzahl der Klassen und der Softmax-Aktivierungsfunktion
cnn.add(Dense(number_of_classes, activation="softmax"))

# Kompilieren des Neuronalen Netzes. Verlustfunktion ist die kategorielle Kreuzentropie und die Performmace wird durch die Accuracy bestimmt
# als Optimierungsverfahren wird rmsprop also root mean square propagation verwendet
cnn.compile(loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"])

# Trainiere das Neuronale Netz mit den Trainingsbildern und ihren Labels. Die Anzahl Epochen beträgt 2 und die Batch-Size beträgt 128.
# die Validierung soll auf den Testdatensätzen stattfinden
start_time = time.time()

cnn.fit(train_images,
        train_labels,
        epochs=40,
        verbose=0,
        batch_size=32,
        validation_data=(test_images, test_labels))

time_fitting = time.time() - start_time
print("fitting time:", time_fitting)


# Accuracy
plt.plot(cnn.history.history['acc'])
plt.plot(cnn.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# loss
plt.plot(cnn.history.history['loss'])
plt.plot(cnn.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


time_end = time.time() - beginn
print("execution_time:", time_end)

print(cnn.summary())