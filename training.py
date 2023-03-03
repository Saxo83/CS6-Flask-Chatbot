# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context #<----use if there are any issues

import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

lematizer = nltk.WordNetLemmatizer()

intents = json.loads(open('data.json').read())

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.', '-', '_', '/', '\'', '=', '+', '<', '>', '&', '£', '$', '%', '@', ';', '*']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print('BEFORE OUTPUT ENDS')

words = [lematizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lematizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    # print(bag)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# print (training)

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = tf.keras.optimizers.legacy.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=64, verbose=1)
model.save('chatbot_model.model', history)


# class TargetCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         accuracy_target = 1.00
#         loss_target = 0.001

#         if logs.get('accuracy') >= accuracy_target and logs.get('loss') <= loss_target:
#             print(f"\nAchieved {accuracy_target * 100}% accuracy and a loss below {loss_target}")
#             self.model.stop_training = True

# target = TargetCallback()