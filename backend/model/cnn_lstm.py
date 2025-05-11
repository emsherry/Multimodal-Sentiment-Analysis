# backend/model/cnn_lstm.py

from tensorflow.keras import layers, models

def build_cnn_lstm(input_shape=(128, 256, 1), num_classes=8):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Reshape((30, -1)))  # Tune based on spectrogram size
    
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
