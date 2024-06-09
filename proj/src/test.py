import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Carica il modello addestrato
model = load_model('res/modello_cnr_alzheimer.h5')


# Funzione per prevedere la categoria data un'immagine
def predict_category(image_path, model):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizza l'immagine

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Mappa l'indice della classe predetta alla sua etichetta
    labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    predicted_label = labels[predicted_class[0]]

    return predicted_label


# Testa l'immagine e ottieni la categoria predetta
image_path = '../dataset/test/NonDemented/26 (88).jpg'
predicted_category = predict_category(image_path, model)
print("Predicted Category:", predicted_category)
