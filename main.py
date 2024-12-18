# Gerekli Kütüphaneleri Yükleyin
!pip install tensorflow matplotlib numpy

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import matplotlib.pyplot as plt

# MNIST Veri Setini Yükle
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5  # [-1, 1] aralığına normalize et
x_train = np.expand_dims(x_train, axis=-1)  # Kanal boyutunu ekle

BUFFER_SIZE = 60000
BATCH_SIZE = 128

# Veri setini karıştır ve batch'e böl
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator Modeli
def build_generator():
    model = Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = build_generator()

# Discriminator Modeli
def build_discriminator():
    model = Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

discriminator = build_discriminator()

# Kayıp Fonksiyonları ve Optimizasyon
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Eğitim Adımı
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Eğitim Döngüsü
EPOCHS = 5

for epoch in range(EPOCHS):
    for image_batch in dataset:
        train_step(image_batch)

    # Eğitim sırasında bazı üretilen görüntüleri göster
    noise = tf.random.normal([16, 100])
    generated_images = generator(noise, training=False)

    fig, axes = plt.subplots(1, 16, figsize=(16, 2))
    for i in range(16):
        axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    plt.show()

# LSTM Modeli
def build_lstm():
    model = Sequential([
        layers.LSTM(256, input_shape=(None, 100), return_sequences=True),  # Daha fazla nöron
        layers.LSTM(128),  # Daha güçlü LSTM katmanı
        layers.Dense(100)  # Çıktı gürültü boyutu
    ])
    return model

lstm = build_lstm()

# LSTM ile Zaman Serisi Oluşturma
def generate_sequence(generator, lstm, steps=10):
    sequence = []
    noise = tf.random.normal([1, 100])  # İlk gürültü vektörü

    for _ in range(steps):
        sequence.append(noise.numpy().flatten())  # Gürültüyü düzleştir
        lstm_input = np.array(sequence).reshape(1, len(sequence), 100)  # LSTM için uygun şekil
        noise = lstm.predict(lstm_input, verbose=0)  # LSTM tahmini
        noise = noise.flatten()  # Çıktıyı tek boyutlu hale getir
        noise = np.resize(noise, (1, 100))  # Çıktıyı (1, 100) boyutuna getir
        noise += tf.random.normal(noise.shape, mean=0.0, stddev=0.01)  # Gürültü ekle
        noise = tf.convert_to_tensor(noise)  # TensorFlow tensörüne dönüştür
    
    return sequence


# Üretilen Görüntüleri Görselleştir
def plot_images(sequence, generator):
    plt.figure(figsize=(10, 10))
    for i, noise in enumerate(sequence):
        generated_image = generator(np.array(noise).reshape(1, 100), training=False) 
        plt.subplot(1, len(sequence), i + 1)
        plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

# Sekans Üretimi ve Görselleştirme
sequence = generate_sequence(generator, lstm)
plot_images(sequence, generator)

# Veri setinden rastgele örnekler görselleştiriliyor
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()
