import tensorflow as tf
from data_loading import load_data
from model import build_vgg_model
import os
import datetime

# Chargement des données
(x_train, y_train), (x_test, y_test) = load_data()

# Construction du modèle
model = build_vgg_model(x_train.shape[1:])

# Compilation du modèle avec normalisation des gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Affichage du modèle
model.summary()

# Configuration de TensorBoard
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Configuration d'EarlyStopping et de ReduceLROnPlateau
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Entraînement du modèle avec les callbacks
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback])

# Évaluer la perte initiale avec un petit échantillon
initial_loss = model.evaluate(x_train[:10], y_train[:10], verbose=0)
print(f"Initial loss: {initial_loss}")

# Overfitting intentionnel sur un petit échantillon
model.fit(x_train[:500], y_train[:500], epochs=10, verbose=0)

# Appliquer une planification de learning rate
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch / 30))
model.fit(x_train, y_train, epochs=5, callbacks=[lr_schedule], verbose=0)

# Recherche en grille pour test différentes combinaisons d'hyperparamètres
results = {}
log_dir_base = "logs/grid_search/"

for lr in [1e-3, 1e-4, 1e-5]:
    for batch_size in [32, 64]:
        # Re-construction du modèle pour chaque combinaison d'hyperparamètres
        model = build_vgg_model(x_train.shape[1:])
        
        # Technique de normalisation des gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        log_dir = os.path.join(log_dir_base, f"lr_{lr}_bs_{batch_size}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        print(f"Training with learning rate: {lr} and batch size: {batch_size}")
        history_grid = model.fit(x_train, y_train,
                                 epochs=5,
                                 batch_size=batch_size,
                                 validation_data=(x_test, y_test),
                                 callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback],
                                 verbose=0)
        
        final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)
        results[(lr, batch_size)] = (final_loss, final_accuracy)
        
        print(f"Results for lr={lr}, batch_size={batch_size}: Loss={final_loss:.4f}, Accuracy={final_accuracy:.4f}")

print(results)

# Sauvegarde du modèle final
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "vgg_cifar10.h5"))

print(f"Model trained and saved at 'models/vgg_cifar10.h5'")
print(f"TensorBoard logs are saved at '{log_dir}'")

