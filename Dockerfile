# Utilisez une image de base compatible
FROM debian:bullseye-slim

# Installer les dépendances nécessaires
RUN apt-get update && apt-get install -y --fix-missing \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ajouter un alias pour python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Installer TensorFlow et TensorBoard
RUN pip3 install tensorflow tensorboard

# Définir le répertoire de travail
WORKDIR /app

# Copier le reste des fichiers du projet
COPY . .

# Exposer le port pour TensorBoard
EXPOSE 6006

# Commande par défaut pour démarrer TensorBoard
CMD ["tensorboard", "--logdir=logs", "--host=0.0.0.0"]

