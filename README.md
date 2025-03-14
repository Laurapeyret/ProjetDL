#ProjetDL

#Objectif

Ce projet met en œuvre un modèle de réseau de neurones convolutifs (VGG-16) pour la classification d'images à grande échelle sur le dataset CIFAR-10. 
L'objectif est de démontrer que l'augmentation de la profondeur des réseaux neuronaux améliore les performances en reconnaissance d'image.

#Structure du Projet
Le projet est organisé comme suit :

src/ : Contient les fichiers sources du projet.
data_loading.py : Chargement et transformation du dataset.
model.py : Définition de l'architecture du modèle VGG-16.
train.py : Script d'entraînement du modèle et configuration de TensorBoard.
logs/ : Contient les logs d'entraînement.
Dockerfile : Configuration de l'image Docker.
README.md : Documentation du projet.

#Installation des Dépendances
Certains fichiers nécessaires sont trop volumineux pour être inclus directement dans le repository GitHub. Pour recréer l'environnement

#Configuration Python

Créer un environnement virtuel et installer les dépendances :

python3 -m venv venv
source venv/bin/activate    # Pour macOS/Linux
pip install -r requirements.txt

#Configuration Docker
1. Construire l'image Docker :
docker build -t my_project .

2. Exécuter l'entraînement du modèle :
docker run my_project python3 src/train.py

#Auteur
Laura PEYRET



#Utilisation de TensorBoard
Pour visualiser les courbes d'apprentissage enregistrées par TensorBoard, il faut utiliser la commande suivante :
docker run -p 6006:6006 -v $(pwd)/logs_local_copy:/logs my_project tensorboard --logdir=/logs --host=0.0.0.0
Ensuite, accédez à http://localhost:6006 dans votre navigateur.
