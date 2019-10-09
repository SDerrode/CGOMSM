# Filtrage des données de Seattle par CGOFMSM

Température de l'air et de la surface de la route à Seattle : https://data.seattle.gov/Transportation/Road-Weather-Information-Stations/egc4-d24i

## Description des programmes

   - **all_to_individuals.py** : programme permettant d'extraire de la base *./input/road-weather-information-stations.csv*, les informations sur les différents points de mesure. Nous nous concentrons sur le point appelé *JoseRizalBridgeNorth*.

   - **PlotTemperatures.py** : génères des figures de températures de l'air de de la surface au sol pour le point de mesure. Les résultats sont enregistrés dans le répertoire *./figures*.

   - **PlotSignals.py** : contient toutes les fonctions de plot des fichiers *Signal\*.py* décrit ci-dessous.

   - **Signal\*.py** : les programmes a exécuter dans l'ordre suivant pour le filtrage:

      1. **SignalResampling.py**: Ré-échantillonnage du signal pour réduire le nombre de données.

      2. **SignalGT.py**: Génère la vérité-terrain pour X et pour R, servant pour l'apprentissage réalisé dans l'étape suivante, et enregistre la figure *./figures/\*_GT.png*.

      3. **SignalLearning.py**: Ce script génère le fichier des paramètres de Y|R (dans un fichier appelé *Signal.param*) ainsi que les paramètres du modèle flou utilisé pour P(R1, R2). Il génère finalement la ligne de commande requise pour lancer les programmes de filtrage flou. Typiquement :

        *python3 CGOFMSM_Signals.py ./Parameters/Signal.param 4:0.15588404684200421:16.823468372812314:0.006825938566553447:0.006825938566553447 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_all_resample_5209_GT.csv 3 2 1*
      
      4. **SignalComparing.py**: Calcul des MSE permettant de comparer le signal restauré et le signal originel

Commande *python3 CGOFMSM_Signals.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_all_resample_15624_GT.csv 3 2 1*, à comparer avec cet algo : https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition
