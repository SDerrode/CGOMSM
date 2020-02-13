# Filtrage des données de Zied (2 séries de 10 000 valeurs représentant X et Y)

## Description des programmes

Les programmes à exécuter dans l'ordre suivant pour le filtrage:

  1. **python3 SignalExtracting.py**: Extraction d'un sous-échantillon des séries temporelles pour réduire le nombre de données.

  2. **python3 SignalGT.py**: Génère la vérité-terrain pour R avec Y, servant pour l'apprentissage réalisé dans l'étape suivante, et enregistre la figure *./figures/\*_GT.png*.

  **python3 SignalGT2.py**: Génère la vérité-terrain pour R avec X, servant pour l'apprentissage réalisé dans l'étape suivante, et enregistre la figure *./figures/\*_GT.png*.

  3. **python3 SignalLearning.py**: Ce script génère le fichier des paramètres de Y|R (dans un fichier appelé *Signal.param*) ainsi que les paramètres du modèle flou utilisé pour P(R1, R2). Il génère finalement la ligne de commande requise pour lancer les programmes de filtrage flou. Typiquement :

     *python3 CGOFMSM_Signals.py ./Parameters/SignalDOE.param 4:0.18353174603174605:0.8091517857142845:0.13333333333333353:0.13333333333333353 1,1,0 ./Data/OpenEI/BuildingTemp/input/building1retail_June_Week_672_GT.csv 1,3,5 2 1*
      
  4. **python3 SignalComparing.py**: Calcul des MSE permettant de comparer le signal restauré et le signal originel

Commande *python3 CGOFMSM_Signals.py Parameters/Signal.param 2:0.07:0.24:0.09 1,1,0 ./Data/Kaggle/input/JoseRizalBridgeNorth_all_resample_15624_GT.csv 3 2 1*, à comparer avec cet algo : https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition
