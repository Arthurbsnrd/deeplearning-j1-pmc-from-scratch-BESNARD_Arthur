# PMC From Scratch - Jour 1

Ce dépôt contient l'implémentation d'un Perceptron Multicouche (PMC) construit "from scratch" en utilisant uniquement NumPy.

## Avancement

### Phase 1 : Neurone unique, forward pass et calcul d'erreur
- [x] Implémentation de la fonction `sigmoid`
- [x] Implémentation du `forward pass` pour un neurone
- [x] Calcul de la loss (Binary Cross-Entropy)
- [x] Test avec des poids manuels

### Phase 2 : Descente de gradient à la main, loss par epoch
- [x] Implémentation de la boucle d'entraînement (epochs)
- [x] Calcul des gradients analytiques pour la BCE et sigmoid
- [x] Mise à jour des poids avec la descente de gradient manuelle
- [x] Tracé et sauvegarde de la courbe de convergence de la loss

### Phase 3 : XOR, réseau 2-2-1, décision non linéaire
- [x] Implémentation du réseau 2-2-1
- [x] Entraînement jusqu'à 100% d'accuracy
- [x] Tracé de la frontière de décision (phase3_xor_boundary.png)

### Phase 4 : Spirale, réseau profond ReLU
- [x] Génération du dataset spirale (400 points, bruit 0.15)
- [x] Réseau 2-64-64-1 avec ReLU (couches cachées) et sigmoid (sortie)
- [x] Backpropagation manuelle sur 3 couches
- [x] Entraînement jusqu'à 100% d'accuracy
- [x] Frontière de décision et courbe de loss (phase4_spirale.png)

---
*Ce README sera mis à jour à chaque phase du projet.*
