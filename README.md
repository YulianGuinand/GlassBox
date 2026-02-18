# GlassBox 

> **Démocratiser l'intelligence artificielle en transformant l'apprentissage profond en une expérience tactile, visuelle et immédiate.**

GlassBox est une application desktop (basée sur navigateur) qui lève le voile sur la "boîte noire" de l'IA. Elle permet de concevoir, entraîner et diagnostiquer des réseaux de neurones sans écrire une seule ligne de code, tout en conservant la puissance native de **PyTorch**.

---

## Vision

L'objectif n'est pas seulement de construire des modèles, mais de **comprendre par l'action**. 
> GlassBox s'adresse aux étudiants, chercheurs et développeurs souhaitant prototyper rapidement ou visualiser les mécanismes internes du Deep Learning.

### Fonctionnalités Clés

* **Modélisation Intuitive** : Assemblez des briques logiques (Couches Linéaires, Convolutions, Dropout) comme des LEGO. Le système gère automatiquement la compatibilité des dimensions matricielles.
* **Data Management Intégré** : Ingestion de fichiers CSV/Parquet, détection automatique des types, normalisation et encodage via Scikit-learn.
* **Entraînement Temps Réel** : Visualisez la descente de gradient et les courbes de perte (Loss) en direct. Ajustez le Learning Rate à la volée sans redémarrer le script.
* **Diagnostics Avancés** : Cartes de saillance (Saliency Maps), histogrammes de poids et matrices de confusion pour interpréter les décisions du modèle.
* **Interopérabilité** : Exportez vos modèles au format standard `.pth` (PyTorch) pour une mise en production immédiate.

---

## Stack Technique

* **Moteur** : Python 3.11+, PyTorch (CUDA/MPS supporté).
* **Interface** : Streamlit.
* **Data** : Pandas, Scikit-learn, NumPy.
* **Visualisation** : Plotly Interactive.

---

## Installation & Démarrage

### Prérequis
* Python 3.11 ou supérieur.
* Un gestionnaire de paquets (recommandé : `uv` ou `pip`).

### Installation rapide

```bash
# Cloner le dépôt
git clone [https://github.com/YulianGuinand/glassbox.git](https://github.com/YulianGuinand/glassbox.git)
cd glassbox

# Créer l'environnement virtuel et installer les dépendances
uv venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
uv run installation/install.py
```

## Lancer l'application
```bash
uv run streamlit run main.py
```

## Roadmap

Le développement suit 4 phases strictes :

1. **Fondations & Data** : Ingestion robuste et preprocessing (Actuel).
2. **Model Factory** : Générateur dynamique de classes nn.Module.
3. **Training Engine** : Boucle d'entraînement threadée et asynchrone.
4. **Intelligence Interprétative** : Outils de diagnostic visuel.

## Licence et Crédits

**Auteur** : Yulian Guinand 
**Concept** : GlassBox - Laboratoire éducatif et outil de prototypage
