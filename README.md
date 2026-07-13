# Phoenix-LLM34 🚀

**Application desktop puissante pour modèles AI avec Ollama**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)

## 📋 Table des matières

- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Pipeline de traitement](#pipeline-de-traitement)
- [Dépannage](#dépannage)
- [Contribuer](#contribuer)
- [Licence](#licence)

## 🎯 Présentation

Phoenix-LLM34 est une application desktop complète pour interagir avec des modèles d'IA via Ollama. Elle offre une interface utilisateur moderne et intuitive avec des fonctionnalités avancées de traitement, de gestion des modèles et de conversations.

### ✨ Caractéristiques principales

- **Interface utilisateur élégante** - Design moderne avec thème sombre
- **Gestion complète des modèles** - Téléchargement, chargement, suppression
- **Conversations persistantes** - Sauvegarde et restauration des conversations
- **RAG intégré** - Retrieval Augmented Generation pour des réponses contextuelles
- **Pipeline de traitement** - Traitement optimisé avec multiplexage
- **Support audio** - Enregistrement et traitement vocal
- **Recherche web** - Intégration de la recherche en ligne
- **Multi-threading** - Performance optimisée pour les calculs intensifs
- **Barres de progression** - Suivi en temps réel des opérations

## 🚀 Fonctionnalités détaillées

### Gestion des modèles
- Recherche de modèles sur Ollama.com
- Support de multiples formats : GGUF, SafeTensors, Diffusers, Aria2c, Exl2
- Téléchargement avec barre de progression
- Quantization : q4, q6, q8, f16, f32
- Chargement et suppression de modèles

### Interface utilisateur
- **Panel central** : Sélection de modèle, zone de réponse, zone de prompt
- **Outils de prompt** : Micro, téléchargement de fichiers, recherche web
- **Panel gauche** : Gestion des conversations, recherche de modèles, RAG, CoWork
- **Barres de progression** : Téléchargement, traitement, calcul, génération

### Pipeline de traitement
Input → TurboQuant → Hash → Multiplexeur → Process → Démultiplexeur → Dehash → Output

text

### Fonctionnalités avancées
- **RAG (Retrieval Augmented Generation)** : Contexte enrichi par des documents
- **CoWork** : Collaboration entre agents AI
- **Accès réseau** : Recherche web intégrée
- **Multi-threading** : Traitement parallèle optimisé

## 🏗️ Architecture

### Structure du projet
phoenix-llm34/
├── main.py # Application principale
├── models_manager.py # Gestionnaire de modèles
├── rag_system.py # Système RAG
├── processing_pipeline.py # Pipeline de traitement
├── requirements.txt # Dépendances Python
├── run.sh # Script de lancement
├── install.sh # Script d'installation
├── README.md # Documentation
└── INSTALL.md # Guide d'installation

text

### Composants clés
- **ConfigManager** : Gestion de la configuration
- **OllamaManager** : Interface avec Ollama API
- **ConversationManager** : Gestion des conversations
- **RAGSystem** : Système de recherche augmentée
- **ProcessingPipeline** : Pipeline de traitement optimisé
- **GenerationWorker** : Thread de génération avec signaux PyQt

## 📦 Prérequis

### Matériel
- **Processeur** : Intel i7-6820 ou équivalent
- **RAM** : 32 GB minimum recommandé
- **GPU** : NVIDIA RTX 5060 8GB + GTX 970 (support multi-GPU)
- **Stockage** : 50 GB minimum pour les modèles

### Logiciels
- **Système** : Debian 13 ou compatible
- **Python** : 3.10 ou supérieur
- **Ollama** : Dernière version
- **X11/Wayland** : Environnement graphique

## 🔧 Installation

Pour une installation complète, suivez le guide dans [INSTALL.md](INSTALL.md).

### Installation rapide
```bash
# Cloner le dépôt
git clone https://github.com/yourusername/phoenix-llm34.git
cd phoenix-llm34

# Lancer l'installation
chmod +x install.sh
./install.sh

# Démarrer l'application
phoenix-llm34
Installation manuelle
bash
# Installer les dépendances système
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances Python
pip install -r requirements.txt

# Lancer l'application
python3 main.py
🎮 Utilisation
Premier lancement
Démarrer Ollama : ollama serve

Lancer Phoenix-LLM34 : phoenix-llm34 ou python3 main.py

Télécharger un modèle : Rechercher et télécharger depuis l'interface

Charger un modèle : Sélectionner dans la liste des modèles disponibles

Démarrer une conversation : Saisir un prompt et cliquer sur Submit

Raccourcis clavier
Ctrl+N : Nouvelle conversation

Ctrl+S : Sauvegarder la conversation

Ctrl+Q : Quitter l'application

Enter : Soumettre le prompt (avec Shift pour nouvelle ligne)

Interface utilisateur
Panel gauche
Conversations : Créer, sauvegarder, supprimer

Modèles : Rechercher, télécharger, charger, supprimer

RAG : Activer/désactiver, gérer les documents

CoWork : Activer/désactiver la collaboration

Accès réseau : Activer la recherche web

Panel central
Sélection de modèle : Choisir le modèle actif

Zone de réponse : Affichage des réponses du modèle

Zone de prompt : Saisie des questions

Barre d'outils : Micro, upload, recherche web

Boutons : Submit et Stop

Barre de progression
Téléchargement : Progression des téléchargements

Traitement : Progression du traitement

Calcul intensif : Progression des calculs

Génération AI : Progression de la génération


--------------


🐛 Dépannage
Problèmes courants
Ollama ne démarre pas
bash
# Vérifier l'installation
ollama --version

# Démarrer manuellement
ollama serve

# Vérifier le statut
curl http://localhost:11434/api/tags
Erreur de mémoire
bash
# Limiter l'utilisation mémoire
export OLLAMA_NUM_PARALLEL=2

# Utiliser une quantification plus légère
# Sélectionner q4 ou q6 dans l'interface
Problèmes graphiques
bash
# Réinitialiser la configuration
rm -rf ~/.phoenix-llm34/config.json

# Vérifier les drivers GPU
nvidia-smi
Logs de l'application
bash
# Consulter les logs
tail -f phoenix-llm34.log

# Logs détaillés
export PYTHONDEBUG=1
python3 main.py