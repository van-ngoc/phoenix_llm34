# phoenix-llm34/INSTALL.md
# Guide d'installation de Phoenix-LLM34 🔧

## 📋 Table des matières

1. [Prérequis système](#prérequis-système)
2. [Installation rapide](#installation-rapide)
3. [Installation détaillée](#installation-détaillée)
4. [Configuration](#configuration)
5. [Démarrage](#démarrage)
6. [Optimisation](#optimisation)
7. [Désinstallation](#désinstallation)
8. [Dépannage](#dépannage)

## 💻 Prérequis système

### Configuration minimale
- **Système d'exploitation** : Debian 13 (ou compatible)
- **Processeur** : Intel Core i5 ou équivalent
- **RAM** : 8 GB minimum (32 GB recommandé)
- **Espace disque** : 20 GB minimum
- **GPU** : NVIDIA avec support CUDA (optionnel mais recommandé)
- **Réseau** : Connexion internet pour le téléchargement des modèles

### Configuration optimale (testée)
CPU: Intel i7-6820HQ
RAM: 32 GB DDR4
GPU: NVIDIA RTX 5060 8GB + GTX 970
OS: Debian 13 (Bookworm)
Espace: 50 GB SSD

text

### Dépendances système
```bash
# Paquets nécessaires
- python3 (>= 3.10)
- python3-pip
- python3-venv
- python3-dev
- git
- curl
- wget
- build-essential
- libgl1-mesa-glx
- libegl1-mesa
- libxcb-xinerama0
- libxcb-icccm4
- libxcb-image0
- libxcb-keysyms1
- libxcb-randr0
- libxcb-render-util0
- libxcb-shape0
- libxcb-xfixes0
- libxcb-xkb1
- libxkbcommon-x11-0


-------------------

# 1. Cloner le dépôt

git clone https://github.com/van-ngoc/phoenix_llm34.git


cd phoenix-llm34

# 2. Installer les dépendances système
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev git curl wget build-essential

# 3. Installer PyQt6 et ses dépendances
sudo apt install -y libgl1-mesa-glx libegl1-mesa libxcb-xinerama0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xfixes0 libxcb-xkb1 libxkbcommon-x11-0

# 4. Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 5. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 6. Installer les dépendances Python
pip install --upgrade pip
pip install -r requirements.txt

# 7. Créer les répertoires de configuration
mkdir -p ~/.phoenix-llm34/{models,conversations,rag,cache}

# 8. Créer le fichier de configuration
cat > ~/.phoenix-llm34/config.json << 'EOF'
{
    "ollama_url": "http://localhost:11434",
    "default_model": "",
    "quantization": "q4",
    "theme": "dark",
    "system_prompt": "You are a helpful AI assistant.",
    "temperature": 0.7,
    "max_tokens": 2048,
    "clear_prompt_after_submit": true,
    "rag_enabled": false,
    "cowork_enabled": false,
    "web_search_enabled": false,
    "audio_enabled": true
}
EOF

# 9. Créer le script de lancement
cat > phoenix-llm34 << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 main.py "$@"
EOF
chmod +x phoenix-llm34

----------------

✅ Vérification de l'installation
Tests de base
bash
# Vérifier Python
python3 --version

# Vérifier les packages
pip list | grep -E "PyQt6|requests|beautifulsoup4"

# Vérifier Ollama
ollama list

# Vérifier les fichiers de configuration
ls -la ~/.phoenix-llm34/