# phoenix-llm34/install.sh
#!/bin/bash
# Script d'installation complet pour Phoenix-LLM34

set -e

echo "=========================================="
echo "  Phoenix-LLM34 - Installation v1.0.0"
echo "=========================================="
echo ""

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonctions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Vérifier le système
print_info "Vérification du système..."
if [ ! -f /etc/debian_version ]; then
    print_error "Ce script est conçu pour Debian"
    exit 1
fi
print_success "Système compatible"

# Vérifier les prérequis
print_info "Vérification des prérequis..."

# Python
if ! command -v python3 &> /dev/null; then
    print_info "Installation de Python3..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv python3-dev
fi
print_success "Python3 installé"

# Git
if ! command -v git &> /dev/null; then
    print_info "Installation de Git..."
    sudo apt install -y git
fi
print_success "Git installé"

# PyQt6
print_info "Installation des dépendances système pour PyQt6..."
sudo apt install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0

# Ollama
if ! command -v ollama &> /dev/null; then
    print_info "Installation d'Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
print_success "Ollama installé"

# Créer les répertoires
print_info "Création des répertoires..."
mkdir -p ~/.phoenix-llm34/{models,conversations,rag,cache}
mkdir -p /opt/phoenix-llm34

# Créer l'environnement virtuel
print_info "Création de l'environnement virtuel..."
cd /opt/phoenix-llm34
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
print_info "Installation des dépendances Python..."
pip install --upgrade pip
pip install -r /path/to/requirements.txt

# Créer le fichier de configuration
print_info "Création du fichier de configuration..."
if [ ! -f ~/.phoenix-llm34/config.json ]; then
    cat > ~/.phoenix-llm34/config.json << 'EOF'
{
    "ollama_url": "http://localhost:11434",
    "default_model": "",
    "quantization": "q4",
    "theme": "dark",
    "system_prompt": "You are a helpful AI assistant.",
    "temperature": 0.7,
    "max_tokens": 2048,
    "rag_enabled": false,
    "cowork_enabled": false,
    "web_search_enabled": false,
    "audio_enabled": true
}
EOF
fi

# Créer le fichier de lancement
print_info "Création du script de lancement..."
cat > /usr/local/bin/phoenix-llm34 << 'EOF'
#!/bin/bash
cd /opt/phoenix-llm34
source venv/bin/activate
python3 main.py "$@"
EOF

chmod +x /usr/local/bin/phoenix-llm34

# Créer le fichier .desktop
print_info "Création du fichier .desktop..."
cat > ~/.local/share/applications/phoenix-llm34.desktop << 'EOF'
[Desktop Entry]
Name=Phoenix-LLM34
Comment=Application desktop pour modèles AI avec Ollama
Exec=/usr/local/bin/phoenix-llm34
Icon=phoenix-llm34
Terminal=false
Type=Application
Categories=Utility;Development;
StartupNotify=true
EOF

print_success "Installation terminée !"
echo ""
echo "=========================================="
echo "  Phoenix-LLM34 est prêt à l'emploi !"
echo "=========================================="
echo ""
echo "Pour lancer l'application:"
echo "  phoenix-llm34"
echo ""
echo "Ou depuis le répertoire:"
echo "  cd /opt/phoenix-llm34"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""