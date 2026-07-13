# phoenix-llm34/run.sh
#!/bin/bash
# Script de lancement pour Phoenix-LLM34

echo "=========================================="
echo "  Phoenix-LLM34 v1.0.0"
echo "=========================================="
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python3 n'est pas installé"
    exit 1
fi

# Vérifier les dépendances
echo "Vérification des dépendances..."
if ! python3 -c "import PyQt6" 2>/dev/null; then
    echo "Installation des dépendances..."
    pip3 install -r requirements.txt
fi

# Vérifier Ollama
if ! command -v ollama &> /dev/null; then
    echo "Erreur: Ollama n'est pas installé"
    echo "Installez Ollama depuis: https://ollama.com"
    exit 1
fi

# Démarrer l'application
echo "Démarrage de Phoenix-LLM34..."
python3 main.py "$@"