# phoenix-llm34/main.py
#!/usr/bin/env python3
"""
Phoenix-LLM34 - Application desktop pour modèles AI avec Ollama
Version: 1.0.0 - CORRIGÉE (effacement du prompt après submit)
"""

import sys
import os
import json
import threading
import queue
import time
import hashlib
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phoenix-llm34.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports PyQt6
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# Imports pour l'audio
import pyaudio
import wave
import numpy as np

# Imports pour les requêtes
import requests
from bs4 import BeautifulSoup

# Imports pour le traitement
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Constants
APP_NAME = "Phoenix-LLM34"
VERSION = "1.0.0"
OLLAMA_API = "http://localhost:11434"
MODEL_TYPES = ["gguf", "safetensors", "diffusers", "aria2c", "exl2"]
QUANTIZATIONS = ["q4", "q6", "q8", "f16", "f32"]
MAX_WORKERS = min(mp.cpu_count(), 8)

class ConfigManager:
    """Gestionnaire de configuration"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".phoenix-llm34"
        self.config_file = self.config_dir / "config.json"
        self.models_dir = self.config_dir / "models"
        self.conversations_dir = self.config_dir / "conversations"
        self.rag_dir = self.config_dir / "rag"
        self.cache_dir = self.config_dir / "cache"
        
        self._create_directories()
        self.config = self._load_config()
    
    def _create_directories(self):
        """Crée les répertoires nécessaires"""
        for dir_path in [self.config_dir, self.models_dir, self.conversations_dir, 
                        self.rag_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Charge la configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {
            "ollama_url": OLLAMA_API,
            "default_model": "",
            "quantization": "q4",
            "theme": "dark",
            "system_prompt": "You are a helpful AI assistant.",
            "temperature": 0.7,
            "max_tokens": 2048,
            "clear_prompt_after_submit": True  # Nouvelle option
        }
    
    def save_config(self):
        """Sauvegarde la configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save_config()

class OllamaManager:
    """Gestionnaire d'Ollama"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.api_url = config.get("ollama_url", OLLAMA_API)
        self.process = None
        self.models_cache = {}
        
    def check_ollama(self) -> bool:
        """Vérifie si Ollama est en cours d'exécution"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_ollama(self):
        """Démarre Ollama"""
        if not self.check_ollama():
            try:
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                time.sleep(2)
                return self.check_ollama()
            except:
                return False
        return True
    
    def list_models(self) -> List[Dict]:
        """Liste les modèles installés"""
        try:
            response = requests.get(f"{self.api_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except:
            pass
        return []
    
    def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Télécharge un modèle"""
        try:
            response = requests.post(
                f"{self.api_url}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if progress_callback and "progress" in data:
                            progress_callback(data["progress"])
                        if "error" in data:
                            return False
                return True
        except Exception as e:
            logger.error(f"Erreur pull model: {e}")
        return False
    
    def delete_model(self, model_name: str) -> bool:
        """Supprime un modèle"""
        try:
            response = requests.delete(
                f"{self.api_url}/api/delete",
                json={"name": model_name}
            )
            return response.status_code == 200
        except:
            return False
    
    def generate(self, model: str, prompt: str, stream: bool = False, 
                 system_prompt: str = None, temperature: float = 0.7):
        """Génère une réponse - Générateur pour le streaming"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "temperature": temperature
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if stream:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json=payload,
                    stream=True
                )
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                yield data
                            except:
                                continue
                else:
                    yield {"error": f"Erreur: {response.status_code}"}
            else:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json=payload
                )
                if response.status_code == 200:
                    yield response.json()
                else:
                    yield {"error": f"Erreur: {response.status_code}"}
        except Exception as e:
            yield {"error": str(e)}

class Conversation:
    """Classe pour les conversations"""
    
    def __init__(self, name: str = None):
        self.id = str(int(time.time()))
        self.name = name or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.messages = []
        self.created_at = datetime.now()
        self.model = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "model": self.model
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        conv = cls(data.get("name"))
        conv.id = data.get("id")
        conv.messages = data.get("messages", [])
        conv.created_at = datetime.fromisoformat(data.get("created_at"))
        conv.model = data.get("model", "")
        return conv
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

class ConversationManager:
    """Gestionnaire de conversations"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.conversations_dir = config.conversations_dir
        self.current_conversation = None
        self.conversations = self.load_all()
    
    def load_all(self) -> Dict[str, Conversation]:
        """Charge toutes les conversations"""
        conversations = {}
        for file_path in self.conversations_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    conv = Conversation.from_dict(data)
                    conversations[conv.id] = conv
            except:
                pass
        return conversations
    
    def create_conversation(self, name: str = None) -> Conversation:
        """Crée une nouvelle conversation"""
        conv = Conversation(name)
        self.conversations[conv.id] = conv
        self.current_conversation = conv
        return conv
    
    def save_conversation(self, conv: Conversation):
        """Sauvegarde une conversation"""
        file_path = self.conversations_dir / f"{conv.id}.json"
        with open(file_path, 'w') as f:
            json.dump(conv.to_dict(), f, indent=2)
    
    def delete_conversation(self, conv_id: str):
        """Supprime une conversation"""
        if conv_id in self.conversations:
            file_path = self.conversations_dir / f"{conv_id}.json"
            if file_path.exists():
                file_path.unlink()
            del self.conversations[conv_id]
            if self.current_conversation and self.current_conversation.id == conv_id:
                self.current_conversation = None
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        return self.conversations.get(conv_id)

# Classes de pipeline
class PipelineStage:
    """Étape du pipeline de traitement"""
    
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def process(self, data: Any) -> Any:
        """À implémenter par les sous-classes"""
        return data
    
    def start(self):
        """Démarre le traitement"""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Arrête le traitement"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _run(self):
        """Boucle de traitement"""
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = self.process(data)
                if result is not None:
                    self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur dans le pipeline: {e}")

class TurboQuantStage(PipelineStage):
    """Étape de quantification turbo"""
    
    def __init__(self, quantization: str = "q4"):
        super().__init__()
        self.quantization = quantization
    
    def process(self, data: Any) -> Any:
        # Simulation de quantification
        if isinstance(data, dict):
            data["quantization"] = self.quantization
            data["quantized_at"] = datetime.now().isoformat()
        return data

class HashStage(PipelineStage):
    """Étape de hachage"""
    
    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Hacher le contenu
            content_str = json.dumps(data, sort_keys=True)
            data["hash"] = hashlib.sha256(content_str.encode()).hexdigest()
        return data

class MultiplexerStage(PipelineStage):
    """Étape de multiplexage"""
    
    def __init__(self, num_workers: int = 4):
        super().__init__()
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def process(self, data: Any) -> Any:
        # Multiplexer vers plusieurs workers
        futures = []
        for i in range(self.num_workers):
            future = self.executor.submit(self._worker_process, data, i)
            futures.append(future)
        
        # Attendre les résultats
        results = [f.result() for f in futures]
        return results
    
    def _worker_process(self, data: Any, worker_id: int) -> Any:
        # Simuler un traitement par worker
        if isinstance(data, dict):
            data[f"worker_{worker_id}"] = f"processed_by_{worker_id}"
        return data

class ProcessStage(PipelineStage):
    """Étape de traitement"""
    
    def process(self, data: Any) -> Any:
        # Traitement des données
        if isinstance(data, list):
            processed = []
            for item in data:
                if isinstance(item, dict):
                    item["processed"] = True
                    item["processed_at"] = datetime.now().isoformat()
                    processed.append(item)
            return processed
        return data

class DemultiplexerStage(PipelineStage):
    """Étape de démultiplexage"""
    
    def process(self, data: Any) -> Any:
        if isinstance(data, list):
            # Combiner les résultats des workers
            combined = {}
            for item in data:
                if isinstance(item, dict):
                    combined.update(item)
            return combined
        return data

class DehashStage(PipelineStage):
    """Étape de déhachage"""
    
    def process(self, data: Any) -> Any:
        if isinstance(data, dict):
            # Vérifier le hash
            if "hash" in data:
                content_str = json.dumps(data, sort_keys=True)
                content_str = content_str.replace(data["hash"], "")
                content_str = content_str.replace('"hash": "",', "")
                if content_str.strip():
                    new_hash = hashlib.sha256(content_str.encode()).hexdigest()
                    if new_hash == data["hash"]:
                        data["hash_valid"] = True
                    else:
                        data["hash_valid"] = False
        return data

class OutputStage(PipelineStage):
    """Étape de sortie"""
    
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
    
    def process(self, data: Any) -> Any:
        if self.callback:
            self.callback(data)
        return data

class ProcessingPipeline:
    """Pipeline de traitement complet"""
    
    def __init__(self, quantization: str = "q4"):
        self.stages = []
        self.create_pipeline(quantization)
        self.running = False
        self.thread = None
    
    def create_pipeline(self, quantization: str):
        """Crée le pipeline avec toutes les étapes"""
        self.stages = []
        
        # Créer les étapes
        stages = [
            TurboQuantStage(quantization),
            HashStage(),
            MultiplexerStage(MAX_WORKERS),
            ProcessStage(),
            DemultiplexerStage(),
            DehashStage()
        ]
        
        # Connecter les étapes
        for i in range(len(stages) - 1):
            stages[i].output_queue = stages[i + 1].input_queue
        
        self.stages = stages
    
    def start(self):
        """Démarre le pipeline"""
        self.running = True
        for stage in self.stages:
            stage.start()
        
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Arrête le pipeline"""
        self.running = False
        for stage in self.stages:
            stage.stop()
        if self.thread:
            self.thread.join(timeout=1)
    
    def process(self, data: Any, callback=None) -> Any:
        """Traite des données à travers le pipeline"""
        if not self.stages:
            return None
        
        # Ajouter une étape de sortie avec callback
        output_stage = OutputStage(callback)
        self.stages[-1].output_queue = output_stage.input_queue
        
        # Démarrer l'étape de sortie
        output_stage.start()
        
        # Envoyer les données
        self.stages[0].input_queue.put(data)
        
        # Attendre le résultat
        if callback is None:
            try:
                result = output_stage.output_queue.get(timeout=30)
                output_stage.stop()
                return result
            except:
                return None
        else:
            return None
    
    def _run(self):
        """Boucle principale du pipeline"""
        while self.running:
            time.sleep(0.1)

# Classe pour la génération de réponse avec signaux
class GenerationWorker(QThread):
    """Thread de génération de réponse avec signaux PyQt"""
    
    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, ollama: OllamaManager, model: str, prompt: str, 
                 system_prompt: str = None, temperature: float = 0.7):
        super().__init__()
        self.ollama = ollama
        self.model = model
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.is_running = True
    
    def run(self):
        """Exécute la génération"""
        try:
            full_response = ""
            chunk_count = 0
            
            for chunk in self.ollama.generate(
                self.model, 
                self.prompt, 
                stream=True,
                system_prompt=self.system_prompt,
                temperature=self.temperature
            ):
                if not self.is_running:
                    break
                    
                if "error" in chunk:
                    self.error_occurred.emit(chunk["error"])
                    break
                elif "response" in chunk:
                    text = chunk["response"]
                    full_response += text
                    chunk_count += 1
                    
                    # Émettre le chunk
                    self.response_chunk.emit(text)
                    
                    # Mettre à jour la progression
                    if chunk_count % 5 == 0:
                        self.progress_update.emit(min(100, chunk_count))
                        
                elif "done" in chunk:
                    self.progress_update.emit(100)
                    break
            
            # Émettre la réponse complète
            self.response_complete.emit(full_response)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Arrête la génération"""
        self.is_running = False

class MainWindow(QMainWindow):
    """Fenêtre principale de l'application"""
    
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.ollama = OllamaManager(self.config)
        self.conversation_manager = ConversationManager(self.config)
        self.pipeline = ProcessingPipeline(self.config.get("quantization", "q4"))
        self.generation_worker = None
        self.current_prompt = ""  # Stocker le prompt pour référence
        
        self.setup_ui()
        self.setup_pipeline()
        self.load_models()
        self.load_conversations()
        
        # Démarrer Ollama si nécessaire
        if not self.ollama.check_ollama():
            self.ollama.start_ollama()
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.setGeometry(100, 100, 1400, 900)
        
        # Style amélioré
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f1a;
                color: #e0e0e0;
            }
            QWidget {
                background-color: #1a1a2e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e94560, stop:1 #c0392b);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6b81, stop:1 #e74c3c);
            }
            QPushButton:pressed {
                background: #c0392b;
            }
            QPushButton:disabled {
                background: #2d3561;
                color: #666;
            }
            QLineEdit, QTextEdit, QPlainTextEdit {
                background-color: #0f0f1a;
                border: 1px solid #2d3561;
                border-radius: 6px;
                padding: 8px;
                color: #e0e0e0;
                selection-background-color: #e94560;
            }
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #e94560;
            }
            QComboBox {
                background-color: #0f0f1a;
                border: 1px solid #2d3561;
                border-radius: 6px;
                padding: 8px;
                color: #e0e0e0;
                min-height: 30px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #e0e0e0;
                margin-right: 8px;
            }
            QListWidget, QTableWidget {
                background-color: #0f0f1a;
                border: 1px solid #2d3561;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item, QTableWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected, QTableWidget::item:selected {
                background-color: #e94560;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2d3561;
            }
            QProgressBar {
                background-color: #0f0f1a;
                border: 1px solid #2d3561;
                border-radius: 6px;
                text-align: center;
                height: 20px;
                color: white;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e94560, stop:1 #ff6b81);
                border-radius: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #2d3561;
                border-radius: 6px;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #0f0f1a;
                padding: 8px 16px;
                border-radius: 6px 6px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #e94560;
                color: white;
            }
            QSplitter::handle {
                background-color: #2d3561;
            }
            QMenuBar {
                background-color: #0f0f1a;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #e94560;
            }
            QMenu {
                background-color: #1a1a2e;
                border: 1px solid #2d3561;
            }
            QMenu::item:selected {
                background-color: #e94560;
            }
            QScrollBar:vertical {
                background-color: #0f0f1a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #2d3561;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #e94560;
            }
            QGroupBox {
                border: 1px solid #2d3561;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #2d3561;
                background-color: #0f0f1a;
            }
            QCheckBox::indicator:checked {
                background-color: #e94560;
                border: 1px solid #e94560;
            }
        """)
        
        # Widget central et layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Splitter principal
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Panel gauche
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        main_splitter.setSizes([350, 1050])
        
        # Panel central
        central_panel = self.create_central_panel()
        main_splitter.addWidget(central_panel)
        
        # Panel bas (progress bars)
        bottom_panel = self.create_bottom_panel()
        self.statusBar().addPermanentWidget(bottom_panel)
        
        # Barre de menu
        self.create_menu_bar()
        
        # Initialiser les conversations
        self.update_conversations_list()
    
    def create_menu_bar(self):
        """Crée la barre de menu"""
        menubar = self.menuBar()
        
        # Menu Fichier
        file_menu = menubar.addMenu("&Fichier")
        
        new_action = QAction("Nouvelle conversation", self)
        new_action.triggered.connect(self.new_conversation)
        file_menu.addAction(new_action)
        
        save_action = QAction("Sauvegarder", self)
        save_action.triggered.connect(self.save_current_conversation)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quitter", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Menu Modèles
        models_menu = menubar.addMenu("&Modèles")
        
        refresh_action = QAction("Rafraîchir la liste", self)
        refresh_action.triggered.connect(self.load_models)
        models_menu.addAction(refresh_action)
        
        models_menu.addSeparator()
        
        download_action = QAction("Télécharger un modèle", self)
        download_action.triggered.connect(self.download_model)
        models_menu.addAction(download_action)
        
        # Menu Paramètres
        settings_menu = menubar.addMenu("&Paramètres")
        
        theme_action = QAction("Changer de thème", self)
        theme_action.triggered.connect(self.toggle_theme)
        settings_menu.addAction(theme_action)
        
        rag_action = QAction("Gérer RAG", self)
        rag_action.triggered.connect(self.manage_rag)
        settings_menu.addAction(rag_action)
        
        # Option pour effacer le prompt après submit
        clear_action = QAction("Effacer le prompt après envoi", self)
        clear_action.setCheckable(True)
        clear_action.setChecked(self.config.get("clear_prompt_after_submit", True))
        clear_action.triggered.connect(lambda: self.toggle_clear_prompt(clear_action))
        settings_menu.addAction(clear_action)
    
    def create_left_panel(self):
        """Crée le panel gauche"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Section Conversations
        conv_group = QGroupBox("Conversations")
        conv_layout = QVBoxLayout(conv_group)
        
        # Boutons de conversation
        conv_buttons_layout = QHBoxLayout()
        new_conv_btn = QPushButton("Nouvelle")
        new_conv_btn.clicked.connect(self.new_conversation)
        save_conv_btn = QPushButton("Sauvegarder")
        save_conv_btn.clicked.connect(self.save_current_conversation)
        delete_conv_btn = QPushButton("Supprimer")
        delete_conv_btn.clicked.connect(self.delete_current_conversation)
        
        conv_buttons_layout.addWidget(new_conv_btn)
        conv_buttons_layout.addWidget(save_conv_btn)
        conv_buttons_layout.addWidget(delete_conv_btn)
        conv_layout.addLayout(conv_buttons_layout)
        
        # Liste des conversations
        self.conversations_list = QListWidget()
        self.conversations_list.itemClicked.connect(self.load_conversation)
        conv_layout.addWidget(self.conversations_list)
        
        layout.addWidget(conv_group)
        
        # Section Modèles
        models_group = QGroupBox("Modèles")
        models_layout = QVBoxLayout(models_group)
        
        # Recherche de modèles
        search_layout = QHBoxLayout()
        self.model_search = QLineEdit()
        self.model_search.setPlaceholderText("Rechercher des modèles...")
        search_btn = QPushButton("🔍")
        search_btn.clicked.connect(self.search_models)
        search_layout.addWidget(self.model_search)
        search_layout.addWidget(search_btn)
        models_layout.addLayout(search_layout)
        
        # Liste des modèles trouvés
        self.models_list = QListWidget()
        self.models_list.itemDoubleClicked.connect(self.download_selected_model)
        models_layout.addWidget(self.models_list)
        
        # Boutons d'action
        model_buttons_layout = QHBoxLayout()
        download_btn = QPushButton("Télécharger")
        download_btn.clicked.connect(self.download_selected_model)
        load_btn = QPushButton("Charger")
        load_btn.clicked.connect(self.load_selected_model)
        delete_btn = QPushButton("Supprimer")
        delete_btn.clicked.connect(self.delete_selected_model)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_operation)
        
        model_buttons_layout.addWidget(download_btn)
        model_buttons_layout.addWidget(load_btn)
        model_buttons_layout.addWidget(delete_btn)
        model_buttons_layout.addWidget(stop_btn)
        models_layout.addLayout(model_buttons_layout)
        
        # Filtres de type de modèle
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(MODEL_TYPES)
        type_layout.addWidget(self.model_type_combo)
        models_layout.addLayout(type_layout)
        
        layout.addWidget(models_group)
        
        # Section RAG
        rag_group = QGroupBox("RAG & CoWork")
        rag_layout = QVBoxLayout(rag_group)
        
        rag_buttons_layout = QHBoxLayout()
        self.rag_check = QCheckBox("Activer RAG")
        self.rag_check.stateChanged.connect(self.toggle_rag)
        rag_buttons_layout.addWidget(self.rag_check)
        
        self.cowork_check = QCheckBox("CoWork")
        self.cowork_check.stateChanged.connect(self.toggle_cowork)
        rag_buttons_layout.addWidget(self.cowork_check)
        
        rag_layout.addLayout(rag_buttons_layout)
        
        # Accès réseau
        net_group = QGroupBox("Accès Réseau")
        net_layout = QHBoxLayout(net_group)
        self.web_search_check = QCheckBox("Recherche Web")
        net_layout.addWidget(self.web_search_check)
        rag_layout.addWidget(net_group)
        
        layout.addWidget(rag_group)
        
        # Espace pour pousser les éléments vers le haut
        layout.addStretch()
        
        return panel
    
    def create_central_panel(self):
        """Crée le panel central"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Section sélection de modèle
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Modèle:"))
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        model_select_layout.addWidget(self.model_combo)
        
        self.load_model_btn = QPushButton("Charger le modèle")
        self.load_model_btn.clicked.connect(self.load_selected_model_from_combo)
        model_select_layout.addWidget(self.load_model_btn)
        
        model_select_layout.addStretch()
        layout.addLayout(model_select_layout)
        
        # Zone de réponse du modèle
        response_group = QGroupBox("Réponse du modèle")
        response_layout = QVBoxLayout(response_group)
        
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setMinimumHeight(250)
        self.response_area.setPlaceholderText("La réponse du modèle s'affichera ici...")
        response_layout.addWidget(self.response_area)
        
        layout.addWidget(response_group)
        
        # Zone d'input
        input_group = QGroupBox("Prompt")
        input_layout = QVBoxLayout(input_group)
        
        self.input_area = QPlainTextEdit()
        self.input_area.setPlaceholderText("Entrez votre prompt ici...")
        self.input_area.setMaximumHeight(120)
        input_layout.addWidget(self.input_area)
        
        # Barre d'outils d'input
        tools_layout = QHBoxLayout()
        
        # Micro
        self.micro_btn = QPushButton("🎤")
        self.micro_btn.setToolTip("Enregistrer audio")
        self.micro_btn.clicked.connect(self.toggle_recording)
        tools_layout.addWidget(self.micro_btn)
        
        # Bouton + (upload fichier)
        upload_btn = QPushButton("+")
        upload_btn.setToolTip("Joindre un fichier")
        upload_btn.clicked.connect(self.upload_file)
        tools_layout.addWidget(upload_btn)
        
        # Recherche web
        self.web_search_btn = QPushButton("🌐")
        self.web_search_btn.setToolTip("Recherche web")
        self.web_search_btn.clicked.connect(self.web_search)
        tools_layout.addWidget(self.web_search_btn)
        
        tools_layout.addStretch()
        
        # Boutons Submit et Stop
        self.submit_btn = QPushButton("▶ Submit")
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00b894, stop:1 #00a086);
                padding: 10px 30px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #00d2a0, stop:1 #00b894);
            }
        """)
        self.submit_btn.clicked.connect(self.submit_prompt)
        tools_layout.addWidget(self.submit_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e17055, stop:1 #d63031);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff7675, stop:1 #e17055);
            }
        """)
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)  # Désactivé par défaut
        tools_layout.addWidget(self.stop_btn)
        
        input_layout.addLayout(tools_layout)
        layout.addWidget(input_group)
        
        return panel
    
    def create_bottom_panel(self):
        """Crée le panel du bas avec les barres de progression"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 5, 10, 5)
        
        # Barre de progression téléchargement
        download_layout = QHBoxLayout()
        download_layout.addWidget(QLabel("Téléchargement:"))
        self.download_progress = QProgressBar()
        self.download_progress.setMaximum(100)
        download_layout.addWidget(self.download_progress)
        layout.addLayout(download_layout)
        
        # Barre de progression traitement
        process_layout = QHBoxLayout()
        process_layout.addWidget(QLabel("Traitement:"))
        self.process_progress = QProgressBar()
        self.process_progress.setMaximum(100)
        process_layout.addWidget(self.process_progress)
        layout.addLayout(process_layout)
        
        # Barre de progression calcul intensif
        compute_layout = QHBoxLayout()
        compute_layout.addWidget(QLabel("Calcul intensif:"))
        self.compute_progress = QProgressBar()
        self.compute_progress.setMaximum(100)
        compute_layout.addWidget(self.compute_progress)
        layout.addLayout(compute_layout)
        
        # Barre de progression génération
        gen_layout = QHBoxLayout()
        gen_layout.addWidget(QLabel("Génération AI:"))
        self.gen_progress = QProgressBar()
        self.gen_progress.setMaximum(100)
        gen_layout.addWidget(self.gen_progress)
        layout.addLayout(gen_layout)
        
        return panel
    
    # Méthodes fonctionnelles
    def setup_pipeline(self):
        """Configure le pipeline de traitement"""
        quantization = self.config.get("quantization", "q4")
        self.pipeline = ProcessingPipeline(quantization)
        self.pipeline.start()
    
    def load_models(self):
        """Charge la liste des modèles disponibles"""
        self.model_combo.clear()
        
        if self.ollama.check_ollama():
            models = self.ollama.list_models()
            for model in models:
                name = model.get("name", "")
                size = model.get("size", 0)
                self.model_combo.addItem(f"{name} ({self.format_size(size)})")
        
        # Ajouter les modèles téléchargés localement
        config = ConfigManager()
        for model_dir in config.models_dir.glob("*"):
            if model_dir.is_dir():
                # Vérifier si c'est un modèle
                if (model_dir / "model.gguf").exists():
                    self.model_combo.addItem(f"{model_dir.name} (local)")
    
    def format_size(self, size):
        """Formate la taille en octets"""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"
    
    def load_conversations(self):
        """Charge les conversations"""
        self.conversations_list.clear()
        for conv_id, conv in self.conversation_manager.conversations.items():
            self.conversations_list.addItem(conv.name)
    
    def update_conversations_list(self):
        """Met à jour la liste des conversations"""
        self.load_conversations()
    
    def new_conversation(self):
        """Crée une nouvelle conversation"""
        name, ok = QInputDialog.getText(self, "Nouvelle conversation", 
                                        "Nom de la conversation:")
        if ok and name:
            conv = self.conversation_manager.create_conversation(name)
            self.conversations_list.addItem(conv.name)
            self.response_area.clear()
            self.input_area.clear()  # Effacer aussi le prompt
            self.conversations_list.setCurrentRow(self.conversations_list.count() - 1)
    
    def save_current_conversation(self):
        """Sauvegarde la conversation courante"""
        if self.conversation_manager.current_conversation:
            self.conversation_manager.save_conversation(
                self.conversation_manager.current_conversation
            )
            QMessageBox.information(self, "Succès", "Conversation sauvegardée")
    
    def delete_current_conversation(self):
        """Supprime la conversation courante"""
        current_row = self.conversations_list.currentRow()
        if current_row >= 0:
            conv_id = list(self.conversation_manager.conversations.keys())[current_row]
            if conv_id:
                reply = QMessageBox.question(self, "Confirmation",
                                           "Voulez-vous supprimer cette conversation ?",
                                           QMessageBox.StandardButton.Yes | 
                                           QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.conversation_manager.delete_conversation(conv_id)
                    self.conversations_list.takeItem(current_row)
                    self.response_area.clear()
                    self.input_area.clear()
    
    def load_conversation(self, item):
        """Charge une conversation"""
        conv_id = list(self.conversation_manager.conversations.keys())[
            self.conversations_list.currentRow()
        ]
        conv = self.conversation_manager.get_conversation(conv_id)
        if conv:
            self.conversation_manager.current_conversation = conv
            # Afficher les messages
            self.response_area.clear()
            for msg in conv.messages:
                role = "👤 Vous" if msg["role"] == "user" else "🤖 Assistant"
                self.response_area.append(f"**{role}:** {msg['content']}\n")
            self.input_area.clear()
    
    def search_models(self):
        """Recherche des modèles sur Ollama.com"""
        search_term = self.model_search.text()
        self.models_list.clear()
        self.models_list.addItem("Recherche en cours...")
        
        # Recherche simple (simulée pour l'instant)
        QTimer.singleShot(1000, lambda: self.display_models([
            {"name": "llama2:7b", "description": "Modèle Llama 2 7B", "downloads": "1M+"},
            {"name": "mistral:7b", "description": "Modèle Mistral 7B", "downloads": "500K+"},
            {"name": "codellama:7b", "description": "Modèle Code Llama 7B", "downloads": "300K+"},
            {"name": "phi:2.7b", "description": "Modèle Phi-2 2.7B", "downloads": "200K+"},
            {"name": "gemma:2b", "description": "Modèle Gemma 2B", "downloads": "150K+"},
        ]))
    
    def display_models(self, models):
        """Affiche les modèles trouvés"""
        self.models_list.clear()
        for model in models:
            item_text = f"{model['name']} - {model['downloads']} downloads"
            if model.get('description'):
                item_text += f" - {model['description']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, model)
            self.models_list.addItem(item)
    
    def download_selected_model(self):
        """Télécharge le modèle sélectionné"""
        current_item = self.models_list.currentItem()
        if current_item:
            model_data = current_item.data(Qt.ItemDataRole.UserRole)
            if model_data:
                model_name = model_data.get("name")
                if model_name:
                    self.download_model(model_name)
    
    def download_model(self, model_name=None):
        """Télécharge un modèle"""
        if not model_name:
            model_name, ok = QInputDialog.getText(self, "Télécharger un modèle",
                                                "Nom du modèle:")
            if not ok or not model_name:
                return
        
        # Vérifier si Ollama est disponible
        if not self.ollama.check_ollama():
            if not self.ollama.start_ollama():
                QMessageBox.critical(self, "Erreur", 
                                   "Impossible de démarrer Ollama")
                return
        
        # Démarrer le téléchargement
        self.download_progress.setValue(0)
        
        def update_progress(progress):
            if isinstance(progress, dict) and "completed" in progress:
                total = progress.get("total", 1)
                completed = progress.get("completed", 0)
                pct = int((completed / total) * 100)
                self.download_progress.setValue(pct)
            elif isinstance(progress, (int, float)):
                self.download_progress.setValue(int(progress))
        
        # Télécharger dans un thread
        self.download_thread = threading.Thread(
            target=self._download_model_thread,
            args=(model_name, update_progress)
        )
        self.download_thread.daemon = True
        self.download_thread.start()
    
    def _download_model_thread(self, model_name, progress_callback):
        """Thread de téléchargement"""
        try:
            success = self.ollama.pull_model(model_name, progress_callback)
            if success:
                self.download_progress.setValue(100)
                self.load_models()
                QMessageBox.information(self, "Succès", 
                                      f"Modèle {model_name} téléchargé avec succès")
            else:
                self.download_progress.setValue(0)
                QMessageBox.critical(self, "Erreur",
                                   f"Échec du téléchargement de {model_name}")
        except Exception as e:
            self.download_progress.setValue(0)
            QMessageBox.critical(self, "Erreur", f"Erreur: {str(e)}")
    
    def load_selected_model(self):
        """Charge le modèle sélectionné"""
        current_item = self.models_list.currentItem()
        if current_item:
            model_data = current_item.data(Qt.ItemDataRole.UserRole)
            if model_data:
                model_name = model_data.get("name")
                self.load_model(model_name)
    
    def load_selected_model_from_combo(self):
        """Charge le modèle sélectionné dans la combo box"""
        current_text = self.model_combo.currentText()
        if current_text:
            # Extraire le nom du modèle
            model_name = current_text.split(" (")[0] if " (" in current_text else current_text
            self.load_model(model_name)
    
    def load_model(self, model_name):
        """Charge un modèle"""
        if not model_name:
            return
        
        # Vérifier si Ollama est disponible
        if not self.ollama.check_ollama():
            if not self.ollama.start_ollama():
                QMessageBox.critical(self, "Erreur",
                                   "Impossible de démarrer Ollama")
                return
        
        # Vérifier si le modèle existe
        models = self.ollama.list_models()
        model_found = False
        for model in models:
            if model.get("name") == model_name:
                model_found = True
                break
        
        if not model_found:
            # Essayer de trouver dans les modèles locaux
            config = ConfigManager()
            model_path = config.models_dir / model_name
            if not model_path.exists():
                QMessageBox.critical(self, "Erreur",
                                   f"Modèle {model_name} non trouvé")
                return
        
        # Enregistrer le modèle
        self.config.set("default_model", model_name)
        self.load_models()
        
        # Sélectionner dans la combo
        for i in range(self.model_combo.count()):
            if model_name in self.model_combo.itemText(i):
                self.model_combo.setCurrentIndex(i)
                break
        
        QMessageBox.information(self, "Succès",
                              f"Modèle {model_name} chargé avec succès")
    
    def delete_selected_model(self):
        """Supprime le modèle sélectionné"""
        current_item = self.models_list.currentItem()
        if current_item:
            model_data = current_item.data(Qt.ItemDataRole.UserRole)
            if model_data:
                model_name = model_data.get("name")
                if model_name:
                    reply = QMessageBox.question(self, "Confirmation",
                                               f"Supprimer le modèle {model_name} ?",
                                               QMessageBox.StandardButton.Yes |
                                               QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        self.delete_model(model_name)
    
    def delete_model(self, model_name):
        """Supprime un modèle"""
        if self.ollama.delete_model(model_name):
            self.load_models()
            QMessageBox.information(self, "Succès",
                                  f"Modèle {model_name} supprimé")
        else:
            QMessageBox.critical(self, "Erreur",
                               f"Impossible de supprimer {model_name}")
    
    def stop_operation(self):
        """Arrête l'opération en cours"""
        self.stop_generation()
        QMessageBox.information(self, "Info", "Arrêt demandé")
    
    def toggle_recording(self):
        """Active/désactive l'enregistrement audio"""
        # Simulé pour l'instant
        self.micro_btn.setStyleSheet("background-color: #e74c3c;" if self.micro_btn.text() == "🎤" else "")
        self.micro_btn.setText("🔴" if self.micro_btn.text() == "🎤" else "🎤")
    
    def upload_file(self):
        """Télécharge un fichier pour le prompt"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un fichier",
            "", "Tous les fichiers (*.*);;Images (*.png *.jpg *.jpeg *.gif);;Vidéo (*.mp4 *.avi *.mkv);;Audio (*.mp3 *.wav *.flac);;Texte (*.txt *.md)"
        )
        if file_path:
            # Ajouter le chemin du fichier dans le prompt
            current_text = self.input_area.toPlainText()
            new_text = current_text + f"\n\n[Fichier: {file_path}]"
            self.input_area.setPlainText(new_text)
    
    def web_search(self):
        """Effectue une recherche web"""
        query = self.input_area.toPlainText()
        if query:
            # Simuler une recherche web
            self.response_area.append("\n**Recherche Web:**\n")
            self.response_area.append("- Résultat 1: Lorem ipsum dolor sit amet...")
            self.response_area.append("- Résultat 2: Consectetur adipiscing elit...")
            self.response_area.append("- Résultat 3: Sed do eiusmod tempor...\n")
    
    def toggle_clear_prompt(self, action):
        """Active/désactive l'effacement automatique du prompt"""
        self.config.set("clear_prompt_after_submit", action.isChecked())
    
    def submit_prompt(self):
        """Soumet le prompt au modèle"""
        prompt = self.input_area.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Veuillez entrer un prompt")
            return
        
        # Récupérer le modèle sélectionné
        model_text = self.model_combo.currentText()
        if not model_text:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner un modèle")
            return
        
        model_name = model_text.split(" (")[0] if " (" in model_text else model_text
        
        # Vérifier si le modèle est disponible
        if not self.ollama.check_ollama():
            if not self.ollama.start_ollama():
                QMessageBox.critical(self, "Erreur", "Ollama n'est pas disponible")
                return
        
        # Stocker le prompt pour référence
        self.current_prompt = prompt
        
        # AJOUT: Effacer le champ de saisie après avoir récupéré le prompt
        # Vérifier si l'option est activée
        if self.config.get("clear_prompt_after_submit", True):
            self.input_area.clear()
        
        # Ajouter à la conversation
        if self.conversation_manager.current_conversation:
            self.conversation_manager.current_conversation.add_message("user", prompt)
        
        # Afficher le prompt dans la zone de réponse
        self.response_area.append(f"\n**👤 Vous:** {prompt}\n")
        self.response_area.append("**🤖 Assistant:** ")
        
        # Désactiver les boutons
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("⏳ Génération...")
        self.stop_btn.setEnabled(True)
        
        # Configurer les paramètres
        system_prompt = self.config.get("system_prompt")
        temperature = self.config.get("temperature", 0.7)
        
        # Créer et démarrer le worker de génération
        self.generation_worker = GenerationWorker(
            self.ollama, model_name, prompt, system_prompt, temperature
        )
        
        # Connecter les signaux
        self.generation_worker.response_chunk.connect(self.on_response_chunk)
        self.generation_worker.response_complete.connect(self.on_response_complete)
        self.generation_worker.error_occurred.connect(self.on_generation_error)
        self.generation_worker.progress_update.connect(self.gen_progress.setValue)
        
        # Démarrer
        self.generation_worker.start()
    
    def on_response_chunk(self, chunk: str):
        """Reçoit un chunk de réponse"""
        # Ajouter le chunk à la zone de réponse
        cursor = self.response_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.response_area.setTextCursor(cursor)
        self.response_area.ensureCursorVisible()
        
        # Mettre à jour la progression
        self.gen_progress.setValue(min(100, self.gen_progress.value() + 1))
    
    def on_response_complete(self, full_response: str):
        """Termine la génération"""
        # Réactiver les boutons
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("▶ Submit")
        self.stop_btn.setEnabled(False)
        
        # Ajouter à la conversation
        if self.conversation_manager.current_conversation:
            self.conversation_manager.current_conversation.add_message("assistant", full_response)
            self.conversation_manager.save_conversation(
                self.conversation_manager.current_conversation
            )
        
        # Ajouter un saut de ligne
        self.response_area.append("\n")
        
        # Réinitialiser la progression
        self.gen_progress.setValue(0)
    
    def on_generation_error(self, error: str):
        """Gère les erreurs de génération"""
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("▶ Submit")
        self.stop_btn.setEnabled(False)
        self.gen_progress.setValue(0)
        
        QMessageBox.critical(self, "Erreur", f"Erreur de génération: {error}")
    
    def stop_generation(self):
        """Arrête la génération en cours"""
        if self.generation_worker and self.generation_worker.isRunning():
            self.generation_worker.stop()
            self.generation_worker.quit()
            self.generation_worker.wait()
            self.generation_worker = None
            
            self.submit_btn.setEnabled(True)
            self.submit_btn.setText("▶ Submit")
            self.stop_btn.setEnabled(False)
            self.gen_progress.setValue(0)
            
            self.response_area.append("\n*Génération arrêtée*\n")
    
    def toggle_theme(self):
        """Change le thème"""
        current_theme = self.config.get("theme", "dark")
        new_theme = "light" if current_theme == "dark" else "dark"
        self.config.set("theme", new_theme)
        QMessageBox.information(self, "Thème", f"Thème changé en: {new_theme}")
    
    def toggle_rag(self, state):
        """Active/désactive le RAG"""
        if state == Qt.CheckState.Checked:
            QMessageBox.information(self, "RAG", "RAG activé")
        else:
            QMessageBox.information(self, "RAG", "RAG désactivé")
    
    def toggle_cowork(self, state):
        """Active/désactive le CoWork"""
        if state == Qt.CheckState.Checked:
            QMessageBox.information(self, "CoWork", "CoWork activé")
        else:
            QMessageBox.information(self, "CoWork", "CoWork désactivé")
    
    def manage_rag(self):
        """Gère les documents RAG"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Ajouter un document RAG",
            "", "Fichiers texte (*.txt *.md *.pdf);;Tous les fichiers (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    QMessageBox.information(self, "Succès",
                                          f"Document ajouté au RAG: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Erreur: {str(e)}")
    
    def closeEvent(self, event):
        """Gère la fermeture de l'application"""
        # Arrêter la génération si en cours
        if self.generation_worker and self.generation_worker.isRunning():
            self.generation_worker.stop()
            self.generation_worker.wait()
        
        # Arrêter le pipeline
        self.pipeline.stop()
        
        # Sauvegarder la conversation
        self.save_current_conversation()
        
        event.accept()

def main():
    """Point d'entrée principal"""
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)
    
    # Créer et afficher la fenêtre principale
    window = MainWindow()
    window.show()
    
    # Exécuter l'application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()