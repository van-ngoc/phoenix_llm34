# phoenix-llm34/models_manager.py
"""
Gestionnaire spécialisé pour les modèles AI
Support: GGUF, SafeTensors, Diffusers, Aria2c, Exl2
"""

import os
import json
import hashlib
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue
import time

@dataclass
class ModelInfo:
    """Informations sur un modèle"""
    name: str
    type: str  # gguf, safetensors, diffusers, aria2c, exl2
    size: int
    path: str
    quantizations: List[str]
    download_url: Optional[str] = None
    description: str = ""
    version: str = "1.0.0"
    digest: str = ""

class ModelManager:
    """Gestionnaire de modèles avancé"""
    
    def __init__(self, config):
        self.config = config
        self.models_dir = config.models_dir
        self.models: Dict[str, ModelInfo] = {}
        self.download_queue = queue.Queue()
        self.download_threads = []
        self.is_downloading = False
        
        self._load_local_models()
    
    def _load_local_models(self):
        """Charge les modèles locaux"""
        for model_dir in self.models_dir.glob("*"):
            if model_dir.is_dir():
                # Chercher différents formats
                for ext in [".gguf", ".safetensors"]:
                    for file_path in model_dir.glob(f"*{ext}"):
                        try:
                            model_info = self._parse_model_file(file_path)
                            if model_info:
                                self.models[model_info.name] = model_info
                        except:
                            continue
    
    def _parse_model_file(self, file_path: Path) -> Optional[ModelInfo]:
        """Parse un fichier modèle"""
        # Déterminer le type
        ext = file_path.suffix.lower()
        if ext == ".gguf":
            model_type = "gguf"
        elif ext == ".safetensors":
            model_type = "safetensors"
        else:
            return None
        
        # Obtenir les infos
        name = file_path.stem
        size = file_path.stat().st_size
        
        # Déterminer les quantizations disponibles
        quantizations = []
        quant_dir = file_path.parent / "quantizations"
        if quant_dir.exists():
            for q_file in quant_dir.glob("*.gguf"):
                # Extraire le type de quantization
                q_name = q_file.stem.split("_")[-1] if "_" in q_file.stem else "unknown"
                if q_name in ["q4", "q6", "q8", "f16", "f32"]:
                    quantizations.append(q_name)
        
        return ModelInfo(
            name=name,
            type=model_type,
            size=size,
            path=str(file_path),
            quantizations=quantizations or ["q4"]
        )
    
    def search_model_online(self, search_term: str = "") -> List[Dict]:
        """Recherche des modèles en ligne"""
        # Utiliser Hugging Face API ou autres
        try:
            if search_term:
                url = f"https://huggingface.co/api/models?search={search_term}"
            else:
                url = "https://huggingface.co/api/models"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for model in data[:20]:  # Limiter à 20 résultats
                    results.append({
                        "name": model.get("modelId", ""),
                        "description": model.get("description", ""),
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "type": "gguf"  # Par défaut
                    })
                return results
        except Exception as e:
            print(f"Erreur de recherche: {e}")
        
        return []
    
    def download_model(self, model_name: str, model_type: str = "gguf",
                      quantization: str = "q4",
                      progress_callback=None) -> bool:
        """Télécharge un modèle"""
        try:
            # Créer le répertoire du modèle
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # URL de téléchargement selon le type
            if model_type == "gguf":
                url = f"https://huggingface.co/{model_name}/resolve/main/model.q{quantization}.gguf"
            else:
                url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors"
            
            # Télécharger le modèle
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            file_path = model_dir / f"model.{'gguf' if model_type == 'gguf' else 'safetensors'}"
            
            downloaded = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress)
            
            if progress_callback:
                progress_callback(100)
            
            # Ajouter aux modèles locaux
            self._load_local_models()
            return True
            
        except Exception as e:
            print(f"Erreur de téléchargement: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Supprime un modèle"""
        model_dir = self.models_dir / model_name
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            if model_name in self.models:
                del self.models[model_name]
            return True
        return False
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Récupère les infos d'un modèle"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[ModelInfo]:
        """Liste tous les modèles"""
        return list(self.models.values())
    
    def convert_model(self, model_name: str, target_quantization: str) -> bool:
        """Convertit un modèle vers une autre quantization"""
        # TODO: Implémenter la conversion
        return False
    
    def start_download_queue(self):
        """Démarre la file de téléchargement"""
        self.is_downloading = True
        for _ in range(3):  # 3 threads de téléchargement
            thread = threading.Thread(target=self._download_worker)
            thread.daemon = True
            thread.start()
            self.download_threads.append(thread)
    
    def _download_worker(self):
        """Worker de téléchargement"""
        while self.is_downloading:
            try:
                task = self.download_queue.get(timeout=1)
                if task:
                    model_name = task.get("name")
                    model_type = task.get("type", "gguf")
                    quantization = task.get("quantization", "q4")
                    callback = task.get("callback")
                    
                    self.download_model(model_name, model_type, quantization, callback)
                    self.download_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erreur dans le worker: {e}")
    
    def stop_download(self):
        """Arrête les téléchargements"""
        self.is_downloading = False
        for thread in self.download_threads:
            thread.join(timeout=2)