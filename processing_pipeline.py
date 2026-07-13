# phoenix-llm34/processing_pipeline.py
"""
Pipeline de traitement complet avec toutes les étapes
"""

import queue
import threading
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ProcessingContext:
    """Contexte de traitement"""
    id: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"
    metadata: Dict = field(default_factory=dict)
    results: Dict = field(default_factory=dict)

class ProcessingStage:
    """Étape de traitement de base"""
    
    def __init__(self, name: str):
        self.name = name
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.stats = {
            "processed": 0,
            "errors": 0,
            "total_time": 0
        }
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Traite le contexte"""
        raise NotImplementedError
    
    def start(self):
        """Démarre l'étape"""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Arrête l'étape"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run(self):
        """Boucle de traitement principale"""
        while self.running:
            try:
                context = self.input_queue.get(timeout=0.1)
                start_time = time.time()
                try:
                    result = self.process(context)
                    self.output_queue.put(result)
                    self.stats["processed"] += 1
                    self.stats["total_time"] += time.time() - start_time
                except Exception as e:
                    context.status = "error"
                    context.metadata["error"] = str(e)
                    self.output_queue.put(context)
                    self.stats["errors"] += 1
                    logger.error(f"Erreur dans {self.name}: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur inattendue dans {self.name}: {e}")

class TurboQuantStage(ProcessingStage):
    """Étape de quantification ultra-rapide"""
    
    def __init__(self, quantization: str = "q4"):
        super().__init__("TurboQuant")
        self.quantization = quantization
        self.quantization_map = {
            "q4": 4,
            "q6": 6,
            "q8": 8,
            "f16": 16,
            "f32": 32
        }
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Applique la quantification"""
        context.metadata["quantization"] = self.quantization
        context.metadata["bits"] = self.quantization_map.get(self.quantization, 4)
        
        # Simulation de quantification
        if isinstance(context.data, dict):
            # Quantifier les nombres
            for key, value in context.data.items():
                if isinstance(value, (int, float)):
                    if key not in ["id", "timestamp", "status"]:
                        # Quantification simplifiée
                        bits = self.quantization_map.get(self.quantization, 4)
                        max_val = 2 ** bits
                        context.data[key] = float(int(value * max_val)) / max_val
        
        context.status = "quantized"
        return context

class HashStage(ProcessingStage):
    """Étape de hachage"""
    
    def __init__(self, algorithm: str = "sha256"):
        super().__init__("Hash")
        self.algorithm = algorithm
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Hache le contenu"""
        # Sérialiser le contenu
        content_str = json.dumps(context.data, sort_keys=True, default=str)
        
        # Calculer le hash
        hash_obj = hashlib.sha256(content_str.encode())
        context.metadata["hash"] = hash_obj.hexdigest()
        context.metadata["hash_algorithm"] = self.algorithm
        
        context.status = "hashed"
        return context

class MultiplexerStage(ProcessingStage):
    """Étape de multiplexage"""
    
    def __init__(self, num_workers: int = 4):
        super().__init__("Multiplexer")
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Multiplexe le traitement"""
        # Diviser les données
        chunks = self._split_data(context.data)
        
        # Traiter chaque chunk en parallèle
        futures = []
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self._process_chunk, chunk, i)
            futures.append(future)
        
        # Récupérer les résultats
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur dans le multiplexeur: {e}")
        
        # Combiner les résultats
        context.data = self._combine_results(results)
        context.metadata["num_workers"] = self.num_workers
        context.metadata["num_chunks"] = len(chunks)
        
        context.status = "multiplexed"
        return context
    
    def _split_data(self, data: Any) -> List[Any]:
        """Divise les données en chunks"""
        if isinstance(data, list):
            # Diviser la liste en chunks
            chunk_size = max(1, len(data) // self.num_workers)
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, dict):
            # Diviser le dictionnaire
            items = list(data.items())
            chunk_size = max(1, len(items) // self.num_workers)
            chunks = []
            for i in range(0, len(items), chunk_size):
                chunks.append(dict(items[i:i + chunk_size]))
            return chunks
        else:
            # Donnée unique
            return [data] * self.num_workers
    
    def _process_chunk(self, chunk: Any, worker_id: int) -> Any:
        """Traite un chunk avec un worker"""
        # Simuler un traitement
        time.sleep(0.01)  # Simuler le travail
        if isinstance(chunk, list):
            return [self._process_item(item, worker_id) for item in chunk]
        elif isinstance(chunk, dict):
            return {key: self._process_item(value, worker_id) for key, value in chunk.items()}
        else:
            return self._process_item(chunk, worker_id)
    
    def _process_item(self, item: Any, worker_id: int) -> Any:
        """Traite un élément individuel"""
        if isinstance(item, dict):
            item["worker_id"] = worker_id
            item["processed_at"] = time.time()
        return item
    
    def _combine_results(self, results: List[Any]) -> Any:
        """Combine les résultats des workers"""
        if not results:
            return []
        
        if isinstance(results[0], list):
            combined = []
            for r in results:
                combined.extend(r)
            return combined
        elif isinstance(results[0], dict):
            combined = {}
            for r in results:
                combined.update(r)
            return combined
        else:
            return results

class ProcessStage(ProcessingStage):
    """Étape de traitement principal"""
    
    def __init__(self, model_name: str = None):
        super().__init__("Process")
        self.model_name = model_name
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Traite les données"""
        # Simulation de traitement intensif
        if isinstance(context.data, dict):
            # Analyser les données
            if "prompt" in context.data:
                # Traitement du prompt
                context.data["processed_prompt"] = self._process_prompt(
                    context.data["prompt"]
                )
            if "images" in context.data:
                # Traitement des images
                context.data["processed_images"] = self._process_images(
                    context.data["images"]
                )
        
        context.status = "processed"
        context.metadata["model"] = self.model_name
        return context
    
    def _process_prompt(self, prompt: str) -> str:
        """Traite le prompt"""
        # Nettoyer et optimiser le prompt
        prompt = prompt.strip()
        # Ajouter des instructions contextuelles
        return prompt
    
    def _process_images(self, images: List) -> List:
        """Traite les images"""
        # Simuler le traitement d'images
        processed = []
        for img in images:
            processed.append({
                "original": img,
                "processed": True,
                "timestamp": time.time()
            })
        return processed

class DemultiplexerStage(ProcessingStage):
    """Étape de démultiplexage"""
    
    def __init__(self):
        super().__init__("Demultiplexer")
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Démultiplexe les résultats"""
        # Nettoyer et organiser les résultats
        if isinstance(context.data, list):
            # Combiner les listes
            context.data = self._merge_lists(context.data)
        elif isinstance(context.data, dict):
            # Organiser le dictionnaire
            context.data = self._organize_dict(context.data)
        
        context.status = "demultiplexed"
        return context
    
    def _merge_lists(self, data_list: List) -> List:
        """Fusionne des listes"""
        if not data_list:
            return []
        
        merged = []
        for item in data_list:
            if isinstance(item, list):
                merged.extend(item)
            else:
                merged.append(item)
        return merged
    
    def _organize_dict(self, data: Dict) -> Dict:
        """Organise un dictionnaire"""
        organized = {}
        for key, value in data.items():
            # Nettoyer les clés
            clean_key = key.replace("_", " ").title()
            organized[clean_key] = value
        return organized

class DehashStage(ProcessingStage):
    """Étape de déhachage"""
    
    def __init__(self):
        super().__init__("Dehash")
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Vérifie et déhache les données"""
        if "hash" in context.metadata:
            # Vérifier l'intégrité
            expected_hash = context.metadata["hash"]
            content_str = json.dumps(context.data, sort_keys=True, default=str)
            actual_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            context.metadata["hash_verified"] = expected_hash == actual_hash
            context.metadata["hash_check_time"] = time.time()
        
        context.status = "completed"
        return context

class OutputStage(ProcessingStage):
    """Étape de sortie"""
    
    def __init__(self, callback: Optional[Callable] = None):
        super().__init__("Output")
        self.callback = callback
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Prépare la sortie"""
        context.status = "output"
        context.metadata["output_time"] = time.time()
        context.metadata["total_time"] = context.metadata["output_time"] - context.timestamp
        
        if self.callback:
            try:
                self.callback(context)
            except Exception as e:
                logger.error(f"Erreur dans le callback de sortie: {e}")
        
        return context

class ProcessingPipeline:
    """Pipeline de traitement complet"""
    
    def __init__(self, quantization: str = "q4", num_workers: int = 4):
        self.quantization = quantization
        self.num_workers = num_workers
        self.stages = []
        self.running = False
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "total_time": 0
        }
        
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Construit le pipeline complet"""
        # Créer toutes les étapes
        stages = [
            TurboQuantStage(self.quantization),
            HashStage(),
            MultiplexerStage(self.num_workers),
            ProcessStage(),
            DemultiplexerStage(),
            DehashStage(),
            OutputStage()
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
        
        # Thread de monitoring
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """Arrête le pipeline"""
        self.running = False
        for stage in self.stages:
            stage.stop()
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
    
    def process(self, data: Any, callback: Optional[Callable] = None) -> ProcessingContext:
        """Traite des données"""
        # Créer le contexte
        context = ProcessingContext(
            id=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            data=data
        )
        
        # Ajouter le callback à l'étape de sortie
        if callback:
            self.stages[-1].callback = callback
        
        # Envoyer au pipeline
        self.stages[0].input_queue.put(context)
        
        # Si pas de callback, attendre le résultat
        if not callback:
            try:
                result = self.stages[-1].output_queue.get(timeout=60)
                self._update_stats(result)
                return result
            except queue.Empty:
                context.status = "timeout"
                return context
        
        return context
    
    def _monitor(self):
        """Monitor le pipeline"""
        while self.running:
            time.sleep(5)
            # Log des statistiques
            total_processed = sum(stage.stats["processed"] for stage in self.stages)
            total_errors = sum(stage.stats["errors"] for stage in self.stages)
            total_time = sum(stage.stats["total_time"] for stage in self.stages)
            
            logger.info(f"Pipeline stats - Processed: {total_processed}, "
                       f"Errors: {total_errors}, Time: {total_time:.2f}s")
    
    def _update_stats(self, context: ProcessingContext):
        """Met à jour les statistiques"""
        self.stats["total_processed"] += 1
        if context.status == "error":
            self.stats["total_errors"] += 1
        self.stats["total_time"] += context.metadata.get("total_time", 0)
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du pipeline"""
        return {
            "pipeline": self.stats,
            "stages": {
                stage.name: stage.stats for stage in self.stages
            }
        }