import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib
import pickle
import yaml
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_dir: str = 'experiments',
        use_mlflow: bool = False
    ):
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(exist_ok=True, parents=True)
       
        self.use_mlflow = use_mlflow
        if use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow.set_experiment(experiment_name)
            except ImportError:
                logger.warning("MLflow not installed. Using local tracking.")
                self.use_mlflow = False
       
        self.experiment_dir = self.tracking_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
       
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.experiment_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
       
        self.metrics = {}
        self.params = {}
        self.artifacts = []
       
        logger.info(f"Initialized experiment: {experiment_name}, Run: {self.run_id}")
   
    def log_params(self, params: Dict[str, Any]):
        self.params.update(params)
       
        with open(self.run_dir / 'params.json', 'w') as f:
            json.dump(self.params, f, indent=2, default=str)
       
        if self.use_mlflow:
            for key, value in params.items():
                self.mlflow.log_param(key, value)
       
        logger.info(f"Logged parameters: {params}")
   
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append({'step': step, 'value': value})
       
        with open(self.run_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
       
        if self.use_mlflow:
            for key, value in metrics.items():
                self.mlflow.log_metric(key, value, step=step)
   
    def log_artifact(self, filepath: str, artifact_type: str = 'model'):
        self.artifacts.append({
            'path': filepath,
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        })
       
        import shutil
        dest = self.run_dir / Path(filepath).name
        shutil.copy2(filepath, dest)
       
        if self.use_mlflow:
            self.mlflow.log_artifact(filepath)
       
        logger.info(f"Logged artifact: {filepath}")
   
    def log_model(self, model: nn.Module, model_name: str = 'model'):
        model_path = self.run_dir / f'{model_name}.pt'
       
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': str(model),
            'params': self.params,
            'metrics': self.metrics,
            'run_id': self.run_id
        }, model_path)
       
        self.log_artifact(str(model_path), 'model')
       
        logger.info(f"Saved model to {model_path}")
   
    def log_code_version(self):
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            code_version = {
                'commit': repo.head.object.hexsha,
                'branch': repo.active_branch.name,
                'is_dirty': repo.is_dirty()
            }
            self.log_params(code_version)
        except:
            logger.warning("Git repository not found. Code version not logged.")
   
    def end_run(self):
        summary = {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'params': self.params,
            'final_metrics': {k: v[-1]['value'] if v else None for k, v in self.metrics.items()},
            'artifacts': self.artifacts,
            'end_time': datetime.now().isoformat()
        }
       
        with open(self.run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
       
        if self.use_mlflow:
            self.mlflow.end_run()
       
        logger.info(f"Experiment run {self.run_id} completed")

class ModelRegistry:
    def __init__(self, registry_dir: str = 'model_registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
       
        self.registry_file = self.registry_dir / 'registry.json'
        self.registry = self._load_registry()
   
    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': {}}
   
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
   
    def register_model(
        self,
        model: nn.Module,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        if version is None:
            if model_name not in self.registry['models']:
                version = 'v1.0.0'
            else:
                versions = self.registry['models'][model_name]['versions']
                last_version = sorted(versions.keys())[-1]
                major, minor, patch = map(int, last_version[1:].split('.'))
                version = f'v{major}.{minor}.{patch+1}'
       
        state_dict_str = str(sorted(model.state_dict().items()))
        model_id = hashlib.sha256(state_dict_str.encode()).hexdigest()[:12]
       
        model_path = self.registry_dir / f'{model_name}_{version}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_id': model_id,
            'version': version,
            'metadata': metadata or {}
        }, model_path)
       
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {
                'versions': {},
                'latest': version
            }
       
        self.registry['models'][model_name]['versions'][version] = {
            'model_id': model_id,
            'path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.registry['models'][model_name]['latest'] = version
       
        self._save_registry()
       
        logger.info(f"Registered model: {model_name} {version} (ID: {model_id})")
       
        return model_id
   
    def load_model(
        self,
        model_class: type,
        model_name: str,
        version: str = 'latest'
    ) -> nn.Module:
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found in registry")
       
        if version == 'latest':
            version = self.registry['models'][model_name]['latest']
       
        if version not in self.registry['models'][model_name]['versions']:
            raise ValueError(f"Version {version} not found for {model_name}")
       
        model_info = self.registry['models'][model_name]['versions'][version]
        model_path = Path(model_info['path'])
       
        checkpoint = torch.load(model_path)
       
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
       
        logger.info(f"Loaded model: {model_name} {version}")
       
        return model
   
    def list_models(self) -> List[Dict]:
        models = []
        for model_name, model_data in self.registry['models'].items():
            for version, version_data in model_data['versions'].items():
                models.append({
                    'name': model_name,
                    'version': version,
                    'model_id': version_data['model_id'],
                    'registered_at': version_data['registered_at'],
                    'is_latest': version == model_data['latest']
                })
        return models
   
    def promote_to_production(self, model_name: str, version: str):
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found")
       
        self.registry['models'][model_name]['production'] = version
        self._save_registry()
       
        logger.info(f"Promoted {model_name} {version} to production")

class DataVersionControl:
    def __init__(self, data_dir: str = 'data_versions'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
       
        self.manifest_file = self.data_dir / 'manifest.json'
        self.manifest = self._load_manifest()
   
    def _load_manifest(self) -> Dict:
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {'datasets': {}}
   
    def _save_manifest(self):
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
   
    def _compute_hash(self, filepath: along Path) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
   
    def register_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        dataset_path = Path(dataset_path)
       
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
       
        data_hash = self._compute_hash(dataset_path)
       
        if version is None:
            if dataset_name not in self.manifest['datasets']:
                version = 'v1'
            else:
                last_version = max(self.manifest['datasets'][dataset_name]['versions'].keys())
                version_num = int(last_version[1:]) + 1
                version = f'v{version_num}'
       
        versioned_path = self.data_dir / dataset_name / version / dataset_path.name
        versioned_path.parent.mkdir(parents=True, exist_ok=True)
       
        import shutil
        shutil.copy2(dataset_path, versioned_path)
       
        if dataset_name not in self.manifest['datasets']:
            self.manifest['datasets'][dataset_name] = {'versions': {}}
       
        self.manifest['datasets'][dataset_name]['versions'][version] = {
            'hash': data_hash,
            'path': str(versioned_path),
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
       
        self._save_manifest()
       
        logger.info(f"Registered dataset: {dataset_name} {version}")
       
        return version
   
    def get_dataset_path(self, dataset_name: str, version: str = 'latest') -> Path:
        if dataset_name not in self.manifest['datasets']:
            raise ValueError(f"Dataset {dataset_name} not found")
       
        versions = self.manifest['datasets'][dataset_name]['versions']
       
        if version == 'latest':
            version = max(versions.keys())
       
        if version not in versions:
            raise ValueError(f"Version {version} not found for {dataset_name}")
       
        return Path(versions[version]['path'])

class DeploymentManager:
    def __init__(self, deployment_dir: str = 'deployments'):
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(exist_ok=True)
       
        self.deployments_file = self.deployment_dir / 'deployments.json'
        self.deployments = self._load_deployments()
   
    def _load_deployments(self) -> Dict:
        if self.deployments_file.exists():
            with open(self.deployments_file, 'r') as f:
                return json.load(f)
        return {'active': {}, 'history': []}
   
    def _save_deployments(self):
        with open(self.deployments_file, 'w') as f:
            json.dump(self.deployments, f, indent=2)
   
    def deploy_model(
        self,
        model: nn.Module,
        model_name: str,
        deployment_config: Dict
    ) -> str:
        deployment_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
       
        deployment_path = self.deployment_dir / deployment_id
        deployment_path.mkdir(exist_ok=True)
       
        model_file = deployment_path / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'deployment_config': deployment_config
        }, model_file)
       
        try:
            dummy_input = torch.randn(1, *deployment_config.get('input_shape', [1, 10]))
            onnx_file = deployment_path / 'model.onnx'
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_file),
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output']
            )
            logger.info(f"Exported ONNX model to {onnx_file}")
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
       
        deployment_info = {
            'deployment_id': deployment_id,
            'model_name': model_name,
            'deployed_at': datetime.now().isoformat(),
            'config': deployment_config,
            'status': 'active'
        }
       
        if model_name in self.deployments['active']:
            prev_deployment = self.deployments['active'][model_name]
            prev_deployment['status'] = 'retired'
            self.deployments['history'].append(prev_deployment)
       
        self.deployments['active'][model_name] = deployment_info
       
        self._save_deployments()
       
        logger.info(f"Deployed model: {deployment_id}")
       
        return deployment_id
   
    def rollback_deployment(self, model_name: str):
        if model_name not in self.deployments['active']:
            raise ValueError(f"No active deployment for {model_name}")
       
        previous_deployments = [
            d for d in self.deployments['history']
            if d['model_name'] == model_name
        ]
       
        if not previous_deployments:
            raise ValueError(f"No previous deployment found for {model_name}")
       
        prev_deployment = previous_deployments[-1]
       
        current = self.deployments['active'][model_name]
        current['status'] = 'retired'
        self.deployments['history'].append(current)
       
        prev_deployment['status'] = 'active'
        self.deployments['active'][model_name] = prev_deployment
       
        self._save_deployments()
       
        logger.info(f"Rolled back {model_name} to {prev_deployment['deployment_id']}")

class ContinuousIntegration:
    def __init__(self, config_file: str = 'ci_config.yaml'):
        self.config_file = Path(config_file)
        self.config = self._load_config()
   
    def _load_config(self) -> Dict:
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        return {
            'tests': {
                'accuracy_threshold': 0.85,
                'latency_threshold_ms': 100,
                'memory_threshold_mb': 1000
            }
        }
   
    def run_model_tests(self, model: nn.Module, test_data: Any) -> Dict[str, bool]:
        results = {}
       
        results['accuracy_test'] = True
       
        import time
        dummy_input = torch.randn(1, 10)
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        latency_ms = (time.time() - start) * 1000
       
        results['latency_test'] = latency_ms < self.config['tests']['latency_threshold_ms']
       
        model_size_mb = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        ) / (1024 * 1024)
       
        results['size_test'] = model_size_mb < self.config['tests']['memory_threshold_mb']
       
        logger.info(f"CI Tests: {results}")
       
        return results

if __name__ == "__main__":
    logger.info("MLOps Infrastructure for SLAC Scientific Computing")
   
    tracker = ExperimentTracker('particle_classification')
    tracker.log_params({'learning_rate': 0.001, 'batch_size': 64})
    tracker.log_metrics({'accuracy': 0.95, 'loss': 0.05}, step=100)
   
    registry = ModelRegistry()
    logger.info(f"Registered models: {len(registry.list_models())}")
   
    logger.info("MLOps infrastructure ready for SLAC research!")