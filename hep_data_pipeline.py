import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ParticleFeatures:
    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    energy: np.ndarray
    charge: np.ndarray
    pdgid: np.ndarray
    mass: np.ndarray

@dataclass
class EventFeatures:
    particles: ParticleFeatures
    met: float
    n_jets: int
    n_leptons: int
    event_id: int
    label: int

class ROOTDataLoader:
    def __init__(self, use_uproot: bool = True):
        self.use_uproot = use_uproot
        if use_uproot:
            try:
                import uproot
                self.uproot = uproot
            except ImportError:
                logger.warning("uproot not installed. Install with: pip install uproot")
                self.use_uproot = False
   
    def load_root_file(self, filepath: str, tree_name: str = "Events") -> Dict:
        if not self.use_uproot:
            raise ImportError("uproot required for ROOT file processing")
       
        logger.info(f"Loading ROOT file: {filepath}")
       
        with self.uproot.open(filepath) as file:
            tree = file[tree_name]
           
            data = {
                'particles': self._extract_particles(tree),
                'jets': self._extract_jets(tree),
                'leptons': self._extract_leptons(tree),
                'met': self._extract_met(tree),
                'event_info': self._extract_event_info(tree)
            }
       
        logger.info(f"Loaded {len(data['event_info'])} events")
        return data
   
    def _extract_particles(self, tree) -> Dict:
        try:
            particles = {
                'pt': tree['Particle_PT'].array(library="np"),
                'eta': tree['Particle_Eta'].array(library="np"),
                'phi': tree['Particle_Phi'].array(library="np"),
                'energy': tree['Particle_E'].array(library="np"),
                'charge': tree['Particle_Charge'].array(library="np"),
                'pdgid': tree['Particle_PID'].array(library="np"),
            }
            return particles
        except KeyError as e:
            logger.warning(f"Some particle branches not found: {e}")
            return {}
   
    def _extract_jets(self, tree) -> Dict:
        try:
            jets = {
                'pt': tree['Jet_PT'].array(library="np"),
                'eta': tree['Jet_Eta'].array(library="np"),
                'phi': tree['Jet_Phi'].array(library="np"),
                'mass': tree['Jet_Mass'].array(library="np"),
                'btag': tree['Jet_BTag'].array(library="np") if 'Jet_BTag' in tree else None
            }
            return jets
        except KeyError:
            return {}
   
    def _extract_leptons(self, tree) -> Dict:
        leptons = {}
       
        try:
            leptons['electrons'] = {
                'pt': tree['Electron_PT'].array(library="np"),
                'eta': tree['Electron_Eta'].array(library="np"),
                'phi': tree['Electron_Phi'].array(library="np"),
                'charge': tree['Electron_Charge'].array(library="np")
            }
        except KeyError:
            pass
       
        try:
            leptons['muons'] = {
                'pt': tree['Muon_PT'].array(library="np"),
                'eta': tree['Muon_Eta'].array(library="np"),
                'phi': tree['Muon_Phi'].array(library="np"),
                'charge': tree['Muon_Charge'].array(library="np")
            }
        except KeyError:
            pass
       
        return leptons
   
    def _extract_met(self, tree) -> np.ndarray:
        try:
            return tree['MissingET_MET'].array(library="np")
        except KeyError:
            return None
   
    def _extract_event_info(self, tree) -> Dict:
        try:
            return {
                'event_number': tree['Event_Number'].array(library="np"),
                'weight': tree['Event_Weight'].array(library="np") if 'Event_Weight' in tree else None
            }
        except KeyError:
            return {'event_number': np.arange(len(tree))}

class HDF5DataProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
   
    def create_hdf5_dataset(
        self,
        output_path: str,
        data_dict: Dict[str, np.ndarray],
        compression: str = 'gzip'
    ):
        logger.info(f"Creating HDF5 dataset: {output_path}")
       
        with h5py.File(output_path, 'w') as f:
            for key, data in data_dict.items():
                f.create_dataset(
                    key,
                    data=data,
                    compression=compression,
                    chunks=True
                )
                logger.info(f"Saved {key}: shape {data.shape}")
   
    def load_hdf5_dataset(
        self,
        filepath: str,
        keys: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        logger.info(f"Loading HDF5 dataset: {filepath}")
       
        data = {}
        with h5py.File(filepath, 'r') as f:
            keys_to_load = keys if keys else list(f.keys())
           
            for key in keys_to_load:
                if key in f:
                    data[key] = f[key][:]
                    logger.info(f"Loaded {key}: shape {data[key].shape}")
       
        return data
   
    def iterate_hdf5_batches(
        self,
        filepath: str,
        batch_size: int = 100,
        keys: Optional[List[str]] = None
    ):
        with h5py.File(filepath, 'r') as f:
            keys_to_load = keys if keys else list(f.keys())
           
            first_key = keys_to_load[0]
            dataset_length = len(f[first_key])
           
            for start_idx in range(0, dataset_length, batch_size):
                end_idx = min(start_idx + batch_size, dataset_length)
               
                batch = {}
                for key in keys_to_load:
                    batch[key] = f[key][start_idx:end_idx]
               
                yield batch

class PhysicsFeatureEngineering:
    @staticmethod
    def compute_invariant_mass(
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        energy: np.ndarray
    ) -> np.ndarray:
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
       
        p_squared = px**2 + py**2 + pz**2
        mass_squared = energy**2 - p_squared
       
        mass_squared = np.maximum(mass_squared, 0)
       
        return np.sqrt(mass_squared)
   
    @staticmethod
    def compute_delta_r(
        eta1: np.ndarray,
        phi1: np.ndarray,
        eta2: np.ndarray,
        phi2: np.ndarray
    ) -> np.ndarray:
        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2
       
        delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
        delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
       
        return np.sqrt(delta_eta**2 + delta_phi**2)
   
    @staticmethod
    def compute_transverse_mass(
        pt1: np.ndarray,
        phi1: np.ndarray,
        pt2: np.ndarray,
        phi2: np.ndarray
    ) -> np.ndarray:
        delta_phi = phi1 - phi2
        delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
        delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
       
        mt = np.sqrt(2 * pt1 * pt2 * (1 - np.cos(delta_phi)))
       
        return mt
   
    @staticmethod
    def compute_jet_substructure(
        jet_constituents_pt: np.ndarray,
        jet_constituents_eta: np.ndarray,
        jet_constituents_phi: np.ndarray
    ) -> Dict[str, float]:
        return {
            'n_constituents': len(jet_constituents_pt),
            'pt_dispersion': np.std(jet_constituents_pt)
        }
   
    @staticmethod
    def apply_detector_smearing(
        features: np.ndarray,
        resolution: float = 0.1
    ) -> np.ndarray:
        noise = np.random.normal(0, resolution * features, features.shape)
        return features + noise

class EventSelectionCriteria:
    def __init__(self):
        self.cuts = {}
   
    def add_cut(self, name: str, condition_func):
        self.cuts[name] = condition_func
   
    def apply_cuts(self, events: Dict) -> Tuple[Dict, np.ndarray]:
        mask = np.ones(len(events['event_info']['event_number']), dtype=bool)
       
        for cut_name, cut_func in self.cuts.items():
            cut_mask = cut_func(events)
            logger.info(f"Cut '{cut_name}': {cut_mask.sum()}/{len(cut_mask)} events pass")
            mask = mask & cut_mask
       
        filtered_events = self._apply_mask(events, mask)
       
        logger.info(f"Total passing: {mask.sum()}/{len(mask)} events")
       
        return filtered_events, mask
   
    def _apply_mask(self, events: Dict, mask: np.ndarray) -> Dict:
        filtered = {}
        for key, value in events.items():
            if isinstance(value, dict):
                filtered[key] = self._apply_mask(value, mask)
            elif isinstance(value, np.ndarray):
                filtered[key] = value[mask]
            else:
                filtered[key] = value
        return filtered

class DataNormalization:
    def __init__(self):
        self.stats = {}
   
    def fit(self, data: Dict[str, np.ndarray]):
        for key, values in data.items():
            if values.dtype in [np.float32, np.float64]:
                self.stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
   
    def transform(
        self,
        data: Dict[str, np.ndarray],
        method: str = 'standard'
    ) -> Dict[str, np.ndarray]:
        normalized = {}
       
        for key, values in data.items():
            if key not in self.stats:
                normalized[key] = values
                continue
           
            stats = self.stats[key]
           
            if method == 'standard':
               
                normalized[key] = (values - stats['mean']) / (stats['std'] + 1e-8)
           
            elif method == 'minmax':
                normalized[key] = (values - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
           
            elif method == 'robust':
                median = np.median(values)
                q75, q25 = np.percentile(values, [75, 25])
                iqr = q75 - q25
                normalized[key] = (values - median) / (iqr + 1e-8)
           
            else:
                normalized[key] = values
       
        return normalized
   
    def save_stats(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
   
    def load_stats(self, filepath: str):
        with open(filepath, 'r') as f:
            self.stats = json.load(f)

class SLACDataPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.root_loader = ROOTDataLoader()
        self.hdf5_processor = HDF5DataProcessor()
        self.feature_engineer = PhysicsFeatureEngineering()
        self.normalizer = DataNormalization()
        self.event_selector = EventSelectionCriteria()
   
    def setup_event_selection(self):
        self.event_selector.add_cut(
            'min_pt',
            lambda events: np.any(events['particles']['pt'] > 20, axis=1)
        )
       
        self.event_selector.add_cut(
            'fiducial_eta',
            lambda events: np.all(np.abs(events['particles']['eta']) < 2.5, axis=1)
        )
   
    def process_dataset(
        self,
        input_files: List[str],
        output_path: str,
        file_format: str = 'root'
    ):
        logger.info(f"Processing {len(input_files)} files...")
       
        all_events = []
       
        for filepath in input_files:
            logger.info(f"Processing {filepath}")
           
            if file_format == 'root':
                events = self.root_loader.load_root_file(filepath)
            elif file_format == 'hdf5':
                events = self.hdf5_processor.load_hdf5_dataset(filepath)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
           
            selected_events, mask = self.event_selector.apply_cuts(events)
           
            all_events.append(selected_events)
       
        combined_events = self._combine_events(all_events)
       
        engineered_features = self._engineer_features(combined_events)
       
        self.normalizer.fit(engineered_features)
        normalized_features = self.normalizer.transform(engineered_features)
       
        self.hdf5_processor.create_hdf5_dataset(
            output_path,
            normalized_features,
            compression='gzip'
        )
       
        self.normalizer.save_stats(output_path.replace('.h5', '_stats.json'))
       
        logger.info(f"Dataset processing complete. Saved to {output_path}")
   
    def _combine_events(self, event_list: List[Dict]) -> Dict:
        combined = {}
       
        for key in event_list[0].keys():
            if isinstance(event_list[0][key], dict):
                combined[key] = self._combine_events([e[key] for e in event_list])
            elif isinstance(event_list[0][key], np.ndarray):
                combined[key] = np.concatenate([e[key] for e in event_list])
       
        return combined
   
    def _engineer_features(self, events: Dict) -> Dict[str, np.ndarray]:
        features = {}
       
        for key in ['pt', 'eta', 'phi', 'energy']:
            if key in events.get('particles', {}):
                features[key] = events['particles'][key]
       
        if all(k in features for k in ['pt', 'eta', 'phi', 'energy']):
            features['mass'] = self.feature_engineer.compute_invariant_mass(
                features['pt'], features['eta'],
                features['phi'], features['energy']
            )
       
        return features

if __name__ == "__main__":
    logger.info("High-Energy Physics Data Pipeline for SLAC")
   
    config = {
        'input_format': 'root',
        'output_format': 'hdf5',
        'normalization': 'standard',
        'event_selection': True
    }
   
    pipeline = SLACDataPipeline(config)
    pipeline.setup_event_selection()
   
    logger.info("Pipeline initialized and ready for SLAC data processing")
    logger.info("Use pipeline.process_dataset() to process raw files")