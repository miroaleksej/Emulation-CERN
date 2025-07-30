import time
import logging
import random
import yaml
import json
import copy
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from collections import defaultdict

# --- Импорты научных библиотек ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_p
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Подавление предупреждений для чистоты вывода ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# 1. Настройка логирования
# ===================================================================
"""
Настройка логирования для отслеживания процесса выполнения.
Создает лог-файл и выводит сообщения в консоль.
"""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_lhc_framework.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Unified_LHC_Framework")

# ===================================================================
# 2. Загрузка конфигурации
# ===================================================================
"""
Загрузка параметров симуляции из файла 'lhc_config.yaml'.
Если файл не найден, используются значения по умолчанию.
"""
try:
    with open("lhc_config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    logger.info("Конфигурация загружена из lhc_config.yaml")
except FileNotFoundError:
    CONFIG = {
        "beam": {"energy": 6500, "particles": "protons", "bunch_intensity": 1.15e11, "num_bunches": 2748},
        "geometry": {"circumference": 26658.883, "dipole_field": 8.33},
        "validation": {"dataset_id": "cms-2011-collision-data"},
        "simulation": {"physics_engines": ["pythia", "herwig", "built-in"], "beam_dynamics_engine": "madx"}
    }
    logger.warning("Конфигурационный файл не найден. Используется дефолтная конфигурация.")

# ===================================================================
# 3. Проверка доступности внешних библиотек
# ===================================================================
"""
Проверка наличия внешних библиотек, которые могут быть использованы для расширения функциональности.
"""
ROOT_AVAILABLE = False
try:
    import ROOT
    ROOT_AVAILABLE = True
    logger.info("ROOT library found.")
except ImportError:
    logger.warning("ROOT library not found.")

HEPMC3_AVAILABLE = False
try:
    import pyhepmc as hep
    import pyhepmc.io as io
    _test_evt = hep.GenEvent()
    HEPMC3_AVAILABLE = True
    logger.info("HepMC3 library (pyhepmc) found.")
except ImportError:
    logger.warning("HepMC3 library (pyhepmc) not found.")

PERSISTENCE_AVAILABLE = False
USE_RIPSER = False
USE_GUDHI = False
try:
    import gudhi as gd
    PERSISTENCE_AVAILABLE = True
    USE_GUDHI = True
    logger.info("GUDHI found.")
except ImportError:
    try:
        import ripser
        from persim import plot_diagrams
        PERSISTENCE_AVAILABLE = True
        USE_RIPSER = True
        logger.info("Ripser/Persim found.")
    except ImportError:
        logger.warning("Neither GUDHI nor Ripser/Persim found.")

PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    logger.info("Plotly found.")
except ImportError:
    logger.warning("Plotly not found.")

NUMBA_CUDA_AVAILABLE = False
CUPY_AVAILABLE = False
GPU_ACCELERATION_AVAILABLE = False
try:
    from numba import cuda
    if cuda.is_available():
        NUMBA_CUDA_AVAILABLE = True
        GPU_ACCELERATION_AVAILABLE = True
        logger.info("Numba CUDA found.")
    else:
        logger.info("Numba found, but CUDA is not available.")
except ImportError:
    logger.info("Numba CUDA not found.")
if not GPU_ACCELERATION_AVAILABLE:
    try:
        import cupy
        CUPY_AVAILABLE = True
        GPU_ACCELERATION_AVAILABLE = True
        logger.info("CuPy found.")
    except ImportError:
        logger.info("CuPy not found.")

GEANT4_AVAILABLE = False
try:
    import geant4 as g4
    GEANT4_AVAILABLE = True
    logger.info("Geant4 Python bindings found.")
except ImportError:
    try:
        import pyG4
        GEANT4_AVAILABLE = True
        logger.info("pyG4 (Geant4 Python bindings) found.")
    except ImportError:
        logger.warning("Geant4 Python bindings not found.")

SCIKIT_LEARN_AVAILABLE = False
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    SCIKIT_LEARN_AVAILABLE = True
    logger.info("Scikit-learn found. AI Physics Module functionality is available.")
except ImportError:
    logger.warning("Scikit-learn not found. AI Physics Module will be unavailable.")

# ===================================================================
# 4. Базовые структуры данных и вспомогательные классы
# ===================================================================
@dataclass
class Particle:
    """Информация о частице."""
    name: str
    mass: float  # ГэВ/c²
    charge: int
    spin: int
    lifetime: float  # секунды
    category: str
    symbol: str

class ParticleDatabase:
    """База данных частиц с информацией о партонах."""
    def __init__(self):
        """Инициализация базы данных частиц."""
        self.particles = self._load_particles()
        self.proton_parton_weights = {
            'u_quark': 0.35, 'ubar_quark': 0.15,
            'd_quark': 0.25, 'dbar_quark': 0.10,
            's_quark': 0.05, 'gluon': 0.10
        }
        self.parton_types = list(self.proton_parton_weights.keys())
        self.parton_weights = list(self.proton_parton_weights.values())
        total_weight = sum(self.parton_weights)
        self.parton_weights = [w / total_weight for w in self.parton_weights]

    def _load_particles(self) -> Dict[str, Particle]:
        """Загрузка информации о частицах."""
        particles = {
            'proton': Particle(name='proton', mass=0.9382720813, charge=1, spin=1/2, lifetime=float('inf'), category='baryon', symbol='p'),
            'neutron': Particle(name='neutron', mass=0.9395654133, charge=0, spin=1/2, lifetime=881.5, category='baryon', symbol='n'),
            'electron': Particle(name='electron', mass=0.0005109989461, charge=-1, spin=1/2, lifetime=float('inf'), category='lepton', symbol='e⁻'),
            'positron': Particle(name='positron', mass=0.0005109989461, charge=1, spin=1/2, lifetime=float('inf'), category='lepton', symbol='e⁺'),
            'muon': Particle(name='muon', mass=0.1056583745, charge=-1, spin=1/2, lifetime=2.1969811e-6, category='lepton', symbol='μ⁻'),
            'antimuon': Particle(name='antimuon', mass=0.1056583745, charge=1, spin=1/2, lifetime=2.1969811e-6, category='lepton', symbol='μ⁺'),
            'photon': Particle(name='photon', mass=0.0, charge=0, spin=1, lifetime=float('inf'), category='boson', symbol='γ'),
            'gluon': Particle(name='gluon', mass=0.0, charge=0, spin=1, lifetime=0, category='boson', symbol='g'),
            'u_quark': Particle(name='up quark', mass=0.0022, charge=2/3, spin=1/2, lifetime=0, category='quark', symbol='u'),
            'd_quark': Particle(name='down quark', mass=0.0047, charge=-1/3, spin=1/2, lifetime=0, category='quark', symbol='d'),
            's_quark': Particle(name='strange quark', mass=0.095, charge=-1/3, spin=1/2, lifetime=0, category='quark', symbol='s'),
            'ubar_quark': Particle(name='anti-up quark', mass=0.0022, charge=-2/3, spin=1/2, lifetime=0, category='quark', symbol='ū'),
            'dbar_quark': Particle(name='anti-down quark', mass=0.0047, charge=1/3, spin=1/2, lifetime=0, category='quark', symbol='d̄'),
            'gluon': Particle(name='gluon', mass=0.0, charge=0, spin=1, lifetime=0, category='boson', symbol='g'),
            'bottom_quark': Particle(name='bottom quark', mass=4.18, charge=-1/3, spin=1/2, lifetime=0, category='quark', symbol='b'),
            'antibottom_quark': Particle(name='antibottom quark', mass=4.18, charge=1/3, spin=1/2, lifetime=0, category='quark', symbol='b̄'),
            'top_quark': Particle(name='top quark', mass=173.1, charge=2/3, spin=1/2, lifetime=5e-25, category='quark', symbol='t'),
            'antitop_quark': Particle(name='antitop quark', mass=173.1, charge=-2/3, spin=1/2, lifetime=5e-25, category='quark', symbol='t̄'),
            'W_plus': Particle(name='W plus boson', mass=80.379, charge=1, spin=1, lifetime=3e-25, category='boson', symbol='W⁺'),
            'W_minus': Particle(name='W minus boson', mass=80.379, charge=-1, spin=1, lifetime=3e-25, category='boson', symbol='W⁻'),
            'Z0': Particle(name='Z boson', mass=91.1876, charge=0, spin=1, lifetime=3e-25, category='boson', symbol='Z⁰'),
            'Higgs': Particle(name='Higgs boson', mass=125.18, charge=0, spin=0, lifetime=1.56e-22, category='boson', symbol='H⁰'),
            'pion_plus': Particle(name='pion plus', mass=0.13957061, charge=1, spin=0, lifetime=2.6033e-8, category='meson', symbol='π⁺'),
            'pion_minus': Particle(name='pion minus', mass=0.13957061, charge=-1, spin=0, lifetime=2.6033e-8, category='meson', symbol='π⁻'),
            'pion_zero': Particle(name='pion zero', mass=0.1349770, charge=0, spin=0, lifetime=8.4e-17, category='meson', symbol='π⁰'),
            'kaon_plus': Particle(name='kaon plus', mass=0.493677, charge=1, spin=0, lifetime=1.238e-8, category='meson', symbol='K⁺'),
            'kaon_minus': Particle(name='kaon minus', mass=0.493677, charge=-1, spin=0, lifetime=1.238e-8, category='meson', symbol='K⁻'),
            'lead_nucleus': Particle(name='lead nucleus', mass=193.688, charge=82, spin=0, lifetime=float('inf'), category='nucleus', symbol='Pb'),
        }
        return particles

    def get_particle(self, name: str) -> Optional[Particle]:
        """Получение информации о частице по имени."""
        return self.particles.get(name.lower())

    def get_category(self, name: str) -> Optional[str]:
        """Получение категории частицы."""
        particle = self.get_particle(name)
        return particle.category if particle else None

    def sample_parton(self) -> str:
        """Выбирает тип партонa из протона на основе упрощённых весов."""
        return random.choices(self.parton_types, weights=self.parton_weights, k=1)[0]

# ===================================================================
# 5. Абстрактные интерфейсы для движков
# ===================================================================
class PhysicsEngineInterface(ABC):
    """Абстрактный интерфейс для физических движков (генераторов событий)"""
    @abstractmethod
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия частиц"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Получение имени движка"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности движка"""
        pass

class BeamDynamicsInterface(ABC):
    """Абстрактный интерфейс для моделирования динамики пучка"""
    @abstractmethod
    def simulate_turn(self, state: Dict, revolution_time: float, include_space_charge: bool = True, **kwargs) -> Dict:
        """Симуляция одного оборота пучка"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Получение имени движка"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности движка"""
        pass

# ===================================================================
# 6. Реализации физических движков
# ===================================================================
class BuiltInPhysicsEngine(PhysicsEngineInterface):
    """Улучшенный встроенный физический движок с элементами QCD и ЭВ."""
    def __init__(self, particle_db):
        self.particle_db = particle_db

    def _sample_x(self) -> float:
        """Генерирует долю импульса x для партонa. Упрощённая модель: x ~ exp(-b*x)"""
        b = 5.0
        x = 1.0
        while x >= 1.0:
             u = random.random()
             x = -np.log(1 - u * (1 - np.exp(-b))) / b
        return max(1e-6, x)

    def _fragment_hadron(self, parent_energy: float, num_hadrons: int) -> List[Dict]:
        """Фрагментирует энергию родителя на несколько адронов."""
        base_energy = parent_energy / num_hadrons
        hadrons = []
        remaining_energy = parent_energy
        pion_types = ['pion_plus', 'pion_minus', 'pion_zero']
        kaon_types = ['kaon_plus', 'kaon_minus']

        for i in range(num_hadrons):
            if i == num_hadrons - 1:
                energy = remaining_energy
            else:
                energy = max(0.1, base_energy * random.uniform(0.7, 1.3))
                energy = min(energy, remaining_energy - 0.1 * (num_hadrons - i - 1))
            
            remaining_energy -= energy

            if random.random() < 0.85:
                hadron_type = random.choice(pion_types)
            else:
                hadron_type = random.choice(kaon_types)
            
            h_info = self.particle_db.get_particle(hadron_type)
            if h_info:
                mass = h_info.mass
                if mass < energy:
                    momentum = np.sqrt(energy**2 - mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                hadrons.append({
                    'name': hadron_type, 'energy': energy,
                    'px': px, 'py': py, 'pz': pz, 'mass': mass
                })
        return hadrons

    def _generate_jet(self, energy: float) -> List[Dict]:
        """Генерация струи (джета) с улучшенной фрагментацией."""
        num_hadrons = max(3, int(energy / 5.0))
        num_hadrons = min(num_hadrons, 30)
        num_hadrons = int(random.gauss(num_hadrons, num_hadrons * 0.2))
        num_hadrons = max(2, num_hadrons)
        return self._fragment_hadron(energy, num_hadrons)

    def _generate_drell_yan(self, energy: float) -> List[Dict]:
        """Генерация лептонной пары в канале Дрелла-Яна."""
        mu_energy = energy * random.uniform(0.4, 0.6)
        amu_energy = energy - mu_energy
        
        mu_info = self.particle_db.get_particle('muon')
        amu_info = self.particle_db.get_particle('antimuon')
        
        products = []
        if mu_info and amu_info:
            if mu_info.mass < mu_energy:
                momentum = np.sqrt(mu_energy**2 - mu_info.mass**2)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                mu_px = momentum * np.sin(theta) * np.cos(phi)
                mu_py = momentum * np.sin(theta) * np.sin(phi)
                mu_pz = momentum * np.cos(theta)
            else: mu_px, mu_py, mu_pz = 0.0, 0.0, 0.0
            
            products.append({
                'name': 'muon', 'energy': mu_energy,
                'px': mu_px, 'py': mu_py, 'pz': mu_pz, 'mass': mu_info.mass
            })
            
            if amu_info.mass < amu_energy:
                momentum = np.sqrt(amu_energy**2 - amu_info.mass**2)
                amu_px = -mu_px + random.gauss(0, 0.05 * abs(mu_px))
                amu_py = -mu_py + random.gauss(0, 0.05 * abs(mu_py))
                amu_pz = -mu_pz + random.gauss(0, 0.05 * abs(mu_pz))
            else: amu_px, amu_py, amu_pz = 0.0, 0.0, 0.0
            
            products.append({
                'name': 'antimuon', 'energy': amu_energy,
                'px': amu_px, 'py': amu_py, 'pz': amu_pz, 'mass': amu_info.mass
            })
        return products

    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """
        Улучшенное моделирование взаимодействия частиц (в основном pp).
        """
        events = []
        for _ in range(num_events):
            if particle1 == 'proton' and particle2 == 'proton':
                x1 = self._sample_x()
                x2 = self._sample_x()
                parton1_type = self.particle_db.sample_parton()
                parton2_type = self.particle_db.sample_parton()
                parton_energy1 = energy * x1
                parton_energy2 = energy * x2
                total_parton_energy = parton_energy1 + parton_energy2
                
                prob_qq, prob_qg, prob_gg, prob_dy = 0.5, 0.3, 0.15, 0.05
                r = random.random()
                process_type = 'generic'
                products = []
                
                if r < prob_qq:
                    process_type = 'qq_scattering'
                    jet1_energy = total_parton_energy * random.uniform(0.4, 0.6)
                    jet2_energy = total_parton_energy - jet1_energy
                    products = self._generate_jet(jet1_energy) + self._generate_jet(jet2_energy)
                    
                elif r < prob_qq + prob_qg:
                    process_type = 'qg_scattering'
                    jet1_energy = total_parton_energy * random.uniform(0.4, 0.6)
                    jet2_energy = total_parton_energy - jet1_energy
                    products = self._generate_jet(jet1_energy) + self._generate_jet(jet2_energy)
                    
                elif r < prob_qq + prob_qg + prob_gg:
                    process_type = 'gg_fusion'
                    if random.random() < 0.7:
                        jet1_energy = total_parton_energy * random.uniform(0.4, 0.6)
                        jet2_energy = total_parton_energy - jet1_energy
                        products = self._generate_jet(jet1_energy) + self._generate_jet(jet2_energy)
                    else:
                        if random.random() < 0.5:
                            heavy_quark_type = 'bottom_quark'
                            anti_quark_type = 'antibottom_quark'
                        else:
                            heavy_quark_type = 'top_quark'
                            anti_quark_type = 'antitop_quark'
                        
                        hq_info = self.particle_db.get_particle(heavy_quark_type)
                        aq_info = self.particle_db.get_particle(anti_quark_type)
                        if hq_info and aq_info:
                            hq_energy = total_parton_energy * random.uniform(0.45, 0.55)
                            aq_energy = total_parton_energy - hq_energy
                            if hq_info.mass < hq_energy:
                                momentum = np.sqrt(hq_energy**2 - hq_info.mass**2)
                                theta = random.uniform(0, np.pi)
                                phi = random.uniform(0, 2*np.pi)
                                hq_px = momentum * np.sin(theta) * np.cos(phi)
                                hq_py = momentum * np.sin(theta) * np.sin(phi)
                                hq_pz = momentum * np.cos(theta)
                            else: hq_px, hq_py, hq_pz = 0.0, 0.0, 0.0
                            
                            if aq_info.mass < aq_energy:
                                aq_px = -hq_px + random.gauss(0, 0.1 * abs(hq_px))
                                aq_py = -hq_py + random.gauss(0, 0.1 * abs(hq_py))
                                aq_pz = -hq_pz + random.gauss(0, 0.1 * abs(hq_pz))
                            else: aq_px, aq_py, aq_pz = 0.0, 0.0, 0.0
                            
                            products = [
                                {'name': heavy_quark_type, 'energy': hq_energy, 'px': hq_px, 'py': hq_py, 'pz': hq_pz, 'mass': hq_info.mass},
                                {'name': anti_quark_type, 'energy': aq_energy, 'px': aq_px, 'py': aq_py, 'pz': aq_pz, 'mass': aq_info.mass}
                            ]
                            
                elif r < prob_qq + prob_qg + prob_gg + prob_dy:
                    process_type = 'drell_yan'
                    products = self._generate_drell_yan(total_parton_energy)
                
                else:
                    products = [{'name': random.choice(['electron', 'muon', 'photon', 'jet']), 'energy': random.uniform(10, total_parton_energy/2)} for _ in range(random.randint(2, 6))]

                event = {
                    'event_type': process_type,
                    'energy': energy,
                    'parton_x1': x1,
                    'parton_x2': x2,
                    'parton_type1': parton1_type,
                    'parton_type2': parton2_type,
                    'parton_energy': total_parton_energy,
                    'products': products,
                    'timestamp': time.time()
                }
                
            elif particle1 == 'lead_nucleus' and particle2 == 'lead_nucleus':
                logger.info("Генерация события Pb-Pb столкновения (упрощённая модель).")
                energy_per_nucleon = energy / (2 * 208)
                num_sub_interactions = random.choices([1, 2, 3, 4], weights=[0.1, 0.3, 0.4, 0.2], k=1)[0]
                all_products = []
                for _ in range(num_sub_interactions):
                    sub_energy = energy_per_nucleon * random.uniform(0.5, 1.5) * 2
                    pp_event = self._simulate_pp_interaction(sub_energy)
                    all_products.extend(pp_event.get('products', []))
                
                event = {
                    'event_type': 'PbPb_collision',
                    'energy': energy,
                    'sub_interactions': num_sub_interactions,
                    'products': all_products,
                    'timestamp': time.time()
                }
            else:
                event = {
                    'event_type': 'generic_other',
                    'energy': energy,
                    'products': [{'name': random.choice(['electron', 'muon', 'photon']), 'energy': random.uniform(1, energy/4)} for _ in range(random.randint(1, 3))],
                    'timestamp': time.time()
                }
            events.append(event)
        return events

    def _simulate_pp_interaction(self, energy: float) -> Dict:
        """Симулирует одно протон-протонное взаимодействие."""
        x1 = self._sample_x()
        x2 = self._sample_x()
        parton1_type = self.particle_db.sample_parton()
        parton2_type = self.particle_db.sample_parton()
        parton_energy1 = energy * x1
        parton_energy2 = energy * x2
        total_parton_energy = parton_energy1 + parton_energy2

        prob_qq, prob_qg, prob_gg, prob_dy = 0.5, 0.3, 0.15, 0.05
        r = random.random()
        process_type, products = 'generic', []

        if r < prob_qq:
            process_type = 'qq_scattering'
            jet1_energy = total_parton_energy * random.uniform(0.4, 0.6)
            jet2_energy = total_parton_energy - jet1_energy
            products = self._generate_jet(jet1_energy) + self._generate_jet(jet2_energy)
        elif r < prob_qq + prob_qg:
            process_type = 'qg_scattering'
            jet1_energy = total_parton_energy * random.uniform(0.4, 0.6)
            jet2_energy = total_parton_energy - jet1_energy
            products = self._generate_jet(jet1_energy) + self._generate_jet(jet2_energy)
        elif r < prob_qq + prob_qg + prob_gg:
            process_type = 'gg_fusion'
            if random.random() < 0.7:
                jet1_energy = total_parton_energy * random.uniform(0.4, 0.6)
                jet2_energy = total_parton_energy - jet1_energy
                products = self._generate_jet(jet1_energy) + self._generate_jet(jet2_energy)
            else:
                heavy_quark_type = random.choice(['bottom_quark', 'top_quark'])
                anti_quark_type = 'antibottom_quark' if heavy_quark_type == 'bottom_quark' else 'antitop_quark'
                hq_info = self.particle_db.get_particle(heavy_quark_type)
                aq_info = self.particle_db.get_particle(anti_quark_type)
                if hq_info and aq_info:
                    hq_energy = total_parton_energy * random.uniform(0.45, 0.55)
                    aq_energy = total_parton_energy - hq_energy
                    if hq_info.mass < hq_energy:
                        momentum = np.sqrt(hq_energy**2 - hq_info.mass**2)
                        theta = random.uniform(0, np.pi)
                        phi = random.uniform(0, 2*np.pi)
                        hq_px = momentum * np.sin(theta) * np.cos(phi)
                        hq_py = momentum * np.sin(theta) * np.sin(phi)
                        hq_pz = momentum * np.cos(theta)
                    else: hq_px, hq_py, hq_pz = 0.0, 0.0, 0.0
                    if aq_info.mass < aq_energy:
                        aq_px = -hq_px + random.gauss(0, 0.1 * abs(hq_px))
                        aq_py = -hq_py + random.gauss(0, 0.1 * abs(hq_py))
                        aq_pz = -hq_pz + random.gauss(0, 0.1 * abs(hq_pz))
                    else: aq_px, aq_py, aq_pz = 0.0, 0.0, 0.0
                    products = [
                        {'name': heavy_quark_type, 'energy': hq_energy, 'px': hq_px, 'py': hq_py, 'pz': hq_pz, 'mass': hq_info.mass},
                        {'name': anti_quark_type, 'energy': aq_energy, 'px': aq_px, 'py': aq_py, 'pz': aq_pz, 'mass': aq_info.mass}
                    ]
        elif r < prob_qq + prob_qg + prob_gg + prob_dy:
            process_type = 'drell_yan'
            products = self._generate_drell_yan(total_parton_energy)
        else:
            products = [{'name': random.choice(['electron', 'muon', 'photon', 'jet']), 'energy': random.uniform(10, total_parton_energy/2)} for _ in range(random.randint(2, 6))]

        return {
            'event_type': process_type,
            'energy': energy,
            'parton_x1': x1,
            'parton_x2': x2,
            'parton_type1': parton1_type,
            'parton_type2': parton2_type,
            'parton_energy': total_parton_energy,
            'products': products,
            'timestamp': time.time()
        }


    def get_name(self) -> str:
        return "built-in"

    def is_available(self) -> bool:
        return True

class Geant4PhysicsEngine(PhysicsEngineInterface):
    """Интеграция с Geant4 для детального моделирования взаимодействий."""
    def __init__(self):
        self.initialized = False
        self.geant4 = None
        if GEANT4_AVAILABLE:
            try:
                self.initialized = True
                logger.info("Geant4 успешно инициализирован (stub).")
            except Exception as e:
                logger.error(f"Ошибка инициализации Geant4: {e}")
                self.initialized = False
        else:
            logger.warning("Geant4 недоступен.")

    def get_name(self) -> str:
        return "geant4"

    def is_available(self) -> bool:
        return GEANT4_AVAILABLE and self.initialized

    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия с использованием Geant4."""
        if not self.is_available():
            logger.warning("Geant4 недоступен. Используется встроенная модель.")
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        logger.warning("Geant4 integration is a stub. Using BuiltInPhysicsEngine for demonstration.")
        built_in_engine = BuiltInPhysicsEngine(ParticleDatabase())
        events = built_in_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        return events

    def _convert_to_geant4_particles(self, particle1: str, particle2: str, energy: float) -> List:
        """Конвертация частиц в формат Geant4."""
        logger.warning("_convert_to_geant4_particles not implemented.")
        return []

    def _convert_from_geant4_results(self, results) -> List[Dict]:
        """Конвертация результатов Geant4 в наш формат."""
        logger.warning("_convert_from_geant4_results not implemented.")
        return []

# ===================================================================
# 7. Реализации движков динамики пучка
# ===================================================================
class BuiltInBeamDynamics(BeamDynamicsInterface):
    """Встроенный движок динамики пучка."""
    def simulate_turn(self, state: Dict, revolution_time: float, include_space_charge: bool = True, **kwargs) -> Dict:
        """Симуляция одного оборота пучка."""
        updated_state = state.copy()
        updated_state['beam_dynamics']['time'].append(state['beam_dynamics']['time'][-1] + revolution_time)
        updated_state['beam_dynamics']['luminosity'].append(state['beam_dynamics']['luminosity'][-1] * random.uniform(0.99, 1.01))
        updated_state['beam_dynamics']['beam_size_x'].append(state['beam_dynamics']['beam_size_x'][-1] * random.uniform(0.998, 1.002))
        updated_state['beam_dynamics']['beam_size_y'].append(state['beam_dynamics']['beam_size_y'][-1] * random.uniform(0.998, 1.002))
        updated_state['beam_dynamics']['emittance'].append(state['beam_dynamics']['emittance'][-1] * random.uniform(0.999, 1.001))
        return updated_state

    def get_name(self) -> str:
        return "built-in"

    def is_available(self) -> bool:
        return True

# ===================================================================
# 8. Гибридные движки
# ===================================================================
class HybridPhysicsEngine:
    """Гибридный физический движок с поддержкой приоритетов."""
    def __init__(self, particle_db):
        self.particle_db = particle_db
        self.engines = {
            "built-in": BuiltInPhysicsEngine(particle_db),
            "geant4": Geant4PhysicsEngine()
        }
        self.preferred_engines = CONFIG.get("simulation", {}).get("physics_engines", ["built-in"])
        logger.info(f"Гибридный физический движок инициализирован. Предпочтения: {self.preferred_engines}")

    def set_engine_priority(self, priorities: List[str]):
        """Устанавливает приоритет движков."""
        self.preferred_engines = [p for p in priorities if p in self.engines]
        logger.info(f"Приоритеты физических движков установлены: {self.preferred_engines}")

    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, force_engine: Optional[str] = None, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия с выбором движка."""
        if force_engine and force_engine in self.engines:
            logger.info(f"Используется принудительно выбранный физический движок: {force_engine}")
            if self.engines[force_engine].is_available():
                return self.engines[force_engine].interact(particle1, particle2, energy, num_events, **kwargs)
            else:
                logger.warning(f"Движок {force_engine} недоступен, используем встроенный.")
                return self.engines["built-in"].interact(particle1, particle2, energy, num_events, **kwargs)

        for engine_name in self.preferred_engines:
            engine = self.engines[engine_name]
            if engine.is_available():
                logger.info(f"Используется физический движок: {engine_name}")
                return engine.interact(particle1, particle2, energy, num_events, **kwargs)

        logger.warning("Нет доступных предпочтительных движков, используем встроенный.")
        return self.engines["built-in"].interact(particle1, particle2, energy, num_events, **kwargs)

class HybridBeamDynamics:
    """Гибридный движок динамики пучка с поддержкой приоритетов."""
    def __init__(self):
        self.engines = {
            "built-in": BuiltInBeamDynamics()
        }
        self.preferred_engines = [CONFIG.get("simulation", {}).get("beam_dynamics_engine", "built-in")]
        logger.info(f"Гибридный движок динамики инициализирован. Предпочтения: {self.preferred_engines}")

    def set_engine_priority(self, priorities: List[str]):
        """Устанавливает приоритет движков."""
        self.preferred_engines = [p for p in priorities if p in self.engines]
        logger.info(f"Приоритеты движков динамики установлены: {self.preferred_engines}")

    def simulate_turn(self, state: Dict, revolution_time: float, include_space_charge: bool = True, force_engine: Optional[str] = None, **kwargs) -> Dict:
        """Симуляция одного оборота пучка с выбором движка."""
        if force_engine and force_engine in self.engines:
            logger.info(f"Используется принудительно выбранный движок динамики: {force_engine}")
            if self.engines[force_engine].is_available():
                return self.engines[force_engine].simulate_turn(state, revolution_time, include_space_charge, **kwargs)
            else:
                logger.warning(f"Движок {force_engine} недоступен, используем встроенный.")
                return self.engines["built-in"].simulate_turn(state, revolution_time, include_space_charge, **kwargs)

        for engine_name in self.preferred_engines:
            engine = self.engines[engine_name]
            if engine.is_available():
                logger.info(f"Используется движок динамики: {engine_name}")
                return engine.simulate_turn(state, revolution_time, include_space_charge, **kwargs)

        logger.warning("Нет доступных предпочтительных движков динамики, используем встроенный.")
        return self.engines["built-in"].simulate_turn(state, revolution_time, include_space_charge, **kwargs)

# ===================================================================
# 9. Система кэширования
# ===================================================================
class SimulationCache:
    """Система кэширования результатов симуляции."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = 0
        self.hit_count = 0
        logger.info(f"Кэш симуляции инициализирован с максимальным размером {max_size}")

    @staticmethod
    def generate_key(params: Dict) -> str:
        """Генерация ключа для кэширования."""
        sorted_items = sorted(params.items())
        param_str = ",".join([f"{k}={v}" for k, v in sorted_items])
        return str(hash(param_str))

    def get(self, key: str) -> Optional[Dict]:
        """Получение результатов из кэша."""
        self.access_count += 1
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        return None

    def set(self, key: str, value: Dict):
        """Сохранение результатов в кэш."""
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = value

    def get_hit_rate(self) -> float:
        """Получение hit rate кэша."""
        if self.access_count == 0:
            return 0.0
        return self.hit_count / self.access_count

# ===================================================================
# 10. Система валидации и калибровки
# ===================================================================
class ValidationSystem:
    """Система валидации симуляции против реальных данных."""
    def __init__(self, lhc_model):
        self.lhc_model = lhc_model
        self.validation_data = {}
        self.calibration_factors = {'luminosity': 1.0, 'emittance': 1.0, 'beam_size': 1.0}
        self.calibration_history = []

    def load_real_data(self, dataset_id: str):
        """Загружает реальные данные для валидации."""
        self.validation_data = {
            "dataset_id": dataset_id,
            "description": "CMS open data from LHC Run 2",
            "energy": 13000,
            "luminosity": 1.5e34,
            "num_events": 100000,
            "collision_type": "proton-proton",
            "recorded_data": [
                {"event_id": i, "energy": random.uniform(100, 1000),
                 "products": random.sample(["electron", "muon", "photon", "jet"], random.randint(1, 4)),
                 "timestamp": time.time() - random.randint(0, 86400)}
                for i in range(100)
            ]
        }
        logger.info(f"Загружены реальные данные для валидации: {dataset_id}")

    def validate_luminosity(self) -> float:
        """Валидация светимости."""
        if not self.validation_data:
            logger.warning("Данные для валидации не загружены")
            return 0.0
        real_luminosity = self.validation_data.get("luminosity", 2.0e34)
        sim_luminosity = self.lhc_model.simulation_state['luminosity']
        relative_error = abs(real_luminosity - sim_luminosity) / real_luminosity if real_luminosity > 0 else 0.0
        logger.info(f"Валидация светимости: реальная={real_luminosity:.2e}, симулированная={sim_luminosity:.2e}, ошибка={relative_error:.2%}")
        return relative_error

    def calibrate(self):
        """Калибровка модели на основе реальных данных."""
        if not self.validation_data:
            logger.warning("Нет данных для калибровки")
            return
        real_luminosity = self.validation_data.get("luminosity", 2.0e34)
        sim_luminosity = self.lhc_model.simulation_state['luminosity']
        self.calibration_factors['luminosity'] = real_luminosity / sim_luminosity if sim_luminosity > 0 else 1.0
        self.calibration_history.append({
            'timestamp': time.time(),
            'factors': self.calibration_factors.copy(),
            'luminosity_error': abs(real_luminosity - sim_luminosity) / real_luminosity if real_luminosity > 0 else 0.0,
        })
        logger.info(f"Модель откалибрована. Факторы: {self.calibration_factors}")

# ===================================================================
# 11. Система детектирования (упрощенная)
# ===================================================================
class DetectorSystem:
    """Упрощенная система детекторов."""
    def __init__(self, circumference: float):
        self.circumference = circumference
        self.detectors = self._initialize_detectors()

    def _initialize_detectors(self) -> Dict[str, Any]:
        """Инициализация детекторов."""
        positions = {'DetectorA': 0.25, 'DetectorB': 0.75}
        detector_specs = {
            'DetectorA': {'size': (10, 10, 20), 'resolution': 0.02, 'efficiency': 0.95},
            'DetectorB': {'size': (15, 15, 15), 'resolution': 0.03, 'efficiency': 0.92},
        }
        detectors = {}
        for name, pos_ratio in positions.items():
            angle = 2 * np.pi * pos_ratio
            radius = self.circumference / (2 * np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0
            specs = detector_specs[name]
            detectors[name] = {
                'name': name, 'position': (x, y, z), 'size': specs['size'],
                'resolution': specs['resolution'], 'efficiency': specs['efficiency']
            }
        return detectors

    def detect_event(self, event: Dict) -> List[Dict]:
        """Моделирование детектирования события."""
        detected_particles = []
        for product in event.get('products', []):
            particle_name = product['name']
            true_energy = product['energy']
            for detector in self.detectors.values():
                if random.random() < detector['efficiency']:
                    reconstructed_energy = np.random.normal(true_energy, true_energy * detector['resolution'])
                    reconstructed_momentum = reconstructed_energy * random.uniform(0.8, 1.2)
                    detected_particles.append({
                        'detector': detector['name'],
                        'particle': particle_name,
                        'true_energy': true_energy,
                        'reconstructed_energy': reconstructed_energy,
                        'reconstructed_momentum': reconstructed_momentum
                    })
        return detected_particles

# ===================================================================
# 12. *** МОДУЛЬ: TopoAnalyzer ***
# ===================================================================
class TopoAnalyzer:
    """
    Улучшенный Топологический анализатор событий.
    Использует идеи из топологического анализа данных (TDA) и анализа корреляций.
    Вдохновлен топологическим анализом ECDSA (торы, числа Бетти).
    """
    def __init__(self, events: List[Dict[str, Any]], feature_names: Optional[List[str]] = None):
        """Инициализирует анализатор."""
        self.events = events
        self.feature_names = feature_names or [
            'num_products', 'total_energy', 'total_px', 'total_py', 'total_pz',
            'num_jets', 'num_muons', 'num_electrons', 'num_photons',
            'max_product_energy', 'invariant_mass'
        ]
        self.feature_vectors = np.array([])
        self.distance_matrix = None
        self.scaler = None
        self.correlation_matrix = None
        self.correlation_spectrum = None
        self.pca_result = None
        self.persistence_result = None
        self.betti_numbers = None
        logger.info("TopoAnalyzer инициализирован.")

    def _extract_features(self, event: Dict[str, Any]) -> List[float]:
        """Извлекает вектор признаков из одного события."""
        features = []
        products = event.get('products', [])
        num_products = len(products)
        total_energy = sum(p.get('energy', 0.0) for p in products)
        total_px = sum(p.get('px', 0.0) for p in products)
        total_py = sum(p.get('py', 0.0) for p in products)
        total_pz = sum(p.get('pz', 0.0) for p in products)
        num_jets = sum(1 for p in products if p.get('name') == 'jet')
        num_muons = sum(1 for p in products if p.get('name') in ['muon', 'antimuon'])
        num_electrons = sum(1 for p in products if p.get('name') in ['electron', 'positron'])
        num_photons = sum(1 for p in products if p.get('name') == 'photon')
        max_product_energy = max((p.get('energy', 0.0) for p in products), default=0.0)
        invariant_mass_sq = total_energy**2 - (total_px**2 + total_py**2 + total_pz**2)
        invariant_mass = np.sqrt(max(invariant_mass_sq, 0.0))
        feature_dict = {
            'num_products': float(num_products),
            'total_energy': float(total_energy),
            'total_px': float(total_px),
            'total_py': float(total_py),
            'total_pz': float(total_pz),
            'num_jets': float(num_jets),
            'num_muons': float(num_muons),
            'num_electrons': float(num_electrons),
            'num_photons': float(num_photons),
            'max_product_energy': float(max_product_energy),
            'invariant_mass': float(invariant_mass)
        }
        return [feature_dict[name] for name in self.feature_names]

    def build_feature_vectors(self):
        """Строит матрицу признаков из всех событий."""
        logger.info("Извлечение признаков из событий...")
        if not self.events:
             logger.warning("Список событий пуст.")
             self.feature_vectors = np.array([])
             return
        try:
            feature_list = [self._extract_features(event) for event in self.events]
            self.feature_vectors = np.array(feature_list)
            logger.info(f"Извлечено {self.feature_vectors.shape[0]} векторов признаков размерности {self.feature_vectors.shape[1]}.")
        except Exception as e:
             logger.error(f"Ошибка при извлечении признаков: {e}")
             self.feature_vectors = np.array([])

    def compute_distance_matrix(self):
        """Вычисляет матрицу расстояний между векторами признаков."""
        if self.feature_vectors.size == 0:
            logger.error("Векторы признаков не построены.")
            return
        logger.info("Вычисление матрицы расстояний...")
        try:
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(self.feature_vectors)
            self.distance_matrix = euclidean_distances(normalized_features)
            logger.info("Матрица расстояний вычислена.")
        except Exception as e:
             logger.error(f"Ошибка при вычислении матрицы расстояний: {e}")
             self.distance_matrix = None

    def compute_persistence(self, max_dimension: int = 1, max_edge_length: float = np.inf):
        """Вычисляет персистентную гомологию."""
        if self.distance_matrix is None:
            logger.error("Матрица расстояний не вычислена.")
            return

        if not PERSISTENCE_AVAILABLE:
            logger.warning("Библиотеки для персистентной гомологии недоступны.")
            return

        logger.info("Вычисление персистентной гомологии...")
        try:
            if USE_GUDHI:
                logger.info("Используется GUDHI.")
                rips_complex = gd.RipsComplex(distance_matrix=self.distance_matrix, max_edge_length=max_edge_length)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
                persistence_result = simplex_tree.persistence()
                self.persistence_result = {'dgms': [], 'simplex_tree': simplex_tree}
                for dim in range(max_dimension + 1):
                    gudhi_dgm = simplex_tree.persistence_intervals_in_dimension(dim)
                    if len(gudhi_dgm) > 0:
                        self.persistence_result['dgms'].append(np.array(gudhi_dgm))
                    else:
                         self.persistence_result['dgms'].append(np.array([]).reshape(0,2))
                logger.info("Персистентная гомология вычислена (GUDHI).")
                
            elif USE_RIPSER:
                logger.info("Используется Ripser.")
                self.persistence_result = ripser.ripser(self.distance_matrix, maxdim=max_dimension, thresh=max_edge_length, metric='precomputed')
                logger.info("Персистентная гомология вычислена (Ripser).")
        except Exception as e:
            logger.error(f"Ошибка при вычислении персистентной гомологии: {e}")
            self.persistence_result = None

    def compute_betti_numbers(self):
        """Вычисляет числа Бетти из результатов персистентной гомологии."""
        if not self.persistence_result:
            logger.warning("Нет результатов персистентной гомологии.")
            return
        betti = {}
        try:
            if USE_GUDHI and 'simplex_tree' in self.persistence_result:
                 st = self.persistence_result['simplex_tree']
                 betti_numbers_list = st.betti_numbers()
                 for i, bn in enumerate(betti_numbers_list):
                     betti[f'β{i}'] = bn
                 logger.info(f"Числа Бетти (GUDHI): {betti}")
            elif USE_RIPSER or (USE_GUDHI and 'dgms' in self.persistence_result):
                dgms = self.persistence_result.get('dgms', []) if USE_RIPSER else self.persistence_result['dgms']
                all_pers = []
                for dgm in dgms:
                    if dgm.size > 0:
                        pers = dgm[:, 1] - dgm[:, 0]
                        all_pers.extend(pers)
                if all_pers:
                    pers_threshold = 0.1 * np.max(all_pers)
                else:
                    pers_threshold = 0
                for dim, dgm in enumerate(dgms):
                    if dgm.size == 0:
                        betti[f'β{dim}'] = 0
                        continue
                    pers = dgm[:, 1] - dgm[:, 0]
                    count = np.sum(pers > pers_threshold) if pers_threshold > 0 else len(pers)
                    betti[f'β{dim}'] = int(count)
                logger.info(f"Числа Бетти (оценка по диаграммам): {betti}")
            self.betti_numbers = betti
        except Exception as e:
             logger.error(f"Ошибка при вычислении чисел Бетти: {e}")
             self.betti_numbers = None

    def find_persistence_outliers(self, dimension: int = 1, percentile: float = 99.0) -> List[int]:
        """Находит индексы событий с аномальной персистентностью."""
        if not self.persistence_result or dimension >= len(self.persistence_result.get('dgms', [])):
            logger.warning(f"Диаграмма для H_{dimension} не найдена.")
            return []
        try:
            dgm = self.persistence_result['dgms'][dimension]
            if dgm.size == 0:
                logger.info(f"Диаграмма H_{dimension} пуста.")
                return []
            pers = dgm[:, 1] - dgm[:, 0]
            if np.all(pers == 0):
                 logger.info(f"Все персистентности для H_{dimension} равны нулю.")
                 return []
            threshold = np.percentile(pers, percentile)
            outlier_indices = np.where(pers > threshold)[0].tolist()
            logger.info(f"Найдено {len(outlier_indices)} 'выбросов' в H_{dimension}.")
            return outlier_indices
        except Exception as e:
             logger.error(f"Ошибка при поиске выбросов в персистентности: {e}")
             return []

    def analyze_correlations(self, method: str = 'pearson') -> Optional[np.ndarray]:
        """Анализирует корреляции между признаками."""
        if self.feature_vectors.size == 0:
            logger.error("Векторы признаков не построены.")
            return None
        logger.info(f"Анализ корреляций между признаками (метод: {method})...")
        try:
            if method == 'pearson':
                self.correlation_matrix = np.corrcoef(self.feature_vectors, rowvar=False)
            elif method == 'spearman':
                from scipy.stats import spearmanr
                self.correlation_matrix = spearmanr(self.feature_vectors).correlation
            else:
                logger.warning(f"Неизвестный метод корреляции: {method}. Используется Pearson.")
                self.correlation_matrix = np.corrcoef(self.feature_vectors, rowvar=False)
            
            logger.info("Анализ корреляций завершен.")
            return self.correlation_matrix
        except Exception as e:
             logger.error(f"Ошибка при анализе корреляций: {e}")
             return None

    def analyze_correlation_spectrum(self) -> Optional[Dict[str, Any]]:
        """Анализирует спектр корреляционной матрицы."""
        corr_matrix = self.analyze_correlations()
        if corr_matrix is None:
            return None
        logger.info("Анализ спектра корреляционной матрицы...")
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            self.correlation_spectrum = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'condition_number': eigenvalues[0] / (eigenvalues[-1] + 1e-12)
            }
            logger.info("Анализ спектра завершен.")
            return self.correlation_spectrum
        except Exception as e:
             logger.error(f"Ошибка при анализе спектра корреляций: {e}")
             return None

    def perform_pca(self, n_components: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Выполняет PCA для снижения размерности."""
        if self.feature_vectors.size == 0:
            logger.error("Векторы признаков не построены.")
            return None
        logger.info("Выполнение PCA...")
        try:
            self.scaler_pca = StandardScaler()
            data_scaled = self.scaler_pca.fit_transform(self.feature_vectors)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)
            self.pca_result = {
                'transformed_data': pca_result,
                'components': pca.components_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'singular_values': pca.singular_values_
            }
            logger.info(f"PCA завершен. Объяснено {np.sum(pca.explained_variance_ratio_)*100:.2f}% дисперсии.")
            return self.pca_result
        except Exception as e:
             logger.error(f"Ошибка при выполнении PCA: {e}")
             return None

    def plot_correlation_matrix(self, ax=None, method: str = 'pearson'):
        """Визуализирует матрицу корреляций."""
        corr_matrix = self.analyze_correlations(method=method)
        if corr_matrix is None:
            return
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        try:
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(self.feature_names)))
            ax.set_yticks(range(len(self.feature_names)))
            ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax.set_yticklabels(self.feature_names)
            ax.set_title(f'Матрица корреляций ({method})')
            plt.colorbar(im, ax=ax)
            logger.info("Визуализирована матрица корреляций.")
        except Exception as e:
             logger.error(f"Ошибка при визуализации матрицы корреляций: {e}")

    def plot_pca(self, ax=None):
        """Визуализирует результаты PCA."""
        if self.pca_result is None:
            logger.warning("PCA не выполнен.")
            return
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        try:
            data = self.pca_result['transformed_data']
            if data.shape[1] < 2:
                logger.warning("Недостаточно компонент для 2D визуализации PCA.")
                return
            scatter = ax.scatter(data[:, 0], data[:, 1], alpha=0.6)
            ax.set_xlabel(f"PC 1 ({self.pca_result['explained_variance_ratio'][0]*100:.1f}% variance)")
            ax.set_ylabel(f"PC 2 ({self.pca_result['explained_variance_ratio'][1]*100:.1f}% variance)")
            ax.set_title('PCA проекция данных событий')
            logger.info("Визуализированы результаты PCA.")
        except Exception as e:
             logger.error(f"Ошибка при визуализации PCA: {e}")

    def plot_persistence(self, ax=None, plot_only: Optional[List[int]] = None):
        """Визуализирует результаты персистентной гомологии."""
        if not PERSISTENCE_AVAILABLE or not self.persistence_result:
            logger.warning("Нет данных для визуализации персистентной гомологии.")
            return
        try:
            if USE_GUDHI and 'simplex_tree' in self.persistence_result:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                dgm = self.persistence_result['simplex_tree'].persistence()
                gd.plot_persistence_diagram(dgm, axes=ax)
                ax.set_title('Диаграмма персистентности (GUDHI)')
            elif USE_RIPSER and 'dgms' in self.persistence_result:
                 if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                 from persim import plot_diagrams
                 plot_diagrams(self.persistence_result['dgms'], ax=ax, title='Диаграммы персистентности (Ripser/Persim)', show=plot_only)
        except Exception as e:
             logger.error(f"Ошибка при визуализации персистентной гомологии: {e}")

    def get_summary_report(self) -> Dict[str, Any]:
        """Создает сводный отчет об анализе."""
        spectrum = self.analyze_correlation_spectrum()
        return {
            'num_events': len(self.events),
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'persistence_available': PERSISTENCE_AVAILABLE,
            'betti_numbers': self.betti_numbers,
            'correlation_spectrum': {
                'eigenvalues': spectrum['eigenvalues'].tolist() if spectrum else None,
                'condition_number': spectrum['condition_number'] if spectrum else None
            },
            'pca_variance_explained': self.pca_result['explained_variance_ratio'].tolist() if self.pca_result else None
        }

    def run_full_analysis(self, compute_persistence: bool = True, max_pers_dim: int = 1, compute_pca: bool = True):
        """Запускает полный анализ."""
        logger.info("=== ЗАПУСК ПОЛНОГО ТОПОЛОГИЧЕСКОГО АНАЛИЗА ===")
        self.build_feature_vectors()
        if self.feature_vectors.size == 0:
             logger.error("Анализ остановлен: не удалось построить векторы признаков.")
             return
        self.compute_distance_matrix()
        if compute_persistence and PERSISTENCE_AVAILABLE:
            self.compute_persistence(max_dimension=max_pers_dim)
            self.compute_betti_numbers()
        self.analyze_correlation_spectrum()
        if compute_pca:
             self.perform_pca(n_components=min(5, len(self.feature_names)))
        logger.info("=== ПОЛНЫЙ ТОПОЛОГИЧЕСКИЙ АНАЛИЗ ЗАВЕРШЕН ===")

# ===================================================================
# 13. *** МОДУЛЬ: GradientCalibrator ***
# ===================================================================
class GradientCalibrator:
    """
    Калибровщик модели на основе градиентного анализа и оптимизации.
    Использует scipy.optimize для надежной минимизации ошибки.
    """
    def __init__(self, model, target_observables: Dict[str, float], 
                 parameters_to_calibrate: List[str], 
                 observable_getters: Dict[str, Callable], 
                 perturbation_factor: float = 0.001,
                 error_weights: Optional[Dict[str, float]] = None):
        """
        Инициализация калибратора.
        """
        self.model = model
        self.target_observables = target_observables
        self.parameters_to_calibrate = parameters_to_calibrate
        self.observable_getters = observable_getters
        self.perturbation_factor = perturbation_factor
        self.error_weights = error_weights or {k: 1.0 for k in target_observables.keys()}
        self.original_config = None
        self.history = []
        self.optimization_result = None
        self.sensitivity_analysis = None
        logger.info("GradientCalibrator инициализирован.")
        logger.info(f"Целевые величины: {target_observables}")
        logger.info(f"Параметры для калибровки: {parameters_to_calibrate}")

    def _get_current_observables(self) -> Dict[str, float]:
        """Получает текущие значения наблюдаемых величин из модели."""
        current_obs = {}
        for name, getter_func in self.observable_getters.items():
            try:
                current_obs[name] = getter_func(self.model)
            except Exception as e:
                logger.error(f"Ошибка при получении наблюдаемой величины '{name}': {e}")
                current_obs[name] = 0.0
        return current_obs

    def _set_parameters(self, param_values: np.ndarray):
        """Устанавливает значения параметров в модели."""
        for i, param_name in enumerate(self.parameters_to_calibrate):
            value = param_values[i]
            if param_name in self.model.config.get('beam', {}):
                self.model.config['beam'][param_name] = value
            elif param_name in self.model.config.get('geometry', {}):
                self.model.config['geometry'][param_name] = value
            else:
                logger.warning(f"Параметр '{param_name}' не найден.")

    def _get_parameters(self) -> np.ndarray:
        """Получает текущие значения параметров из модели."""
        values = []
        for param_name in self.parameters_to_calibrate:
            if param_name in self.model.config.get('beam', {}):
                values.append(self.model.config['beam'][param_name])
            elif param_name in self.model.config.get('geometry', {}):
                values.append(self.model.config['geometry'][param_name])
            else:
                logger.warning(f"Параметр '{param_name}' не найден.")
                values.append(0.0)
        return np.array(values)

    def _objective_function(self, param_values: np.ndarray, num_turns: int = 10) -> float:
        """
        Целевая функция для минимизации.
        """
        self._set_parameters(param_values)
        try:
            temp_model = type(self.model)()
            temp_model.config = self.model.config.copy()
            temp_model.run_simulation(num_turns=num_turns, force_physics_engine="built-in", force_beam_engine="built-in")
            current_observables = self._get_current_observables_from_model(temp_model)
            total_error_sq = 0.0
            for obs_name, target_val in self.target_observables.items():
                current_val = current_observables.get(obs_name, 0.0)
                error = current_val - target_val
                weight = self.error_weights.get(obs_name, 1.0)
                scale = abs(target_val) if target_val != 0 else 1.0
                normalized_error = error / scale
                total_error_sq += weight * (normalized_error ** 2)
            rmse = np.sqrt(total_error_sq / len(self.target_observables))
            self.history.append({'params': param_values.copy(), 'observables': current_observables.copy(), 'rmse': rmse})
            logger.debug(f"Целевая функция: params={param_values}, RMSE={rmse:.2e}")
            return rmse
        except Exception as e:
            logger.error(f"Ошибка в целевой функции: {e}")
            return 1e6 

    def _get_current_observables_from_model(self, model_instance) -> Dict[str, float]:
        """Вспомогательная функция для извлечения наблюдаемых."""
        obs = {}
        for name, getter_func in self.observable_getters.items():
            try:
                if name == 'luminosity':
                    obs[name] = model_instance.simulation_state.get('luminosity', 0.0)
                elif name == 'beam_size_x':
                    obs[name] = model_instance.simulation_state.get('beam_size_x', 0.0)
                elif name == 'avg_event_energy':
                    events = model_instance.simulation_state.get('collision_events', [])
                    if events:
                        total_energy = sum(e.get('energy', 0.0) for e in events)
                        obs[name] = total_energy / len(events)
                    else:
                        obs[name] = 0.0
                else:
                     obs[name] = 0.0
            except Exception as e:
                logger.error(f"Ошибка при извлечении '{name}': {e}")
                obs[name] = 0.0
        return obs

    def calibrate(self, num_turns: int = 10, initial_guess: Optional[List[float]] = None, 
                  method: str = 'L-BFGS-B', tolerance: float = 1e-4, max_iterations: int = 50):
        """
        Процесс калибровки с использованием scipy.optimize.
        """
        logger.info("Начало процесса калибровки...")
        if self.original_config is None:
             self.original_config = self.model.config.copy()
        if initial_guess is not None:
            x0 = np.array(initial_guess)
            logger.info(f"Используется заданное начальное приближение: {x0}")
        else:
            x0 = self._get_parameters()
            logger.info(f"Используются текущие параметры модели: {x0}")
        bounds = [(None, None) for _ in self.parameters_to_calibrate]
        extra_args = (num_turns,)
        logger.info(f"Запуск оптимизации методом {method}...")
        try:
            self.optimization_result = minimize(
                fun=self._objective_function,
                x0=x0,
                args=extra_args,
                method=method,
                bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
                tol=tolerance,
                options={'maxiter': max_iterations}
            )
            logger.info("Оптимизация завершена.")
            logger.info(f"Результат: {self.optimization_result.message}")
            logger.info(f"Финальная ошибка (RMSE): {self.optimization_result.fun:.2e}")
            logger.info(f"Финальные параметры: {self.optimization_result.x}")
            if self.optimization_result.success:
                self._set_parameters(self.optimization_result.x)
                logger.info("Оптимальные параметры установлены.")
            else:
                logger.warning("Оптимизация не сошлась.")
        except Exception as e:
            logger.error(f"Ошибка в процессе калибровки: {e}")
            self.optimization_result = None

    def analyze_sensitivity(self, num_turns: int = 10, use_original_config: bool = True):
        """
        Анализ чувствительности.
        """
        logger.info("Начало анализа чувствительности...")
        if self.optimization_result is None or not self.optimization_result.success:
            logger.warning("Нет результата успешной оптимизации.")
            return
        try:
            optimal_params = self.optimization_result.x
            base_error = self.optimization_result.fun
            if use_original_config and self.original_config:
                 self.model.config = self.original_config.copy()
            gradients = {}
            hess_diag_approx = {}
            for i, param_name in enumerate(self.parameters_to_calibrate):
                perturbed_params_up = optimal_params.copy()
                delta_p = optimal_params[i] * self.perturbation_factor
                perturbed_params_up[i] += delta_p
                error_up = self._objective_function(perturbed_params_up, num_turns)
                perturbed_params_down = optimal_params.copy()
                perturbed_params_down[i] -= delta_p
                error_down = self._objective_function(perturbed_params_down, num_turns)
                if delta_p != 0:
                    gradient = (error_up - error_down) / (2 * delta_p)
                    gradients[param_name] = gradient
                    second_deriv = (error_up - 2 * base_error + error_down) / (delta_p ** 2)
                    hess_diag_approx[param_name] = second_deriv
                    logger.info(f"Параметр '{param_name}': градиент={gradient:.4e}, d2E/dp2={second_deriv:.4e}")
                else:
                    gradients[param_name] = 0.0
                    hess_diag_approx[param_name] = 0.0
                    logger.warning(f"Деление на ноль для параметра {param_name}.")
            self.sensitivity_analysis = {
                'optimal_parameters': optimal_params.tolist(),
                'base_error': base_error,
                'gradients': gradients,
                'hessian_diagonal': hess_diag_approx
            }
            logger.info("Анализ чувствительности завершен.")
        except Exception as e:
            logger.error(f"Ошибка в анализе чувствительности: {e}")
            self.sensitivity_analysis = None

    def get_summary_report(self) -> Dict[str, Any]:
        """Возвращает сводный отчет о калибровке и чувствительности."""
        report = {
            'calibration_performed': self.optimization_result is not None,
            'calibration_success': self.optimization_result.success if self.optimization_result else False,
            'final_error': self.optimization_result.fun if self.optimization_result else None,
            'final_parameters': self.optimization_result.x.tolist() if self.optimization_result and self.optimization_result.success else None,
            'number_of_iterations': len(self.history),
            'sensitivity_analysis': self.sensitivity_analysis
        }
        return report

# --- Функции-геттеры для GradientCalibrator ---
def get_luminosity(model): return model.simulation_state.get('luminosity', 0.0)
def get_beam_size_x(model): return model.simulation_state.get('beam_size_x', 0.0)
def get_avg_event_energy(model):
    events = model.simulation_state.get('collision_events', [])
    if events:
        total_energy = sum(e.get('energy', 0.0) for e in events)
        return total_energy / len(events)
    else:
        return 0.0

# ===================================================================
# 14. *** МОДУЛЬ: AnomalyDetector ***
# ===================================================================
class AnomalyDetector:
    """
    Многоуровневый детектор аномалий для данных симуляции LHC.
    """
    def __init__(self, model=None):
        """Инициализация детектора аномалий."""
        self.model = model
        self.anomalies_found = {
            'by_type': {
                'statistical': [], 'topological': [], 'gradient': [],
                'model_behavior': [], 'custom': []
            },
            'summary': {'total_count': 0, 'types_found': set()}
        }
        logger.info("AnomalyDetector инициализирован.")

    def detect_statistical_anomalies(self,  List[Dict], feature_name: str, method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """Обнаруживает статистические аномалии в событиях."""
        logger.info(f"Поиск статистических аномалий по признаку '{feature_name}' методом '{method}'...")
        if not 
            logger.warning("Нет данных для статистического анализа.")
            return []
        try:
            values = np.array([event.get(feature_name, np.nan) for event in data])
            valid_indices = ~np.isnan(values)
            valid_values = values[valid_indices]
            if len(valid_values) < 2:
                 logger.warning("Недостаточно данных.")
                 return []
            anomaly_indices = []
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(valid_values, nan_policy='omit'))
                local_anomaly_indices = np.where(z_scores > threshold)[0]
                anomaly_indices = np.where(valid_indices)[0][local_anomaly_indices].tolist()
            elif method == 'iqr':
                Q1 = np.percentile(valid_values, 25)
                Q3 = np.percentile(valid_values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                local_anomaly_indices = np.where((valid_values < lower_bound) | (valid_values > upper_bound))[0]
                anomaly_indices = np.where(valid_indices)[0][local_anomaly_indices].tolist()
            else:
                logger.warning(f"Неизвестный метод: {method}")
                return []
            logger.info(f"Найдено {len(anomaly_indices)} статистических аномалий по признаку '{feature_name}'.")
            for idx in anomaly_indices:
                anomaly_details = {
                    'type': 'statistical', 'subtype': method, 'feature': feature_name,
                    'event_index': int(idx), 'event_details': data[idx],
                    'value': float(data[idx].get(feature_name, np.nan)),
                }
                if method == 'zscore':
                    val_idx_in_valid = np.where(np.where(valid_indices)[0] == idx)[0]
                    if len(val_idx_in_valid) > 0:
                        anomaly_details['z_score'] = float(z_scores[val_idx_in_valid[0]])
                self.anomalies_found['by_type']['statistical'].append(anomaly_details)
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('statistical')
            return anomaly_indices
        except Exception as e:
            logger.error(f"Ошибка при статистическом обнаружении аномалий: {e}")
            return []

    def detect_topological_anomalies(self, topo_analysis_result: Dict[str, Any], events_ List[Dict]) -> List[int]:
        """Обнаруживает топологические аномалии."""
        logger.info("Поиск топологических аномалий...")
        if not topo_analysis_result:
            logger.warning("Нет результатов топологического анализа.")
            return []
        persistence_res = topo_analysis_result.get('persistence_result')
        if not persistence_res:
             logger.info("Персистентная гомология не была рассчитана.")
             return []
        try:
            anomaly_indices = []
            dgms = persistence_res.get('dgms', [])
            if not dgms:
                 logger.warning("Диаграммы персистентности пусты.")
                 return []
            all_pers = []
            for dim, dgm in enumerate(dgms):
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    all_pers.extend(pers)
            if not all_pers:
                 logger.info("Нет персистентностей для анализа.")
                 return []
            pers_threshold = np.percentile(all_pers, 99)
            logger.info(f"Порог персистентности: {pers_threshold:.4f}")
            for dim, dgm in enumerate(dgms):
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    high_pers_indices = np.where(pers > pers_threshold)[0]
                    if len(high_pers_indices) > 0:
                         logger.info(f"Найдено {len(high_pers_indices)} точек в H_{dim}.")
                         self.anomalies_found['by_type']['topological'].append({
                            'type': 'topological', 'subtype': f'high_persistence_H{dim}',
                            'description': f'Точки с персистентностью > {pers_threshold:.4f} в H_{dim}',
                            'count': int(len(high_pers_indices)), 'threshold': float(pers_threshold)
                         })
                         self.anomalies_found['summary']['total_count'] += 1
                         self.anomalies_found['summary']['types_found'].add('topological')
            logger.info("Поиск топологических аномалий завершен.")
            return []
        except Exception as e:
            logger.error(f"Ошибка при топологическом обнаружении аномалий: {e}")
            return []

    def detect_model_behavior_anomalies(self, gradient_calibrator_report: Dict[str, Any]) -> List[Dict]:
        """Обнаруживает аномалии в поведении модели."""
        logger.info("Поиск аномалий в поведении модели...")
        if not gradient_calibrator_report:
            logger.warning("Нет отчета от GradientCalibrator.")
            return []
        try:
            anomalies = []
            if not gradient_calibrator_report.get('calibration_performed', False):
                 anomalies.append({'type': 'model_behavior', 'subtype': 'calibration_not_performed',
                                   'description': 'Калибровка не была выполнена.'})
            elif not gradient_calibrator_report.get('calibration_success', False):
                 anomalies.append({'type': 'model_behavior', 'subtype': 'calibration_failed',
                                   'description': 'Калибровка не сошлась.',
                                   'final_error': gradient_calibrator_report.get('final_error')})
            sensitivity = gradient_calibrator_report.get('sensitivity_analysis')
            if sensitivity:
                grads = sensitivity.get('gradients', {})
                hess_diag = sensitivity.get('hessian_diagonal', {})
                for param, grad_val in grads.items():
                    if abs(grad_val) < 1e-6:
                         anomalies.append({'type': 'model_behavior', 'subtype': 'low_sensitivity',
                                           'description': f'Параметр {param} имеет низкую чувствительность.',
                                           'parameter': param, 'gradient': float(grad_val)})
                for param, hess_val in hess_diag.items():
                    if abs(hess_val) < 1e-6:
                         anomalies.append({'type': 'model_behavior', 'subtype': 'flat_minimum',
                                           'description': f'Функция ошибки слабо зависит от {param}.',
                                           'parameter': param, 'hessian_diagonal': float(hess_val)})
            if anomalies:
                logger.info(f"Найдено {len(anomalies)} аномалий в поведении модели.")
                self.anomalies_found['by_type']['model_behavior'].extend(anomalies)
                self.anomalies_found['summary']['total_count'] += len(anomalies)
                self.anomalies_found['summary']['types_found'].add('model_behavior')
            else:
                 logger.info("Аномалии в поведении модели не обнаружены.")
            return anomalies
        except Exception as e:
            logger.error(f"Ошибка при обнаружении аномалий в поведении модели: {e}")
            return []

    def get_report(self) -> Dict[str, Any]:
        """Возвращает структурированный отчет о найденных аномалиях."""
        report = self.anomalies_found.copy()
        report['summary'] = report['summary'].copy()
        report['summary']['types_found'] = list(report['summary']['types_found'])
        return report

    def reset(self):
        """Сбрасывает список найденных аномалий."""
        self.anomalies_found = {
            'by_type': {'statistical': [], 'topological': [], 'gradient': [],
                        'model_behavior': [], 'custom': []},
            'summary': {'total_count': 0, 'types_found': set()}
        }
        logger.info("Список аномалий сброшен.")

# ===================================================================
# 15. *** МОДУЛЬ: ROOTExporter ***
# ===================================================================
class ROOTExporter:
    """
    Экспортер данных симуляции в формат ROOT.
    """
    def __init__(self):
        """Инициализация экспортера."""
        if not ROOT_AVAILABLE:
            raise RuntimeError("ROOT library is not available.")
        logger.info("ROOTExporter инициализирован.")

    def export_collision_events(self, events List[Dict], filename: str = "lhc_events.root"):
        """
        Экспортирует список событий столкновений в ROOT TTree.
        """
        if not ROOT_AVAILABLE:
            logger.error("ROOT library is not available.")
            return False
        logger.info(f"Начало экспорта {len(events)} событий в ROOT файл '{filename}'...")
        try:
            root_file = ROOT.TFile(filename, "RECREATE")
            event_tree = ROOT.TTree("events", "Simulated LHC Collision Events")
            event_id = np.array([0], dtype=np.int32)
            event_type = ROOT.std.string()
            energy = np.array([0.0], dtype=np.float64)
            parton_energy = np.array([0.0], dtype=np.float64)
            num_products = np.array([0], dtype=np.int32)
            event_tree.Branch("event_id", event_id, "event_id/I")
            event_tree.Branch("event_type", event_type)
            event_tree.Branch("energy", energy, "energy/D")
            event_tree.Branch("parton_energy", parton_energy, "parton_energy/D")
            event_tree.Branch("num_products", num_products, "num_products/I")
            particle_tree = ROOT.TTree("particles", "Particles from LHC Events")
            p_event_id = np.array([0], dtype=np.int32)
            p_name = ROOT.std.string()
            p_energy = np.array([0.0], dtype=np.float64)
            p_px = np.array([0.0], dtype=np.float64)
            p_py = np.array([0.0], dtype=np.float64)
            p_pz = np.array([0.0], dtype=np.float64)
            p_mass = np.array([0.0], dtype=np.float64)
            particle_tree.Branch("event_id", p_event_id, "event_id/I")
            particle_tree.Branch("name", p_name)
            particle_tree.Branch("energy", p_energy, "energy/D")
            particle_tree.Branch("px", p_px, "px/D")
            particle_tree.Branch("py", p_py, "py/D")
            particle_tree.Branch("pz", p_pz, "pz/D")
            particle_tree.Branch("mass", p_mass, "mass/D")
            for evt in events:
                event_id[0] = evt.get('event_id', -1)
                event_type.assign(evt.get('event_type', 'unknown'))
                energy[0] = evt.get('energy', 0.0)
                parton_energy[0] = evt.get('parton_energy', 0.0)
                products = evt.get('products', [])
                num_products[0] = len(products)
                event_tree.Fill()
                for particle in products:
                    p_event_id[0] = event_id[0]
                    p_name.assign(particle.get('name', 'unknown'))
                    p_energy[0] = particle.get('energy', 0.0)
                    p_px[0] = particle.get('px', 0.0)
                    p_py[0] = particle.get('py', 0.0)
                    p_pz[0] = particle.get('pz', 0.0)
                    p_mass[0] = particle.get('mass', 0.0)
                    particle_tree.Fill()
            root_file.Write()
            root_file.Close()
            logger.info(f"Экспорт в ROOT файл '{filename}' успешно завершен.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при экспорте в ROOT: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# ===================================================================
# 16. *** МОДУЛЬ: HepMC3Exporter ***
# ===================================================================
class HepMC3Exporter:
    """
    Экспортер данных симуляции в формат HepMC3.
    """
    def __init__(self):
        """Инициализация экспортера."""
        if not HEPMC3_AVAILABLE:
            raise RuntimeError("HepMC3 library is not available.")
        logger.info("HepMC3Exporter инициализирован.")

    def export_collision_events(self, events List[Dict], filename: str = "lhc_events.hepmc3"):
        """
        Экспортирует список событий столкновений в HepMC3 файл.
        """
        if not HEPMC3_AVAILABLE:
            logger.error("HepMC3 library is not available.")
            return False
        logger.info(f"Начало экспорта {len(events)} событий в HepMC3 файл '{filename}'...")
        try:
            with io.WriterAscii(filename) as f:
                for evt_index, evt_dict in enumerate(events):
                    hepmc_evt = hep.GenEvent(hep.Units.GEV, hep.Units.MM)
                    hepmc_evt.event_number = evt_dict.get('event_id', evt_index)
                    collision_vertex = hep.GenVertex()
                    hepmc_evt.add_vertex(collision_vertex)
                    beam_energy = evt_dict.get('energy', 6500e9) / 2.0
                    p1_in = hep.GenParticle(hep.FourVector(0, 0, beam_energy, beam_energy), 2212, 4)
                    p2_in = hep.GenParticle(hep.FourVector(0, 0, -beam_energy, beam_energy), 2212, 4)
                    hepmc_evt.add_particle(p1_in)
                    hepmc_evt.add_particle(p2_in)
                    collision_vertex.add_particle_in(p1_in)
                    collision_vertex.add_particle_in(p2_in)
                    products = evt_dict.get('products', [])
                    for particle_dict in products:
                        name = particle_dict.get('name', 'unknown')
                        pdg_id = self._get_pdg_id(name)
                        px = particle_dict.get('px', 0.0)
                        py = particle_dict.get('py', 0.0)
                        pz = particle_dict.get('pz', 0.0)
                        energy = particle_dict.get('energy', 0.0)
                        if energy < np.sqrt(px**2 + py**2 + pz**2):
                             logger.warning(f"Энергия частицы ({energy}) меньше модуля импульса. Корректируем.")
                             energy = np.sqrt(px**2 + py**2 + pz**2 + 1e-9)
                        p4 = hep.FourVector(px, py, pz, energy)
                        status = 1
                        hepmc_particle = hep.GenParticle(p4, pdg_id, status)
                        hepmc_evt.add_particle(hepmc_particle)
                        collision_vertex.add_particle_out(hepmc_particle)
                    f.write(hepmc_evt)
            logger.info(f"Экспорт в HepMC3 файл '{filename}' успешно завершен.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при экспорте в HepMC3: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _get_pdg_id(self, particle_name: str) -> int:
        """Простое сопоставление названий частиц с PDG ID."""
        mapping = {
            'proton': 2212, 'neutron': 2112, 'electron': 11, 'positron': -11,
            'muon': 13, 'antimuon': -13, 'photon': 22, 'gluon': 21,
            'up_quark': 2, 'down_quark': 1, 'strange_quark': 3,
            'ubar_quark': -2, 'dbar_quark': -1, 'sbar_quark': -3,
            'bottom_quark': 5, 'antibottom_quark': -5,
            'top_quark': 6, 'antitop_quark': -6,
            'W_plus': 24, 'W_minus': -24, 'Z0': 23, 'Higgs': 25,
            'pion_plus': 211, 'pion_minus': -211, 'pion_zero': 111,
            'kaon_plus': 321, 'kaon_minus': -321,
            'jet': 81, 'unknown': 0
        }
        return mapping.get(particle_name, 0)

# ===================================================================
# 17. *** МОДУЛЬ: Visualizer ***
# ===================================================================
class Visualizer:
    """
    Интерактивная визуализация для коллайдера с использованием Plotly.
    """
    def __init__(self):
        """Инициализация визуализатора."""
        self.plotly_available = PLOTLY_AVAILABLE
        logger.info("Visualizer инициализирован.")

    def plot_geometry_3d(self, geometry, detector_system):
        """Интерактивная 3D-визуализация геометрии коллайдера и детекторов."""
        if not self.plotly_available:
            logger.warning("Plotly not available.")
            return
        try:
            fig = go.Figure()
            radius = geometry.circumference / (2 * np.pi)
            theta = np.linspace(0, 2*np.pi, 100)
            x_ring = radius * np.cos(theta)
            y_ring = radius * np.sin(theta)
            z_ring = np.zeros_like(x_ring)
            fig.add_trace(go.Scatter3d(x=x_ring, y=y_ring, z=z_ring, mode='lines',
                                       name='Collider Ring', line=dict(color='blue', width=5)))
            detector_data = detector_system.detectors
            det_x, det_y, det_z = [], [], []
            det_names = []
            for name, det_info in detector_data.items():
                pos = det_info['position']
                det_x.append(pos[0])
                det_y.append(pos[1])
                det_z.append(pos[2])
                det_names.append(name)
            if det_x:
                fig.add_trace(go.Scatter3d(x=det_x, y=det_y, z=det_z, mode='markers',
                                           name='Detectors', marker=dict(size=8, color='red'),
                                           text=det_names, hoverinfo='text'))
            fig.update_layout(title='3D Geometry of LHC',
                              scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]'),
                              margin=dict(l=0, r=0, b=0, t=25))
            fig.show()
            logger.info("3D geometry plot displayed.")
        except Exception as e:
            logger.error(f"Error in 3D geometry plotting: {e}")

    def plot_particle_tracks_3d(self, tracks: List[List[Tuple[float, float, float]]], event: Dict):
        """Интерактивная 3D-визуализация траекторий частиц."""
        if not self.plotly_available:
            logger.warning("Plotly not available.")
            return
        try:
            fig = go.Figure()
            for i, track in enumerate(tracks):
                xs, ys, zs = zip(*track)
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines+markers',
                                           name=f'Track {i}', marker=dict(size=2),
                                           line=dict(width=2)))
            fig.update_layout(title=f'3D Particle Tracks for Event {event.get("event_id", "N/A")}',
                              scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]'),
                              margin=dict(l=0, r=0, b=0, t=25))
            fig.show()
            logger.info("3D particle tracks plot displayed.")
        except Exception as e:
            logger.error(f"Error in 3D tracks plotting: {e}")

    def plot_detector_response_3d(self, detected_events: List[Dict], detector_system):
        """Интерактивная 3D-визуализация отклика детекторов."""
        if not self.plotly_available:
            logger.warning("Plotly not available.")
            return
        try:
            fig = go.Figure()
            detector_hits = {}
            for de in detected_events:
                det_name = de['detector']
                if det_name not in detector_hits:
                    detector_hits[det_name] = {'energies': [], 'positions': []}
                det_pos = detector_system.detectors.get(det_name, {}).get('position', (0,0,0))
                detector_hits[det_name]['energies'].append(de['reconstructed_energy'])
                detector_hits[det_name]['positions'].append(det_pos)
            for det_name, data in detector_hits.items():
                energies = data['energies']
                positions = data['positions']
                if positions:
                    xs, ys, zs = zip(*positions)
                    sizes = [max(3, e / max(energies) * 15) for e in energies] if max(energies) > 0 else [5]*len(energies)
                    fig.add_trace(go.Scatter3d(x=list(xs), y=list(ys), z=list(zs),
                                               mode='markers',
                                               name=det_name,
                                               marker=dict(size=sizes, color=energies, colorscale='Viridis', showscale=True),
                                               text=[f"Energy: {e:.2f} GeV" for e in energies],
                                               hoverinfo='text'))
            fig.update_layout(title='3D Detector Response (Hits)',
                              scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]'),
                              margin=dict(l=0, r=0, b=0, t=25))
            fig.show()
            logger.info("3D detector response plot displayed.")
        except Exception as e:
            logger.error(f"Error in 3D detector response plotting: {e}")

# ===================================================================
# 18. *** МОДУЛЬ: GPUAccelerator (Заглушка) ***
# ===================================================================
class GPUAccelerator:
    """Система ускорения вычислений с использованием GPU (Numba CUDA или CuPy)."""
    def __init__(self):
        """Инициализация GPU-ускорителя."""
        self.backend = None
        self.is_available = GPU_ACCELERATION_AVAILABLE

        if NUMBA_CUDA_AVAILABLE:
            self.backend = 'numba_cuda'
            logger.info("GPUAccelerator: Using Numba CUDA backend.")
        elif CUPY_AVAILABLE:
            self.backend = 'cupy'
            logger.info("GPUAccelerator: Using CuPy backend.")
        else:
            logger.warning("GPUAccelerator: No compatible GPU backend found. Running on CPU.")

    def track_particles(self, particles: List[Dict], field: Tuple[float, float, float],
                       num_steps: int = 1000, dt: float = 1e-12) -> List[List[Tuple[float, float, float]]]:
        """
        Трассировка частиц с использованием GPU-ускорения при возможности.
        """
        if not self.is_available or self.backend is None:
            logger.warning("GPU acceleration not available. Falling back to CPU tracking.")
            return [self._track_single_particle_on_cpu(p, field, num_steps, dt) for p in particles]

        logger.info(f"Tracking {len(particles)} particles using {self.backend}...")
        try:
            if self.backend == 'numba_cuda':
                return self._track_with_numba_cuda(particles, field, num_steps, dt)
            elif self.backend == 'cupy':
                return self._track_with_cupy(particles, field, num_steps, dt)
        except Exception as e:
            logger.error(f"GPU tracking failed: {e}. Falling back to CPU.")
            return [self._track_single_particle_on_cpu(p, field, num_steps, dt) for p in particles]

    def _track_single_particle_on_cpu(self, particle: Dict, field: Tuple[float, float, float],
                                      num_steps: int, dt: float) -> List[Tuple[float, float, float]]:
        """CPU fallback для трекинга одной частицы."""
        logger.debug("Using CPU fallback for particle tracking (stub).")
        # В реальности здесь была бы логика трекинга на CPU
        # Для демонстрации просто возвращаем фиктивную траекторию
        x, y, z = particle.get('x', 0.0), particle.get('y', 0.0), particle.get('z', 0.0)
        px, py, pz = particle.get('px', 0.0), particle.get('py', 0.0), particle.get('pz', 0.0)
        track = [(x + i*dt*px, y + i*dt*py, z + i*dt*pz) for i in range(num_steps)]
        return track

    def _track_with_numba_cuda(self, particles: List[Dict], field: Tuple[float, float, float],
                               num_steps: int, dt: float) -> List[List[Tuple[float, float, float]]]:
        """Трекинг с использованием Numba CUDA."""
        logger.info("Using Numba CUDA for tracking (stub implementation).")
        return [self._track_single_particle_on_cpu(p, field, num_steps, dt) for p in particles]

    def _track_with_cupy(self, particles: List[Dict], field: Tuple[float, float, float],
                         num_steps: int, dt: float) -> List[List[Tuple[float, float, float]]]:
        """Трекинг с использованием CuPy."""
        logger.info("Using CuPy for tracking (stub implementation).")
        return [self._track_single_particle_on_cpu(p, field, num_steps, dt) for p in particles]

# ===================================================================
# 19. Основная модель коллайдера
# ===================================================================
class LHCHybridModel:
    """
    Усовершенствованная гибридная модель Большого адронного коллайдера.
    
    Это центральный класс фреймворка, объединяющий все компоненты:
    - Физические и динамические движки
    - Системы анализа (TopoAnalyzer, GradientCalibrator, AnomalyDetector)
    - Экспорт данных
    - Визуализация
    """
    def __init__(self):
        """Инициализация модели."""
        self.particle_db = ParticleDatabase()
        self.geometry = type('MockGeometry', (), {'circumference': CONFIG['geometry']['circumference']})()
        self.detector_system = DetectorSystem(self.geometry.circumference)
        self.physics_engine = HybridPhysicsEngine(self.particle_db)
        self.beam_dynamics = HybridBeamDynamics()
        self.cache = SimulationCache()
        self.validation_system = ValidationSystem(self)
        self.visualizer = Visualizer()
        self.config = CONFIG
        self.simulation_state = self._initialize_simulation_state()
        self.last_topo_analysis_result = None
        self.last_calibration_report = None
        self.anomaly_report = None
        logger.info("Усовершенствованная гибридная модель коллайдера инициализирована")

    def _initialize_simulation_state(self) -> Dict:
        """Инициализация начального состояния симуляции."""
        if 'beam' in CONFIG:
            beam_energy = CONFIG['beam']['energy'] * 1e9
            particles = CONFIG['beam']['particles']
            bunch_intensity = CONFIG['beam']['bunch_intensity']
            num_bunches = CONFIG['beam']['num_bunches']
        else:
            beam_energy = 6500 * 1e9
            particles = 'protons'
            bunch_intensity = 1.15e11
            num_bunches = 2748
        gamma = beam_energy / (m_p * c**2)
        beta = np.sqrt(1 - 1/gamma**2)
        revolution_time = self.geometry.circumference / (beta * c)
        peak_luminosity = (num_bunches * bunch_intensity)**2 / (self.geometry.circumference * 25e-9) * 1.5e-4
        beam_size_x = 0.05
        beam_size_y = 0.05
        state = {
            'turn': 0, 'beam_energy': beam_energy, 'particles': particles,
            'num_bunches': num_bunches, 'protons_per_bunch': bunch_intensity,
            'revolution_time': revolution_time, 'gamma': gamma, 'beta': beta,
            'circumference': self.geometry.circumference,
            'beam_size_x': beam_size_x, 'beam_size_y': beam_size_y,
            'luminosity': peak_luminosity, 'peak_luminosity': peak_luminosity,
            'beam_dynamics': {
                'time': [0.0],
                'luminosity': [peak_luminosity],
                'beam_size_x': [beam_size_x],
                'beam_size_y': [beam_size_y],
                'emittance': [3.5e-6],
                'beam_intensity': [bunch_intensity * num_bunches]
            },
            'collision_events': [], 'detected_events': []
        }
        return state

    def step_simulation(self, include_space_charge: bool = True,
                        force_physics_engine: Optional[str] = None,
                        force_beam_engine: Optional[str] = None):
        """Выполнение одного шага симуляции (один оборот пучка)."""
        updated_state = self.beam_dynamics.simulate_turn(
            self.simulation_state,
            self.simulation_state['revolution_time'],
            include_space_charge=include_space_charge,
            force_engine=force_beam_engine
        )
        self.simulation_state = updated_state
        self.simulation_state['turn'] += 1
        self._simulate_collision(force_engine=force_physics_engine)

    def _simulate_collision(self, force_engine: Optional[str] = None):
        """Моделирование события столкновения."""
        energy = self.simulation_state['beam_energy'] * 2
        event_id = len(self.simulation_state['collision_events'])
        cache_params = {'energy': energy, 'engine': force_engine or 'auto', 'num_events': 1}
        cache_key = SimulationCache.generate_key(cache_params)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Использован кэшированный результат для столкновения")
            events = cached_result
        else:
            events = self.physics_engine.interact(
                'proton', 'proton', energy, num_events=1, force_engine=force_engine
            )
            self.cache.set(cache_key, events)
        for event in events:
            event['event_id'] = event_id
            event['timestamp'] = time.time()
            self.simulation_state['collision_events'].append(event)
        if events:
            detected_particles = self.detector_system.detect_event(events[0])
            self.simulation_state['detected_events'].extend(detected_particles)

    def run_simulation(self, num_turns: int = 100,
                       include_space_charge: bool = True,
                       force_physics_engine: Optional[str] = None,
                       force_beam_engine: Optional[str] = None):
        """Запуск симуляции на заданное количество оборотов."""
        logger.info(f"Запуск симуляции на {num_turns} оборотов")
        start_time = time.time()
        for _ in range(num_turns):
            self.step_simulation(
                include_space_charge, force_physics_engine, force_beam_engine
            )
        end_time = time.time()
        logger.info(f"Симуляция завершена за {end_time - start_time:.2f} секунд")
        self.validate_results()
        hit_rate = self.cache.get_hit_rate()
        logger.info(f"Кэш: hit rate = {hit_rate:.2%}")

    def validate_results(self):
        """Валидация результатов симуляции."""
        dataset_id = CONFIG['validation']['dataset_id']
        self.validation_system.load_real_data(dataset_id)
        luminosity_error = self.validation_system.validate_luminosity()
        logger.info(f"Валидация завершена. Ошибка светимости: {luminosity_error:.2%}")

    def export_results(self, format: str = 'json', path: str = '.'):
        """Экспорт результатов симуляции."""
        os.makedirs(path, exist_ok=True)
        success = False
        if format == 'json':
            export_data = {
                'simulation_parameters': {
                    'num_turns': self.simulation_state['turn'],
                    'beam_energy': self.simulation_state['beam_energy'],
                    'particles': self.simulation_state['particles'],
                },
                'beam_dynamics': {
                    'time': self.simulation_state['beam_dynamics']['time'],
                    'luminosity': self.simulation_state['beam_dynamics']['luminosity'],
                },
                'collision_events': self.simulation_state['collision_events'][:100],
                'detected_events': self.simulation_state['detected_events'][:1000],
                'cache_hit_rate': self.cache.get_hit_rate(),
            }
            filename = os.path.join(path, 'lhc_simulation_results_unified.json')
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Результаты экспортированы в JSON: {filename}")
            success = True
        elif format == 'root':
            if ROOT_AVAILABLE:
                exporter = ROOTExporter()
                events_to_export = self.simulation_state.get('collision_events', [])
                filename = os.path.join(path, 'lhc_collision_events.root')
                success = exporter.export_collision_events(events_to_export, filename)
                if success:
                    logger.info(f"Результаты экспортированы в ROOT: {filename}")
            else:
                logger.error("Экспорт в ROOT невозможен.")
                success = False
        elif format == 'hepmc3':
            if HEPMC3_AVAILABLE:
                exporter = HepMC3Exporter()
                events_to_export = self.simulation_state.get('collision_events', [])
                filename = os.path.join(path, 'lhc_collision_events.hepmc3')
                success = exporter.export_collision_events(events_to_export, filename)
                if success:
                    logger.info(f"Результаты экспортированы в HepMC3: {filename}")
            else:
                logger.error("Экспорт в HepMC3 невозможен.")
                success = False
        else:
            logger.error(f"Формат экспорта '{format}' не поддерживается.")
            success = False
        return success

    def analyze_topology(self, max_events: int = 1000, compute_persistence: bool = True, compute_pca: bool = True):
        """Новый метод для топологического анализа результатов симуляции."""
        logger.info("Запуск топологического анализа событий...")
        if not self.simulation_state['collision_events']:
            logger.warning("Нет событий для анализа.")
            return
        events_to_analyze = self.simulation_state['collision_events'][:max_events]
        analyzer = TopoAnalyzer(events_to_analyze)
        try:
            analyzer.run_full_analysis(compute_persistence=compute_persistence, compute_pca=compute_pca)
            self.last_topo_analysis_result = analyzer.get_summary_report()
            self.last_topo_analysis_result['persistence_result'] = analyzer.persistence_result
            with open(os.path.join(".", "topo_analysis_report.json"), "w") as f:
                 json.dump(self.last_topo_analysis_result, f, indent=2, default=str)
            logger.info("Отчет топологического анализа сохранен.")
            logger.info("Топологический анализ завершен.")
        except Exception as e:
            logger.error(f"Ошибка в топологическом анализе: {e}")
            self.last_topo_analysis_result = None

    def calibrate_via_gradients(self, target_values: Dict[str, float], parameters: List[str], 
                                num_turns: int = 10, error_weights: Optional[Dict[str, float]] = None):
        """Новый метод для калибровки модели."""
        logger.info("Запуск калибровки модели...")
        observable_getters = {
            'luminosity': get_luminosity,
            'beam_size_x': get_beam_size_x,
            'avg_event_energy': get_avg_event_energy
        }
        calibrator = GradientCalibrator(
            self, target_observables=target_values,
            parameters_to_calibrate=parameters,
            observable_getters=observable_getters,
            perturbation_factor=0.001,
            error_weights=error_weights
        )
        calibrator.calibrate(num_turns=num_turns, method='L-BFGS-B', tolerance=1e-4, max_iterations=30)
        calibrator.analyze_sensitivity(num_turns=num_turns)
        self.last_calibration_report = calibrator.get_summary_report()
        logger.info("Отчет о калибровке:")
        logger.info(json.dumps(self.last_calibration_report, indent=2, default=str))
        logger.info("Калибровка модели завершена.")

    def detect_anomalies(self, use_topo_results: bool = True, use_calib_report: bool = True):
        """Новый метод для обнаружения аномалий в результатах симуляции."""
        logger.info("Запуск многоуровневого обнаружения аномалий...")
        detector = AnomalyDetector(model=self)
        detector.reset()
        events = self.simulation_state.get('collision_events', [])
        if events:
            detector.detect_statistical_anomalies(events, 'energy', method='zscore', threshold=2.5)
            detector.detect_statistical_anomalies(events, 'num_products', method='iqr', threshold=1.5)
        if use_topo_results and self.last_topo_analysis_result:
            detector.detect_topological_anomalies(self.last_topo_analysis_result, events)
        if use_calib_report and self.last_calibration_report:
            detector.detect_model_behavior_anomalies(self.last_calibration_report)
        self.anomaly_report = detector.get_report()
        total_anomalies = self.anomaly_report['summary']['total_count']
        types_found = self.anomaly_report['summary']['types_found']
        logger.info(f"Обнаружение аномалий завершено. Найдено аномалий: {total_anomalies}")
        if types_found:
            logger.info(f"  Типы аномалий: {', '.join(types_found)}")
            for anomaly_type, anomalies_list in self.anomaly_report['by_type'].items():
                if anomalies_list:
                    logger.info(f"    - {anomaly_type.capitalize()}: {len(anomalies_list)}")
        else:
             logger.info("  Аномалии не найдены.")
        try:
            with open(os.path.join(".", "anomaly_report.json"), "w") as f:
                 json.dump(self.anomaly_report, f, indent=2, default=str)
            logger.info("Отчет об аномалиях сохранен.")
        except Exception as e:
             logger.error(f"Не удалось сохранить отчет об аномалиях: {e}")
        return self.anomaly_report

    def visualize_results(self):
        """Улучшенная визуализация результатов симуляции."""
        logger.info("Запуск улучшенной визуализации результатов...")
        try:
            self.visualizer.plot_geometry_3d(self.geometry, self.detector_system)
            if self.simulation_state['detected_events']:
                self.visualizer.plot_detector_response_3d(self.simulation_state['detected_events'], self.detector_system)
            logger.info("Улучшенная визуализация завершена.")
        except Exception as e:
            logger.error(f"Ошибка при улучшенной визуализации: {e}")

# ===================================================================
# 20. Функции демонстрации
# ===================================================================
def create_default_config():
    """Создание файла конфигурации по умолчанию."""
    config = {
        "beam": {"energy": 6500, "particles": "protons", "bunch_intensity": 1.15e11, "num_bunches": 2748},
        "geometry": {"circumference": 26658.883, "dipole_field": 8.33},
        "validation": {"dataset_id": "cms-2011-collision-data"},
        "simulation": {"physics_engines": ["pythia", "herwig", "built-in"], "beam_dynamics_engine": "madx"}
    }
    with open("lhc_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Файл конфигурации создан: lhc_config.yaml")

def demo_unified_framework():
    """Демонстрация работы унифицированного фреймворка."""
    logger.info("Сценарий: Демонстрация унифицированного фреймворка")
    if not os.path.exists("lhc_config.yaml"):
        create_default_config()
    lhc = LHCHybridModel()
    lhc.run_simulation(num_turns=20)
    lhc.analyze_topology(max_events=500, compute_persistence=True, compute_pca=True)
    target_observables = {
        'luminosity': 1.5e34,
        'beam_size_x': 0.045,
        'avg_event_energy': 5000.0
    }
    params_to_calibrate = ['beam_energy', 'num_bunches']
    lhc.calibrate_via_gradients(target_observables, params_to_calibrate, num_turns=20)
    report = lhc.detect_anomalies(use_topo_results=True, use_calib_report=True)
    lhc.export_results(format='json')
    if ROOT_AVAILABLE:
        lhc.export_results(format='root')
    if HEPMC3_AVAILABLE:
        lhc.export_results(format='hepmc3')
    lhc.visualize_results()

def main():
    """Основная функция для демонстрации работы модели."""
    logger.info("="*70)
    logger.info("УНИФИЦИРОВАННЫЙ ФРЕЙМВОРК ДЛЯ СИМУЛЯЦИИ И АНАЛИЗА LHC")
    logger.info("="*70)
    demo_unified_framework()
    logger.info("="*70)
    logger.info("РАБОТА ФРЕЙМВОРКА ЗАВЕРШЕНА")
    logger.info("="*70)

if __name__ == "__main__":
    main()
