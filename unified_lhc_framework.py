import os
import time
import json
import logging
import random
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# Подавление предупреждений для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# 1. Настройка логирования
# ===================================================================
"""Настройка логирования для отслеживания процесса выполнения.
Создает лог-файл и выводит сообщения в консоль."""
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("unified_lhc_framework.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("Unified_LHC_Framework")

# ===================================================================
# 2. Загрузка конфигурации
# ===================================================================
"""Загрузка параметров симуляции из файла 'lhc_config.yaml'.
Если файл не найден, используются значения по умолчанию."""
try:
    with open("lhc_config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    logger.info("Конфигурация загружена из lhc_config.yaml")
except FileNotFoundError:
    CONFIG = {
        'beam': {
            'energy': 6500,  # в ГэВ
            'particles': 'protons',
            'bunch_intensity': 1.15e11,
            'num_bunches': 2748
        },
        'geometry': {
            'circumference': 26659,  # в метрах
            'bending_radius': 2800
        },
        'simulation': {
            'num_turns': 1000,
            'revolution_time': 88.9e-6  # в секундах
        },
        'validation': {
            'dataset_id': 'CMS_OpenData_2018'
        }
    }
    logger.warning("Файл конфигурации не найден. Используются значения по умолчанию.")

# ===================================================================
# 3. Проверка зависимостей
# ===================================================================
try:
    import ROOT
    ROOT_AVAILABLE = True
    logger.info("ROOT библиотека доступна.")
except ImportError:
    ROOT_AVAILABLE = False
    logger.warning("ROOT библиотека недоступна.")

try:
    import gudhi
    GUDHI_AVAILABLE = True
    USE_GUDHI = True
    USE_RIPSER = False
    logger.info("GUDHI библиотека доступна.")
except ImportError:
    GUDHI_AVAILABLE = False
    USE_GUDHI = False
    logger.warning("GUDHI библиотека недоступна.")

try:
    import ripser
    RIPSER_AVAILABLE = True
    if not USE_GUDHI:
        USE_RIPSER = True
    logger.info("Ripser библиотека доступна.")
except ImportError:
    RIPSER_AVAILABLE = False
    logger.warning("Ripser библиотека недоступна.")

try:
    import persim
    PERSISTENCE_AVAILABLE = GUDHI_AVAILABLE or RIPSER_AVAILABLE
    logger.info("Библиотеки для персистентной гомологии доступны.")
except ImportError:
    PERSISTENCE_AVAILABLE = False
    logger.warning("Библиотеки для персистентной гомологии недоступны.")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    logger.info("Plotly библиотека доступна.")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly библиотека недоступна.")

try:
    import numba
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
    logger.info("Numba CUDA доступен.")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    logger.warning("Numba CUDA недоступен.")

try:
    import cupy
    CUPY_AVAILABLE = True
    logger.info("CuPy доступен.")
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy недоступен.")

GPU_ACCELERATION_AVAILABLE = NUMBA_CUDA_AVAILABLE or CUPY_AVAILABLE

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
            'u_quark': 0.35, 
            'ubar_quark': 0.15,
            'd_quark': 0.25, 
            'dbar_quark': 0.10,
            's_quark': 0.05, 
            'gluon': 0.10
        }
        self.parton_types = list(self.proton_parton_weights.keys())
        self.parton_weights = list(self.proton_parton_weights.values())
    
    def _load_particles(self) -> Dict[str, Particle]:
        """Загружает базовые частицы."""
        particles = {
            'proton': Particle('proton', 0.938, 1, 1, float('inf'), 'baryon', 'p'),
            'electron': Particle('electron', 0.000511, -1, 1, float('inf'), 'lepton', 'e-'),
            'muon': Particle('muon', 0.105, -1, 1, 2.2e-6, 'lepton', 'μ-'),
            'antimuon': Particle('antimuon', 0.105, 1, 1, 2.2e-6, 'lepton', 'μ+'),
            'photon': Particle('photon', 0.0, 0, 1, float('inf'), 'boson', 'γ'),
            'u_quark': Particle('u_quark', 0.0023, 2/3, 1, float('inf'), 'quark', 'u'),
            'ubar_quark': Particle('ubar_quark', 0.0023, -2/3, 1, float('inf'), 'quark', 'ū'),
            'd_quark': Particle('d_quark', 0.0048, -1/3, 1, float('inf'), 'quark', 'd'),
            'dbar_quark': Particle('dbar_quark', 0.0048, 1/3, 1, float('inf'), 'quark', 'đ'),
            's_quark': Particle('s_quark', 0.095, -1/3, 1, float('inf'), 'quark', 's'),
            'gluon': Particle('gluon', 0.0, 0, 1, float('inf'), 'boson', 'g'),
            'top_quark': Particle('top_quark', 173.1, 2/3, 1, 5.0e-25, 'quark', 't'),
            'antitop_quark': Particle('antitop_quark', 173.1, -2/3, 1, 5.0e-25, 'quark', 't̄'),
            'W_plus': Particle('W_plus', 80.4, 1, 1, 3.0e-25, 'boson', 'W+'),
            'W_minus': Particle('W_minus', 80.4, -1, 1, 3.0e-25, 'boson', 'W-'),
            'Z0': Particle('Z0', 91.2, 0, 1, 3.0e-25, 'boson', 'Z'),
            'Higgs': Particle('Higgs', 125.1, 0, 0, 1.6e-22, 'boson', 'H'),
            'pion_plus': Particle('pion_plus', 0.140, 1, 0, 2.6e-8, 'meson', 'π+'),
            'pion_minus': Particle('pion_minus', 0.140, -1, 0, 2.6e-8, 'meson', 'π-'),
            'pion_zero': Particle('pion_zero', 0.135, 0, 0, 8.5e-17, 'meson', 'π0'),
            'kaon_plus': Particle('kaon_plus', 0.494, 1, 0, 1.2e-8, 'meson', 'K+'),
            'kaon_minus': Particle('kaon_minus', 0.494, -1, 0, 1.2e-8, 'meson', 'K-'),
            'jet': Particle('jet', 0.0, 0, 0, float('inf'), 'composite', 'jet'),
            'unknown': Particle('unknown', 0.0, 0, 0, float('inf'), 'unknown', 'X')
        }
        return particles
    
    def sample_parton(self) -> str:
        """Случайно выбирает партон согласно весам."""
        return random.choices(self.parton_types, weights=self.parton_weights, k=1)[0]

# Константы для физических вычислений
SPEED_OF_LIGHT = 299792458  # м/с
PROTON_MASS = 0.938  # ГэВ/c²
SMALL_EPSILON = 1e-12  # Для избежания деления на ноль
MAX_X_SAMPLE_ITERATIONS = 100  # Максимальное количество итераций для _sample_x

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
    
    @staticmethod
    def generate_key(params: Dict) -> str:
        """Генерирует уникальный ключ для кэша на основе параметров."""
        sorted_params = sorted(params.items())
        return str(hash(str(sorted_params)))
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша."""
        self.access_count += 1
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Сохраняет значение в кэш."""
        if len(self.cache) >= self.max_size:
            # Простая стратегия удаления: удаляем самую старую запись
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def get_hit_rate(self) -> float:
        """Возвращает долю попаданий в кэш."""
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0

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
    """Абстрактный интерфейс для движков динамики пучка"""
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
        iterations = 0
        
        # Добавлена защита от бесконечного цикла
        while x >= 1.0 and iterations < MAX_X_SAMPLE_ITERATIONS:
            u = random.random()
            x = -np.log(1 - u * (1 - np.exp(-b))) / b
            iterations += 1
        
        # Если превышено максимальное количество итераций, возвращаем безопасное значение
        if iterations >= MAX_X_SAMPLE_ITERATIONS:
            logger.warning(f"Превышено максимальное количество итераций ({MAX_X_SAMPLE_ITERATIONS}) в _sample_x. Возвращаем безопасное значение.")
            return 0.5  # Возвращаем среднее значение для x
        
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
            
            hadrons.append({
                'name': hadron_type,
                'energy': energy,
                'px': random.uniform(-energy/2, energy/2),
                'py': random.uniform(-energy/2, energy/2),
                'pz': random.uniform(-energy/2, energy/2),
                'mass': self.particle_db.particles[hadron_type].mass
            })
        
        return hadrons
    
    def _generate_particles_from_partons(self, parton1_type: str, parton2_type: str, total_energy: float) -> List[Dict]:
        """Генерирует продукты столкновения на основе партонов."""
        process_type = "QCD"
        products = []
        
        # Примеры процессов в зависимости от типов партонов
        if 'quark' in parton1_type and 'quark' in parton2_type:
            if 'u' in parton1_type and 'd' in parton2_type:
                process_type = "W_production"
                # W-бозон распадается на лептоны
                if random.random() < 0.33:
                    products.append({'name': 'W_plus', 'energy': total_energy*0.8})
                    # Распад W+ -> e+ + nu_e
                    products.append({'name': 'positron', 'energy': total_energy*0.4})
                    products.append({'name': 'neutrino_e', 'energy': total_energy*0.4})
                else:
                    # Распад на адроны
                    num_hadrons = random.randint(2, 5)
                    products.extend(self._fragment_hadron(total_energy*0.8, num_hadrons))
        
        elif 'quark' in parton1_type and 'gluon' in parton2_type:
            process_type = "jet_production"
            # Генерация струй
            num_jets = random.randint(1, 3)
            for _ in range(num_jets):
                jet_energy = total_energy * random.uniform(0.1, 0.5)
                products.append({
                    'name': 'jet',
                    'energy': jet_energy,
                    'px': random.uniform(-jet_energy, jet_energy),
                    'py': random.uniform(-jet_energy, jet_energy),
                    'pz': random.uniform(-jet_energy, jet_energy),
                    'mass': 0.0
                })
        
        elif 'gluon' in parton1_type and 'gluon' in parton2_type:
            process_type = "gluon_fusion"
            if random.random() < 0.05:
                # Редкое производство Хиггса через слияние глюонов
                products.append({'name': 'Higgs', 'energy': total_energy*0.9})
                # Распад Хиггса на два фотона
                if random.random() < 0.002:
                    products.append({'name': 'photon', 'energy': total_energy*0.45})
                    products.append({'name': 'photon', 'energy': total_energy*0.45})
                # Распад на б-кварки
                elif random.random() < 0.58:
                    products.append({'name': 'b_quark', 'energy': total_energy*0.45})
                    products.append({'name': 'bbar_quark', 'energy': total_energy*0.45})
                # Распад на W-бозоны
                else:
                    products.append({'name': 'W_plus', 'energy': total_energy*0.45})
                    products.append({'name': 'W_minus', 'energy': total_energy*0.45})
            else:
                # Обычное производство струй
                num_jets = random.randint(2, 4)
                for _ in range(num_jets):
                    jet_energy = total_energy * random.uniform(0.1, 0.4)
                    products.append({
                        'name': 'jet',
                        'energy': jet_energy,
                        'px': random.uniform(-jet_energy, jet_energy),
                        'py': random.uniform(-jet_energy, jet_energy),
                        'pz': random.uniform(-jet_energy, jet_energy),
                        'mass': 0.0
                    })
        
        return products, process_type
    
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия частиц."""
        events = []
        for _ in range(num_events):
            x1 = self._sample_x()
            x2 = self._sample_x()
            parton1_type = self.particle_db.sample_parton()
            parton2_type = self.particle_db.sample_parton()
            parton_energy1 = energy * x1
            parton_energy2 = energy * x2
            total_parton_energy = parton_energy1 + parton_energy2
            
            products, process_type = self._generate_particles_from_partons(
                parton1_type, parton2_type, total_parton_energy
            )
            
            event = {
                'event_id': len(events),
                'process_type': process_type,
                'energy': energy,
                'parton_x1': x1,
                'parton_x2': x2,
                'parton_type1': parton1_type,
                'parton_type2': parton2_type,
                'parton_energy': total_parton_energy,
                'products': products,
                'timestamp': time.time()
            }
            events.append(event)
        
        return events
    
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
                # Имитация инициализации Geant4
                logger.info("Geant4 успешно инициализирован (stub).")
                self.initialized = True
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
        
        # Здесь должен быть реальный вызов Geant4
        # Для демонстрации используем встроенный движок
        logger.info("Используется Geant4 (заглушка)")
        fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
        return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
    
    def _convert_from_geant4_results(self, results) -> List[Dict]:
        """Конвертация результатов Geant4 в наш формат."""
        logger.warning("_convert_from_geant4_results not implemented.")
        return []

class PythiaPhysicsEngine(PhysicsEngineInterface):
    """Интеграция с Pythia для моделирования событий."""
    def __init__(self):
        self.initialized = False
        
        if PYTHIA_AVAILABLE:
            try:
                # Имитация инициализации Pythia
                logger.info("Pythia успешно инициализирован (stub).")
                self.initialized = True
            except Exception as e:
                logger.error(f"Ошибка инициализации Pythia: {e}")
                self.initialized = False
        else:
            logger.warning("Pythia недоступен.")
    
    def get_name(self) -> str:
        return "pythia"
    
    def is_available(self) -> bool:
        return PYTHIA_AVAILABLE and self.initialized
    
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия с использованием Pythia."""
        if not self.is_available():
            logger.warning("Pythia недоступен. Используется встроенная модель.")
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        
        # Здесь должен быть реальный вызов Pythia
        logger.info("Используется Pythia (заглушка)")
        fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
        return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)

class HerwigPhysicsEngine(PhysicsEngineInterface):
    """Интеграция с Herwig для моделирования событий."""
    def __init__(self):
        self.initialized = False
        
        if HERWIG_AVAILABLE:
            try:
                # Имитация инициализации Herwig
                logger.info("Herwig успешно инициализирован (stub).")
                self.initialized = True
            except Exception as e:
                logger.error(f"Ошибка инициализации Herwig: {e}")
                self.initialized = False
        else:
            logger.warning("Herwig недоступен.")
    
    def get_name(self) -> str:
        return "herwig"
    
    def is_available(self) -> bool:
        return HERWIG_AVAILABLE and self.initialized
    
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия с использованием Herwig."""
        if not self.is_available():
            logger.warning("Herwig недоступен. Используется встроенная модель.")
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        
        # Здесь должен быть реальный вызов Herwig
        logger.info("Используется Herwig (заглушка)")
        fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
        return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)

# ===================================================================
# 7. Реализации движков динамики пучка
# ===================================================================
class BuiltInBeamDynamics(BeamDynamicsInterface):
    """Встроенный движок динамики пучка."""
    def simulate_turn(self, state: Dict, revolution_time: float, include_space_charge: bool = True, **kwargs) -> Dict:
        """Симуляция одного оборота пучка."""
        updated_state = state.copy()
        
        # Обновляем время
        updated_state['beam_dynamics']['time'].append(
            updated_state['beam_dynamics']['time'][-1] + revolution_time
        )
        
        # Обновляем светимость (простая модель)
        base_luminosity = CONFIG['beam']['bunch_intensity']**2 * CONFIG['beam']['num_bunches'] / (
            4 * np.pi * updated_state['beam_dynamics']['beam_size_x'][-1] * updated_state['beam_dynamics']['beam_size_y'][-1]
        )
        
        # Добавляем небольшие флуктуации
        luminosity = base_luminosity * random.uniform(0.99, 1.01)
        updated_state['beam_dynamics']['luminosity'].append(luminosity)
        
        # Обновляем размеры пучка с учетом space charge (если включено)
        if include_space_charge and random.random() < 0.1:
            # Space charge эффект увеличивает размеры пучка
            updated_state['beam_dynamics']['beam_size_x'].append(
                updated_state['beam_dynamics']['beam_size_x'][-1] * random.uniform(1.001, 1.005)
            )
            updated_state['beam_dynamics']['beam_size_y'].append(
                updated_state['beam_dynamics']['beam_size_y'][-1] * random.uniform(1.001, 1.005)
            )
        else:
            # Натуральная эволюция размеров пучка
            updated_state['beam_dynamics']['beam_size_x'].append(
                updated_state['beam_dynamics']['beam_size_x'][-1] * random.uniform(0.999, 1.001)
            )
            updated_state['beam_dynamics']['beam_size_y'].append(
                updated_state['beam_dynamics']['beam_size_y'][-1] * random.uniform(0.999, 1.001)
            )
        
        # Обновляем эмиттанс
        updated_state['beam_dynamics']['emittance'].append(
            updated_state['beam_dynamics']['emittance'][-1] * random.uniform(0.999, 1.001)
        )
        
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
            "geant4": Geant4PhysicsEngine(),
            "pythia": PythiaPhysicsEngine(),
            "herwig": HerwigPhysicsEngine()
        }
        self.preferred_engines = ["geant4", "pythia", "herwig", "built-in"]
    
    def set_preferred_engines(self, engine_list: List[str]):
        """Устанавливает порядок предпочтения движков."""
        self.preferred_engines = engine_list
    
    def get_available_engines(self) -> List[str]:
        """Возвращает список доступных движков."""
        return [name for name, engine in self.engines.items() if engine.is_available()]
    
    def simulate_event(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Симулирует событие, используя доступные движки в порядке приоритета."""
        for engine_name in self.preferred_engines:
            if engine_name in self.engines and self.engines[engine_name].is_available():
                return self.engines[engine_name].interact(
                    particle1, particle2, energy, num_events, **kwargs
                )
        
        # Если ни один движок недоступен, используем встроенный
        if "built-in" in self.engines:
            logger.warning("Нет доступных предпочтительных движков, используем встроенный.")
            return self.engines["built-in"].interact(
                particle1, particle2, energy, num_events, **kwargs
            )
        
        # Если вообще нет доступных движков, возвращаем пустой результат
        logger.error("Нет доступных физических движков!")
        return []

# ===================================================================
# 12. *** МОДУЛЬ: TopoAnalyzer ***
# ===================================================================
class TopoAnalyzer:
    """Улучшенный Топологический анализатор событий.
    Использует идеи из топологического анализа данных (TDA) и анализа корреляций.
    Вдохновлен топологическим анализом ECDSA (торы, числа Бетти)."""
    def __init__(self, events: Optional[List[Dict]] = None):
        """Инициализация топологического анализатора."""
        self.events = events if events is not None else []
        self.feature_vectors = np.array([])
        self.distance_matrix = None
        self.persistence_result = None
        self.correlation_spectrum = None
        self.pca_result = None
        self.feature_names = [
            'num_products', 'total_energy', 'total_px', 'total_py', 'total_pz',
            'num_jets', 'num_muons', 'num_antimuons', 'num_electrons', 'num_photons'
        ]
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
        num_muons = sum(1 for p in products if p.get('name') == 'muon')
        num_antimuons = sum(1 for p in products if p.get('name') == 'antimuon')
        num_electrons = sum(1 for p in products if p.get('name') in ['electron', 'positron'])
        num_photons = sum(1 for p in products if p.get('name') == 'photon')
        
        features.extend([
            num_products, total_energy, total_px, total_py, total_pz,
            num_jets, num_muons, num_antimuons, num_electrons, num_photons
        ])
        return features
    
    def build_feature_vectors(self):
        """Строит матрицу признаков для всех событий."""
        if not self.events:
            logger.warning("Нет событий для построения векторов признаков.")
            return
        
        feature_list = []
        for event in self.events:
            features = self._extract_features(event)
            feature_list.append(features)
        
        self.feature_vectors = np.array(feature_list)
        logger.info(f"Построена матрица признаков размером {self.feature_vectors.shape}")
    
    def compute_distance_matrix(self):
        """Вычисляет матрицу расстояний между событиями."""
        if self.feature_vectors.size == 0:
            logger.error("Нет векторов признаков для вычисления матрицы расстояний.")
            return
        
        # Нормализация признаков
        normalized_features = (self.feature_vectors - np.mean(self.feature_vectors, axis=0)) / (
            np.std(self.feature_vectors, axis=0) + SMALL_EPSILON
        )
        
        try:
            self.distance_matrix = euclidean_distances(normalized_features)
            logger.info("Матрица расстояний вычислена.")
        except Exception as e:
            logger.error(f"Ошибка при вычислении матрица расстояний: {e}")
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
                # Создаем фильтрационный комплекс
                rips = gudhi.RipsComplex(distance_matrix=self.distance_matrix, max_edge_length=max_edge_length)
                simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
                self.persistence_result = {'simplex_tree': simplex_tree}
                logger.info("Персистентная гомология вычислена (GUDHI).")
            elif USE_RIPSER:
                logger.info("Используется Ripser.")
                self.persistence_result = ripser.ripser(
                    self.distance_matrix, 
                    maxdim=max_dimension, 
                    thresh=max_edge_length, 
                    metric='precomputed'
                )
                logger.info("Персистентная гомология вычислена (Ripser).")
        except Exception as e:
            logger.error(f"Ошибка при вычислении персистентной гомологии: {e}")
            self.persistence_result = None
    
    def compute_betti_numbers(self):
        """Вычисляет числа Бетти из результатов персистентной гомологии."""
        if not self.persistence_result:
            logger.warning("Нет результатов персистентной гомологии.")
            return None
        
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
                
                # Вычисляем приближенные числа Бетти
                betti = {f'β{i}': np.sum(dgm[:, 1] == np.inf) if i < len(dgms) and dgms[i].size > 0 else 0 
                         for i in range(len(dgms))}
                logger.info(f"Числа Бетти (оценка): {betti}")
            
            return betti
        except Exception as e:
            logger.error(f"Ошибка при вычислении чисел Бетти: {e}")
            return None
    
    def analyze_correlations(self, method: str = 'pearson'):
        """Анализирует корреляции между признаками."""
        if self.feature_vectors.size == 0:
            logger.error("Нет векторов признаков для анализа корреляций.")
            return None
        
        try:
            # Нормализация данных
            normalized = (self.feature_vectors - np.mean(self.feature_vectors, axis=0)) / (
                np.std(self.feature_vectors, axis=0) + SMALL_EPSILON
            )
            
            # Вычисление корреляционной матрицы
            if method == 'pearson':
                corr_matrix = np.corrcoef(normalized, rowvar=False)
            elif method == 'spearman':
                # Для упрощения используем ранги
                ranks = np.argsort(np.argsort(normalized, axis=0), axis=0)
                corr_matrix = np.corrcoef(ranks, rowvar=False)
            else:
                logger.warning(f"Неизвестный метод корреляции: {method}. Используем Pearson.")
                corr_matrix = np.corrcoef(normalized, rowvar=False)
            
            return corr_matrix
        except Exception as e:
            logger.error(f"Ошибка при анализе корреляций: {e}")
            return None
    
    def analyze_correlation_spectrum(self, corr_matrix=None):
        """Анализирует спектр корреляционной матрицы."""
        if corr_matrix is None:
            corr_matrix = self.analyze_correlations(method='pearson')
        
        if corr_matrix is None:
            return None
        
        logger.info("Анализ спектра корреляционной матрицы...")
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Используем константу вместо магического числа
            condition_number = eigenvalues[0] / (eigenvalues[-1] + SMALL_EPSILON)
            
            self.correlation_spectrum = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'condition_number': condition_number
            }
            logger.info("Анализ спектра завершен.")
            return self.correlation_spectrum
        except Exception as e:
            logger.error(f"Ошибка при анализе спектра корреляций: {e}")
            return None
    
    def perform_pca(self, n_components: Optional[int] = None):
        """Выполняет анализ главных компонент."""
        if self.feature_vectors.size == 0:
            logger.error("Нет векторов признаков для PCA.")
            return None
        
        if n_components is None:
            n_components = min(10, self.feature_vectors.shape[1])
        
        try:
            # Нормализация данных
            normalized = (self.feature_vectors - np.mean(self.feature_vectors, axis=0)) / (
                np.std(self.feature_vectors, axis=0) + SMALL_EPSILON
            )
            
            # Выполнение PCA
            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(normalized)
            
            self.pca_result = {
                'transformed_data': transformed_data,
                'explained_variance': pca.explained_variance_,
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
                gudhi.plot_persistence_diagram(dgm, axes=ax)
                ax.set_title('Диаграмма персистентности (GUDHI)')
            elif USE_RIPSER and 'dgms' in self.persistence_result:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                from persim import plot_diagrams
                plot_diagrams(self.persistence_result['dgms'], plot_only=plot_only, ax=ax)
                ax.set_title('Диаграмма персистентности (Ripser)')
            else:
                logger.warning("Не удалось определить формат данных персистентной гомологии.")
        except Exception as e:
            logger.error(f"Ошибка при визуализации персистентной гомологии: {e}")
    
    def analyze_topology(self, events_to_analyze: Optional[List[Dict]] = None, 
                         max_events: int = 500, 
                         compute_persistence: bool = True,
                         max_pers_dim: int = 1,
                         compute_pca: bool = True):
        """Выполняет полный топологический анализ."""
        logger.info("=== ЗАПУСК ПОЛНОГО ТОПОЛОГИЧЕСКОГО АНАЛИЗА ===")
        
        # Используем предоставленные события или сохраненные
        if events_to_analyze is not None:
            self.events = events_to_analyze[:max_events]
        elif self.events:
            self.events = self.events[:max_events]
        else:
            logger.error("Нет событий для анализа.")
            return None
        
        if not self.events:
            logger.error("Список событий пуст.")
            return None
        
        # Построение векторов признаков
        self.build_feature_vectors()
        if self.feature_vectors.size == 0:
            logger.error("Анализ остановлен: не удалось построить векторы признаков.")
            return None
        
        # Вычисление матрицы расстояний
        self.compute_distance_matrix()
        
        # Вычисление персистентной гомологии
        if compute_persistence:
            self.compute_persistence(max_dimension=max_pers_dim)
        
        # Анализ корреляций
        self.analyze_correlation_spectrum()
        
        # PCA
        if compute_pca:
            self.perform_pca()
        
        # Формирование результатов
        results = {
            'num_events_analyzed': len(self.events),
            'feature_names': self.feature_names,
            'correlation_spectrum': {
                'eigenvalues': self.correlation_spectrum['eigenvalues'].tolist() if self.correlation_spectrum else None,
                'condition_number': self.correlation_spectrum['condition_number'] if self.correlation_spectrum else None
            },
            'pca_variance_explained': self.pca_result['explained_variance_ratio'].tolist() if self.pca_result else None,
            'betti_numbers': self.compute_betti_numbers()
        }
        
        logger.info("Топологический анализ завершен.")
        return results
    
    def save_analysis_report(self, filename: str = "topo_analysis_report.json"):
        """Сохраняет отчет топологического анализа в JSON файл."""
        try:
            report = self.analyze_topology()
            if report:
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Отчет топологического анализа сохранен в {filename}.")
                return True
            else:
                logger.error("Не удалось создать отчет топологического анализа.")
                return False
        except Exception as e:
            logger.error(f"Ошибка при сохранении отчета топологического анализа: {e}")
            return False

# ===================================================================
# 13. *** МОДУЛЬ: GradientCalibrator ***
# ===================================================================
class GradientCalibrator:
    """Калибровщик модели на основе градиентного анализа и оптимизации.
    Использует scipy.optimize для надежной минимизации ошибки."""
    def __init__(self, model, target_observables: Dict[str, float],
                 parameters_to_calibrate: List[str],
                 observable_getters: Dict[str, Callable],
                 perturbation_factor: float = 0.001,
                 error_weights: Optional[Dict[str, float]] = None):
        """Инициализация калибратора."""
        self.model = model
        self.target_observables = target_observables
        self.parameters_to_calibrate = parameters_to_calibrate
        self.observable_getters = observable_getters
        self.perturbation_factor = perturbation_factor
        self.error_weights = error_weights or {obs: 1.0 for obs in target_observables}
        self.optimization_result = None
        self.history = []
        self.sensitivity_analysis = None
        logger.info("GradientCalibrator инициализирован.")
    
    def _objective_function(self, param_values: np.ndarray, num_turns: int = 10) -> float:
        """Целевая функция для минимизации - среднеквадратичная ошибка."""
        # Устанавливаем параметры в модели
        self._set_parameters(param_values)
        
        # Запускаем симуляцию
        try:
            self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
            current_observables = self._get_current_observables_from_model(self.model)
        except Exception as e:
            logger.error(f"Ошибка при запуске симуляции для калибровки: {e}")
            return 1e6  # Большое значение, чтобы алгоритм оптимизации избегал этой точки
        
        total_error_sq = 0.0
        for name, target_value in self.target_observables.items():
            if name not in current_observables:
                logger.warning(f"Наблюдаемая величина '{name}' не найдена в текущих результатах.")
                continue
            
            current_value = current_observables[name]
            error = current_value - target_value
            
            # Получаем масштаб для нормализации ошибки
            scale = target_value
            # Используем SMALL_EPSILON вместо магического числа 1e-12
            normalized_error = error / scale if scale > SMALL_EPSILON else error
            
            # Вес ошибки
            weight = self.error_weights.get(name, 1.0)
            total_error_sq += weight * (normalized_error ** 2)
        
        rmse = np.sqrt(total_error_sq / len(self.target_observables))
        self.history.append({
            'params': param_values.copy(), 
            'observables': current_observables.copy(), 
            'rmse': rmse
        })
        logger.debug(f"Целевая функция: params={param_values}, RMSE={rmse:.2e}")
        return rmse
    
    def _get_current_observables_from_model(self, model_instance) -> Dict[str, float]:
        """Вспомогательная функция для извлечения наблюдаемых."""
        obs = {}
        for name, getter in self.observable_getters.items():
            try:
                value = getter(model_instance)
                obs[name] = value
            except Exception as e:
                logger.error(f"Ошибка при получении наблюдаемой величины '{name}': {e}")
                # Вместо установки 0.0, пропускаем эту наблюдаемую
                # Это лучше, чем использовать нулевое значение, которое может быть некорректным
                continue
        return obs
    
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
    
    def calibrate(self, initial_params: Optional[np.ndarray] = None, 
                  method: str = 'L-BFGS-B', 
                  tolerance: float = 1e-6,
                  max_iterations: int = 100,
                  num_turns: int = 10):
        """Запускает процесс калибровки."""
        logger.info("Запуск процесса калибровки...")
        
        # Инициализация начальных параметров
        if initial_params is None:
            initial_params = np.array([
                self.model.config['beam'].get(param, 1.0) 
                if param in self.model.config.get('beam', {}) 
                else self.model.config['geometry'].get(param, 1.0)
                for param in self.parameters_to_calibrate
            ])
        
        # Настройка границ для оптимизации
        bounds = [(None, None) for _ in self.parameters_to_calibrate]
        extra_args = (num_turns,)
        
        logger.info(f"Запуск оптимизации методом {method}...")
        try:
            self.optimization_result = minimize(
                fun=self._objective_function,
                x0=initial_params,
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
        """Анализ чувствительности."""
        logger.info("Начало анализа чувствительности...")
        
        if self.optimization_result is None or not self.optimization_result.success:
            logger.warning("Нет результата успешной оптимизации.")
            return
        
        try:
            # Сохраняем оригинальные параметры
            original_params = np.array([
                self.model.config['beam'].get(param, 1.0) 
                if param in self.model.config.get('beam', {}) 
                else self.model.config['geometry'].get(param, 1.0)
                for param in self.parameters_to_calibrate
            ])
            
            gradients = []
            hess_diag_approx = []
            base_observables = self._get_current_observables_from_model(self.model)
            base_error = self._objective_function(original_params, num_turns)
            
            for i, param_name in enumerate(self.parameters_to_calibrate):
                # Пертурбация вверх
                params_up = original_params.copy()
                params_up[i] *= (1 + self.perturbation_factor)
                self._set_parameters(params_up)
                self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
                observables_up = self._get_current_observables_from_model(self.model)
                error_up = self._objective_function(params_up, num_turns)
                
                # Пертурбация вниз
                params_down = original_params.copy()
                params_down[i] *= (1 - self.perturbation_factor)
                self._set_parameters(params_down)
                self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
                observables_down = self._get_current_observables_from_model(self.model)
                error_down = self._objective_function(params_down, num_turns)
                
                # Градиент
                gradient = (error_up - error_down) / (2 * self.perturbation_factor * original_params[i])
                gradients.append(gradient)
                
                # Диагональ Гессиана (приближение)
                hessian_diag = (error_up - 2 * base_error + error_down) / (
                    (self.perturbation_factor * original_params[i]) ** 2
                )
                hess_diag_approx.append(hessian_diag)
            
            # Восстанавливаем оригинальные параметры, если нужно
            if use_original_config:
                self._set_parameters(original_params)
            
            self.sensitivity_analysis = {
                'parameters': self.parameters_to_calibrate,
                'base_observables': base_observables,
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
            'final_rmse': self.optimization_result.fun if self.optimization_result else None,
            'final_parameters': dict(zip(self.parameters_to_calibrate, self.optimization_result.x)) 
                if self.optimization_result and self.optimization_result.success else None,
            'history': self.history,
            'sensitivity_analysis': self.sensitivity_analysis
        }
        return report

# ===================================================================
# 14. *** МОДУЛЬ: AnomalyDetector ***
# ===================================================================
class AnomalyDetector:
    """Многоуровневый детектор аномалий для данных симуляции LHC."""
    def __init__(self, model=None):
        """Инициализация детектора аномалий."""
        self.model = model
        self.anomalies_found = {
            'by_type': {
                'statistical': [], 
                'topological': [], 
                'gradient': [],
                'model_behavior': [], 
                'custom': []
            },
            'summary': {
                'total_count': 0, 
                'types_found': set()
            }
        }
        logger.info("AnomalyDetector инициализирован.")
    
    def detect_statistical_anomalies(self, data: List[Dict], feature_name: str, 
                                     method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """Обнаруживает статистические аномалии в событиях."""
        logger.info(f"Поиск статистических аномалий по признаку '{feature_name}' методом '{method}'...")
        
        if not data:
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
                mean = np.mean(valid_values)
                std = np.std(valid_values)
                if std > 0:
                    z_scores = np.abs((valid_values - mean) / std)
                    anomaly_mask = z_scores > threshold
                    anomaly_indices = np.where(valid_indices)[0][anomaly_mask].tolist()
            
            elif method == 'iqr':
                q1 = np.percentile(valid_values, 25)
                q3 = np.percentile(valid_values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                anomaly_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
                anomaly_indices = np.where(valid_indices)[0][anomaly_mask].tolist()
            
            else:
                logger.warning(f"Неизвестный метод: {method}. Используем zscore.")
                return self.detect_statistical_anomalies(data, feature_name, 'zscore', threshold)
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['statistical'].extend([
                {'event_index': idx, 'feature': feature_name, 'method': method, 'value': values[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('statistical')
            
            logger.info(f"Найдено {len(anomaly_indices)} статистических аномалий.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при поиске статистических аномалий: {e}")
            return []
    
    def detect_topological_anomalies(self, topo_analysis_result, threshold_percentile: int = 99):
        """Обнаруживает аномалии на основе топологического анализа."""
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
            
            # Используем процентиль вместо жесткого порога
            pers_threshold = np.percentile(all_pers, threshold_percentile)
            
            # Идентифицируем аномальные события (с очень длинной персистентностью)
            for i, (dim, dgm) in enumerate(dgms):
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    anomalous = pers > pers_threshold
                    anomaly_indices.extend(np.where(anomalous)[0].tolist())
            
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['topological'].extend([
                {'event_index': idx, 'persistence_length': all_pers[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('topological')
            
            logger.info(f"Найдено {len(anomaly_indices)} топологических аномалий.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при поиске топологических аномалий: {e}")
            return []
    
    def detect_gradient_anomalies(self, calib_report):
        """Обнаруживает аномалии на основе градиентного анализа."""
        logger.info("Поиск градиентных аномалий...")
        
        if not calib_report or not calib_report.get('sensitivity_analysis'):
            logger.warning("Нет данных градиентного анализа.")
            return []
        
        try:
            sens_analysis = calib_report['sensitivity_analysis']
            gradients = sens_analysis.get('gradients', [])
            hessian_diag = sens_analysis.get('hessian_diagonal', [])
            
            anomaly_indices = []
            
            # Проверяем аномально большие градиенты
            if gradients:
                grad_mean = np.mean(gradients)
                grad_std = np.std(gradients)
                for i, grad in enumerate(gradients):
                    if grad_std > 0 and abs((grad - grad_mean) / grad_std) > 3.0:
                        anomaly_indices.append(i)
            
            # Проверяем аномальные значения диагонали Гессиана
            if hessian_diag:
                hess_mean = np.mean(hessian_diag)
                hess_std = np.std(hessian_diag)
                for i, hess in enumerate(hessian_diag):
                    if hess_std > 0 and abs((hess - hess_mean) / hess_std) > 3.0:
                        if i not in anomaly_indices:
                            anomaly_indices.append(i)
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['gradient'].extend([
                {'parameter_index': idx, 'gradient': gradients[idx], 'hessian_diag': hessian_diag[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('gradient')
            
            logger.info(f"Найдено {len(anomaly_indices)} градиентных аномалий.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при поиске градиентных аномалий: {e}")
            return []
    
    def detect_model_behavior_anomalies(self, model_state_history):
        """Обнаруживает аномалии в поведении модели."""
        logger.info("Поиск аномалий в поведении модели...")
        
        if not model_state_history or len(model_state_history) < 2:
            logger.warning("Недостаточно данных о состоянии модели.")
            return []
        
        try:
            anomaly_indices = []
            
            # Анализируем светимость
            luminosity = [state['beam_dynamics']['luminosity'][-1] for state in model_state_history]
            lum_diff = np.diff(luminosity)
            lum_std = np.std(lum_diff)
            if lum_std > 0:
                lum_zscores = np.abs((lum_diff - np.mean(lum_diff)) / lum_std)
                anomaly_indices.extend(np.where(lum_zscores > 3.0)[0] + 1)  # +1 из-за diff
            
            # Анализируем размеры пучка
            beam_size_x = [state['beam_dynamics']['beam_size_x'][-1] for state in model_state_history]
            beam_size_y = [state['beam_dynamics']['beam_size_y'][-1] for state in model_state_history]
            
            size_x_diff = np.diff(beam_size_x)
            size_y_diff = np.diff(beam_size_y)
            
            size_x_std = np.std(size_x_diff)
            size_y_std = np.std(size_y_diff)
            
            if size_x_std > 0:
                size_x_zscores = np.abs((size_x_diff - np.mean(size_x_diff)) / size_x_std)
                anomaly_indices.extend(np.where(size_x_zscores > 3.0)[0] + 1)
            
            if size_y_std > 0:
                size_y_zscores = np.abs((size_y_diff - np.mean(size_y_diff)) / size_y_std)
                anomaly_indices.extend(np.where(size_y_zscores > 3.0)[0] + 1)
            
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['model_behavior'].extend([
                {'turn': idx, 'luminosity': luminosity[idx], 
                 'beam_size_x': beam_size_x[idx], 'beam_size_y': beam_size_y[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('model_behavior')
            
            logger.info(f"Найдено {len(anomaly_indices)} аномалий поведения модели.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при поиске аномалий поведения модели: {e}")
            return []
    
    def detect_custom_anomalies(self, custom_detector_func, *args, **kwargs):
        """Обнаруживает аномалии с помощью пользовательской функции."""
        logger.info("Поиск пользовательских аномалий...")
        
        try:
            custom_anomalies = custom_detector_func(*args, **kwargs)
            if custom_anomalies:
                self.anomalies_found['by_type']['custom'].extend(custom_anomalies)
                self.anomalies_found['summary']['total_count'] += len(custom_anomalies)
                self.anomalies_found['summary']['types_found'].add('custom')
                logger.info(f"Найдено {len(custom_anomalies)} пользовательских аномалий.")
            return custom_anomalies
        except Exception as e:
            logger.error(f"Ошибка при поиске пользовательских аномалий: {e}")
            return []
    
    def generate_report(self, output_file: str = "anomaly_report.json"):
        """Генерирует отчет об обнаруженных аномалиях."""
        logger.info("Генерация отчета об аномалиях...")
        
        # Преобразуем set в list для сериализации JSON
        self.anomalies_found['summary']['types_found'] = list(self.anomalies_found['summary']['types_found'])
        
        # Логируем информацию об аномалиях
        for anomaly_type, anomalies_list in self.anomalies_found['by_type'].items():
            if anomalies_list:
                logger.info(f" - {anomaly_type.capitalize()}: {len(anomalies_list)}")
            else:
                logger.info(f" - {anomaly_type.capitalize()}: не найдено")
        
        try:
            with open(output_file, "w") as f:
                json.dump(self.anomalies_found, f, indent=2, default=str)
            logger.info(f"Отчет об аномалиях сохранен в {output_file}.")
        except Exception as e:
            logger.error(f"Не удалось сохранить отчет об аномалиях: {e}")
        
        return self.anomalies_found

# ===================================================================
# 15. *** МОДУЛЬ: ROOTExporter ***
# ===================================================================
class ROOTExporter:
    """Экспортер данных симуляции в формат ROOT."""
    def __init__(self):
        """Инициализация экспортера ROOT."""
        self.root_available = ROOT_AVAILABLE
        logger.info(f"ROOTExporter инициализирован. ROOT доступен: {self.root_available}")
    
    def export_collision_events(self, events: List[Dict], filename: str = "lhc_events.root"):
        """Экспортирует события столкновений в ROOT файл."""
        if not self.root_available:
            logger.error("ROOT недоступен. Невозможно экспортировать в ROOT формат.")
            return False
        
        try:
            root_file = ROOT.TFile(filename, "RECREATE")
            tree = ROOT.TTree("CollisionEvents", "События столкновений LHC")
            
            # Определяем переменные для дерева
            event_id = np.array([0], dtype=np.int32)
            process_type = ROOT.std.string()
            energy = np.array([0.0], dtype=np.float64)
            parton_x1 = np.array([0.0], dtype=np.float64)
            parton_x2 = np.array([0.0], dtype=np.float64)
            num_products = np.array([0], dtype=np.int32)
            
            # Создаем ветки
            tree.Branch("event_id", event_id, "event_id/I")
            tree.Branch("process_type", process_type)
            tree.Branch("energy", energy, "energy/D")
            tree.Branch("parton_x1", parton_x1, "parton_x1/D")
            tree.Branch("parton_x2", parton_x2, "parton_x2/D")
            tree.Branch("num_products", num_products, "num_products/I")
            
            # Добавляем ветки для продуктов
            max_products = 100
            p_mass = np.zeros(max_products, dtype=np.float64)
            p_energy = np.zeros(max_products, dtype=np.float64)
            p_px = np.zeros(max_products, dtype=np.float64)
            p_py = np.zeros(max_products, dtype=np.float64)
            p_pz = np.zeros(max_products, dtype=np.float64)
            p_name = [ROOT.std.string() for _ in range(max_products)]
            
            tree.Branch("p_mass", p_mass, f"p_mass[{max_products}]/D")
            tree.Branch("p_energy", p_energy, f"p_energy[{max_products}]/D")
            tree.Branch("p_px", p_px, f"p_px[{max_products}]/D")
            tree.Branch("p_py", p_py, f"p_py[{max_products}]/D")
            tree.Branch("p_pz", p_pz, f"p_pz[{max_products}]/D")
            tree.Branch("p_name", p_name, f"p_name[{max_products}]/string")
            
            # Заполняем дерево
            for i, event in enumerate(events):
                event_id[0] = i
                process_type = ROOT.std.string(event.get('process_type', 'unknown'))
                energy[0] = event.get('energy', 0.0)
                parton_x1[0] = event.get('parton_x1', 0.0)
                parton_x2[0] = event.get('parton_x2', 0.0)
                products = event.get('products', [])
                num_products[0] = min(len(products), max_products)
                
                # Заполняем данные о продуктах
                for j in range(num_products[0]):
                    p = products[j]
                    p_mass[j] = p.get('mass', 0.0)
                    p_energy[j] = p.get('energy', 0.0)
                    p_px[j] = p.get('px', 0.0)
                    p_py[j] = p.get('py', 0.0)
                    p_pz[j] = p.get('pz', 0.0)
                    p_name[j] = ROOT.std.string(p.get('name', 'unknown'))
                
                tree.Fill()
            
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
    """Экспортер данных симуляции в формат HepMC3."""
    def __init__(self):
        """Инициализация экспортера."""
        self.hepmc3_available = HEPMC3_AVAILABLE
        logger.info(f"HepMC3Exporter инициализирован. HepMC3 доступен: {self.hepmc3_available}")
    
    def export_collision_events(self, events: List[Dict], filename: str = "lhc_events.hepmc3"):
        """Экспортирует события столкновений в формат HepMC3."""
        if not self.hepmc3_available:
            logger.error("HepMC3 недоступен. Невозможно экспортировать в HepMC3 формат.")
            return False
        
        try:
            # Здесь должен быть код для экспорта в HepMC3
            # Для примера просто создаем текстовый файл
            with open(filename, 'w') as f:
                f.write("HepMC3 file generated by LHCHybridModel\n")
                f.write("Units: GEV MM\n\n")
                
                for i, event in enumerate(events):
                    f.write(f"E {i} 0 0 0\n")  # Заголовок события
                    
                    # Записываем входные частицы (протоны)
                    f.write(f"P 1 2212 0 0 6500 6500 0 0 0 1 0 0\n")
                    f.write(f"P 2 2212 0 0 -6500 -6500 0 0 0 1 0 0\n")
                    
                    # Записываем продукты столкновения
                    products = event.get('products', [])
                    for j, p in enumerate(products, 3):
                        pdg_id = self._get_pdg_id(p.get('name', 'unknown'))
                        px = p.get('px', 0.0)
                        py = p.get('py', 0.0)
                        pz = p.get('pz', 0.0)
                        e = p.get('energy', 0.0)
                        m = p.get('mass', 0.0)
                        
                        # Формат HepMC3: P ID PDG_ID PX PY PZ E M STATUS
                        f.write(f"P {j} {pdg_id} {px} {py} {pz} {e} {m} 1\n")
                    
                    f.write("\n")
            
            logger.info(f"Экспорт в HepMC3 файл '{filename}' успешно завершен.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при экспорте в HepMC3: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_pdg_id(self, particle_name: str) -> int:
        """Возвращает PDG ID для частицы по имени."""
        mapping = {
            'electron': 11, 'positron': -11,
            'muon': 13, 'antimuon': -13,
            'photon': 22,
            'u_quark': 2, 'ubar_quark': -2,
            'd_quark': 1, 'dbar_quark': -1,
            's_quark': 3, 'sbar_quark': -3,
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
    """Интерактивная визуализация для коллайдера с использованием Plotly."""
    def __init__(self):
        """Инициализация визуализатора."""
        self.plotly_available = PLOTLY_AVAILABLE
        logger.info(f"Visualizer инициализирован. Plotly доступен: {self.plotly_available}")
    
    def plot_geometry_3d(self, geometry, detector_system):
        """Интерактивная 3D-визуализация геометрии коллайдера и детекторов."""
        if not self.plotly_available:
            logger.warning("Plotly недоступен.")
            return
        
        try:
            fig = go.Figure()
            radius = geometry.circumference / (2 * np.pi)
            theta = np.linspace(0, 2*np.pi, 100)
            x_ring = radius * np.cos(theta)
            y_ring = radius * np.sin(theta)
            z_ring = np.zeros_like(x_ring)
            
            fig.add_trace(go.Scatter3d(
                x=x_ring, y=y_ring, z=z_ring, 
                mode='lines', name='Collider Ring', 
                line=dict(color='blue', width=5)
            ))
            
            # Добавляем детекторы
            for detector in detector_system.detectors:
                # Пример: детектор как цилиндр
                detector_radius = radius * 0.9
                detector_z = np.linspace(detector['z_min'], detector['z_max'], 20)
                detector_theta = np.linspace(0, 2*np.pi, 20)
                detector_theta_grid, detector_z_grid = np.meshgrid(detector_theta, detector_z)
                detector_x = detector_radius * np.cos(detector_theta_grid)
                detector_y = detector_radius * np.sin(detector_theta_grid)
                
                fig.add_trace(go.Surface(
                    x=detector_x, y=detector_y, z=detector_z_grid,
                    colorscale=[[0, detector['color']], [1, detector['color']]],
                    opacity=0.5, name=detector['name']
                ))
            
            fig.update_layout(
                title='3D Geometry of LHC and Detectors',
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)'
                )
            )
            
            fig.show()
            logger.info("3D-геометрия коллайдера визуализирована.")
        except Exception as e:
            logger.error(f"Ошибка при 3D-визуализации геометрии: {e}")
    
    def plot_detector_response_3d(self, detected_events, detector_system):
        """Визуализирует отклики детекторов в 3D."""
        if not self.plotly_available:
            logger.warning("Plotly недоступен.")
            return
        
        try:
            fig = go.Figure()
            
            # Добавляем геометрию детекторов (как в plot_geometry_3d)
            # ...
            
            # Добавляем треки частиц
            for event in detected_events[:100]:  # Ограничиваем количество событий для ясности
                for particle in event['particles']:
                    # Пример: простой трек от центра
                    r = np.linspace(0, particle['r_max'], 10)
                    theta = np.full_like(r, particle['theta'])
                    phi = np.full_like(r, particle['phi'])
                    
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        name=f"Particle {particle['id']}",
                        line=dict(width=3)
                    ))
            
            fig.update_layout(title='Detector Responses')
            fig.show()
            logger.info("Отклики детекторов визуализированы.")
        except Exception as e:
            logger.error(f"Ошибка при визуализации откликов детекторов: {e}")

# ===================================================================
# 18. *** МОДУЛЬ: GPUAccelerator (заглушка) ***
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
            logger.warning("GPUAccelerator: Нет доступных GPU бэкендов.")
    
    def is_gpu_available(self) -> bool:
        """Проверяет, доступен ли GPU для ускорения."""
        return self.is_available
    
    def accelerate_function(self, func):
        """Ускоряет функцию с использованием GPU (заглушка)."""
        if not self.is_available:
            logger.warning("GPU недоступен. Используется CPU версия.")
            return func
        
        # В реальной реализации здесь был бы код для ускорения с помощью GPU
        logger.info(f"Функция {func.__name__} ускорена с использованием {self.backend}.")
        return func
    
    def _track_particles_on_gpu(self, particles, field, num_steps, dt):
        """Отслеживает частицы на GPU (заглушка)."""
        if not self.is_available:
            return self._track_single_particle_on_cpu(particles, field, num_steps, dt)
        
        # Заглушка для GPU-ускорения
        logger.warning("GPU ускорение не реализовано. Используется CPU версия.")
        return [self._track_single_particle_on_cpu(p, field, num_steps, dt) for p in particles]
    
    def _track_single_particle_on_cpu(self, particle, field, num_steps, dt):
        """Отслеживает одну частицу на CPU (заглушка)."""
        # Простая симуляция движения частицы в магнитном поле
        trajectory = []
        x, y, z = particle['position']
        px, py, pz = particle['momentum']
        
        for _ in range(num_steps):
            # Простая модель движения в однородном магнитном поле
            B = field.get('strength', 8.3)  # Тесла
            q = particle['charge']
            m = particle['mass']
            
            # Уравнения движения (упрощенные)
            vx = px / m
            vy = py / m
            vz = pz / m
            
            x += vx * dt
            y += vy * dt
            z += vz * dt
            
            # Лоренцева сила (упрощенная)
            px += q * B * vy * dt
            py -= q * B * vx * dt
            
            trajectory.append((x, y, z))
        
        return trajectory

# ===================================================================
# 19. Основная модель коллайдера
# ===================================================================
class LHCHybridModel:
    """Усовершенствованная гибридная модель Большого адронного коллайдера.
    Это центральный класс фреймворка, объединяющий все компоненты:
    - Физические и динамические движки
    - Системы анализа (TopoAnalyzer, GradientCalibrator, AnomalyDetector)
    - Экспорт данных
    - Визуализация"""
    
    def __init__(self):
        """Инициализация модели."""
        self.config = CONFIG
        self.particle_db = ParticleDatabase()
        self.physics_engine = HybridPhysicsEngine(self.particle_db)
        self.beam_dynamics = BuiltInBeamDynamics()
        self.cache = SimulationCache()
        self.topo_analyzer = TopoAnalyzer()
        self.anomaly_detector = AnomalyDetector(self)
        self.geometry = self._initialize_geometry()
        self.detector_system = self._initialize_detector_system()
        self.simulation_state = self._initialize_simulation_state()
        self.calibrator = None
        self.root_exporter = ROOTExporter()
        self.hepmc3_exporter = HepMC3Exporter()
        self.visualizer = Visualizer()
        
        logger.info("Усовершенствованная гибридная модель коллайдера инициализирована")
    
    def _initialize_geometry(self):
        """Инициализация геометрии коллайдера."""
        class Geometry:
            def __init__(self, config):
                self.circumference = config['geometry']['circumference']
                self.bending_radius = config['geometry']['bending_radius']
        
        return Geometry(self.config)
    
    def _initialize_detector_system(self):
        """Инициализация системы детекторов."""
        class DetectorSystem:
            def __init__(self):
                self.detectors = [
                    {'name': 'ATLAS', 'z_min': -5, 'z_max': 5, 'color': 'red'},
                    {'name': 'CMS', 'z_min': -4, 'z_max': 4, 'color': 'green'},
                    {'name': 'ALICE', 'z_min': -3, 'z_max': 3, 'color': 'blue'},
                    {'name': 'LHCb', 'z_min': -2, 'z_max': 2, 'color': 'purple'}
                ]
        
        return DetectorSystem()
    
    def _initialize_simulation_state(self) -> Dict:
        """Инициализация начального состояния симуляции."""
        if 'beam' in self.config:
            beam_energy = self.config['beam']['energy'] * 1e9  # в эВ
            particles = self.config['beam']['particles']
            bunch_intensity = self.config['beam']['bunch_intensity']
            num_bunches = self.config['beam']['num_bunches']
        else:
            beam_energy = 6500 * 1e9  # в эВ
            particles = 'protons'
            bunch_intensity = 1.15e11
            num_bunches = 2748
        
        # Вычисляем релятивистские параметры
        gamma = beam_energy / (PROTON_MASS * 1e9)  # 1e9 для перевода ГэВ в эВ
        beta = np.sqrt(1 - 1/gamma**2)
        revolution_time = self.geometry.circumference / (beta * SPEED_OF_LIGHT)
        
        return {
            'turn': 0,
            'beam_energy': beam_energy,
            'particles': particles,
            'bunch_intensity': bunch_intensity,
            'num_bunches': num_bunches,
            'revolution_time': revolution_time,
            'beam_dynamics': {
                'time': [0.0],
                'luminosity': [1.0e34],  # начальная светимость в см^-2 с^-1
                'beam_size_x': [0.045],  # в метрах
                'beam_size_y': [0.045],
                'emittance': [3.5e-6]  # в м·рад
            },
            'collision_events': [],
            'detected_events': []
        }
    
    def step_simulation(self, include_space_charge: bool = True,
                        force_physics_engine: Optional[str] = None,
                        force_beam_engine: Optional[str] = None):
        """Выполнение одного шага симуляции (один оборот пучка)."""
        # Обновляем динамику пучка
        updated_state = self.beam_dynamics.simulate_turn(
            self.simulation_state,
            self.simulation_state['revolution_time'],
            include_space_charge=include_space_charge,
            force_engine=force_beam_engine
        )
        self.simulation_state = updated_state
        
        # Симулируем столкновения
        self._simulate_collision(force_engine=force_physics_engine)
        
        # Увеличиваем счетчик оборотов
        self.simulation_state['turn'] += 1
    
    def _simulate_collision(self, force_engine: Optional[str] = None):
        """Моделирование события столкновения."""
        energy = self.simulation_state['beam_energy'] * 2  # полная энергия в центре масс
        event_id = len(self.simulation_state['collision_events'])
        
        # Создаем ключ для кэша
        cache_params = {
            'energy': energy, 
            'engine': force_engine or 'auto', 
            'num_events': 1
        }
        cache_key = SimulationCache.generate_key(cache_params)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Использован кэшированный результат для столкновения")
            events = cached_result
        else:
            # Симулируем столкновение
            events = self.physics_engine.simulate_event(
                'proton', 'proton', energy, num_events=1, force_engine=force_engine
            )
            self.cache.set(cache_key, events)
        
        # Добавляем события в историю
        if events:
            self.simulation_state['collision_events'].extend(events)
            
            # Симулируем детектирование
            detected = self._simulate_detection(events)
            self.simulation_state['detected_events'].extend(detected)
    
    def _simulate_detection(self, events: List[Dict]) -> List[Dict]:
        """Симулирует детектирование событий."""
        detected_events = []
        
        for event in events:
            detected_particles = []
            for particle in event['products']:
                # Простая модель детектирования
                if random.random() < 0.9:  # 90% шанс детектирования
                    # Примерные ошибки измерения
                    energy_error = random.gauss(0, 0.05 * particle.get('energy', 0))
                    momentum_error = random.gauss(0, 0.05 * np.sqrt(
                        particle.get('px', 0)**2 + 
                        particle.get('py', 0)**2 + 
                        particle.get('pz', 0)**2
                    ))
                    
                    detected_particles.append({
                        'detector': random.choice([d['name'] for d in self.detector_system.detectors]),
                        'particle': particle.get('name', 'unknown'),
                        'true_energy': particle.get('energy', 0.0),
                        'reconstructed_energy': particle.get('energy', 0.0) + energy_error,
                        'reconstructed_momentum': momentum_error
                    })
            
            detected_events.append({
                'event_id': event['event_id'],
                'timestamp': time.time(),
                'particles': detected_particles
            })
        
        return detected_events
    
    def run_simulation(self, num_turns: int = 100, 
                       include_space_charge: bool = True,
                       force_physics_engine: Optional[str] = None,
                       force_beam_engine: Optional[str] = None):
        """Запуск симуляции на заданное количество оборотов."""
        logger.info(f"Запуск симуляции на {num_turns} оборотов")
        start_time = time.time()
        
        for _ in range(num_turns):
            self.step_simulation(include_space_charge, force_physics_engine, force_beam_engine)
        
        end_time = time.time()
        logger.info(f"Симуляция завершена за {end_time - start_time:.2f} секунд")
        self.validate_results()
        
        hit_rate = self.cache.get_hit_rate()
        logger.info(f"Кэш: hit rate = {hit_rate:.2%}")
    
    def validate_results(self):
        """Валидация результатов симуляции."""
        dataset_id = self.config['validation']['dataset_id']
        logger.info(f"Валидация результатов с использованием набора данных: {dataset_id}")
        
        # Здесь должна быть реальная валидация с использованием реальных данных
        # Для демонстрации просто сравниваем с жестко заданными значениями
        target_luminosity = 1.5e34  # см^-2 с^-1
        current_luminosity = self.simulation_state['beam_dynamics']['luminosity'][-1]
        
        relative_error = abs(current_luminosity - target_luminosity) / target_luminosity
        logger.info(f"Относительная ошибка светимости: {relative_error:.2%}")
    
    def analyze_topology(self, max_events: int = 500, 
                         compute_persistence: bool = True, 
                         max_pers_dim: int = 1, 
                         compute_pca: bool = True):
        """Запускает топологический анализ событий."""
        logger.info("Запуск топологического анализа...")
        
        # Используем последние события
        events_to_analyze = self.simulation_state['collision_events'][-max_events:]
        
        # Выполняем анализ
        self.topo_analyzer.events = events_to_analyze
        results = self.topo_analyzer.analyze_topology(
            compute_persistence=compute_persistence,
            max_pers_dim=max_pers_dim,
            compute_pca=compute_pca
        )
        
        logger.info("Топологический анализ завершен.")
        return results
    
    def calibrate_via_gradients(self, target_values: Dict[str, float], 
                               parameters: List[str],
                               num_turns: int = 10, 
                               error_weights: Optional[Dict[str, float]] = None):
        """Новый метод для калибровки модели."""
        logger.info("Запуск калибровки модели...")
        
        # Определяем геттеры для наблюдаемых
        def get_luminosity(model):
            return model.simulation_state['beam_dynamics']['luminosity'][-1]
        
        def get_beam_size_x(model):
            return model.simulation_state['beam_dynamics']['beam_size_x'][-1]
        
        def get_avg_event_energy(model):
            if not model.simulation_state['collision_events']:
                return 0.0
            total_energy = sum(
                event.get('energy', 0.0) 
                for event in model.simulation_state['collision_events']
            )
            return total_energy / len(model.simulation_state['collision_events'])
        
        observable_getters = {
            'luminosity': get_luminosity,
            'beam_size_x': get_beam_size_x,
            'avg_event_energy': get_avg_event_energy
        }
        
        # Создаем калибратор
        self.calibrator = GradientCalibrator(
            model=self,
            target_observables=target_values,
            parameters_to_calibrate=parameters,
            observable_getters=observable_getters,
            error_weights=error_weights
        )
        
        # Запускаем калибровку
        self.calibrator.calibrate(num_turns=num_turns)
        
        # Анализируем чувствительность
        self.calibrator.analyze_sensitivity(num_turns=num_turns)
        
        logger.info("Калибровка завершена.")
        return self.calibrator.get_summary_report()
    
    def detect_anomalies(self, use_topo_results: bool = True, use_calib_report: bool = True):
        """Обнаруживает аномалии в данных симуляции."""
        logger.info("Запуск обнаружения аномалий...")
        
        # 1. Статистические аномалии
        if self.simulation_state['collision_events']:
            self.anomaly_detector.detect_statistical_anomalies(
                self.simulation_state['collision_events'], 
                'energy', 
                method='zscore'
            )
        
        # 2. Топологические аномалии
        if use_topo_results and hasattr(self, 'topo_analyzer') and self.topo_analyzer.events:
            self.anomaly_detector.detect_topological_anomalies(
                self.topo_analyzer.analyze_topology()
            )
        
        # 3. Градиентные аномалии
        if use_calib_report and self.calibrator:
            self.anomaly_detector.detect_gradient_anomalies(
                self.calibrator.get_summary_report()
            )
        
        # 4. Аномалии поведения модели
        self.anomaly_detector.detect_model_behavior_anomalies(
            [self.simulation_state.copy()]  # В реальной реализации здесь была бы история состояний
        )
        
        # Генерируем отчет
        report = self.anomaly_detector.generate_report()
        logger.info("Обнаружение аномалий завершено.")
        return report
    
    def export_results(self, format: str = 'json', path: str = "."):
        """Экспортирует результаты симуляции в указанный формат."""
        logger.info(f"Экспорт результатов в формат {format}...")
        
        success = False
        if format == 'json':
            try:
                export_data = {
                    'metadata': {
                        'timestamp': time.time(),
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
            except Exception as e:
                logger.error(f"Ошибка при экспорте в JSON: {e}")
        
        elif format == 'root' and self.root_exporter.root_available:
            success = self.root_exporter.export_collision_events(
                self.simulation_state['collision_events'],
                os.path.join(path, 'lhc_events.root')
            )
        
        elif format == 'hepmc3' and self.hepmc3_exporter.hepmc3_available:
            success = self.hepmc3_exporter.export_collision_events(
                self.simulation_state['collision_events'],
                os.path.join(path, 'lhc_events.hepmc3')
            )
        
        else:
            logger.error(f"Формат экспорта '{format}' не поддерживается или недоступен.")
        
        return success
    
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
        'beam': {
            'energy': 6500,  # в ГэВ
            'particles': 'protons',
            'bunch_intensity': 1.15e11,
            'num_bunches': 2748
        },
        'geometry': {
            'circumference': 26659,  # в метрах
            'bending_radius': 2800
        },
        'simulation': {
            'num_turns': 1000,
            'revolution_time': 88.9e-6  # в секундах
        },
        'validation': {
            'dataset_id': 'CMS_OpenData_2018'
        }
    }
    with open("lhc_config.yaml", "w") as f:
        yaml.dump(config, f)
    logger.info("Создан файл конфигурации по умолчанию: lhc_config.yaml")

# ===================================================================
# 21. Основной сценарий
# ===================================================================
if __name__ == "__main__":
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
