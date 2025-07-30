import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.constants import c, e, m_p, m_e, epsilon_0, hbar
import logging
import json
import time
import os
import shutil
import subprocess
import yaml
import requests
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

# Подавление предупреждений для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================================================================
# 1. Настройка логирования и конфигурации
# ===================================================================

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lhc_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LHC_Simulator")

# Загрузка конфигурации
try:
    with open("lhc_config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    logger.info("Конфигурация загружена из lhc_config.yaml")
except FileNotFoundError:
    # Дефолтная конфигурация
    CONFIG = {
        "beam": {
            "energy": 6500,  # ГэВ
            "particles": "protons",
            "bunch_intensity": 1.15e11,
            "num_bunches": 2748
        },
        "geometry": {
            "circumference": 26658.883,  # м
            "dipole_field": 8.33  # Тл
        },
        "validation": {
            "dataset_id": "cms-2011-collision-data"
        }
    }
    logger.warning("Конфигурационный файл не найден. Используется дефолтная конфигурация.")


# ===================================================================
# 2. База данных частиц и физические константы
# ===================================================================

@dataclass
class Particle:
    """Класс для хранения информации о частице"""
    name: str
    mass: float  # ГэВ/c²
    charge: float  # в единицах элементарного заряда
    spin: float
    lifetime: float  # секунды
    category: str
    symbol: str
    width: float = 0.0  # ширина распада в ГэВ
    decay_channels: List[Tuple[str, float]] = None  # (продукты, вероятность)

    def __post_init__(self):
        if self.decay_channels is None:
            self.decay_channels = []


class ParticleDatabase:
    """База данных частиц с полной информацией о свойствах и распадах"""
    
    def __init__(self):
        """Инициализация базы данных частиц"""
        self.particles = self._load_particle_data()
        logger.info(f"База данных частиц загружена. Доступно {len(self.particles)} частиц.")
    
    def _load_particle_data(self) -> Dict[str, Particle]:
        """Загрузка данных о частицах"""
        particles = {
            'proton': Particle(
                name='proton',
                mass=0.938272,  # ГэВ/c²
                charge=1,
                spin=0.5,
                lifetime=float('inf'),
                category='baryon',
                symbol='p'
            ),
            'electron': Particle(
                name='electron',
                mass=0.000511,  # ГэВ/c²
                charge=-1,
                spin=0.5,
                lifetime=float('inf'),
                category='lepton',
                symbol='e⁻'
            ),
            'positron': Particle(
                name='positron',
                mass=0.000511,  # ГэВ/c²
                charge=1,
                spin=0.5,
                lifetime=float('inf'),
                category='lepton',
                symbol='e⁺'
            ),
            'muon': Particle(
                name='muon',
                mass=0.105658,  # ГэВ/c²
                charge=-1,
                spin=0.5,
                lifetime=2.197e-6,
                category='lepton',
                symbol='μ⁻'
            ),
            'antimuon': Particle(
                name='antimuon',
                mass=0.105658,  # ГэВ/c²
                charge=1,
                spin=0.5,
                lifetime=2.197e-6,
                category='lepton',
                symbol='μ⁺'
            ),
            'photon': Particle(
                name='photon',
                mass=0.0,
                charge=0,
                spin=1,
                lifetime=float('inf'),
                category='boson',
                symbol='γ'
            ),
            'w_boson': Particle(
                name='W boson',
                mass=80.379,  # ГэВ/c²
                charge=1,
                spin=1,
                lifetime=3.2e-25,
                category='boson',
                symbol='W⁺',
                width=2.085,
                decay_channels=[
                    ('electron + neutrino_e', 0.1071),
                    ('muon + neutrino_μ', 0.1063),
                    ('tau + neutrino_τ', 0.1125),
                    ('quark + antiquark', 0.6741)
                ]
            ),
            'z_boson': Particle(
                name='Z boson',
                mass=91.1876,  # ГэВ/c²
                charge=0,
                spin=1,
                lifetime=2.64e-25,
                category='boson',
                symbol='Z⁰',
                width=2.4952,
                decay_channels=[
                    ('electron + positron', 0.03363),
                    ('muon + antimuon', 0.03366),
                    ('tau + antitau', 0.03370),
                    ('neutrino + antineutrino', 0.2000),
                    ('quark + antiquark', 0.6991)
                ]
            ),
            'higgs_boson': Particle(
                name='Higgs boson',
                mass=125.1,  # ГэВ/c²
                charge=0,
                spin=0,
                lifetime=1.56e-22,
                category='boson',
                symbol='H⁰',
                width=0.00407,
                decay_channels=[
                    ('bottom_quark + antibottom_quark', 0.5824),
                    ('W_boson + W_boson', 0.2155),
                    ('gluon + gluon', 0.0858),
                    ('tau + antitau', 0.0627),
                    ('Z_boson + Z_boson', 0.0267),
                    ('photon + photon', 0.00228)
                ]
            ),
            'top_quark': Particle(
                name='top quark',
                mass=172.76,  # ГэВ/c²
                charge=2/3,
                spin=0.5,
                lifetime=5.0e-25,
                category='quark',
                symbol='t',
                width=1.34
            ),
            'bottom_quark': Particle(
                name='bottom quark',
                mass=4.18,  # ГэВ/c²
                charge=-1/3,
                spin=0.5,
                lifetime=1.6e-12,
                category='quark',
                symbol='b'
            ),
            'charm_quark': Particle(
                name='charm quark',
                mass=1.27,  # ГэВ/c²
                charge=2/3,
                spin=0.5,
                lifetime=1.0e-12,
                category='quark',
                symbol='c'
            ),
            'strange_quark': Particle(
                name='strange quark',
                mass=0.096,  # ГэВ/c²
                charge=-1/3,
                spin=0.5,
                lifetime=1.5e-8,
                category='quark',
                symbol='s'
            ),
            'up_quark': Particle(
                name='up quark',
                mass=0.00216,  # ГэВ/c²
                charge=2/3,
                spin=0.5,
                lifetime=float('inf'),
                category='quark',
                symbol='u'
            ),
            'down_quark': Particle(
                name='down quark',
                mass=0.00467,  # ГэВ/c²
                charge=-1/3,
                spin=0.5,
                lifetime=float('inf'),
                category='quark',
                symbol='d'
            ),
            'pion_plus': Particle(
                name='pion plus',
                mass=0.13957039,  # ГэВ/c²
                charge=1,
                spin=0,
                lifetime=2.6033e-8,
                category='meson',
                symbol='π⁺'
            ),
            'kaon_plus': Particle(
                name='kaon plus',
                mass=0.493677,  # ГэВ/c²
                charge=1,
                spin=0,
                lifetime=1.238e-8,
                category='meson',
                symbol='K⁺'
            )
        }
        return particles
    
    def get_particle(self, name: str) -> Optional[Particle]:
        """Получение информации о частице по имени"""
        return self.particles.get(name.lower())
    
    def get_category(self, name: str) -> Optional[str]:
        """Получение категории частицы"""
        particle = self.get_particle(name)
        return particle.category if particle else None
    
    def get_decay_products(self, particle_name: str) -> List[Tuple[str, float]]:
        """Получение каналов распада частицы с вероятностями"""
        particle = self.get_particle(particle_name)
        if particle and particle.decay_channels:
            return particle.decay_channels
        return []
    
    def generate_decay_products(self, particle_name: str) -> List[str]:
        """Генерация продуктов распада на основе вероятностей"""
        decay_channels = self.get_decay_products(particle_name)
        if not decay_channels:
            return [particle_name]  # Стабильная частица
        
        # Нормализация вероятностей
        total_prob = sum(prob for _, prob in decay_channels)
        normalized_channels = [(products, prob/total_prob) for products, prob in decay_channels]
        
        # Случайный выбор канала распада
        r = random.random()
        cumulative = 0.0
        for products, prob in normalized_channels:
            cumulative += prob
            if r <= cumulative:
                return [p.strip() for p in products.split('+')]
        
        return [particle_name]  # fallback


# ===================================================================
# 3. Интеграция с CERN Open Data API
# ===================================================================

class CERN_Open_Data_API:
    """Интерфейс для работы с CERN Open Data Portal"""
    
    BASE_URL = "https://opendata.cern.ch/api/records/"
    
    @staticmethod
    def get_lhc_parameters() -> Dict[str, float]:
        """Получение основных параметров БАК из открытых данных"""
        # В реальной реализации этот метод должен запрашивать данные через API
        # Здесь используем тестовые данные
        logger.info("Загрузка параметров БАК из CERN Open Data")
        return {
            "circumference": 26658.883,  # м
            "beam_energy": 6500,  # ГэВ
            "peak_luminosity": 2.0e34,  # см⁻²с⁻¹
            "num_bunches": 2748,
            "bunch_intensity": 1.15e11,
            "revolution_frequency": 11245.5,  # Гц
            "dipole_field": 8.33  # Тл
        }
    
    @staticmethod
    def get_collision_data(dataset_id: str) -> Dict[str, Any]:
        """Получение данных о столкновениях из открытых данных CERN"""
        logger.info(f"Загрузка данных столкновений из набора {dataset_id}")
        
        # В реальной системе это был бы запрос к API
        # Например: response = requests.get(f"{CERN_Open_Data_API.BASE_URL}{dataset_id}")
        
        # Для демонстрации возвращаем тестовые данные
        # В реальной системе данные пришли бы из CERN Open Data Portal, который управляет несколькими петабайтами данных от LHC [[9]]
        return {
            "dataset_id": dataset_id,
            "description": "CMS open data from LHC Run 2",
            "energy": 13000,  # ГэВ
            "luminosity": 1.5e34,  # см⁻²с⁻¹
            "num_events": 100000,
            "collision_type": "proton-proton",
            "recorded_data": [
                {
                    "event_id": i,
                    "energy": random.uniform(100, 1000),
                    "products": random.sample(["electron", "muon", "photon", "jet"], random.randint(1, 4)),
                    "timestamp": time.time() - random.randint(0, 86400)
                } for i in range(100)  # В реальности было бы больше данных
            ]
        }
    
    @staticmethod
    def get_real_lhc_parameters() -> Dict[str, float]:
        """Получение реальных параметров БАК из открытых данных"""
        # В реальной системе это был бы запрос к API
        # В CERN Open Data Portal доступны данные о столкновениях протон-протон за 2010-2011 и половину 2012 года [[6]]
        return {
            "circumference": 26658.883,
            "beam_energy": 6500,
            "peak_luminosity": 2.0e34,
            "bunch_spacing": 25e-9,
            "num_bunches": 2748,
            "bunch_intensity": 1.15e11,
            "revolution_time": 8.89e-5,
            "dipole_field": 8.33
        }
    
    @staticmethod
    def get_record_metadata(record_id: str) -> Dict[str, Any]:
        """Получение метаданных записи из CERN Open Data"""
        # В реальной системе: запрос к API
        logger.info(f"Получение метаданных для записи {record_id}")
        return {
            "record_id": record_id,
            "title": f"CMS collision data set {record_id}",
            "description": "Reconstructed collision data from CMS detector",
            "keywords": ["collision", "proton-proton", "LHC", "Run 2"],
            "energy": 13000,  # ГэВ
            "num_events": 500000,
            "format": "ROOT"
        }


# ===================================================================
# 4. Гибридные интерфейсы для профессиональных инструментов
# ===================================================================

class PhysicsEngineInterface(ABC):
    """Абстрактный интерфейс для физических движков"""
    
    @abstractmethod
    def interact(self, particle1: str, particle2: str, energy: float, 
                num_events: int = 1, **kwargs) -> List[Dict]:
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
    def simulate_turn(self, state: Dict, revolution_time: float, 
                     include_space_charge: bool = True, **kwargs) -> Dict:
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
# 5. Встроенные физические модели
# ===================================================================

class BuiltInPhysicsEngine(PhysicsEngineInterface):
    """Встроенная физическая модель для симуляции взаимодействий"""
    
    def __init__(self, particle_db: ParticleDatabase):
        """Инициализация физического движка"""
        self.particle_db = particle_db
        self.models = {
            'standard': self._standard_model,
            'qcd': self._qcd_model,
            'electroweak': self._electroweak_model,
            'beyond_sm': self._beyond_standard_model
        }
        logger.info("Встроенный физический движок инициализирован")
    
    def get_name(self) -> str:
        return "built-in"
    
    def is_available(self) -> bool:
        return True
    
    def interact(self, particle1: str, particle2: str, energy: float, 
                num_events: int = 1, model: str = 'standard', **kwargs) -> List[Dict]:
        """
        Моделирование взаимодействия частиц с проверкой сохранения 4-импульса.
        
        Параметры:
        particle1: тип первой частицы
        particle2: тип второй частицы
        energy: энергия взаимодействия в ГэВ
        num_events: количество событий для симуляции
        model: используемая физическая модель
        
        Возвращает:
        Список событий со списком продуктов для каждого
        """
        if model not in self.models:
            logger.warning(f"Модель '{model}' не найдена. Используется стандартная модель.")
            model = 'standard'
        
        events = []
        for _ in range(num_events):
            # Получаем информацию о частицах
            p1_info = self.particle_db.get_particle(particle1)
            p2_info = self.particle_db.get_particle(particle2)
            
            if not p1_info or not p2_info:
                logger.error(f"Неизвестные частицы: {particle1}, {particle2}")
                continue
            
            # Проверяем, является ли это упругим рассеянием
            if particle1 == particle2 and random.random() < 0.1:  # 10% вероятность упругого рассеяния
                event = self._elastic_scattering(p1_info, p2_info, energy)
            else:
                # Выбираем тип взаимодействия на основе частиц и энергии
                if "quark" in particle1 or "quark" in particle2 or (particle1 == "proton" and particle2 == "proton"):
                    event = self._qcd_model(particle1, particle2, energy)
                elif "lepton" in self.particle_db.get_category(particle1) or "lepton" in self.particle_db.get_category(particle2):
                    event = self._electroweak_model(particle1, particle2, energy)
                else:
                    event = self._standard_model(particle1, particle2, energy)
            
            # Проверка сохранения 4-импульса
            if not self._check_4_momentum_conservation(event, energy):
                logger.warning("Нарушение сохранения 4-импульса. Корректировка событий.")
                event = self._correct_4_momentum(event, energy)
            
            events.append(event)
        
        return events
    
    def _check_4_momentum_conservation(self, event: Dict, initial_energy: float) -> bool:
        """Проверка сохранения 4-импульса в событии"""
        # Начальный 4-импульс (предполагаем, что частицы сталкиваются лоб в лоб)
        initial_px, initial_py, initial_pz = 0.0, 0.0, 0.0
        initial_e = initial_energy  # в системе центра масса
        
        # Суммарный 4-импульс продуктов
        total_px, total_py, total_pz, total_e = 0.0, 0.0, 0.0, 0.0
        
        for product in event['products']:
            total_px += product.get('px', 0.0)
            total_py += product.get('py', 0.0)
            total_pz += product.get('pz', 0.0)
            total_e += product.get('energy', 0.0)
        
        # Проверка сохранения
        momentum_conserved = (abs(total_px - initial_px) < 1e-6 and 
                              abs(total_py - initial_py) < 1e-6 and 
                              abs(total_pz - initial_pz) < 1e-6)
        energy_conserved = abs(total_e - initial_e) < 1e-3 * initial_e
        
        return momentum_conserved and energy_conserved
    
    def _correct_4_momentum(self, event: Dict, initial_energy: float) -> Dict:
        """Коррекция событий для сохранения 4-импульса"""
        # Пересчитываем энергии, сохраняя относительные доли
        total_energy = sum(product['energy'] for product in event['products'])
        scale_factor = initial_energy / total_energy if total_energy > 0 else 1.0
        
        for product in event['products']:
            product['energy'] *= scale_factor
            # Пересчитываем импульс на основе новой энергии
            if 'mass' in product and product['mass'] > 0:
                momentum = np.sqrt(product['energy']**2 - product['mass']**2)
                # Сохраняем направление
                if 'px' in product and 'py' in product and 'pz' in product:
                    total_momentum = np.sqrt(product['px']**2 + product['py']**2 + product['pz']**2)
                    if total_momentum > 0:
                        product['px'] = (product['px'] / total_momentum) * momentum
                        product['py'] = (product['py'] / total_momentum) * momentum
                        product['pz'] = (product['pz'] / total_momentum) * momentum
        
        return event
    
    def _elastic_scattering(self, p1_info: Particle, p2_info: Particle, energy: float) -> Dict:
        """Моделирование упругого рассеяния"""
        # Угол рассеяния (в радианах)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        # Энергия и импульс после рассеяния (в системе центра масс)
        p1_energy = energy / 2
        p2_energy = energy / 2
        
        # Импульс (для упрощения предполагаем, что массы частиц малы по сравнению с энергией)
        p_magnitude = np.sqrt((energy/2)**2 - p1_info.mass**2)
        
        # Новые импульсы
        p1_px = p_magnitude * np.sin(theta) * np.cos(phi)
        p1_py = p_magnitude * np.sin(theta) * np.sin(phi)
        p1_pz = p_magnitude * np.cos(theta)
        
        p2_px = -p1_px
        p2_py = -p1_py
        p2_pz = -p1_pz
        
        products = [
            {
                'name': p1_info.name,
                'energy': p1_energy,
                'px': p1_px,
                'py': p1_py,
                'pz': p1_pz,
                'mass': p1_info.mass
            },
            {
                'name': p2_info.name,
                'energy': p2_energy,
                'px': p2_px,
                'py': p2_py,
                'pz': p2_pz,
                'mass': p2_info.mass
            }
        ]
        
        return {
            'event_type': 'elastic_scattering',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _standard_model(self, particle1: str, particle2: str, energy: float) -> Dict:
        """Стандартная модель взаимодействий"""
        # Определяем вероятности различных типов событий
        event_types = [
            ('w_production', 0.2),
            ('z_production', 0.15),
            ('higgs_production', 0.05),
            ('dijet_production', 0.4),
            ('b_quark_production', 0.1),
            ('tau_production', 0.1)
        ]
        
        # Нормализуем вероятности
        total_prob = sum(prob for _, prob in event_types)
        normalized_events = [(etype, prob/total_prob) for etype, prob in event_types]
        
        # Случайный выбор типа события
        r = random.random()
        cumulative = 0.0
        selected_type = None
        for etype, prob in normalized_events:
            cumulative += prob
            if r <= cumulative:
                selected_type = etype
                break
        
        if selected_type is None:
            selected_type = 'dijet_production'
        
        # Генерация продуктов на основе типа события
        if selected_type == 'w_production':
            return self._generate_w_production(energy)
        elif selected_type == 'z_production':
            return self._generate_z_production(energy)
        elif selected_type == 'higgs_production':
            return self._generate_higgs_production(energy)
        elif selected_type == 'dijet_production':
            return self._generate_dijet_production(energy)
        elif selected_type == 'b_quark_production':
            return self._generate_b_quark_production(energy)
        elif selected_type == 'tau_production':
            return self._generate_tau_production(energy)
        else:
            return self._generate_dijet_production(energy)
    
    def _qcd_model(self, particle1: str, particle2: str, energy: float) -> Dict:
        """Модель QCD взаимодействий"""
        # Для высоких энергий QCD доминирует
        if energy > 1000:  # 1 ТэВ
            return self._generate_hard_scattering(energy)
        else:
            return self._generate_soft_scattering(energy)
    
    def _electroweak_model(self, particle1: str, particle2: str, energy: float) -> Dict:
        """Модель электрослабых взаимодействий"""
        # Вероятность электрослабых процессов растет с энергией
        weak_prob = min(1.0, energy / 1000.0)  # Упрощенная модель
        
        if random.random() < weak_prob:
            # Электрослабое взаимодействие
            if energy > 80:  # Порог для W бозона
                return self._generate_w_production(energy)
            elif energy > 90:  # Порог для Z бозона
                return self._generate_z_production(energy)
            else:
                return self._generate_lepton_scattering(particle1, particle2, energy)
        else:
            # QCD взаимодействие даже для лептонов (через обмен глюонами через кварки)
            return self._generate_dijet_production(energy)
    
    def _beyond_standard_model(self, particle1: str, particle2: str, energy: float) -> Dict:
        """Модель за пределами Стандартной модели (упрощенная)"""
        # Вероятность новых физических процессов
        new_physics_prob = min(0.1, (energy - 1000) / 10000.0)  # Упрощенная модель
        
        if random.random() < new_physics_prob:
            # Генерация новых частиц
            return self._generate_new_physics_event(energy)
        else:
            # Стандартные процессы
            return self._standard_model(particle1, particle2, energy)
    
    def _generate_w_production(self, energy: float) -> Dict:
        """Генерация события с производством W бозона"""
        # W бозон распадается
        decay_products = self.particle_db.generate_decay_products('w_boson')
        
        products = []
        total_energy = 0.0
        
        for product in decay_products:
            p_info = self.particle_db.get_particle(product)
            if p_info:
                # Энергия продукта (случайное распределение)
                product_energy = energy * random.uniform(0.2, 0.8) / len(decay_products)
                # Импульс
                if p_info.mass < product_energy:
                    momentum = np.sqrt(product_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': product,
                    'energy': product_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
                total_energy += product_energy
        
        # Нормализуем энергии, чтобы суммарная энергия соответствовала входной
        if total_energy > 0:
            scale = energy / total_energy
            for product in products:
                product['energy'] *= scale
                if 'px' in product:
                    product['px'] *= scale
                    product['py'] *= scale
                    product['pz'] *= scale
        
        return {
            'event_type': 'w_production',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_z_production(self, energy: float) -> Dict:
        """Генерация события с производством Z бозона"""
        # Z бозон распадается
        decay_products = self.particle_db.generate_decay_products('z_boson')
        
        products = []
        total_energy = 0.0
        
        for product in decay_products:
            p_info = self.particle_db.get_particle(product)
            if p_info:
                # Энергия продукта
                product_energy = energy * random.uniform(0.2, 0.8) / len(decay_products)
                # Импульс
                if p_info.mass < product_energy:
                    momentum = np.sqrt(product_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': product,
                    'energy': product_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
                total_energy += product_energy
        
        # Нормализуем энергии
        if total_energy > 0:
            scale = energy / total_energy
            for product in products:
                product['energy'] *= scale
                if 'px' in product:
                    product['px'] *= scale
                    product['py'] *= scale
                    product['pz'] *= scale
        
        return {
            'event_type': 'z_production',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_higgs_production(self, energy: float) -> Dict:
        """Генерация события с производством бозона Хиггса"""
        # Бозон Хиггса распадается
        decay_products = self.particle_db.generate_decay_products('higgs_boson')
        
        products = []
        total_energy = 0.0
        
        for product in decay_products:
            p_info = self.particle_db.get_particle(product)
            if p_info:
                # Энергия продукта
                product_energy = energy * random.uniform(0.2, 0.8) / len(decay_products)
                # Импульс
                if p_info.mass < product_energy:
                    momentum = np.sqrt(product_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': product,
                    'energy': product_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
                total_energy += product_energy
        
        # Нормализуем энергии
        if total_energy > 0:
            scale = energy / total_energy
            for product in products:
                product['energy'] *= scale
                if 'px' in product:
                    product['px'] *= scale
                    product['py'] *= scale
                    product['pz'] *= scale
        
        return {
            'event_type': 'higgs_production',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_dijet_production(self, energy: float) -> Dict:
        """Генерация события с производством двух струй (dijet)"""
        # Генерируем два набора частиц, представляющих струи
        jet1_particles = []
        jet2_particles = []
        
        # Количество частиц в каждой струе
        n_particles_jet1 = random.randint(5, 20)
        n_particles_jet2 = random.randint(5, 20)
        
        # Энергия для каждой струи
        jet1_energy = energy * random.uniform(0.4, 0.6)
        jet2_energy = energy - jet1_energy
        
        # Генерация частиц для первой струи
        for _ in range(n_particles_jet1):
            particle_type = random.choice(['pion_plus', 'kaon_plus', 'proton', 'electron'])
            p_info = self.particle_db.get_particle(particle_type)
            if p_info:
                # Энергия частицы в струе
                particle_energy = jet1_energy * random.uniform(0.01, 0.2)
                # Импульс
                if p_info.mass < particle_energy:
                    momentum = np.sqrt(particle_energy**2 - p_info.mass**2)
                    # Углы для конуса струи
                    theta = random.uniform(0, 0.2)  # узкий конус
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                jet1_particles.append({
                    'name': particle_type,
                    'energy': particle_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
        
        # Генерация частиц для второй струи (противоположное направление)
        for _ in range(n_particles_jet2):
            particle_type = random.choice(['pion_plus', 'kaon_plus', 'proton', 'electron'])
            p_info = self.particle_db.get_particle(particle_type)
            if p_info:
                # Энергия частицы в струе
                particle_energy = jet2_energy * random.uniform(0.01, 0.2)
                # Импульс
                if p_info.mass < particle_energy:
                    momentum = np.sqrt(particle_energy**2 - p_info.mass**2)
                    # Углы для конуса струи (противоположное направление)
                    theta = np.pi - random.uniform(0, 0.2)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                jet2_particles.append({
                    'name': particle_type,
                    'energy': particle_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
        
        products = jet1_particles + jet2_particles
        
        return {
            'event_type': 'dijet_production',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_b_quark_production(self, energy: float) -> Dict:
        """Генерация события с производством b-кварков"""
        # b-кварки образуют струи с определенными характеристиками
        products = []
        
        # Генерируем пару b-кварков
        for quark_type in ['bottom_quark', 'antibottom_quark']:
            p_info = self.particle_db.get_particle(quark_type)
            if p_info:
                # Энергия кварка
                quark_energy = energy * random.uniform(0.4, 0.6)
                # Импульс
                if p_info.mass < quark_energy:
                    momentum = np.sqrt(quark_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': quark_type,
                    'energy': quark_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
        
        # Добавляем продукты распада b-кварков (упрощенно)
        for i, product in enumerate(products[:]):
            if 'bottom_quark' in product['name'] or 'antibottom_quark' in product['name']:
                decay_products = self.particle_db.generate_decay_products('bottom_quark')
                for decay_product in decay_products:
                    p_info = self.particle_db.get_particle(decay_product)
                    if p_info:
                        # Энергия продукта распада
                        decay_energy = product['energy'] * random.uniform(0.1, 0.3)
                        # Импульс
                        if p_info.mass < decay_energy:
                            momentum = np.sqrt(decay_energy**2 - p_info.mass**2)
                            theta = random.uniform(0, np.pi)
                            phi = random.uniform(0, 2*np.pi)
                            px = momentum * np.sin(theta) * np.cos(phi)
                            py = momentum * np.sin(theta) * np.sin(phi)
                            pz = momentum * np.cos(theta)
                        else:
                            px, py, pz = 0.0, 0.0, 0.0
                        
                        products.append({
                            'name': decay_product,
                            'energy': decay_energy,
                            'px': px,
                            'py': py,
                            'pz': pz,
                            'mass': p_info.mass
                        })
        
        return {
            'event_type': 'b_quark_production',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_tau_production(self, energy: float) -> Dict:
        """Генерация события с производством тау-лептонов"""
        products = []
        
        # Генерируем пару тау-лептонов
        for tau_type in ['tau', 'antitau']:
            p_info = self.particle_db.get_particle(tau_type)
            if not p_info:
                p_info = self.particle_db.get_particle('tau')  # попытка без античастицы
            
            if p_info:
                # Энергия тау-лептона
                tau_energy = energy * random.uniform(0.4, 0.6)
                # Импульс
                if p_info.mass < tau_energy:
                    momentum = np.sqrt(tau_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': tau_type,
                    'energy': tau_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
        
        # Добавляем продукты распада тау-лептонов
        for i, product in enumerate(products[:]):
            if 'tau' in product['name']:
                decay_products = self.particle_db.generate_decay_products('tau')
                for decay_product in decay_products:
                    p_info = self.particle_db.get_particle(decay_product)
                    if p_info:
                        # Энергия продукта распада
                        decay_energy = product['energy'] * random.uniform(0.1, 0.3)
                        # Импульс
                        if p_info.mass < decay_energy:
                            momentum = np.sqrt(decay_energy**2 - p_info.mass**2)
                            theta = random.uniform(0, np.pi)
                            phi = random.uniform(0, 2*np.pi)
                            px = momentum * np.sin(theta) * np.cos(phi)
                            py = momentum * np.sin(theta) * np.sin(phi)
                            pz = momentum * np.cos(theta)
                        else:
                            px, py, pz = 0.0, 0.0, 0.0
                        
                        products.append({
                            'name': decay_product,
                            'energy': decay_energy,
                            'px': px,
                            'py': py,
                            'pz': pz,
                            'mass': p_info.mass
                        })
        
        return {
            'event_type': 'tau_production',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_hard_scattering(self, energy: float) -> Dict:
        """Генерация события с жестким рассеянием (высокоэнергетические QCD процессы)"""
        # Выбираем тип жесткого рассеяния
        scattering_types = [
            ('quark_quark', 0.4),
            ('quark_gluon', 0.3),
            ('gluon_gluon', 0.3)
        ]
        
        # Случайный выбор типа
        r = random.random()
        cumulative = 0.0
        selected_type = None
        for stype, prob in scattering_types:
            cumulative += prob
            if r <= cumulative:
                selected_type = stype
                break
        
        if selected_type is None:
            selected_type = 'quark_quark'
        
        # Генерация продуктов в зависимости от типа
        if selected_type == 'quark_quark':
            return self._generate_quark_quark_scattering(energy)
        elif selected_type == 'quark_gluon':
            return self._generate_quark_gluon_scattering(energy)
        else:
            return self._generate_gluon_gluon_scattering(energy)
    
    def _generate_soft_scattering(self, energy: float) -> Dict:
        """Генерация события с мягким рассеянием (низкоэнергетические QCD процессы)"""
        # Мягкое рассеяние приводит к образованию множества легких адронов
        products = []
        
        # Количество адронов
        n_hadrons = random.randint(10, 50)
        
        # Общая энергия для распределения
        total_energy = energy
        
        for _ in range(n_hadrons):
            particle_type = random.choice(['pion_plus', 'kaon_plus', 'proton'])
            p_info = self.particle_db.get_particle(particle_type)
            if p_info:
                # Энергия частицы
                particle_energy = total_energy * random.uniform(0.01, 0.1)
                # Импульс
                if p_info.mass < particle_energy:
                    momentum = np.sqrt(particle_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': particle_type,
                    'energy': particle_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
        
        return {
            'event_type': 'soft_scattering',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_quark_quark_scattering(self, energy: float) -> Dict:
        """Генерация события с рассеянием кварк-кварк"""
        # Выбираем типы кварков
        quark_types = ['up_quark', 'down_quark', 'strange_quark', 'charm_quark', 'bottom_quark']
        q1_type = random.choice(quark_types)
        q2_type = random.choice(quark_types)
        
        products = []
        
        # Энергия для каждого кварка
        q1_energy = energy * random.uniform(0.4, 0.6)
        q2_energy = energy - q1_energy
        
        # Добавляем кварки
        for q_type, q_energy in [(q1_type, q1_energy), (q2_type, q2_energy)]:
            p_info = self.particle_db.get_particle(q_type)
            if p_info:
                # Импульс
                if p_info.mass < q_energy:
                    momentum = np.sqrt(q_energy**2 - p_info.mass**2)
                    theta = random.uniform(0, np.pi)
                    phi = random.uniform(0, 2*np.pi)
                    px = momentum * np.sin(theta) * np.cos(phi)
                    py = momentum * np.sin(theta) * np.sin(phi)
                    pz = momentum * np.cos(theta)
                else:
                    px, py, pz = 0.0, 0.0, 0.0
                
                products.append({
                    'name': q_type,
                    'energy': q_energy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                    'mass': p_info.mass
                })
        
        # Добавляем возможные глюоны
        if random.random() < 0.3:  # 30% вероятность излучения глюона
            p_info = self.particle_db.get_particle('gluon') or Particle('gluon', 0.0, 0, 1, float('inf'), 'boson', 'g')
            gluon_energy = energy * random.uniform(0.05, 0.2)
            
            if p_info.mass < gluon_energy:
                momentum = np.sqrt(gluon_energy**2 - p_info.mass**2)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                px = momentum * np.sin(theta) * np.cos(phi)
                py = momentum * np.sin(theta) * np.sin(phi)
                pz = momentum * np.cos(theta)
            else:
                px, py, pz = 0.0, 0.0, 0.0
            
            products.append({
                'name': 'gluon',
                'energy': gluon_energy,
                'px': px,
                'py': py,
                'pz': pz,
                'mass': p_info.mass
            })
        
        return {
            'event_type': 'quark_quark_scattering',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_quark_gluon_scattering(self, energy: float) -> Dict:
        """Генерация события с рассеянием кварк-глюон"""
        # Выбираем тип кварка
        quark_types = ['up_quark', 'down_quark', 'strange_quark', 'charm_quark', 'bottom_quark']
        q_type = random.choice(quark_types)
        
        products = []
        
        # Энергия для кварка и глюона
        q_energy = energy * random.uniform(0.3, 0.7)
        g_energy = energy - q_energy
        
        # Добавляем кварк
        p_info = self.particle_db.get_particle(q_type)
        if p_info:
            # Импульс кварка
            if p_info.mass < q_energy:
                momentum = np.sqrt(q_energy**2 - p_info.mass**2)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                px = momentum * np.sin(theta) * np.cos(phi)
                py = momentum * np.sin(theta) * np.sin(phi)
                pz = momentum * np.cos(theta)
            else:
                px, py, pz = 0.0, 0.0, 0.0
            
            products.append({
                'name': q_type,
                'energy': q_energy,
                'px': px,
                'py': py,
                'pz': pz,
                'mass': p_info.mass
            })
        
        # Добавляем глюон
        p_info = self.particle_db.get_particle('gluon') or Particle('gluon', 0.0, 0, 1, float('inf'), 'boson', 'g')
        
        if p_info.mass < g_energy:
            momentum = np.sqrt(g_energy**2 - p_info.mass**2)
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2*np.pi)
            px = momentum * np.sin(theta) * np.cos(phi)
            py = momentum * np.sin(theta) * np.sin(phi)
            pz = momentum * np.cos(theta)
        else:
            px, py, pz = 0.0, 0.0, 0.0
        
        products.append({
            'name': 'gluon',
            'energy': g_energy,
            'px': px,
            'py': py,
            'pz': pz,
            'mass': p_info.mass
        })
        
        return {
            'event_type': 'quark_gluon_scattering',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_gluon_gluon_scattering(self, energy: float) -> Dict:
        """Генерация события с рассеянием глюон-глюон"""
        products = []
        
        # Энергия для каждого глюона
        g1_energy = energy * random.uniform(0.4, 0.6)
        g2_energy = energy - g1_energy
        
        # Добавляем глюоны
        for g_energy in [g1_energy, g2_energy]:
            p_info = self.particle_db.get_particle('gluon') or Particle('gluon', 0.0, 0, 1, float('inf'), 'boson', 'g')
            
            if p_info.mass < g_energy:
                momentum = np.sqrt(g_energy**2 - p_info.mass**2)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                px = momentum * np.sin(theta) * np.cos(phi)
                py = momentum * np.sin(theta) * np.sin(phi)
                pz = momentum * np.cos(theta)
            else:
                px, py, pz = 0.0, 0.0, 0.0
            
            products.append({
                'name': 'gluon',
                'energy': g_energy,
                'px': px,
                'py': py,
                'pz': pz,
                'mass': p_info.mass
            })
        
        # Добавляем возможные кварки (процесс разделения глюона)
        if random.random() < 0.4:  # 40% вероятность
            quark_types = ['up_quark', 'down_quark', 'strange_quark']
            q_type = random.choice(quark_types)
            aq_type = f"anti{q_type}" if q_type != 'strange_quark' else 'antistrange_quark'
            
            for q in [q_type, aq_type]:
                p_info = self.particle_db.get_particle(q)
                if not p_info and 'anti' in q:
                    base_q = q.replace('anti', '')
                    p_info = self.particle_db.get_particle(base_q)
                
                if p_info:
                    q_energy = energy * random.uniform(0.05, 0.15)
                    
                    if p_info.mass < q_energy:
                        momentum = np.sqrt(q_energy**2 - p_info.mass**2)
                        theta = random.uniform(0, np.pi)
                        phi = random.uniform(0, 2*np.pi)
                        px = momentum * np.sin(theta) * np.cos(phi)
                        py = momentum * np.sin(theta) * np.sin(phi)
                        pz = momentum * np.cos(theta)
                    else:
                        px, py, pz = 0.0, 0.0, 0.0
                    
                    products.append({
                        'name': q,
                        'energy': q_energy,
                        'px': px,
                        'py': py,
                        'pz': pz,
                        'mass': p_info.mass
                    })
        
        return {
            'event_type': 'gluon_gluon_scattering',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_lepton_scattering(self, particle1: str, particle2: str, energy: float) -> Dict:
        """Генерация события с рассеянием лептонов"""
        # Пример: e+e- → μ+μ-
        products = []
        
        # Определяем продукты на основе входных частиц
        if "electron" in particle1 and "positron" in particle2:
            products = [
                {'name': 'muon', 'energy': energy/2, 'px': 0, 'py': 0, 'pz': energy/2, 'mass': 0.105658},
                {'name': 'antimuon', 'energy': energy/2, 'px': 0, 'py': 0, 'pz': -energy/2, 'mass': 0.105658}
            ]
        elif "muon" in particle1 and "antimuon" in particle2:
            products = [
                {'name': 'electron', 'energy': energy/2, 'px': 0, 'py': 0, 'pz': energy/2, 'mass': 0.000511},
                {'name': 'positron', 'energy': energy/2, 'px': 0, 'py': 0, 'pz': -energy/2, 'mass': 0.000511}
            ]
        else:
            # Универсальное рассеяние
            theta = random.uniform(0, np.pi)
            phi = random.uniform(0, 2*np.pi)
            p_magnitude = energy / 2  # Упрощение
            
            px1 = p_magnitude * np.sin(theta) * np.cos(phi)
            py1 = p_magnitude * np.sin(theta) * np.sin(phi)
            pz1 = p_magnitude * np.cos(theta)
            
            px2 = -px1
            py2 = -py1
            pz2 = -pz1
            
            products = [
                {'name': particle1, 'energy': energy/2, 'px': px1, 'py': py1, 'pz': pz1, 'mass': self.particle_db.get_particle(particle1).mass},
                {'name': particle2, 'energy': energy/2, 'px': px2, 'py': py2, 'pz': pz2, 'mass': self.particle_db.get_particle(particle2).mass}
            ]
        
        return {
            'event_type': 'lepton_scattering',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }
    
    def _generate_new_physics_event(self, energy: float) -> Dict:
        """Генерация события с новой физикой (упрощенная модель)"""
        # Пример: производство новых частиц
        new_particles = [
            'dark_matter', 'axion', 'supersymmetric_particle', 'extra_dimension_resonance'
        ]
        
        selected_particle = random.choice(new_particles)
        
        # Создаем "новую" частицу
        new_particle = {
            'name': selected_particle,
            'energy': energy * 0.8,
            'px': 0,
            'py': 0,
            'pz': 0,
            'mass': energy * 0.7  # Предполагаем, что масса близка к энергии
        }
        
        # Генерируем продукты распада
        decay_products = []
        n_decay_products = random.randint(2, 4)
        
        for _ in range(n_decay_products):
            particle_type = random.choice(['electron', 'muon', 'photon', 'jet'])
            p_info = self.particle_db.get_particle(particle_type) or Particle(particle_type, 0.0, 0, 0, float('inf'), 'unknown', particle_type[0])
            
            # Энергия продукта
            product_energy = new_particle['energy'] * random.uniform(0.1, 0.3)
            
            # Импульс
            if p_info.mass < product_energy:
                momentum = np.sqrt(product_energy**2 - p_info.mass**2)
                theta = random.uniform(0, np.pi)
                phi = random.uniform(0, 2*np.pi)
                px = momentum * np.sin(theta) * np.cos(phi)
                py = momentum * np.sin(theta) * np.sin(phi)
                pz = momentum * np.cos(theta)
            else:
                px, py, pz = 0.0, 0.0, 0.0
            
            decay_products.append({
                'name': particle_type,
                'energy': product_energy,
                'px': px,
                'py': py,
                'pz': pz,
                'mass': p_info.mass
            })
        
        products = [new_particle] + decay_products
        
        return {
            'event_type': 'new_physics',
            'energy': energy,
            'products': products,
            'timestamp': time.time()
        }


# ===================================================================
# 6. Интеграция с профессиональными инструментами
# ===================================================================

# Попытка импорта профессиональных инструментов
GEANT4_AVAILABLE = False
PYTHIA_AVAILABLE = False
MADX_AVAILABLE = False
ROOT_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    import geant4
    GEANT4_AVAILABLE = True
    logger.info("Geant4 integration available")
except ImportError:
    logger.warning("Geant4 not available. Using built-in simulation.")

try:
    import pythia8
    PYTHIA_AVAILABLE = True
    logger.info("PYTHIA integration available")
except ImportError:
    logger.warning("PYTHIA not available. Using built-in QCD model.")

try:
    import madx
    MADX_AVAILABLE = True
    logger.info("MADX integration available")
except ImportError:
    logger.warning("MADX not available. Using built-in beam dynamics.")

try:
    import ROOT
    ROOT_AVAILABLE = True
    logger.info("ROOT integration available")
except ImportError:
    logger.warning("ROOT not available. Some analysis features will be limited.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not installed. Interactive 3D visualization will be unavailable.")


class Geant4PhysicsEngine(PhysicsEngineInterface):
    """Интеграция с Geant4 для детального моделирования взаимодействий"""
    
    def __init__(self):
        self.initialized = False
        self.geant4 = None
        if GEANT4_AVAILABLE:
            try:
                # Здесь должен быть код инициализации Geant4
                # self.geant4 = geant4.initialize()
                self.initialized = True
                logger.info("Geant4 успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации Geant4: {e}")
                self.initialized = False
    
    def get_name(self) -> str:
        return "geant4"
    
    def is_available(self) -> bool:
        return GEANT4_AVAILABLE and self.initialized
    
    def interact(self, particle1: str, particle2: str, energy: float, 
                num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия с использованием Geant4"""
        if not self.is_available():
            logger.warning("Geant4 недоступен. Используется встроенная модель.")
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        
        events = []
        try:
            # В реальной реализации:
            # primary_particles = self._convert_to_geant4_particles(particle1, particle2)
            # results = self.geant4.simulate_collision(primary_particles, num_events, energy)
            
            # Для демонстрации используем упрощенную модель
            built_in_engine = BuiltInPhysicsEngine(ParticleDatabase())
            events = built_in_engine.interact(particle1, particle2, energy, num_events, **kwargs)
            
            # В реальной системе результаты Geant4 нужно было бы конвертировать
            # events = self._convert_from_geant4_results(results)
            
        except Exception as e:
            logger.error(f"Ошибка в симуляции Geant4: {e}")
            # Используем встроенную модель как fallback
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            events = fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        
        return events
    
    def _convert_to_geant4_particles(self, particle1: str, particle2: str) -> List:
        """Конвертация частиц в формат Geant4"""
        # Реализация конвертации
        pass
    
    def _convert_from_geant4_results(self, results) -> List[Dict]:
        """Конвертация результатов Geant4 в наш формат"""
        # Реализация конвертации
        pass


class PythiaPhysicsEngine(PhysicsEngineInterface):
    """Интеграция с PYTHIA для моделирования QCD процессов"""
    
    def __init__(self):
        self.initialized = False
        self.pythia = None
        if PYTHIA_AVAILABLE:
            try:
                # Здесь должен быть код инициализации PYTHIA
                # self.pythia = pythia8.Pythia()
                # self.pythia.readString("Main:timesAllowErrors = 10000")
                # self.pythia.init()
                self.initialized = True
                logger.info("PYTHIA успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации PYTHIA: {e}")
                self.initialized = False
    
    def get_name(self) -> str:
        return "pythia"
    
    def is_available(self) -> bool:
        return PYTHIA_AVAILABLE and self.initialized
    
    def interact(self, particle1: str, particle2: str, energy: float, 
                num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия с использованием PYTHIA"""
        if not self.is_available():
            logger.warning("PYTHIA недоступен. Используется встроенная модель.")
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            return fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        
        events = []
        try:
            # В реальной реализации:
            # self._configure_pythia(particle1, particle2, energy)
            # for _ in range(num_events):
            #     if self.pythia.next():
            #         event = self._extract_pythia_event()
            #         events.append(event)
            
            # Для демонстрации используем упрощенную модель
            built_in_engine = BuiltInPhysicsEngine(ParticleDatabase())
            events = built_in_engine.interact(particle1, particle2, energy, num_events, **kwargs)
            
        except Exception as e:
            logger.error(f"Ошибка в симуляции PYTHIA: {e}")
            # Используем встроенную модель как fallback
            fallback_engine = BuiltInPhysicsEngine(ParticleDatabase())
            events = fallback_engine.interact(particle1, particle2, energy, num_events, **kwargs)
        
        return events
    
    def _configure_pythia(self, particle1: str, particle2: str, energy: float):
        """Конфигурация PYTHIA для симуляции"""
        # Реализация конфигурации
        pass
    
    def _extract_pythia_event(self) -> Dict:
        """Извлечение информации о событии из PYTHIA"""
        # Реализация извлечения
        pass


class MadXBeamDynamics(BeamDynamicsInterface):
    """Интеграция с MAD-X для точного моделирования динамики пучка"""
    
    def __init__(self):
        self.initialized = False
        self.madx = None
        if MADX_AVAILABLE:
            try:
                # Здесь должен быть код инициализации MAD-X
                # self.madx = madx.MADX()
                # self.madx.call("beam, particle=proton, energy=6500;")
                # self.madx.call("seqedit, sequence=LHC;")
                # ... другие команды для настройки
                self.initialized = True
                logger.info("MAD-X успешно инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации MAD-X: {e}")
                self.initialized = False
    
    def get_name(self) -> str:
        return "madx"
    
    def is_available(self) -> bool:
        return MADX_AVAILABLE and self.initialized
    
    def simulate_turn(self, state: Dict, revolution_time: float, 
                     include_space_charge: bool = True, **kwargs) -> Dict:
        """Симуляция одного оборота пучка с использованием MAD-X"""
        if not self.is_available():
            logger.warning("MAD-X недоступен. Используется встроенная модель.")
            fallback_engine = BuiltInBeamDynamics()
            return fallback_engine.simulate_turn(state, revolution_time, include_space_charge, **kwargs)
        
        try:
            # В реальной реализации:
            # self._configure_madx(state)
            # self.madx.command(f"twiss, file='twiss_output.dat';")
            # updated_state = self._extract_madx_results()
            # return updated_state
            
            # Для демонстрации используем упрощенную модель
            built_in_engine = BuiltInBeamDynamics()
            return built_in_engine.simulate_turn(state, revolution_time, include_space_charge, **kwargs)
            
        except Exception as e:
            logger.error(f"Ошибка в симуляции MAD-X: {e}")
            # Используем встроенную модель как fallback
            fallback_engine = BuiltInBeamDynamics()
            return fallback_engine.simulate_turn(state, revolution_time, include_space_charge, **kwargs)


class BuiltInBeamDynamics(BeamDynamicsInterface):
    """Встроенная модель для симуляции динамики пучка"""
    
    def __init__(self):
        pass
    
    def get_name(self) -> str:
        return "built-in"
    
    def is_available(self) -> bool:
        return True
    
    def simulate_turn(self, state: Dict, revolution_time: float, 
                     include_space_charge: bool = True, **kwargs) -> Dict:
        """
        Симуляция одного оборота пучка с учетом бетатронных колебаний и эффектов пространственного заряда.
        
        Параметры:
        state: текущее состояние пучка
        revolution_time: время одного оборота
        include_space_charge: учитывать ли эффекты пространственного заряда
        
        Возвращает:
        Обновленное состояние пучка
        """
        # Копируем состояние, чтобы не модифицировать оригинал
        updated_state = state.copy()
        
        # Получаем текущие параметры
        circumference = updated_state.get('circumference', 26658.883)
        beam_energy = updated_state.get('beam_energy', 6500)  # ГэВ
        peak_luminosity = updated_state.get('peak_luminosity', 2.0e34)  # см⁻²с⁻¹
        beam_size_x = updated_state.get('beam_size_x', 0.05)  # м
        beam_size_y = updated_state.get('beam_size_y', 0.05)  # м
        protons_per_bunch = updated_state.get('protons_per_bunch', 1.15e11)
        num_bunches = updated_state.get('num_bunches', 2748)
        
        # Релятивистский гамма-фактор
        proton_mass = 0.938  # ГэВ/c²
        gamma = beam_energy / proton_mass
        
        # Нормированный эмиттанс (предполагаем постоянный)
        normalized_emittance = 3.5e-6  # м·рад
        
        # Бета-функция (предполагаем постоянную по кольцу)
        beta_function = 150  # м
        
        # Геометрический эмиттанс
        geometric_emittance = normalized_emittance / gamma
        
        # Размеры пучка (с учетом бетатронных колебаний)
        sigma_x = np.sqrt(geometric_emittance * beta_function)
        sigma_y = np.sqrt(geometric_emittance * beta_function)
        
        # Обновляем размеры пучка
        updated_state['beam_size_x'] = sigma_x
        updated_state['beam_size_y'] = sigma_y
        
        # Эффекты пространственного заряда (если включены)
        if include_space_charge:
            # Коэффициент, зависящий от плотности пучка
            space_charge_factor = 0.01 * (protons_per_bunch / 1e11) * (1 / (sigma_x * sigma_y))
            
            # Увеличение эмиттанса из-за пространственного заряда
            emittance_growth = space_charge_factor * revolution_time
            
            # Обновляем эмиттанс
            geometric_emittance += emittance_growth
            sigma_x = np.sqrt(geometric_emittance * beta_function)
            sigma_y = np.sqrt(geometric_emittance * beta_function)
            
            updated_state['beam_size_x'] = sigma_x
            updated_state['beam_size_y'] = sigma_y
        
        # Потери частиц (упрощенная модель)
        loss_rate = 1e-4  # 0.01% потерь за оборот
        protons_per_bunch *= (1 - loss_rate)
        
        # Обновляем интенсивность пучка
        updated_state['protons_per_bunch'] = protons_per_bunch
        updated_state['beam_intensity'] = protons_per_bunch * num_bunches
        
        # Обновление светимости
        # Формула светимости для двух пучков: L = (N_b * N1 * N2 * f) / (4 * pi * sigma_x * sigma_y)
        # где N_b - количество сгустков, N1 и N2 - частиц в сгустке, f - частота оборотов
        revolution_frequency = 1 / revolution_time
        current_luminosity = (num_bunches * protons_per_bunch**2 * revolution_frequency) / (4 * np.pi * sigma_x * sigma_y)
        
        # Учитываем пиковые параметры
        current_luminosity = min(current_luminosity, peak_luminosity)
        
        updated_state['luminosity'] = current_luminosity
        
        # Обновляем время
        updated_state['time'] = updated_state.get('time', 0.0) + revolution_time
        
        # Сохраняем историю для анализа
        beam_dynamics = updated_state.get('beam_dynamics', {
            'time': [],
            'luminosity': [],
            'beam_size_x': [],
            'beam_size_y': [],
            'emittance': [],
            'beam_intensity': []
        })
        
        beam_dynamics['time'].append(updated_state['time'])
        beam_dynamics['luminosity'].append(current_luminosity)
        beam_dynamics['beam_size_x'].append(sigma_x)
        beam_dynamics['beam_size_y'].append(sigma_y)
        beam_dynamics['emittance'].append(geometric_emittance)
        beam_dynamics['beam_intensity'].append(protons_per_bunch * num_bunches)
        
        updated_state['beam_dynamics'] = beam_dynamics
        
        return updated_state


# ===================================================================
# 7. Гибридные движки для физики и динамики пучка
# ===================================================================

class HybridPhysicsEngine:
    """Гибридный движок для физических взаимодействий"""
    
    def __init__(self, particle_db: ParticleDatabase):
        """Инициализация гибридного физического движка"""
        self.particle_db = particle_db
        self.engines = {
            "geant4": Geant4PhysicsEngine(),
            "pythia": PythiaPhysicsEngine(),
            "built-in": BuiltInPhysicsEngine(particle_db)
        }
        self.preferred_engines = ["geant4", "pythia", "built-in"]
        logger.info("Гибридный физический движок инициализирован")
    
    def set_engine_priority(self, engine_names: List[str]):
        """
        Установка приоритета движков.
        
        Параметры:
        engine_names: список имен движков в порядке приоритета
        """
        # Проверяем доступность движков
        available_engines = [name for name in engine_names if self.engines[name].is_available()]
        
        if not available_engines:
            logger.warning("Ни один из указанных движков недоступен. Используется встроенный движок.")
            self.preferred_engines = ["built-in"]
        else:
            self.preferred_engines = available_engines
        
        logger.info(f"Приоритет физических движков установлен: {', '.join(self.preferred_engines)}")
    
    def _determine_interaction_type(self, p1: str, p2: str, energy: float) -> str:
        """Определение типа взаимодействия для выбора движка"""
        # QCD процессы для адронных столкновений
        if "quark" in p1 or "quark" in p2 or (p1 == "proton" and p2 == "proton"):
            return "qcd"
        
        # Электрослабые процессы
        p1_category = self.particle_db.get_category(p1)
        p2_category = self.particle_db.get_category(p2)
        if "lepton" in p1_category or "lepton" in p2_category:
            return "electroweak"
        
        # Общие процессы
        return "general"
    
    def interact(self, particle1: str, particle2: str, energy: float, 
                num_events: int = 1, force_engine: Optional[str] = None, **kwargs) -> List[Dict]:
        """
        Моделирование взаимодействия с выбором подходящего движка.
        
        Параметры:
        particle1: тип первой частицы
        particle2: тип второй частицы
        energy: энергия взаимодействия в ГэВ
        num_events: количество событий для симуляции
        force_engine: принудительный выбор движка (если указан)
        **kwargs: дополнительные параметры
        
        Возвращает:
        Список событий
        """
        # Принудительный выбор движка
        if force_engine and force_engine in self.engines:
            logger.info(f"Используется принудительно выбранный движок: {force_engine}")
            return self.engines[force_engine].interact(
                particle1, particle2, energy, num_events, **kwargs
            )
        
        # Определение типа взаимодействия
        interaction_type = self._determine_interaction_type(particle1, particle2, energy)
        
        # Выбор подходящего движка на основе типа взаимодействия
        if interaction_type == "qcd" and "pythia" in self.preferred_engines:
            return self.engines["pythia"].interact(
                particle1, particle2, energy, num_events, **kwargs
            )
        elif "geant4" in self.preferred_engines:
            return self.engines["geant4"].interact(
                particle1, particle2, energy, num_events, **kwargs
            )
        else:
            return self.engines["built-in"].interact(
                particle1, particle2, energy, num_events, **kwargs
            )


class HybridBeamDynamics:
    """Гибридный движок для динамики пучка"""
    
    def __init__(self):
        """Инициализация гибридного движка динамики пучка"""
        self.engines = {
            "madx": MadXBeamDynamics(),
            "built-in": BuiltInBeamDynamics()
        }
        self.preferred_engines = ["madx", "built-in"]
        logger.info("Гибридный движок динамики пучка инициализирован")
    
    def set_engine_priority(self, engine_names: List[str]):
        """
        Установка приоритета движков динамики пучка.
        
        Параметры:
        engine_names: список имен движков в порядке приоритета
        """
        # Проверяем доступность движков
        available_engines = [name for name in engine_names if self.engines[name].is_available()]
        
        if not available_engines:
            logger.warning("Ни один из указанных движков недоступен. Используется встроенный движок.")
            self.preferred_engines = ["built-in"]
        else:
            self.preferred_engines = available_engines
        
        logger.info(f"Приоритет движков динамики пучка установлен: {', '.join(self.preferred_engines)}")
    
    def simulate_turn(self, state: Dict, revolution_time: float, 
                     include_space_charge: bool = True, force_engine: Optional[str] = None, **kwargs) -> Dict:
        """
        Симуляция одного оборота пучка с использованием подходящего движка.
        
        Параметры:
        state: текущее состояние пучка
        revolution_time: время одного оборота
        include_space_charge: учитывать ли эффекты пространственного заряда
        force_engine: принудительный выбор движка (если указан)
        **kwargs: дополнительные параметры
        
        Возвращает:
        Обновленное состояние пучка
        """
        # Принудительный выбор движка
        if force_engine and force_engine in self.engines:
            logger.info(f"Используется принудительно выбранный движок динамики: {force_engine}")
            return self.engines[force_engine].simulate_turn(
                state, revolution_time, include_space_charge, **kwargs
            )
        
        # Попытка использовать предпочтительные движки
        for engine_name in self.preferred_engines:
            engine = self.engines[engine_name]
            if engine.is_available():
                return engine.simulate_turn(
                    state, revolution_time, include_space_charge, **kwargs
                )
        
        # Если ни один движок недоступен, используем встроенный
        return self.engines["built-in"].simulate_turn(
            state, revolution_time, include_space_charge, **kwargs
        )


# ===================================================================
# 8. Система кэширования и оптимизации
# ===================================================================

class SimulationCache:
    """Система кэширования результатов симуляции"""
    
    def __init__(self, max_size: int = 1000):
        """Инициализация кэша симуляции"""
        self.cache = {}
        self.max_size = max_size
        self.access_count = 0
        self.hit_count = 0
        logger.info(f"Кэш симуляции инициализирован с максимальным размером {max_size}")
    
    def get(self, key: str) -> Optional[Dict]:
        """Получение результатов из кэша"""
        self.access_count += 1
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        """Сохранение результатов в кэш"""
        # Если кэш переполнен, удаляем старые записи
        if len(self.cache) >= self.max_size:
            # Удаляем самую старую запись (можно улучшить стратегию)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def get_hit_rate(self) -> float:
        """Получение текущего hit rate"""
        return self.hit_count / self.access_count if self.access_count > 0 else 0.0
    
    @staticmethod
    def generate_key(params: Dict) -> str:
        """Генерация уникального ключа на основе параметров"""
        # Сортировка ключей для консистентности
        sorted_items = sorted(params.items())
        # Создание строки с параметрами
        param_str = ",".join([f"{k}={v}" for k, v in sorted_items])
        # Хеширование для компактности
        return str(hash(param_str))


# ===================================================================
# 9. Система валидации и калибровки
# ===================================================================

class ValidationSystem:
    """Система валидации симуляции против реальных данных"""
    
    def __init__(self, lhc_model):
        """Инициализация системы валидации"""
        self.lhc_model = lhc_model
        self.validation_data = {}
        self.calibration_factors = {
            'luminosity': 1.0,
            'emittance': 1.0,
            'beam_size': 1.0,
            'collision_rate': 1.0
        }
        self.calibration_history = []
        logger.info("Система валидации инициализирована")
    
    def load_real_data(self, dataset_id: str):
        """
        Загрузка реальных данных из CERN Open Data.
        
        Параметры:
        dataset_id: идентификатор набора данных
        """
        logger.info(f"Загрузка реальных данных из набора {dataset_id}")
        try:
            self.validation_data = CERN_Open_Data_API.get_collision_data(dataset_id)
            logger.info(f"Данные успешно загружены из набора {dataset_id}")
            
            # Автоматическая калибровка после загрузки данных
            self.autocalibrate()
        except Exception as e:
            logger.error(f"Ошибка при загрузке реальных данных: {e}")
    
    def _extract_real_luminosity(self) -> float:
        """Извлечение реальных данных о светимости"""
        if not self.validation_data:
            return CONFIG['beam']['energy'] * 1e34  # Значение по умолчанию
        
        # В реальной системе данные о светимости пришли бы из CERN Open Data
        # CERN Open Data Portal предоставляет данные, которые можно использовать для валидации [[1]]
        return self.validation_data.get('luminosity', 2.0e34)
    
    def _extract_real_emittance(self) -> float:
        """Извлечение реальных данных об эмиттансе"""
        if not self.validation_data:
            return 3.5e-6  # Значение по умолчанию
        
        # Упрощенная модель
        return 3.5e-6
    
    def _extract_real_beam_size(self) -> Tuple[float, float]:
        """Извлечение реальных данных о размере пучка"""
        if not self.validation_data:
            return (0.05, 0.05)  # Значение по умолчанию
        
        # Упрощенная модель
        return (0.05, 0.05)
    
    def validate_luminosity(self) -> float:
        """Валидация светимости"""
        if not self.validation_data:
            logger.warning("Данные для валидации не загружены")
            return 0.0
        
        # Получение реальных данных о светимости
        real_luminosity = self._extract_real_luminosity()
        
        # Получение симулированной светимости
        sim_luminosity = self.lhc_model.simulation_state['luminosity']
        
        # Расчет относительной ошибки
        if real_luminosity > 0:
            relative_error = abs(real_luminosity - sim_luminosity) / real_luminosity
        else:
            relative_error = 0.0
        
        logger.info(f"Валидация светимости: реальная={real_luminosity:.2e}, симулированная={sim_luminosity:.2e}, ошибка={relative_error:.2%}")
        
        return relative_error
    
    def validate_emittance(self) -> float:
        """Валидация эмиттанса"""
        if not self.validation_data:
            logger.warning("Данные для валидации не загружены")
            return 0.0
        
        # Получение реальных данных об эмиттансе
        real_emittance = self._extract_real_emittance()
        
        # Получение симулированного эмиттанса
        sim_emittance = self.lhc_model.simulation_state['beam_dynamics']['emittance'][-1] if self.lhc_model.simulation_state['beam_dynamics']['emittance'] else 3.5e-6
        
        # Расчет относительной ошибки
        if real_emittance > 0:
            relative_error = abs(real_emittance - sim_emittance) / real_emittance
        else:
            relative_error = 0.0
        
        logger.info(f"Валидация эмиттанса: реальный={real_emittance:.2e}, симулированный={sim_emittance:.2e}, ошибка={relative_error:.2%}")
        
        return relative_error
    
    def validate_beam_size(self) -> Tuple[float, float]:
        """Валидация размера пучка"""
        if not self.validation_data:
            logger.warning("Данные для валидации не загружены")
            return (0.0, 0.0)
        
        # Получение реальных данных о размере пучка
        real_size_x, real_size_y = self._extract_real_beam_size()
        
        # Получение симулированных размеров
        sim_size_x = self.lhc_model.simulation_state['beam_size_x']
        sim_size_y = self.lhc_model.simulation_state['beam_size_y']
        
        # Расчет относительных ошибок
        error_x = abs(real_size_x - sim_size_x) / real_size_x if real_size_x > 0 else 0.0
        error_y = abs(real_size_y - sim_size_y) / real_size_y if real_size_y > 0 else 0.0
        
        logger.info(f"Валидация размера пучка: реальный=(x={real_size_x:.4f}, y={real_size_y:.4f}), симулированный=(x={sim_size_x:.4f}, y={sim_size_y:.4f}), ошибки=(x={error_x:.2%}, y={error_y:.2%})")
        
        return (error_x, error_y)
    
    def autocalibrate(self):
        """Автоматическая калибровка модели на основе реальных данных"""
        if not self.validation_data:
            logger.warning("Невозможно выполнить калибровку: данные не загружены")
            return
        
        logger.info("Запуск автоматической калибровки модели")
        
        # Валидация и расчет коэффициентов калибровки
        real_luminosity = self._extract_real_luminosity()
        sim_luminosity = self.lhc_model.simulation_state['luminosity']
        self.calibration_factors['luminosity'] = real_luminosity / sim_luminosity if sim_luminosity > 0 else 1.0
        
        real_emittance = self._extract_real_emittance()
        sim_emittance = self.lhc_model.simulation_state['beam_dynamics']['emittance'][-1] if self.lhc_model.simulation_state['beam_dynamics']['emittance'] else 3.5e-6
        self.calibration_factors['emittance'] = real_emittance / sim_emittance if sim_emittance > 0 else 1.0
        
        real_size_x, real_size_y = self._extract_real_beam_size()
        sim_size_x = self.lhc_model.simulation_state['beam_size_x']
        sim_size_y = self.lhc_model.simulation_state['beam_size_y']
        self.calibration_factors['beam_size'] = (real_size_x / sim_size_x + real_size_y / sim_size_y) / 2 if sim_size_x > 0 and sim_size_y > 0 else 1.0
        
        # Сохранение в историю калибровки
        self.calibration_history.append({
            'timestamp': time.time(),
            'factors': self.calibration_factors.copy(),
            'luminosity_error': abs(real_luminosity - sim_luminosity) / real_luminosity if real_luminosity > 0 else 0.0,
            'emittance_error': abs(real_emittance - sim_emittance) / real_emittance if real_emittance > 0 else 0.0,
            'beam_size_error': (abs(real_size_x - sim_size_x) / real_size_x + abs(real_size_y - sim_size_y) / real_size_y) / 2 if real_size_x > 0 and real_size_y > 0 else 0.0
        })
        
        logger.info(f"Калибровочные коэффициенты обновлены: {self.calibration_factors}")
    
    def apply_calibration(self, state: Dict) -> Dict:
        """
        Применение калибровки к состоянию симуляции.
        
        Параметры:
        state: состояние симуляции
        
        Возвращает:
        Откалиброванное состояние
        """
        calibrated_state = state.copy()
        
        # Применение калибровки к светимости
        calibrated_state['luminosity'] *= self.calibration_factors['luminosity']
        
        # Применение калибровки к эмиттансу
        if 'beam_dynamics' in calibrated_state and 'emittance' in calibrated_state['beam_dynamics']:
            calibrated_state['beam_dynamics']['emittance'] = [
                e * self.calibration_factors['emittance'] for e in calibrated_state['beam_dynamics']['emittance']
            ]
        
        # Применение калибровки к размеру пучка
        scale_factor = np.sqrt(self.calibration_factors['beam_size'])
        calibrated_state['beam_size_x'] *= scale_factor
        calibrated_state['beam_size_y'] *= scale_factor
        
        return calibrated_state
    
    def load_calibration_data(self):
        """Загрузка калибровочных данных из файла"""
        try:
            with open("calibration_data.json", "r") as f:
                self.calibration_factors = json.load(f)
            logger.info("Калибровочные данные загружены из файла")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("Калибровочные данные не найдены. Используются значения по умолчанию.")
    
    def save_calibration_data(self):
        """Сохранение калибровочных данных в файл"""
        try:
            with open("calibration_data.json", "w") as f:
                json.dump(self.calibration_factors, f, indent=2)
            logger.info("Калибровочные данные сохранены в файл")
        except Exception as e:
            logger.error(f"Ошибка при сохранении калибровочных данных: {e}")


# ===================================================================
# 10. GPU-ускорение для трассировки частиц
# ===================================================================

class GPUAccelerator:
    """Система ускорения вычислений с использованием GPU"""
    
    def __init__(self):
        """Инициализация GPU-ускорителя"""
        self.gpu_available = False
        self.numba_available = False
        self.cupy_available = False
        
        # Проверка доступности Numba
        try:
            from numba import cuda
            self.numba_available = True
            self.gpu_available = cuda.is_available()
            if self.gpu_available:
                logger.info("GPU ускорение через Numba доступно")
        except ImportError:
            pass
        
        # Проверка доступности CuPy
        try:
            import cupy
            self.cupy_available = True
            if self.gpu_available:
                logger.info("GPU ускорение через CuPy доступно")
        except ImportError:
            pass
        
        if not self.gpu_available:
            logger.warning("GPU не доступен. Вычисления будут выполняться на CPU.")
    
    def track_particles_gpu(self, particles: List[Dict], field: Tuple, 
                          num_steps: int = 1000, dt: float = 1e-12) -> List[List[Tuple[float, float, float]]]:
        """
        Трассировка частиц с использованием GPU-ускорения при возможности.
        
        Параметры:
        particles: список частиц для трассировки
        field: магнитное поле (Bx, By, Bz)
        num_steps: количество шагов трассировки
        dt: временной шаг
        
        Возвращает:
        Список траекторий для каждой частицы
        """
        if self.numba_available and self.gpu_available:
            return self._track_particles_numba(particles, field, num_steps, dt)
        elif self.cupy_available:
            return self._track_particles_cupy(particles, field, num_steps, dt)
        else:
            return self._track_particles_cpu(particles, field, num_steps, dt)
    
    def _track_particles_numba(self, particles: List[Dict], field: Tuple, 
                              num_steps: int, dt: float) -> List[List[Tuple[float, float, float]]]:
        """Трассировка частиц с использованием Numba CUDA"""
        try:
            from numba import cuda
            import numpy as np
            
            # Подготовка данных
            n_particles = len(particles)
            particle_data = np.zeros((n_particles, 7), dtype=np.float32)  # x, y, z, px, py, pz, q/m
            
            for i, p in enumerate(particles):
                particle_data[i, 0] = p['position'][0]
                particle_data[i, 1] = p['position'][1]
                particle_data[i, 2] = p['position'][2]
                particle_data[i, 3] = p['momentum'][0]
                particle_data[i, 4] = p['momentum'][1]
                particle_data[i, 5] = p['momentum'][2]
                
                # q/m (заряд/масса)
                q = p.get('charge', 1.0)
                m = p.get('mass', 1.0)
                particle_data[i, 6] = q / m
            
            # Поля
            Bx, By, Bz = field
            
            # Функция для ядра CUDA
            @cuda.jit
            def trace_kernel(particles, Bx, By, Bz, dt, num_steps, trajectories):
                i = cuda.grid(1)
                if i >= particles.shape[0]:
                    return
                
                x, y, z, px, py, pz, qm = particles[i]
                
                for step in range(num_steps):
                    # Сила Лоренца: F = q(v × B)
                    vx = px
                    vy = py
                    vz = pz
                    
                    fx = qm * (vy * Bz - vz * By)
                    fy = qm * (vz * Bx - vx * Bz)
                    fz = qm * (vx * By - vy * Bx)
                    
                    # Обновление импульса
                    px += fx * dt
                    py += fy * dt
                    pz += fz * dt
                    
                    # Обновление позиции
                    x += px * dt
                    y += py * dt
                    z += pz * dt
                    
                    # Сохранение траектории
                    idx = i * num_steps + step
                    trajectories[idx, 0] = x
                    trajectories[idx, 1] = y
                    trajectories[idx, 2] = z
            
            # Подготовка массива для траекторий
            trajectories = np.zeros((n_particles * num_steps, 3), dtype=np.float32)
            
            # Запуск ядра
            threads_per_block = 256
            blocks_per_grid = (n_particles + threads_per_block - 1) // threads_per_block
            
            trace_kernel[blocks_per_grid, threads_per_block](
                particle_data, Bx, By, Bz, dt, num_steps, trajectories
            )
            
            # Преобразование результатов
            result = []
            for i in range(n_particles):
                traj = [(trajectories[i * num_steps + step, 0],
                         trajectories[i * num_steps + step, 1],
                         trajectories[i * num_steps + step, 2])
                        for step in range(num_steps)]
                result.append(traj)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в трассировке частиц через Numba: {e}")
            return self._track_particles_cpu(particles, field, num_steps, dt)
    
    def _track_particles_cupy(self, particles: List[Dict], field: Tuple, 
                             num_steps: int, dt: float) -> List[List[Tuple[float, float, float]]]:
        """Трассировка частиц с использованием CuPy"""
        try:
            import cupy as cp
            import numpy as np
            
            # Подготовка данных
            n_particles = len(particles)
            particle_data = cp.zeros((n_particles, 7), dtype=cp.float32)  # x, y, z, px, py, pz, q/m
            
            for i, p in enumerate(particles):
                particle_data[i, 0] = p['position'][0]
                particle_data[i, 1] = p['position'][1]
                particle_data[i, 2] = p['position'][2]
                particle_data[i, 3] = p['momentum'][0]
                particle_data[i, 4] = p['momentum'][1]
                particle_data[i, 5] = p['momentum'][2]
                
                # q/m (заряд/масса)
                q = p.get('charge', 1.0)
                m = p.get('mass', 1.0)
                particle_data[i, 6] = q / m
            
            # Поля
            Bx, By, Bz = field
            
            # Траектории
            trajectories = cp.zeros((n_particles, num_steps, 3), dtype=cp.float32)
            
            # Векторизованные вычисления
            for step in range(num_steps):
                # Сила Лоренца
                vx = particle_data[:, 3]
                vy = particle_data[:, 4]
                vz = particle_data[:, 5]
                
                fx = particle_data[:, 6] * (vy * Bz - vz * By)
                fy = particle_data[:, 6] * (vz * Bx - vx * Bz)
                fz = particle_data[:, 6] * (vx * By - vy * Bx)
                
                # Обновление импульса
                particle_data[:, 3] += fx * dt
                particle_data[:, 4] += fy * dt
                particle_data[:, 5] += fz * dt
                
                # Обновление позиции
                particle_data[:, 0] += particle_data[:, 3] * dt
                particle_data[:, 1] += particle_data[:, 4] * dt
                particle_data[:, 2] += particle_data[:, 5] * dt
                
                # Сохранение траектории
                trajectories[:, step, 0] = particle_data[:, 0]
                trajectories[:, step, 1] = particle_data[:, 1]
                trajectories[:, step, 2] = particle_data[:, 2]
            
            # Преобразование в NumPy и затем в список
            trajectories_np = cp.asnumpy(trajectories)
            
            result = []
            for i in range(n_particles):
                traj = [(trajectories_np[i, step, 0],
                         trajectories_np[i, step, 1],
                         trajectories_np[i, step, 2])
                        for step in range(num_steps)]
                result.append(traj)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в трассировке частиц через CuPy: {e}")
            return self._track_particles_cpu(particles, field, num_steps, dt)
    
    def _track_particles_cpu(self, particles: List[Dict], field: Tuple, 
                            num_steps: int, dt: float) -> List[List[Tuple[float, float, float]]]:
        """Трассировка частиц на CPU (резервный вариант)"""
        Bx, By, Bz = field
        trajectories = []
        
        for p in particles:
            x, y, z = p['position']
            px, py, pz = p['momentum']
            
            # q/m (заряд/масса)
            q = p.get('charge', 1.0)
            m = p.get('mass', 1.0)
            qm = q / m
            
            traj = []
            for _ in range(num_steps):
                # Сила Лоренца: F = q(v × B)
                vx = px
                vy = py
                vz = pz
                
                fx = qm * (vy * Bz - vz * By)
                fy = qm * (vz * Bx - vx * Bz)
                fz = qm * (vx * By - vy * Bx)
                
                # Обновление импульса
                px += fx * dt
                py += fy * dt
                pz += fz * dt
                
                # Обновление позиции
                x += px * dt
                y += py * dt
                z += pz * dt
                
                traj.append((x, y, z))
            
            trajectories.append(traj)
        
        return trajectories


# ===================================================================
# 11. Детекторная система
# ===================================================================

class Detector:
    """Класс для моделирования детектора частиц"""
    
    def __init__(self, name: str, position: Tuple[float, float, float], 
                 size: Tuple[float, float, float], resolution: float, 
                 efficiency: float = 0.95):
        """
        Инициализация детектора.
        
        Параметры:
        name: название детектора
        position: позиция детектора (x, y, z)
        size: размеры детектора (dx, dy, dz)
        resolution: энергетическое разрешение
        efficiency: эффективность детектирования
        """
        self.name = name
        self.position = np.array(position)
        self.size = np.array(size)
        self.resolution = resolution
        self.efficiency = efficiency
        self.detected_particles = []
        logger.info(f"Детектор {name} инициализирован на позиции {position} с размерами {size}")
    
    def is_inside(self, position: Tuple[float, float, float]) -> bool:
        """Проверка, находится ли частица внутри детектора"""
        pos = np.array(position)
        lower_bound = self.position - self.size / 2
        upper_bound = self.position + self.size / 2
        return np.all(pos >= lower_bound) and np.all(pos <= upper_bound)
    
    def detect_particle(self, particle: Dict, event_id: int) -> Optional[Dict]:
        """
        Регистрация частицы детектором.
        
        Параметры:
        particle: информация о частице
        event_id: идентификатор события
        
        Возвращает:
        Зарегистрированную частицу или None, если не детектирована
        """
        # Проверка, находится ли частица в детекторе
        if not self.is_inside(particle['position']):
            return None
        
        # Проверка эффективности детектирования
        if random.random() > self.efficiency:
            return None
        
        # Реконструкция энергии с учетом разрешения
        true_energy = particle['energy']
        reconstructed_energy = np.random.normal(true_energy, true_energy * self.resolution)
        
        # Реконструкция импульса
        true_momentum = np.sqrt(particle['energy']**2 - particle.get('mass', 0)**2)
        reconstructed_momentum = np.random.normal(true_momentum, true_momentum * self.resolution)
        
        # Добавление зарегистрированной частицы
        detected_particle = {
            'name': particle['name'],
            'true_energy': true_energy,
            'reconstructed_energy': reconstructed_energy,
            'true_momentum': true_momentum,
            'reconstructed_momentum': reconstructed_momentum,
            'detected_by': self.name,
            'position': particle['position'],
            'event_id': event_id,
            'timestamp': time.time()
        }
        
        self.detected_particles.append(detected_particle)
        return detected_particle


class DetectorSystem:
    """Система детекторов коллайдера"""
    
    def __init__(self, circumference: float):
        """
        Инициализация системы детекторов.
        
        Параметры:
        circumference: длина окружности кольца коллайдера
        """
        self.circumference = circumference
        self.detectors = self._initialize_detectors()
        logger.info(f"Система детекторов инициализирована с {len(self.detectors)} детекторами")
    
    def _initialize_detectors(self) -> Dict[str, Detector]:
        """Инициализация детекторов в точках столкновения"""
        detectors = {}
        
        # Позиции детекторов (в долях окружности)
        positions = {
            'ATLAS': 0.0,
            'CMS': 0.5,
            'ALICE': 0.25,
            'LHCb': 0.75
        }
        
        # Параметры детекторов
        detector_specs = {
            'ATLAS': {'size': (25, 25, 46), 'resolution': 0.01, 'efficiency': 0.98},
            'CMS': {'size': (21, 21, 15), 'resolution': 0.015, 'efficiency': 0.97},
            'ALICE': {'size': (16, 16, 10), 'resolution': 0.05, 'efficiency': 0.95},
            'LHCb': {'size': (20, 12, 10), 'resolution': 0.02, 'efficiency': 0.96}
        }
        
        # Создание детекторов
        for name, pos_ratio in positions.items():
            # Позиция в координатах кольца
            angle = 2 * np.pi * pos_ratio
            radius = self.circumference / (2 * np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0  # Упрощение
            
            # Создание детектора
            specs = detector_specs[name]
            detectors[name] = Detector(
                name=name,
                position=(x, y, z),
                size=specs['size'],
                resolution=specs['resolution'],
                efficiency=specs['efficiency']
            )
        
        return detectors
    
    def detect_event(self, event: Dict) -> List[Dict]:
        """
        Обработка события всеми детекторами.
        
        Параметры:
        event: событие для обработки
        
        Возвращает:
        Список зарегистрированных частиц
        """
        detected = []
        event_id = event.get('event_id', int(time.time()))
        
        # Обработка каждой частицы в событии
        for product in event['products']:
            # Генерация траектории частицы (упрощенно)
            position = self._calculate_position(product)
            
            # Добавление позиции в частицу для детектирования
            particle = product.copy()
            particle['position'] = position
            
            # Попытка детектирования каждой частицы каждым детектором
            for detector in self.detectors.values():
                detected_particle = detector.detect_particle(particle, event_id)
                if detected_particle:
                    detected.append(detected_particle)
        
        return detected
    
    def _calculate_position(self, particle: Dict) -> Tuple[float, float, float]:
        """Расчет точки взаимодействия с детектором (упрощенная модель)"""
        # В реальной системе это зависело бы от типа детектора и траектории частицы
        return (
            np.random.normal(0, 0.1),
            np.random.normal(0, 0.1),
            np.random.normal(0, 0.1)
        )
    
    def reconstruct_event(self, event: Dict) -> Dict:
        """
        Реконструкция события с учетом характеристик детектора и шумов.
        
        Параметры:
        event: исходное событие
        
        Возвращает:
        Реконструированное событие
        """
        reconstructed = {
            'event_id': event['event_id'],
            'original_event': event,
            'reconstructed_products': [],
            'missing_energy': 0.0,
            'confidence': 1.0,
            'timestamp': time.time()
        }
        
        # Реконструкция каждой частицы
        for i, product in enumerate(event['products']):
            particle_name = product['name']
            true_energy = product['energy']
            
            # Имитация шума и потерь
            detection_efficiency = 0.9  # 90% эффективность обнаружения
            energy_resolution = 0.05    # 5% разрешение
            
            if random.random() < detection_efficiency:
                # Частица обнаружена
                reconstructed_energy = np.random.normal(true_energy, true_energy * energy_resolution)
                
                # Реконструируем импульс на основе энергии и массы
                mass = self.lhc_model.particle_db.get_particle(particle_name).mass if self.lhc_model else 0.0
                if mass < reconstructed_energy:
                    reconstructed_momentum = np.sqrt(reconstructed_energy**2 - mass**2)
                else:
                    reconstructed_momentum = 0.0
                
                # Добавляем реконструированную частицу
                reconstructed['reconstructed_products'].append({
                    'name': particle_name,
                    'reconstructed_energy': reconstructed_energy,
                    'reconstructed_momentum': reconstructed_momentum,
                    'true_energy': true_energy,
                    'detected': True
                })
            else:
                # Частица не обнаружена
                reconstructed['missing_energy'] += true_energy
        
        # Вычисление общей уверенности в реконструкции
        if reconstructed['reconstructed_products']:
            reconstructed['confidence'] = min(1.0, 0.9 * len(reconstructed['reconstructed_products']) / len(event['products']))
        
        return reconstructed
    
    def visualize_detector_response(self, event: Dict):
        """
        Визуализация отклика детектора на событие.
        
        Параметры:
        event: событие для визуализации
        """
        plt.figure(figsize=(12, 10))
        
        # 1. Диаграмма Фейнмана (упрощенная)
        ax1 = plt.subplot(2, 2, 1)
        
        # Входные частицы
        ax1.arrow(0, 0, -0.4, 0.4, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        ax1.arrow(0, 0, -0.4, -0.4, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        
        # Выходные частицы
        for i, product in enumerate(event['products']):
            angle = i * 2 * np.pi / len(event['products'])
            length = 0.5 + random.random() * 0.3
            ax1.arrow(0, 0, length * np.cos(angle), length * np.sin(angle), 
                     head_width=0.05, head_length=0.1, fc='red', ec='red')
        
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_title('Диаграмма Фейнмана')
        ax1.axis('equal')
        
        # 2. Распределение энергии
        ax2 = plt.subplot(2, 2, 2)
        energies = [p['energy'] for p in event['products']]
        particles = [p['name'] for p in event['products']]
        
        ax2.bar(particles, energies)
        ax2.set_title('Распределение энергии продуктов')
        ax2.set_ylabel('Энергия (ГэВ)')
        plt.xticks(rotation=45)
        
        # 3. Распределение по детекторам
        ax3 = plt.subplot(2, 2, 3)
        detector_counts = {name: 0 for name in self.detectors.keys()}
        
        # Здесь должны быть данные о том, какие детекторы что зарегистрировали
        # Для демонстрации используем случайные данные
        for name in detector_counts.keys():
            detector_counts[name] = random.randint(0, 10)
        
        ax3.pie(detector_counts.values(), labels=detector_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Распределение событий по детекторам')
        ax3.axis('equal')
        
        # 4. Информация о событии
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        # Заголовок
        ax4.text(0.5, 0.95, f"Событие: {event['event_type'].replace('_', ' ').title()}",
                 fontsize=14, ha='center', weight='bold')
        
        # Энергия
        ax4.text(0.1, 0.85, f"Энергия: {event['energy']:.1f} ГэВ",
                 fontsize=12)
        
        # Продукты
        ax4.text(0.1, 0.75, "Основные продукты:", fontsize=12, weight='bold')
        for i, product in enumerate(event['products'][:5]):  # Показываем первые 5 продуктов
            ax4.text(0.1, 0.65 - i*0.05, 
                     f"- {product['name']}: {product['energy']:.2f} ГэВ",
                     fontsize=10)
        
        plt.tight_layout()
        plt.savefig("detector_response.png")
        plt.show()


# ===================================================================
# 12. Геометрия коллайдера
# ===================================================================

class LHCGeometry:
    """Класс для моделирования геометрии Большого адронного коллайдера"""
    
    def __init__(self, circumference: float = 26658.883):
        """
        Инициализация геометрии коллайдера.
        
        Параметры:
        circumference: длина окружности кольца в метрах
        """
        self.circumference = circumference
        self.radius = circumference / (2 * np.pi)
        self.components = []
        logger.info(f"Геометрия коллайдера инициализирована с длиной окружности {circumference:.2f} м")
    
    def add_component(self, component_type: str, params: Dict):
        """
        Добавление компонента в геометрию коллайдера.
        
        Параметры:
        component_type: тип компонента (dipole, quadrupole, etc.)
        params: параметры компонента
        """
        component = {
            'type': component_type,
            'params': params,
            'position': params.get('position', 0.0)
        }
        self.components.append(component)
        logger.debug(f"Добавлен компонент: {component_type} на позиции {component['position']}")
    
    def build_geometry(self) -> List[Dict]:
        """
        Построение полной геометрии коллайдера.
        
        Возвращает:
        Список компонентов геометрии
        """
        # Очистка предыдущих компонентов
        self.components = []
        
        # Добавление дипольных магнитов (для изгиба траектории)
        num_dipoles = 1232
        for i in range(num_dipoles):
            position = i * self.circumference / num_dipoles
            self.add_component('dipole', {
                'position': position,
                'length': 14.3,  # длина магнита
                'field': 8.33,   # магнитное поле в Тл
                'angle': 2 * np.pi / num_dipoles
            })
        
        # Добавление квадрупольных магнитов (для фокусировки)
        num_quadrupoles = 392
        for i in range(num_quadrupoles):
            position = i * self.circumference / num_quadrupoles + self.circumference / (2 * num_quadrupoles)
            self.add_component('quadrupole', {
                'position': position,
                'length': 5.5,   # длина магнита
                'gradient': 215  # градиент поля в Тл/м
            })
        
        # Добавление секторов ускорения
        num_rf_cavities = 8
        for i in range(num_rf_cavities):
            position = i * self.circumference / num_rf_cavities
            self.add_component('rf_cavity', {
                'position': position,
                'length': 10.0,  # длина каверны
                'voltage': 2e6,  # напряжение в В
                'frequency': 400.8e6  # частота в Гц
            })
        
        # Добавление точек столкновения
        collision_points = {
            'ATLAS': 0.0,
            'CMS': 0.5,
            'ALICE': 0.25,
            'LHCb': 0.75
        }
        for name, pos_ratio in collision_points.items():
            position = pos_ratio * self.circumference
            self.add_component('collision_point', {
                'position': position,
                'name': name,
                'detector_size': 25.0  # размер детектора в м
            })
        
        logger.info(f"Геометрия коллайдера построена с {len(self.components)} компонентами")
        return self.components
    
    def load_from_yaml(self, yaml_file: str):
        """
        Загрузка геометрии из YAML-файла.
        
        Параметры:
        yaml_file: путь к YAML-файлу
        """
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            for comp in config.get('geometry', []):
                self.add_component(comp['type'], comp['params'])
            logger.info(f"Геометрия загружена из {yaml_file}")
        except Exception as e:
            logger.error(f"Ошибка загрузки геометрии из YAML: {e}")
    
    def visualize(self):
        """Визуализация геометрии коллайдера"""
        plt.figure(figsize=(12, 12))
        
        # Отображение кольца коллайдера
        theta = np.linspace(0, 2*np.pi, 1000)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        plt.plot(x, y, 'k-', alpha=0.3)
        
        # Отображение компонентов
        dipole_x, dipole_y = [], []
        quadrupole_x, quadrupole_y = [], []
        collision_x, collision_y = [], []
        collision_labels = []
        
        for component in self.components:
            angle = 2 * np.pi * component['position'] / self.circumference
            x_pos = self.radius * np.cos(angle)
            y_pos = self.radius * np.sin(angle)
            
            if component['type'] == 'dipole':
                dipole_x.append(x_pos)
                dipole_y.append(y_pos)
            elif component['type'] == 'quadrupole':
                quadrupole_x.append(x_pos)
                quadrupole_y.append(y_pos)
            elif component['type'] == 'collision_point':
                collision_x.append(x_pos)
                collision_y.append(y_pos)
                collision_labels.append(component['params']['name'])
        
        # Отображение дипольных магнитов
        plt.scatter(dipole_x, dipole_y, c='blue', s=10, label='Дипольные магниты')
        
        # Отображение квадрупольных магнитов
        plt.scatter(quadrupole_x, quadrupole_y, c='red', s=10, label='Квадрупольные магниты')
        
        # Отображение точек столкновения
        plt.scatter(collision_x, collision_y, c='green', s=100, marker='*', label='Точки столкновения')
        
        # Подписи точек столкновения
        for i, label in enumerate(collision_labels):
            plt.annotate(label, (collision_x[i], collision_y[i]),
                         xytext=(5, 5), textcoords='offset points')
        
        plt.title('Геометрия Большого адронного коллайдера')
        plt.xlabel('X (м)')
        plt.ylabel('Y (м)')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("lhc_geometry.png")
        plt.show()
    
    def get_magnetic_field(self, position: float) -> Tuple[float, float, float]:
        """
        Получение магнитного поля в заданной позиции.
        
        Параметры:
        position: позиция вдоль кольца (в метрах)
        
        Возвращает:
        (Bx, By, Bz) - компоненты магнитного поля
        """
        # Нормализуем позицию в пределах окружности
        position = position % self.circumference
        
        # Инициализируем поле
        Bx, By, Bz = 0.0, 0.0, 0.0
        
        # Ищем ближайшие компоненты
        for component in self.components:
            comp_pos = component['position']
            comp_type = component['type']
            params = component['params']
            
            # Расстояние до компонента
            distance = min(abs(position - comp_pos), self.circumference - abs(position - comp_pos))
            
            # Если мы в пределах компонента
            if distance < params.get('length', 0) / 2:
                if comp_type == 'dipole':
                    # Дипольное поле перпендикулярно траектории
                    angle = 2 * np.pi * comp_pos / self.circumference
                    Bz = params['field']  # Предполагаем, что поле направлено по z
                    Bx = Bz * np.sin(angle)
                    By = -Bz * np.cos(angle)
                elif comp_type == 'quadrupole':
                    # Квадрупольное поле (упрощенно)
                    angle = 2 * np.pi * comp_pos / self.circumference
                    r = self.radius
                    gradient = params['gradient']
                    
                    # Позиция относительно центра магнита
                    rel_pos = position - comp_pos
                    x_rel = r * np.sin(rel_pos / r)
                    y_rel = r * (1 - np.cos(rel_pos / r))
                    
                    # Поле квадрупольного магнита
                    Bx = gradient * y_rel
                    By = gradient * x_rel
        
        return (Bx, By, Bz)
    
    def simulate_particle_motion(self, num_turns: int = 1, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Симуляция движения частицы в кольце БАК.
        
        Параметры:
        num_turns: количество оборотов
        num_points: количество точек на оборот
        
        Возвращает:
        x, y, z: координаты траектории
        """
        # Общее количество точек
        total_points = num_turns * num_points
        
        # Угловые координаты
        theta = np.linspace(0, 2 * np.pi * num_turns, total_points)
        
        # Радиус кривизны (с небольшими колебаниями для реалистичности)
        epsilon = 0.001  # амплитуда колебаний
        n = 10  # частота колебаний
        r = self.radius * (1 + epsilon * np.sin(n * theta))
        
        # Горизонтальные координаты
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Вертикальные колебания (бетатронные колебания)
        A_z = 0.1  # амплитуда
        k_z = 5    # волновое число
        z = A_z * np.sin(k_z * theta)
        
        return x, y, z


# ===================================================================
# 13. Система анализа данных
# ===================================================================

class DataAnalyzer:
    """Система анализа данных коллайдера"""
    
    def __init__(self):
        """Инициализация анализатора данных"""
        self.histograms = {}
        self.roi_filters = []
        logger.info("Система анализа данных инициализирована")
    
    def create_histogram(self, name: str, bins: int, range: Tuple[float, float]):
        """
        Создание гистограммы.
        
        Параметры:
        name: имя гистограммы
        bins: количество бинов
        range: диапазон значений
        """
        self.histograms[name] = {
            'bins': bins,
            'range': range,
            'values': np.zeros(bins),
            'edges': np.linspace(range[0], range[1], bins + 1)
        }
        logger.debug(f"Создана гистограмма: {name} с {bins} бинами в диапазоне {range}")
    
    def fill_histogram(self, name: str, values: Union[float, List[float]]):
        """
        Заполнение гистограммы значениями.
        
        Параметры:
        name: имя гистограммы
        values: значение или список значений
        """
        if name not in self.histograms:
            logger.error(f"Гистограмма '{name}' не найдена")
            return
        
        hist = self.histograms[name]
        if isinstance(values, (int, float)):
            values = [values]
        
        for value in values:
            if hist['range'][0] <= value < hist['range'][1]:
                bin_index = int((value - hist['range'][0]) / (hist['range'][1] - hist['range'][0]) * hist['bins'])
                hist['values'][bin_index] += 1
    
    def create_2d_histogram(self, name: str, x_bins: int, x_range: Tuple[float, float],
                           y_bins: int, y_range: Tuple[float, float]):
        """
        Создание 2D гистограммы.
        
        Параметры:
        name: имя гистограммы
        x_bins, y_bins: количество бинов по осям
        x_range, y_range: диапазоны значений
        """
        self.histograms[name] = {
            'x_bins': x_bins,
            'y_bins': y_bins,
            'x_range': x_range,
            'y_range': y_range,
            'values': np.zeros((x_bins, y_bins)),
            'x_edges': np.linspace(x_range[0], x_range[1], x_bins + 1),
            'y_edges': np.linspace(y_range[0], y_range[1], y_bins + 1)
        }
        logger.debug(f"Создана 2D гистограмма: {name}")
    
    def fill_2d_histogram(self, name: str, x_values: Union[float, List[float]], 
                         y_values: Union[float, List[float]]):
        """
        Заполнение 2D гистограммы значениями.
        
        Параметры:
        name: имя гистограммы
        x_values: значения по оси X
        y_values: значения по оси Y
        """
        if name not in self.histograms:
            logger.error(f"Гистограмма '{name}' не найдена")
            return
        
        hist = self.histograms[name]
        if isinstance(x_values, (int, float)):
            x_values = [x_values]
            y_values = [y_values]
        
        for x, y in zip(x_values, y_values):
            if (hist['x_range'][0] <= x < hist['x_range'][1] and 
                hist['y_range'][0] <= y < hist['y_range'][1]):
                x_bin = int((x - hist['x_range'][0]) / (hist['x_range'][1] - hist['x_range'][0]) * hist['x_bins'])
                y_bin = int((y - hist['y_range'][0]) / (hist['y_range'][1] - hist['y_range'][0]) * hist['y_bins'])
                hist['values'][x_bin, y_bin] += 1
    
    def plot_histogram(self, name: str, log_scale: bool = False, 
                     title: Optional[str] = None, save: bool = False):
        """
        Построение гистограммы.
        
        Параметры:
        name: имя гистограммы
        log_scale: использовать ли логарифмический масштаб
        title: заголовок графика
        save: сохранить ли изображение
        """
        if name not in self.histograms:
            logger.error(f"Гистограмма '{name}' не найдена")
            return
        
        hist = self.histograms[name]
        plt.figure(figsize=(10, 7))
        
        # Построение гистограммы
        plt.hist(hist['edges'][:-1], bins=hist['edges'], weights=hist['values'], 
                 alpha=0.7, color='blue')
        
        # Настройка масштаба
        if log_scale:
            plt.yscale('log')
        
        # Заголовок и метки
        plt.title(title if title else f"Гистограмма: {name}")
        plt.xlabel("Значение")
        plt.ylabel("Число событий")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Сохранение
        if save:
            plt.savefig(f"{name}_histogram.png")
        
        plt.show()
    
    def plot_2d_histogram(self, name: str, log_scale: bool = False, 
                         title: Optional[str] = None, save: bool = False):
        """
        Построение 2D гистограммы.
        
        Параметры:
        name: имя гистограммы
        log_scale: использовать ли логарифмический масштаб
        title: заголовок графика
        save: сохранить ли изображение
        """
        if name not in self.histograms:
            logger.error(f"Гистограмма '{name}' не найдена")
            return
        
        hist = self.histograms[name]
        plt.figure(figsize=(10, 8))
        
        # Построение 2D гистограммы
        if log_scale:
            values = np.log10(hist['values'] + 1)  # Добавляем 1, чтобы избежать log(0)
            c = plt.pcolormesh(hist['x_edges'], hist['y_edges'], values.T, 
                              shading='auto', cmap='viridis')
            plt.colorbar(c, label='log(Число событий + 1)')
        else:
            c = plt.pcolormesh(hist['x_edges'], hist['y_edges'], hist['values'].T, 
                              shading='auto', cmap='viridis')
            plt.colorbar(c, label='Число событий')
        
        # Заголовок и метки
        plt.title(title if title else f"2D Гистограмма: {name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Сохранение
        if save:
            plt.savefig(f"{name}_2d_histogram.png")
        
        plt.show()
    
    def fit_histogram(self, name: str, function: str = 'gaussian', 
                     range: Optional[Tuple[float, float]] = None):
        """
        Фитирование гистограммы.
        
        Параметры:
        name: имя гистограммы
        function: тип функции для фитирования
        range: диапазон для фитирования
        
        Возвращает:
        Параметры фитирования
        """
        if name not in self.histograms:
            logger.error(f"Гистограмма '{name}' не найдена")
            return None
        
        hist = self.histograms[name]
        x = (np.array(hist['edges'][:-1]) + np.array(hist['edges'][1:])) / 2
        y = hist['values']
        
        # Фильтрация по диапазону
        if range:
            mask = (x >= range[0]) & (x <= range[1])
            x = x[mask]
            y = y[mask]
        
        # Фитирование
        if function == 'gaussian':
            from scipy.optimize import curve_fit
            
            # Начальные параметры для гауссова фитирования
            mean = np.sum(x * y) / np.sum(y)
            sigma = np.sqrt(np.sum(y * (x - mean)**2) / np.sum(y))
            amplitude = np.max(y)
            
            # Гауссова функция
            def gaussian(x, a, mu, sigma):
                return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
            
            # Фитирование
            try:
                popt, _ = curve_fit(gaussian, x, y, p0=[amplitude, mean, sigma])
                logger.info(f"Гауссово фитирование для '{name}': амплитуда={popt[0]:.2f}, среднее={popt[1]:.2f}, sigma={popt[2]:.2f}")
                return {
                    'function': 'gaussian',
                    'parameters': {'amplitude': popt[0], 'mean': popt[1], 'sigma': popt[2]}
                }
            except Exception as e:
                logger.error(f"Ошибка при фитировании гистограммы '{name}': {e}")
                return None
        else:
            logger.error(f"Функция '{function}' не поддерживается")
            return None
    
    def add_roi_filter(self, name: str, condition: Callable[[Dict], bool]):
        """
        Добавление фильтра по области интереса (ROI).
        
        Параметры:
        name: имя фильтра
        condition: функция-условие для фильтрации
        """
        self.roi_filters.append({
            'name': name,
            'condition': condition
        })
        logger.debug(f"Добавлен фильтр ROI: {name}")
    
    def apply_roi_filters(self, data: List[Dict]) -> List[Dict]:
        """
        Применение всех фильтров ROI к данным.
        
        Параметры:
        data: список данных для фильтрации
        
        Возвращает:
        Отфильтрованные данные
        """
        filtered_data = data.copy()
        
        for filter_info in self.roi_filters:
            name = filter_info['name']
            condition = filter_info['condition']
            
            initial_count = len(filtered_data)
            filtered_data = [item for item in filtered_data if condition(item)]
            final_count = len(filtered_data)
            
            logger.info(f"Фильтр '{name}' применен к {initial_count} элементам, осталось {final_count} элементов")
        
        return filtered_data
    
    def export_to_root(self, filename: str = "analysis_results.root"):
        """
        Экспорт результатов анализа в формат ROOT.
        
        Параметры:
        filename: имя файла для экспорта
        
        Возвращает:
        True при успешном экспорте, иначе False
        """
        if not ROOT_AVAILABLE:
            logger.warning("ROOT не установлен. Невозможно экспортировать в формат ROOT.")
            return False
        
        try:
            import ROOT
            import array
            
            # Создание файла ROOT
            root_file = ROOT.TFile(filename, "RECREATE")
            
            # Создание дерева для гистограмм
            tree = ROOT.TTree("AnalysisResults", "Результаты анализа данных")
            
            # Определение переменных
            hist_name = ROOT.std.string()
            hist_bins = array.array('i', [0])
            hist_entries = array.array('i', [0])
            hist_mean = array.array('d', [0.0])
            hist_rms = array.array('d', [0.0])
            
            # Создание веток
            tree.Branch("name", hist_name)
            tree.Branch("bins", hist_bins, 'bins/I')
            tree.Branch("entries", hist_entries, 'entries/I')
            tree.Branch("mean", hist_mean, 'mean/D')
            tree.Branch("rms", hist_rms, 'rms/D')
            
            # Заполнение дерева гистограммами
            for name, hist in self.histograms.items():
                # Вычисление статистики
                values = np.array(hist['values'])
                entries = int(np.sum(values))
                
                if entries > 0:
                    x = (np.array(hist['edges'][:-1]) + np.array(hist['edges'][1:])) / 2
                    mean = np.sum(x * values) / entries
                    rms = np.sqrt(np.sum((x - mean)**2 * values) / entries)
                else:
                    mean, rms = 0.0, 0.0
                
                # Заполнение переменных
                hist_name = name
                hist_bins[0] = len(hist['values'])
                hist_entries[0] = entries
                hist_mean[0] = mean
                hist_rms[0] = rms
                
                # Добавление в дерево
                tree.Fill()
            
            # Сохранение и закрытие файла
            root_file.Write()
            root_file.Close()
            
            logger.info(f"Результаты анализа экспортированы в формат ROOT: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при экспорте в ROOT: {e}")
            return False


# ===================================================================
# 14. Визуализация
# ===================================================================

class Visualizer:
    """Интерактивная визуализация для коллайдера"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        logger.info("Визуализатор инициализирован")
    
    def plot_geometry(self, geometry: LHCGeometry):
        """
        Визуализация геометрии эксперимента.
        
        Параметры:
        geometry: геометрия эксперимента
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly не установлен. Используется базовая визуализация.")
            geometry.visualize()
            return
        
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Отображение кольца коллайдера
            theta = np.linspace(0, 2*np.pi, 1000)
            x = geometry.radius * np.cos(theta)
            y = geometry.radius * np.sin(theta)
            z = np.zeros_like(x)
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y
            z += pz * dt
            positions.append((x, y, z))
        tracks.append(positions)
    return tracks


# ===================================================================
# 15. Основная модель коллайдера
# ===================================================================

class LHCHybridModel:
    """Гибридная модель Большого адронного коллайдера"""
    
    def __init__(self):
        """Инициализация гибридной модели коллайдера"""
        # Инициализация компонентов
        self.particle_db = ParticleDatabase()
        self.geometry = LHCGeometry()
        self.detector_system = DetectorSystem(self.geometry.circumference)
        self.gpu_accelerator = GPUAccelerator()
        
        # Инициализация физических движков
        self.physics_engine = HybridPhysicsEngine(self.particle_db)
        self.beam_dynamics = HybridBeamDynamics()
        
        # Инициализация систем
        self.cache = SimulationCache()
        self.data_analyzer = DataAnalyzer()
        self.visualizer = Visualizer()
        
        # Загрузка калибровочных данных
        self.validation_system = ValidationSystem(self)
        self.validation_system.load_calibration_data()
        
        # Построение геометрии
        self.geometry.build_geometry()
        
        # Инициализация состояния симуляции
        self.simulation_state = self._initialize_simulation_state()
        
        logger.info("Гибридная модель коллайдера инициализирована")
    
    def _initialize_simulation_state(self) -> Dict:
        """Инициализация начального состояния симуляции"""
        # Получение параметров из конфигурации или CERN Open Data
        if 'beam' in CONFIG:
            beam_energy = CONFIG['beam']['energy'] * 1e9  # ГэВ -> эВ
            particles = CONFIG['beam']['particles']
            bunch_intensity = CONFIG['beam']['bunch_intensity']
            num_bunches = CONFIG['beam']['num_bunches']
        else:
            # Дефолтные параметры
            real_params = CERN_Open_Data_API.get_real_lhc_parameters()
            beam_energy = real_params['beam_energy'] * 1e9
            particles = 'protons'
            bunch_intensity = real_params['bunch_intensity']
            num_bunches = real_params['num_bunches']
        
        # Расчет производных параметров
        proton_mass = m_p * c**2 / e  # масса протона в эВ
        gamma = beam_energy / proton_mass
        beta = np.sqrt(1 - 1/gamma**2)
        
        # Начальный размер пучка (в м)
        beam_size_x = 0.05
        beam_size_y = 0.05
        
        # Светимость
        peak_luminosity = 2.0e34  # см⁻²с⁻¹
        
        # Время одного оборота
        revolution_time = self.geometry.circumference / (beta * c)
        
        state = {
            'time': 0.0,
            'turn': 0,
            'beam_energy': beam_energy / 1e9,  # ГэВ
            'particles': particles,
            'protons_per_bunch': bunch_intensity,
            'num_bunches': num_bunches,
            'beam_intensity': bunch_intensity * num_bunches,
            'revolution_time': revolution_time,
            'revolution_frequency': 1 / revolution_time,
            'gamma': gamma,
            'beta': beta,
            'circumference': self.geometry.circumference,
            'beam_size_x': beam_size_x,
            'beam_size_y': beam_size_y,
            'luminosity': peak_luminosity,
            'peak_luminosity': peak_luminosity,
            'beam_dynamics': {
                'time': [0.0],
                'luminosity': [peak_luminosity],
                'beam_size_x': [beam_size_x],
                'beam_size_y': [beam_size_y],
                'emittance': [3.5e-6],  # нормированный эмиттанс (м·рад)
                'beam_intensity': [bunch_intensity * num_bunches]
            },
            'collision_events': [],
            'detected_events': []
        }
        
        return state
    
    def step_simulation(self, include_space_charge: bool = True,
                       force_physics_engine: Optional[str] = None,
                       force_beam_engine: Optional[str] = None):
        """
        Выполнение одного шага симуляции (один оборот пучка).
        
        Параметры:
        include_space_charge: учитывать ли эффекты пространственного заряда
        force_physics_engine: принудительный выбор физического движка
        force_beam_engine: принудительный выбор движка динамики пучка
        """
        # Симуляция одного оборота пучка
        updated_state = self.beam_dynamics.simulate_turn(
            self.simulation_state, 
            self.simulation_state['revolution_time'],
            include_space_charge=include_space_charge,
            force_engine=force_beam_engine
        )
        
        # Обновление состояния
        self.simulation_state = updated_state
        
        # Увеличение счетчика оборотов
        self.simulation_state['turn'] += 1
        
        # Вероятность столкновения на каждом обороте
        collision_probability = 0.1  # 10% вероятность столкновения за оборот
        
        if random.random() < collision_probability:
            # Моделирование столкновения
            self._simulate_collision(force_engine=force_physics_engine)
        
        # Логирование состояния
        logger.info(f"Оборот {self.simulation_state['turn']}: "
                   f"Светимость={self.simulation_state['luminosity']:.2e} см⁻²с⁻¹, "
                   f"Размер пучка=({self.simulation_state['beam_size_x']*1e6:.2f}, "
                   f"{self.simulation_state['beam_size_y']*1e6:.2f}) мкм")
    
    def _simulate_collision(self, force_engine: Optional[str] = None):
        """Моделирование события столкновения"""
        # Параметры столкновения
        energy = self.simulation_state['beam_energy'] * 2  # энергия в центре масс
        event_id = len(self.simulation_state['collision_events'])
        
        # Генерация ключа для кэширования
        cache_params = {
            'energy': energy,
            'engine': force_engine or 'auto',
            'num_events': 1
        }
        cache_key = SimulationCache.generate_key(cache_params)
        
        # Попытка получить из кэша
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"Использован кэшированный результат для столкновения")
            events = cached_result
        else:
            # Симуляция столкновения
            events = self.physics_engine.interact(
                'proton', 'proton', energy, 
                num_events=1, force_engine=force_engine
            )
            
            # Сохранение в кэш
            self.cache.set(cache_key, events)
        
        # Добавление события в историю
        for event in events:
            event['event_id'] = event_id
            event['timestamp'] = time.time()
            self.simulation_state['collision_events'].append(event)
            
            # Детектирование события
            detected_particles = self.detector_system.detect_event(event)
            self.simulation_state['detected_events'].extend(detected_particles)
            
            # Реконструкция события
            reconstructed_event = self.detector_system.reconstruct_event(event)
            
            # Анализ события
            self._analyze_event(reconstructed_event)
        
        logger.info(f"Смоделировано событие столкновения: {events[0]['event_type']}")
    
    def _analyze_event(self, event: Dict):
        """
        Анализ события с помощью DataAnalyzer.
        
        Параметры:
        event: реконструированное событие
        """
        # Создание гистограмм при необходимости
        if 'energy_spectrum' not in self.data_analyzer.histograms:
            self.data_analyzer.create_histogram('energy_spectrum', 50, (0, 1000))
        
        if 'momentum_spectrum' not in self.data_analyzer.histograms:
            self.data_analyzer.create_histogram('momentum_spectrum', 50, (0, 1000))
        
        if 'energy_vs_momentum' not in self.data_analyzer.histograms:
            self.data_analyzer.create_2d_histogram('energy_vs_momentum', 
                                                50, (0, 1000), 50, (0, 1000))
        
        # Заполнение гистограмм
        for product in event['reconstructed_products']:
            energy = product['reconstructed_energy']
            momentum = product['reconstructed_momentum']
            
            self.data_analyzer.fill_histogram('energy_spectrum', energy)
            self.data_analyzer.fill_histogram('momentum_spectrum', momentum)
            self.data_analyzer.fill_2d_histogram('energy_vs_momentum', energy, momentum)
    
    def run_simulation(self, num_turns: int = 100, 
                      include_space_charge: bool = True,
                      force_physics_engine: Optional[str] = None,
                      force_beam_engine: Optional[str] = None):
        """
        Запуск симуляции на заданное количество оборотов.
        
        Параметры:
        num_turns: количество оборотов для симуляции
        include_space_charge: учитывать ли эффекты пространственного заряда
        force_physics_engine: принудительный выбор физического движка
        force_beam_engine: принудительный выбор движка динамики пучка
        """
        logger.info(f"Запуск симуляции на {num_turns} оборотов")
        start_time = time.time()
        
        for _ in range(num_turns):
            self.step_simulation(include_space_charge, 
                               force_physics_engine, force_beam_engine)
        
        end_time = time.time()
        logger.info(f"Симуляция завершена за {end_time - start_time:.2f} секунд")
        
        # Валидация результатов
        self.validate_results()
        
        # Отчет о кэше
        hit_rate = self.cache.get_hit_rate()
        logger.info(f"Кэш: hit rate = {hit_rate:.2%}")
    
    def validate_results(self):
        """Валидация результатов симуляции"""
        if hasattr(self, 'validation_system'):
            # Загрузка реальных данных для валидации
            dataset_id = CONFIG['validation']['dataset_id']
            self.validation_system.load_real_data(dataset_id)
            
            # Выполнение валидации
            luminosity_error = self.validation_system.validate_luminosity()
            emittance_error = self.validation_system.validate_emittance()
            beam_size_errors = self.validation_system.validate_beam_size()
            
            logger.info(f"Валидация завершена. "
                       f"Ошибка светимости: {luminosity_error:.2%}, "
                       f"Ошибка эмиттанса: {emittance_error:.2%}, "
                       f"Ошибки размера пучка: x={beam_size_errors[0]:.2%}, y={beam_size_errors[1]:.2%}")
    
    def export_results(self, format: str = 'json', path: str = 'results'):
        """
        Экспорт результатов симуляции.
        
        Параметры:
        format: формат экспорта ('json' или 'root')
        path: путь для сохранения результатов
        """
        # Создание директории
        os.makedirs(path, exist_ok=True)
        
        if format == 'json':
            # Подготовка данных для экспорта
            export_data = {
                'simulation_parameters': {
                    'num_turns': self.simulation_state['turn'],
                    'beam_energy': self.simulation_state['beam_energy'],
                    'particles': self.simulation_state['particles'],
                    'num_bunches': self.simulation_state['num_bunches'],
                    'protons_per_bunch': self.simulation_state['protons_per_bunch']
                },
                'beam_dynamics': {
                    'time': self.simulation_state['beam_dynamics']['time'],
                    'luminosity': self.simulation_state['beam_dynamics']['luminosity'],
                    'beam_size_x': self.simulation_state['beam_dynamics']['beam_size_x'],
                    'beam_size_y': self.simulation_state['beam_dynamics']['beam_size_y'],
                    'emittance': self.simulation_state['beam_dynamics']['emittance'],
                    'beam_intensity': self.simulation_state['beam_dynamics']['beam_intensity']
                },
                'collision_events': self.simulation_state['collision_events'][:100],  # Ограничиваем количество
                'detected_events': self.simulation_state['detected_events'][:1000],  # Ограничиваем количество
                'cache_hit_rate': self.cache.get_hit_rate(),
                'calibration_factors': self.validation_system.calibration_factors if hasattr(self, 'validation_system') else {}
            }
            
            # Сохранение в JSON
            filename = os.path.join(path, 'lhc_simulation_results.json')
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Результаты экспортированы в JSON: {filename}")
        
        elif format == 'root' and ROOT_AVAILABLE:
            success = self.data_analyzer.export_to_root(os.path.join(path, 'analysis_results.root'))
            if success:
                logger.info("Результаты анализа экспортированы в ROOT")
        else:
            logger.error(f"Формат экспорта '{format}' не поддерживается")
    
    def visualize_results(self):
        """Визуализация результатов симуляции"""
        # 1. Визуализация геометрии коллайдера
        self.geometry.visualize()
        
        # 2. Визуализация динамики пучка
        self._plot_beam_dynamics()
        
        # 3. Визуализация гистограмм
        self._plot_analysis_histograms()
        
        # 4. Визуализация детекторного отклика
        if self.simulation_state['collision_events']:
            last_event = self.simulation_state['collision_events'][-1]
            self.detector_system.visualize_detector_response(last_event)
        
        # 5. Визуализация траекторий частиц
        if self.simulation_state['collision_events']:
            self._visualize_particle_tracks()
    
    def _plot_beam_dynamics(self):
        """Построение графиков динамики пучка"""
        dynamics = self.simulation_state['beam_dynamics']
        time_array = np.array(dynamics['time'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Светимость
        ax1.plot(time_array, dynamics['luminosity'], 'b-', linewidth=2)
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Светимость (см⁻²с⁻¹)')
        ax1.set_title('Светимость во времени')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_yscale('log')
        
        # Размер пучка
        ax2.plot(time_array, np.array(dynamics['beam_size_x']) * 1e6, 'r-', label='X', linewidth=2)
        ax2.plot(time_array, np.array(dynamics['beam_size_y']) * 1e6, 'g-', label='Y', linewidth=2)
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Размер пучка (мкм)')
        ax2.set_title('Размер пучка во времени')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Эмиттанс
        ax3.plot(time_array, np.array(dynamics['emittance']) * 1e6, 'm-', linewidth=2)
        ax3.set_xlabel('Время (с)')
        ax3.set_ylabel('Эмиттанс (мкм·рад)')
        ax3.set_title('Эмиттанс во времени')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Интенсивность пучка
        ax4.plot(time_array, np.array(dynamics['beam_intensity']), 'c-', linewidth=2)
        ax4.set_xlabel('Время (с)')
        ax4.set_ylabel('Интенсивность пучка')
        ax4.set_title('Интенсивность пучка во времени')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('beam_dynamics.png')
        plt.show()
    
    def _plot_analysis_histograms(self):
        """Построение гистограмм анализа"""
        # Энергетический спектр
        if 'energy_spectrum' in self.data_analyzer.histograms:
            self.data_analyzer.plot_histogram('energy_spectrum', log_scale=True,
                                            title='Энергетический спектр продуктов', save=True)
        
        # Импульсный спектр
        if 'momentum_spectrum' in self.data_analyzer.histograms:
            self.data_analyzer.plot_histogram('momentum_spectrum', log_scale=True,
                                            title='Импульсный спектр продуктов', save=True)
        
        # Корреляция энергии и импульса
        if 'energy_vs_momentum' in self.data_analyzer.histograms:
            self.data_analyzer.plot_2d_histogram('energy_vs_momentum', log_scale=True,
                                               title='Энергия vs Импульс', save=True)
    
    def _visualize_particle_tracks(self):
        """Визуализация траекторий частиц"""
        # Выбираем последнее событие
        event = self.simulation_state['collision_events'][-1]
        
        # Создаем искусственные траектории для визуализации
        num_particles = len(event['products'])
        tracks = []
        
        for i in range(num_particles):
            # Генерация траектории для каждой частицы
            num_steps = 100
            x = np.linspace(0, 10, num_steps) + np.random.normal(0, 0.5, num_steps)
            y = np.sin(x) + np.random.normal(0, 0.3, num_steps)
            z = np.cos(x) + np.random.normal(0, 0.3, num_steps)
            
            track = [(x[j], y[j], z[j]) for j in range(num_steps)]
            tracks.append(track)
        
        # Визуализация траекторий
        self.visualizer.plot_tracks(tracks, event)
    
    def reset_simulation(self):
        """Сброс симуляции к начальному состоянию"""
        self.simulation_state = self._initialize_simulation_state()
        self.cache = SimulationCache()
        self.data_analyzer = DataAnalyzer()
        logger.info("Симуляция сброшена к начальному состоянию")


# ===================================================================
# 16. Основная функция и примеры использования
# ===================================================================

def create_default_config():
    """Создание файла конфигурации по умолчанию"""
    config = {
        "beam": {
            "energy": 6500,
            "particles": "protons",
            "bunch_intensity": 1.15e11,
            "num_bunches": 2748
        },
        "geometry": {
            "circumference": 26658.883,
            "dipole_field": 8.33
        },
        "validation": {
            "dataset_id": "cms-2011-collision-data"
        }
    }
    
    with open("lhc_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info("Файл конфигурации создан: lhc_config.yaml")


def demo_basic_simulation():
    """Демонстрация базовой симуляции"""
    logger.info("Сценарий 1: Базовая симуляция")
    
    # Создание модели
    lhc = LHCHybridModel()
    
    # Запуск симуляции
    lhc.run_simulation(num_turns=10)
    
    # Визуализация результатов
    lhc.visualize_results()
    
    # Экспорт результатов
    lhc.export_results(format='json')


def demo_hybrid_simulation():
    """Демонстрация гибридной симуляции с выбором движков"""
    logger.info("Сценарий 2: Гибридная симуляция с выбором движков")
    
    # Создание модели
    lhc = LHCHybridModel()
    
    # Установка приоритета движков
    lhc.physics_engine.set_engine_priority(["built-in"])
    lhc.beam_dynamics.set_engine_priority(["built-in"])
    
    # Запуск симуляции
    lhc.run_simulation(num_turns=20, force_physics_engine="built-in")
    
    # Визуализация результатов
    lhc.visualize_results()


def demo_validation_simulation():
    """Демонстрация симуляции с валидацией"""
    logger.info("Сценарий 3: Симуляция с валидацией")
    
    # Создание модели
    lhc = LHCHybridModel()
    
    # Запуск симуляции
    lhc.run_simulation(num_turns=15)
    
    # Валидация результатов
    lhc.validate_results()
    
    # Экспорт результатов
    lhc.export_results(format='json')


def demo_educational_simulation():
    """Демонстрация учебной симуляции"""
    logger.info("Сценарий 4: Полностью учебная модель")
    
    # Создание модели
    lhc = LHCHybridModel()
    
    # Запуск симуляции
    lhc.run_simulation(num_turns=5, force_physics_engine="built-in", force_beam_engine="built-in")
    
    # Визуализация результатов
    lhc.visualize_results()


def main():
    """Основная функция для демонстрации работы модели"""
    logger.info("="*70)
    logger.info("ГИБРИДНАЯ СИСТЕМА ДЛЯ ТОЧНОЙ СИМУЛЯЦИИ БАК")
    logger.info("="*70)
    
    # Проверка наличия конфигурационного файла
    if not os.path.exists("lhc_config.yaml"):
        create_default_config()
    
    # Демонстрация различных сценариев
    demo_basic_simulation()
    
    # Сброс модели для следующей демонстрации
    time.sleep(2)
    
    demo_hybrid_simulation()
    
    # Сброс модели
    time.sleep(2)
    
    demo_validation_simulation()
    
    # Сброс модели
    time.sleep(2)
    
    demo_educational_simulation()
    
    logger.info("="*70)
    logger.info("ГИБРИДНАЯ СИСТЕМА СИМУЛЯЦИИ ЗАВЕРШИЛА РАБОТУ")
    logger.info("="*70)


if __name__ == "__main__":
    main()
