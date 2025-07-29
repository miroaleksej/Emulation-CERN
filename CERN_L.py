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
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Подавление предупреждений для чистоты вывода
warnings.filterwarnings("ignore", category=UserWarning)

# Попытка импорта Plotly для улучшенной визуализации
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Предупреждение: Plotly не установлен. Интерактивная 3D-визуализация будет недоступна.")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lhc_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LHC_Simulation")

# ===================================================================
# 1. Определение базы данных элементарных частиц
# ===================================================================
class ParticleDatabase:
    """База данных элементарных частиц с их физическими параметрами"""
    
    def __init__(self):
        """Инициализация базы данных частиц"""
        self.particles = self._load_particle_data()
        logger.info(f"Particle database initialized with {len(self.particles)} particles")
    
    def _load_particle_data(self) -> Dict[str, Dict]:
        """Загрузка данных об элементарных частицах"""
        # Данные из Particle Data Group (2023)
        return {
            'proton': {
                'mass': 938.27208816,  # МэВ/c²
                'mass_uncertainty': 0.00000029,  # Неопределенность в МэВ/c²
                'charge': 1,  # В единицах элементарного заряда
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'baryon',
                'symbol': 'p',
                'width': 0.0  # ширина распада в МэВ
            },
            'electron': {
                'mass': 0.51099895000,  # МэВ/c²
                'mass_uncertainty': 0.00000000015,
                'charge': -1,
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'lepton',
                'symbol': 'e⁻',
                'width': 0.0
            },
            'muon': {
                'mass': 105.6583755,  # МэВ/c²
                'mass_uncertainty': 0.0000023,
                'charge': -1,
                'spin': 0.5,
                'lifetime': 2.1969811e-6,  # секунды
                'lifetime_uncertainty': 0.0000022e-6,
                'category': 'lepton',
                'symbol': 'μ⁻',
                'width': 2.9991e-19  # ширина распада в ГэВ
            },
            'tau': {
                'mass': 1776.86,  # МэВ/c²
                'mass_uncertainty': 0.12,
                'charge': -1,
                'spin': 0.5,
                'lifetime': 2.903e-13,  # секунды
                'lifetime_uncertainty': 0.005e-13,
                'category': 'lepton',
                'symbol': 'τ⁻',
                'width': 2.2689e-15  # ширина распада в ГэВ
            },
            'neutrino_e': {
                'mass': 0.0000001,  # МэВ/c² (верхняя граница)
                'charge': 0,
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'lepton',
                'symbol': 'ν_e',
                'width': 0.0
            },
            'neutrino_mu': {
                'mass': 0.0000001,  # МэВ/c² (верхняя граница)
                'charge': 0,
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'lepton',
                'symbol': 'ν_μ',
                'width': 0.0
            },
            'neutrino_tau': {
                'mass': 0.0000001,  # МэВ/c² (верхняя граница)
                'charge': 0,
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'lepton',
                'symbol': 'ν_τ',
                'width': 0.0
            },
            'photon': {
                'mass': 0.0,
                'charge': 0,
                'spin': 1,
                'lifetime': float('inf'),
                'category': 'gauge boson',
                'symbol': 'γ',
                'width': 0.0
            },
            'gluon': {
                'mass': 0.0,
                'charge': 0,
                'spin': 1,
                'lifetime': float('inf'),
                'category': 'gauge boson',
                'symbol': 'g',
                'width': 0.0
            },
            'w_boson': {
                'mass': 80379,  # МэВ/c²
                'mass_uncertainty': 12,
                'charge': 1,
                'spin': 1,
                'lifetime': 3.2e-25,  # секунды
                'category': 'gauge boson',
                'symbol': 'W⁺',
                'width': 2085  # ширина распада в МэВ
            },
            'z_boson': {
                'mass': 91187.6,  # МэВ/c²
                'mass_uncertainty': 2.1,
                'charge': 0,
                'spin': 1,
                'lifetime': 2.6e-25,  # секунды
                'category': 'gauge boson',
                'symbol': 'Z⁰',
                'width': 2495.2  # ширина распада в МэВ
            },
            'higgs_boson': {
                'mass': 125.25,  # ГэВ/c² (точное значение)
                'mass_uncertainty': 0.17,  # Неопределенность в ГэВ/c²
                'charge': 0,
                'spin': 0,
                'lifetime': 1.56e-22,  # секунды
                'lifetime_uncertainty': 0.08e-22,
                'category': 'scalar boson',
                'symbol': 'H⁰',
                'width': 4.07e-3  # ширина распада в ГэВ
            },
            'top_quark': {
                'mass': 172.69,  # ГэВ/c²
                'mass_uncertainty': 0.30,
                'charge': 2/3,
                'spin': 0.5,
                'lifetime': 5.0e-25,  # секунды
                'category': 'quark',
                'symbol': 't',
                'width': 1.34  # ширина распада в ГэВ
            },
            'bottom_quark': {
                'mass': 4.18,  # ГэВ/c²
                'mass_uncertainty': 0.03,
                'charge': -1/3,
                'spin': 0.5,
                'lifetime': 1.6e-12,  # секунды
                'category': 'quark',
                'symbol': 'b',
                'width': 0.0  # Стабильный на уровне коллайдера
            },
            'charm_quark': {
                'mass': 1.27,  # ГэВ/c²
                'mass_uncertainty': 0.02,
                'charge': 2/3,
                'spin': 0.5,
                'lifetime': 1.0e-12,  # секунды
                'category': 'quark',
                'symbol': 'c',
                'width': 0.0  # Стабильный на уровне коллайдера
            },
            'strange_quark': {
                'mass': 0.096,  # ГэВ/c²
                'charge': -1/3,
                'spin': 0.5,
                'lifetime': 1.5e-8,  # секунды
                'category': 'quark',
                'symbol': 's',
                'width': 0.0  # Стабильный на уровне коллайдера
            },
            'up_quark': {
                'mass': 0.00216,  # ГэВ/c²
                'charge': 2/3,
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'quark',
                'symbol': 'u',
                'width': 0.0
            },
            'down_quark': {
                'mass': 0.00467,  # ГэВ/c²
                'charge': -1/3,
                'spin': 0.5,
                'lifetime': float('inf'),
                'category': 'quark',
                'symbol': 'd',
                'width': 0.0
            },
            'pion_plus': {
                'mass': 139.57039,  # МэВ/c²
                'mass_uncertainty': 0.00018,
                'charge': 1,
                'spin': 0,
                'lifetime': 2.6033e-8,  # секунды
                'lifetime_uncertainty': 0.0005e-8,
                'category': 'meson',
                'symbol': 'π⁺',
                'width': 2.528e-8  # ширина распада в эВ
            },
            'kaon_plus': {
                'mass': 493.677,  # МэВ/c²
                'mass_uncertainty': 0.013,
                'charge': 1,
                'spin': 0,
                'lifetime': 1.238e-8,  # секунды
                'lifetime_uncertainty': 0.002e-8,
                'category': 'meson',
                'symbol': 'K⁺',
                'width': 5.317e-8  # ширина распада в эВ
            },
            'b_hadron': {
                'mass': 5.279,  # ГэВ/c² (B⁰ мезон)
                'charge': 0,
                'spin': 0,
                'lifetime': 1.519e-12,  # секунды
                'category': 'hadron',
                'symbol': 'B⁰',
                'decay_modes': {
                    'D- + pi+': 0.0027,
                    'J/psi + K*': 0.059,
                    'pion+ + pion-': 0.026,
                    'pion+ + pion- + pion0': 0.094,
                    'pion+ + pion- + pion+ + pion-': 0.026,
                    'electron+ + neutrino_e': 0.00001,
                    'muon+ + neutrino_mu': 0.00001
                }
            },
            'c_hadron': {
                'mass': 1.869,  # ГэВ/c² (D⁰ мезон)
                'charge': 0,
                'spin': 0,
                'lifetime': 0.4101e-12,  # секунды
                'category': 'hadron',
                'symbol': 'D⁰',
                'decay_modes': {
                    'K- + pi+': 0.0395,
                    'K- + pi+ + pi0': 0.144,
                    'K- + pi+ + pi+ + pi-': 0.082,
                    'K- + pi+ + pi0 + pi0': 0.072,
                    'K- + K+': 0.004
                }
            }
        }
    
    def get_particle(self, name: str) -> Dict:
        """
        Получение данных о частице по имени.
        Параметры:
        name: имя частицы
        Возвращает:
        Словарь с параметрами частицы
        """
        if name not in self.particles:
            logger.warning(f"Particle '{name}' not found in database. Using proton as default.")
            return self.particles['proton']
        return self.particles[name]
    
    def get_mass(self, name: str) -> float:
        """
        Получение массы частицы в МэВ/c².
        Параметры:
        name: имя частицы
        Возвращает:
        Масса в МэВ/c²
        """
        mass = self.get_particle(name)['mass']
        # Если масса уже в ГэВ, конвертируем в МэВ
        if mass < 1000 and name not in ['electron', 'muon', 'pion_plus', 'kaon_plus']:
            return mass * 1000
        return mass
    
    def get_mass_in_gev(self, name: str) -> float:
        """
        Получение массы частицы в ГэВ/c².
        Параметры:
        name: имя частицы
        Возвращает:
        Масса в ГэВ/c²
        """
        mass = self.get_particle(name)['mass']
        # Если масса в МэВ, конвертируем в ГэВ
        if mass > 1000 or name in ['electron', 'muon', 'pion_plus', 'kaon_plus']:
            return mass / 1000
        return mass
    
    def get_charge(self, name: str) -> float:
        """
        Получение заряда частицы в единицах элементарного заряда.
        Параметры:
        name: имя частицы
        Возвращает:
        Заряд в единицах e
        """
        return self.get_particle(name)['charge']
    
    def get_spin(self, name: str) -> float:
        """
        Получение спина частицы.
        Параметры:
        name: имя частицы
        Возвращает:
        Спин
        """
        return self.get_particle(name)['spin']
    
    def get_lifetime(self, name: str) -> float:
        """
        Получение времени жизни частицы в секундах.
        Параметры:
        name: имя частицы
        Возвращает:
        Время жизни в секундах
        """
        return self.get_particle(name)['lifetime']
    
    def get_category(self, name: str) -> str:
        """
        Получение категории частицы.
        Параметры:
        name: имя частицы
        Возвращает:
        Категория частицы
        """
        return self.get_particle(name)['category']
    
    def get_symbol(self, name: str) -> str:
        """
        Получение символа частицы.
        Параметры:
        name: имя частицы
        Возвращает:
        Символ частицы
        """
        return self.get_particle(name)['symbol']
    
    def list_particles_by_category(self, category: str) -> List[str]:
        """
        Получение списка частиц по категории.
        Параметры:
        category: категория частиц
        Возвращает:
        Список имен частиц
        """
        return [name for name, data in self.particles.items() if data['category'] == category]
    
    def get_particle_decay_products(self, name: str) -> List[Tuple[str, float]]:
        """
        Получение возможных продуктов распада частицы с реальными вероятностями.
        Параметры:
        name: имя частицы
        Возвращает:
        Список кортежей (имя продукта, вероятность)
        """
        # Реальные данные о распадах из Particle Data Group (2023)
        decays = {
            'muon': [
                ('electron', 0.989348), 
                ('neutrino_e', 0.989348), 
                ('antineutrino_mu', 0.989348)
            ],
            'tau': [
                ('electron', 0.1781), ('muon', 0.1739), 
                ('neutrino_e', 0.1781), ('neutrino_mu', 0.1739), ('neutrino_tau', 0.6479),
                ('pion_plus', 0.1082), ('pion_minus', 0.00001), 
                ('kaon_plus', 0.00696), ('kaon_minus', 0.000005)
            ],
            'w_boson': [
                ('electron', 0.1071), ('positron', 0.1071),
                ('muon', 0.1063), ('antimuon', 0.1063),
                ('tau', 0.1138), ('antitau', 0.1138),
                ('up_quark', 0.320), ('down_antiquark', 0.320),
                ('charm_quark', 0.320), ('strange_antiquark', 0.320),
                ('top_quark', 0.0), ('bottom_antiquark', 0.0)  # Топ-кварк слишком тяжелый
            ],
            'z_boson': [
                ('electron', 0.03363), ('positron', 0.03363),
                ('muon', 0.03366), ('antimuon', 0.03366),
                ('tau', 0.03369), ('antitau', 0.03369),
                ('neutrino_e', 0.0664), ('antineutrino_e', 0.0664),
                ('neutrino_mu', 0.0664), ('antineutrino_mu', 0.0664),
                ('neutrino_tau', 0.0664), ('antineutrino_tau', 0.0664),
                ('down_quark', 0.1524), ('up_antiquark', 0.1524),
                ('strange_quark', 0.1524), ('up_antiquark', 0.1524),
                ('bottom_quark', 0.1512), ('up_antiquark', 0.1512),
                ('up_quark', 0.118), ('down_antiquark', 0.118),
                ('charm_quark', 0.118), ('down_antiquark', 0.118)
            ],
            'higgs_boson': [
                ('bottom_quark', 0.577), ('antibottom_quark', 0.577),
                ('w_boson', 0.215), ('w_boson', 0.215),
                ('gluon', 0.0858), ('gluon', 0.0858),
                ('tau', 0.0627), ('antitau', 0.0627),
                ('charm_quark', 0.0311), ('anticharm_quark', 0.0311),
                ('z_boson', 0.0265), ('z_boson', 0.0265),
                ('photon', 0.00227), ('photon', 0.00227),
                ('muon', 0.000244), ('antimuon', 0.000244),
                ('electron', 0.0000005), ('positron', 0.0000005)
            ],
            'top_quark': [
                ('w_boson', 0.999), ('bottom_quark', 0.999)
            ],
            'pion_plus': [
                ('muon', 0.999877), ('antineutrino_mu', 0.999877)
            ],
            'kaon_plus': [
                ('muon', 0.635), ('antineutrino_mu', 0.635),
                ('pion_plus', 0.056), ('pion_zero', 0.056),
                ('electron', 0.0507), ('neutrino_e', 0.0507),
                ('pion_plus', 0.033), ('pion_zero', 0.033), ('pion_zero', 0.033)
            ],
            'b_hadron': [
                ('c_quark', 0.41), ('w_boson', 0.41),
                ('pion_plus', 0.0027), ('d_quark', 0.0027),
                ('jpsi', 0.059), ('k_star', 0.059),
                ('pion_plus', 0.026), ('pion_minus', 0.026)
            ],
            'c_hadron': [
                ('s_quark', 0.144), ('pion_plus', 0.144),
                ('s_quark', 0.082), ('pion_plus', 0.082), ('pion_plus', 0.082), ('pion_minus', 0.082),
                ('k_minus', 0.0395), ('pion_plus', 0.0395)
            ]
        }
        return decays.get(name, [])
    
    def get_decay_branching_ratios(self, name: str) -> Dict[str, float]:
        """
        Получение коэффициентов ветвления для распадов частицы.
        Параметры:
        name: имя частицы
        Возвращает:
        Словарь с коэффициентами ветвления
        """
        decay_products = self.get_particle_decay_products(name)
        branching_ratios = {}
        for product, probability in decay_products:
            # Если продукт уже есть в словаре, суммируем вероятности
            if product in branching_ratios:
                branching_ratios[product] += probability
            else:
                branching_ratios[product] = probability
        return branching_ratios
    
    def plot_particle_properties(self, name: str):
        """
        Визуализация свойств частицы.
        Параметры:
        name: имя частицы
        """
        particle = self.get_particle(name)
        plt.figure(figsize=(12, 8))
        # Масса
        plt.subplot(2, 2, 1)
        mass = particle['mass']
        if 'mass_uncertainty' in particle and particle['mass_uncertainty'] > 0:
            mass_str = f"{mass:.2f} ± {particle['mass_uncertainty']:.2f}"
        else:
            mass_str = f"{mass:.2f}"
        if mass > 1000:
            mass_str += " ГэВ/c²"
        else:
            mass_str += " МэВ/c²"
        plt.text(0.5, 0.5, mass_str, 
                 fontsize=20, ha='center', va='center')
        plt.title(f"Масса: {particle['symbol']}")
        plt.axis('off')
        # Заряд
        plt.subplot(2, 2, 2)
        charge = particle['charge']
        color = 'blue' if charge > 0 else 'red' if charge < 0 else 'green'
        plt.scatter(0.5, 0.5, s=abs(charge)*500, color=color)
        plt.text(0.5, 0.3, f"Заряд: {charge}e", 
                 ha='center', va='center', fontsize=12)
        plt.title(f"Заряд: {particle['symbol']}")
        plt.axis('off')
        # Спин
        plt.subplot(2, 2, 3)
        spin = particle['spin']
        plt.pie([spin, 1-spin], 
                colors=['skyblue', 'lightgray'],
                startangle=90,
                autopct=lambda p: f"{spin}" if p > 0 else "")
        plt.title(f"Спин: {particle['symbol']}")
        plt.axis('equal')
        # Время жизни
        plt.subplot(2, 2, 4)
        lifetime = particle['lifetime']
        if lifetime == float('inf'):
            lifetime_str = "стабильна"
        elif lifetime > 1:
            lifetime_str = f"{lifetime:.2f} с"
        elif lifetime > 1e-3:
            lifetime_str = f"{lifetime*1000:.2f} мс"
        elif lifetime > 1e-6:
            lifetime_str = f"{lifetime*1e6:.2f} мкс"
        elif lifetime > 1e-9:
            lifetime_str = f"{lifetime*1e9:.2f} нс"
        else:
            lifetime_str = f"{lifetime*1e12:.2f} пс"
        plt.text(0.5, 0.5, lifetime_str, 
                 fontsize=16, ha='center', va='center')
        plt.title(f"Время жизни: {particle['symbol']}")
        plt.axis('off')
        plt.suptitle(f"Свойства частицы: {particle['symbol']} ({name})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"particle_{name}_properties.png")
        logger.info(f"Particle properties visualization saved to 'particle_{name}_properties.png'")
    
    def save_to_json(self, filename: str = "particle_database.json"):
        """
        Сохранение базы данных частиц в JSON-файл.
        Параметры:
        filename: имя файла для сохранения
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.particles, f, indent=4)
            logger.info(f"Particle database saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving particle database: {e}")

# ===================================================================
# 2. Конструктор геометрии эксперимента (OpenParticleLab)
# ===================================================================
class GeometryBuilder:
    """Конструктор геометрии эксперимента для OpenParticleLab"""
    
    def __init__(self):
        """Инициализация конструктора геометрии"""
        self.components = []
        logger.info("Geometry builder initialized")
    
    def add_component(self, comp_type: str, params: Dict):
        """
        Добавление компонента геометрии: 'detector', 'magnet', 'target', 'beamline'
        Параметры:
        comp_type: тип компонента
        params: параметры компонента
        """
        component = {
            'type': comp_type,
            'params': params,
            'position': params.get('position', (0, 0, 0)),
            'orientation': params.get('orientation', (0, 0, 0))
        }
        self.components.append(component)
        logger.info(f"Added {comp_type} component at position {component['position']}")
    
    def build(self) -> List[Dict]:
        """
        Построение геометрии эксперимента
        Возвращает:
        Список компонентов геометрии
        """
        logger.info(f"Geometry built with {len(self.components)} components")
        return self.components
    
    def load_from_yaml(self, yaml_file: str):
        """
        Загрузка геометрии из YAML-файла
        Параметры:
        yaml_file: путь к YAML-файлу
        """
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
                for comp in config.get('geometry', []):
                    self.add_component(comp['type'], comp['params'])
            logger.info(f"Geometry loaded from {yaml_file}")
        except Exception as e:
            logger.error(f"Error loading geometry from YAML: {e}")
    
    def visualize(self):
        """Визуализация геометрии эксперимента"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Цвета для разных типов компонентов
        colors = {
            'detector': 'blue',
            'magnet': 'red',
            'target': 'green',
            'beamline': 'purple'
        }
        
        # Рисование компонентов
        for comp in self.components:
            x, y, z = comp['position']
            comp_type = comp['type']
            
            if comp_type == 'detector':
                size = comp['params'].get('size', (1, 1, 1))
                # Рисуем прямоугольник для детектора
                r = [-size[0]/2, size[0]/2]
                for s, e in [(s, e) for s in r for e in r]:
                    ax.plot3D([x+s, x+e], [y-r[0], y-r[0]], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+s, x+e], [y+r[0], y+r[0]], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+s, x+e], [y-r[0], y-r[0]], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+s, x+e], [y+r[0], y+r[0]], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y+s, y+e], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y+s, y+e], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y+s, y+e], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y+s, y+e], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y-r[1], y-r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y-r[1], y-r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y+r[1], y+r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y+r[1], y+r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
            elif comp_type == 'magnet':
                radius = comp['params'].get('radius', 1.0)
                height = comp['params'].get('height', 2.0)
                # Рисуем цилиндр для магнита
                phi = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(-height/2, height/2, 10)
                X = np.outer(np.cos(phi), np.ones(len(z_cyl))) * radius + x
                Y = np.outer(np.sin(phi), np.ones(len(z_cyl))) * radius + y
                Z = np.outer(np.ones(len(phi)), z_cyl) + z
                ax.plot_surface(X, Y, Z, color=colors[comp_type], alpha=0.3)
            elif comp_type == 'target':
                size = comp['params'].get('size', (0.1, 0.1, 0.1))
                ax.scatter(x, y, z, s=size[0]*1000, c=colors[comp_type], label=comp_type)
            elif comp_type == 'beamline':
                length = comp['params'].get('length', 10.0)
                # Рисуем линию для пучка
                ax.plot([x, x+length], [y, y], [z, z], 'k-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title('Экспериментальная геометрия')
        plt.legend()
        plt.savefig('experiment_geometry.png')
        logger.info("Experiment geometry visualization saved to 'experiment_geometry.png'")

# ===================================================================
# 3. Движок физических процессов (OpenParticleLab)
# ===================================================================
class PhysicsEngine:
    """Движок физических процессов для OpenParticleLab"""
    
    def __init__(self, particle_db: ParticleDatabase):
        """Инициализация физического движка"""
        self.particle_db = particle_db
        self.models = {
            'standard': self._standard_model,
            'qcd': self._qcd_model,
            'electroweak': self._electroweak_model,
            'beyond_sm': self._beyond_standard_model
        }
        logger.info("Physics engine initialized with multiple theoretical models")
    
    def interact(self, particle1: str, particle2: str, energy: float, model: str = 'standard') -> List[Dict]:
        """
        Моделирование взаимодействия частиц.
        Параметры:
        particle1: тип первой частицы
        particle2: тип второй частицы
        energy: энергия взаимодействия в эВ
        model: используемая теоретическая модель
        Возвращает:
        Список продуктов взаимодействия
        """
        if model not in self.models:
            logger.warning(f"Model '{model}' not found. Using 'standard' model.")
            model = 'standard'
        
        logger.info(f"Simulating interaction: {particle1} + {particle2} at {energy/1e12:.2f} TeV using {model} model")
        return self.models[model](particle1, particle2, energy)
    
    def _standard_model(self, p1: str, p2: str, energy: float) -> List[Dict]:
        """Стандартная модель физики частиц"""
        products = []
        
        # Определение типа взаимодействия
        if p1 == p2 == 'proton':
            # Протон-протонные столкновения
            if energy > 1e12:  # 1 ТэВ
                # Вероятности различных процессов
                process_prob = {
                    'elastic': 0.25,
                    'inelastic': 0.65,
                    'higgs_production': 0.00000015 * (energy/1e12 - 0.125),
                    'top_production': 0.00000008 * (energy/1e12 - 0.173),
                    'z_production': 0.0000012 * (energy/1e12 - 0.091),
                    'w_production': 0.000002 * (energy/1e12 - 0.08)
                }
                
                # Нормализация вероятностей
                total = sum(process_prob.values())
                for proc in process_prob:
                    process_prob[proc] /= total
                
                # Выбор процесса
                r = np.random.random()
                cum_prob = 0
                selected_process = None
                for proc, prob in process_prob.items():
                    cum_prob += prob
                    if r < cum_prob:
                        selected_process = proc
                        break
                
                # Генерация продуктов в зависимости от процесса
                if selected_process == 'elastic':
                    # Упругое рассеяние
                    products.append({'name': 'proton', 'energy': energy * 0.49, 'momentum': energy * 0.49 / c})
                    products.append({'name': 'proton', 'energy': energy * 0.49, 'momentum': energy * 0.49 / c})
                
                elif selected_process == 'inelastic':
                    # Неупругое рассеяние
                    num_products = np.random.randint(5, 20)
                    for _ in range(num_products):
                        particle = np.random.choice(['pion_plus', 'kaon_plus', 'muon', 'electron', 'photon'])
                        fraction = np.random.uniform(0.01, 0.1)
                        products.append({
                            'name': particle,
                            'energy': energy * fraction,
                            'momentum': energy * fraction / c
                        })
                
                elif selected_process == 'higgs_production':
                    # Производство бозона Хиггса
                    products.append({'name': 'higgs_boson', 'energy': energy * 0.8, 'momentum': energy * 0.8 / c})
                    # Распад бозона Хиггса
                    decay_products = self.particle_db.get_particle_decay_products('higgs_boson')
                    for particle, prob in decay_products:
                        if np.random.random() < prob:
                            fraction = np.random.uniform(0.05, 0.2)
                            products.append({
                                'name': particle,
                                'energy': energy * fraction,
                                'momentum': energy * fraction / c
                            })
                
                elif selected_process in ['top_production', 'z_production', 'w_production']:
                    # Производство тяжелых бозонов
                    boson = 'top_quark' if selected_process == 'top_production' else \
                            'z_boson' if selected_process == 'z_production' else 'w_boson'
                    products.append({'name': boson, 'energy': energy * 0.7, 'momentum': energy * 0.7 / c})
                    # Распад бозона
                    decay_products = self.particle_db.get_particle_decay_products(boson)
                    for particle, prob in decay_products:
                        if np.random.random() < prob:
                            fraction = np.random.uniform(0.05, 0.2)
                            products.append({
                                'name': particle,
                                'energy': energy * fraction,
                                'momentum': energy * fraction / c
                            })
        
        elif p1 == 'electron' and p2 == 'positron':
            # e+e- аннигиляция
            if energy > 1e10:  # 10 ГэВ
                # Создание Z-бозона
                products.append({'name': 'z_boson', 'energy': energy * 0.95, 'momentum': 0})
                # Распад Z-бозона
                decay_products = self.particle_db.get_particle_decay_products('z_boson')
                for particle, prob in decay_products:
                    if np.random.random() < prob:
                        fraction = np.random.uniform(0.05, 0.3)
                        products.append({
                            'name': particle,
                            'energy': energy * fraction,
                            'momentum': energy * fraction / c
                        })
        
        return products
    
    def _qcd_model(self, p1: str, p2: str, energy: float) -> List[Dict]:
        """Квантовая хромодинамика (QCD) модель"""
        # Реалистичная QCD модель для адронных столкновений
        products = []
        
        if p1 == p2 == 'proton':
            # Генерация струй (jets)
            num_jets = np.random.randint(2, 4)
            jet_energies = []
            
            # Распределение энергии между струями
            remaining_energy = energy * 0.9
            for i in range(num_jets):
                if i == num_jets - 1:
                    jet_energy = remaining_energy
                else:
                    fraction = np.random.uniform(0.1, 0.4)
                    jet_energy = energy * fraction
                    remaining_energy -= jet_energy
                    jet_energies.append(jet_energy)
            
            # Генерация частиц в струях
            for jet_energy in jet_energies:
                # Тип струи (световые кварки, глюоны, b-кварки)
                jet_type = np.random.choice(['light', 'gluon', 'b', 'c'], p=[0.6, 0.3, 0.07, 0.03])
                
                # Количество частиц в струе
                num_particles = max(1, int(np.random.normal(10, 5)))
                
                for _ in range(num_particles):
                    # Выбор частицы в струе
                    if jet_type == 'light':
                        particle = np.random.choice(['pion_plus', 'pion_minus', 'kaon_plus', 'proton'])
                    elif jet_type == 'gluon':
                        particle = np.random.choice(['pion_plus', 'pion_minus', 'gluon'])
                    elif jet_type == 'b':
                        particle = np.random.choice(['pion_plus', 'b_hadron'])
                    else:  # 'c'
                        particle = np.random.choice(['pion_plus', 'c_hadron'])
                    
                    # Доля энергии для частицы
                    particle_fraction = np.random.beta(1, 3)
                    particle_energy = jet_energy * particle_fraction
                    
                    products.append({
                        'name': particle,
                        'energy': particle_energy,
                        'momentum': particle_energy / c,
                        'source': 'jet'
                    })
        
        return products
    
    def _electroweak_model(self, p1: str, p2: str, energy: float) -> List[Dict]:
        """Электрослабая модель"""
        products = []
        
        # Электрослабые взаимодействия
        if p1 == 'electron' and p2 == 'electron':
            # e-e- рассеяние
            if energy > 1e9:  # 1 ГэВ
                # Обмен Z-бозоном
                products.append({'name': 'electron', 'energy': energy * 0.49, 'momentum': energy * 0.49 / c})
                products.append({'name': 'electron', 'energy': energy * 0.49, 'momentum': energy * 0.49 / c})
                # Излучение бозона W
                if np.random.random() < 0.05 * (energy/1e9):
                    products.append({'name': 'w_boson', 'energy': energy * 0.02, 'momentum': 0})
        
        elif p1 == 'neutrino_e' and p2 == 'electron':
            # Упругое рассеяние нейтрино
            if energy > 1e8:  # 0.1 ГэВ
                products.append({'name': 'neutrino_e', 'energy': energy * 0.48, 'momentum': energy * 0.48 / c})
                products.append({'name': 'electron', 'energy': energy * 0.48, 'momentum': energy * 0.48 / c})
                # Иногда создается W-бозон
                if np.random.random() < 0.1 * (energy/1e9):
                    products.append({'name': 'w_boson', 'energy': energy * 0.04, 'momentum': 0})
        
        return products
    
    def _beyond_standard_model(self, p1: str, p2: str, energy: float) -> List[Dict]:
        """Модель за пределами Стандартной модели"""
        products = []
        
        # Гипотетические новые физические явления
        if energy > 10e12:  # 10 ТэВ
            # Вероятность новых физических явлений растет с энергией
            new_physics_prob = 0.00000001 * (energy/1e12 - 10)
            
            if np.random.random() < new_physics_prob:
                # Создание гипотетических частиц
                hypothetical_particles = ['dark_matter', 'axion', 'graviton', 'extra_dimension']
                for particle in hypothetical_particles:
                    if np.random.random() < 0.3:
                        fraction = np.random.uniform(0.05, 0.15)
                        products.append({
                            'name': particle,
                            'energy': energy * fraction,
                            'momentum': energy * fraction / c,
                            'source': 'beyond_sm'
                        })
                
                # Также могут быть созданы известные частицы
                products.append({'name': 'higgs_boson', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c})
                products.append({'name': 'z_boson', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c})
                products.append({'name': 'w_boson', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c})
        
        return products

# ===================================================================
# 4. Симулятор отклика детектора (OpenParticleLab)
# ===================================================================
class DetectorSimulator:
    """Симулятор отклика детектора для OpenParticleLab"""
    
    def __init__(self, geometry: List[Dict], particle_db: ParticleDatabase):
        """Инициализация симулятора детектора"""
        self.geometry = geometry
        self.particle_db = particle_db
        self.detectors = self._initialize_detectors()
        logger.info("Detector simulator initialized with experimental geometry")
    
    def _initialize_detectors(self) -> Dict:
        """Инициализация детекторов на основе геометрии"""
        detectors = {}
        for component in self.geometry:
            if component['type'] == 'detector':
                params = component['params']
                detectors[params.get('name', f'detector_{len(detectors)}')] = {
                    'position': component['position'],
                    'size': params.get('size', (1, 1, 1)),
                    'efficiency': params.get('efficiency', 0.95),
                    'resolution': params.get('resolution', 0.01),
                    'material': params.get('material', 'silicon'),
                    'subdetectors': params.get('subdetectors', []),
                    'energy_resolution': params.get('energy_resolution', {
                        'electromagnetic': lambda E: np.sqrt(0.03**2 + (0.005/E)**2) if E > 0 else 0.03,
                        'hadronic': lambda E: np.sqrt(0.1**2 + (0.03/E)**2) if E > 0 else 0.1
                    }),
                    'tracking_resolution': params.get('tracking_resolution', lambda pT: 0.007 * (1 + pT/150))
                }
        return detectors
    
    def detect(self, particles: List[Dict], event_id: int) -> List[Dict]:
        """
        Регистрация частиц детектором.
        Параметры:
        particles: список частиц для детектирования
        event_id: ID события
        Возвращает:
        Список зарегистрированных частиц
        """
        detected = []
        
        for p in particles:
            particle_name = p['name']
            true_energy = p['energy']
            true_momentum = p['momentum']
            
            # Определение, детектируется ли частица
            detected_by = []
            for detector_name, detector in self.detectors.items():
                # Эффективность детектирования зависит от типа частицы и поддетектора
                efficiency = detector['efficiency']
                if 'lepton' in self.particle_db.get_category(particle_name):
                    efficiency *= 1.1  # Лептоны лучше детектируются
                elif 'quark' in self.particle_db.get_category(particle_name):
                    efficiency *= 0.8  # Кварки хуже детектируются напрямую
                
                if np.random.random() < efficiency:
                    detected_by.append(detector_name)
            
            if not detected_by:
                # Частица не была обнаружена (например, нейтрино)
                continue
            
            # Реконструируемая энергия с учетом разрешения детектора
            if 'photon' in particle_name or 'electron' in particle_name:
                resolution_func = self.detectors[detected_by[0]]['energy_resolution']['electromagnetic']
            else:
                resolution_func = self.detectors[detected_by[0]]['energy_resolution']['hadronic']
            
            # Стохастический компонент разрешения
            stochastic = np.random.normal(0, resolution_func(true_energy/1e9))
            reconstructed_energy = true_energy * (1 + stochastic)
            
            # Разрешение по импульсу для заряженных частиц
            if self.particle_db.get_charge(particle_name) != 0:
                pT = true_momentum * np.random.uniform(0.5, 1.0)  # Поперечный импульс
                momentum_resolution = self.detectors[detected_by[0]]['tracking_resolution'](pT/1e9)
                reconstructed_momentum = true_momentum * (1 + np.random.normal(0, momentum_resolution))
            else:
                reconstructed_momentum = reconstructed_energy / c
            
            # Добавление зарегистрированной частицы
            detected.append({
                'name': particle_name,
                'true_energy': true_energy,
                'reconstructed_energy': reconstructed_energy,
                'true_momentum': true_momentum,
                'reconstructed_momentum': reconstructed_momentum,
                'detected_by': detected_by,
                'position': self._calculate_position(p),
                'event_id': event_id,
                'timestamp': time.time()
            })
        
        return detected
    
    def _calculate_position(self, particle: Dict) -> Tuple[float, float, float]:
        """Расчет точки взаимодействия с детектором"""
        # Упрощенная модель для примера
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
            true_momentum = product['momentum']
            
            # Определение, детектируется ли частица
            detected = False
            for detector_name, detector in self.detectors.items():
                # Эффективность детектирования зависит от типа частицы и поддетектора
                efficiency = detector['efficiency']
                if 'lepton' in self.particle_db.get_category(particle_name):
                    efficiency *= 1.1  # Лептоны лучше детектируются
                elif 'quark' in self.particle_db.get_category(particle_name):
                    efficiency *= 0.8  # Кварки хуже детектируются напрямую
                    
                if np.random.random() < efficiency:
                    detected = True
                    break
            
            if not detected:
                # Частица не была обнаружена (например, нейтрино)
                reconstructed['missing_energy'] += true_energy
                continue
            
            # Реконструируемая энергия с учетом разрешения детектора
            if 'photon' in particle_name or 'electron' in particle_name:
                resolution_func = self.detectors[detector_name]['energy_resolution']['electromagnetic']
            else:
                resolution_func = self.detectors[detector_name]['energy_resolution']['hadronic']
            
            # Стохастический компонент разрешения
            stochastic = np.random.normal(0, resolution_func(true_energy/1e9))
            reconstructed_energy = true_energy * (1 + stochastic)
            
            # Разрешение по импульсу для заряженных частиц
            if self.particle_db.get_charge(particle_name) != 0:
                pT = true_momentum * np.random.uniform(0.5, 1.0)  # Поперечный импульс
                momentum_resolution = self.detectors[detector_name]['tracking_resolution'](pT/1e9)
                reconstructed_momentum = true_momentum * (1 + np.random.normal(0, momentum_resolution))
            else:
                reconstructed_momentum = reconstructed_energy / c
            
            # Добавление реконструированной частицы
            reconstructed['reconstructed_products'].append({
                'name': particle_name,
                'reconstructed_energy': reconstructed_energy,
                'reconstructed_momentum': reconstructed_momentum,
                'true_energy': true_energy,
                'true_momentum': true_momentum,
                'detector': detector_name,
                'position': self._calculate_position(product)
            })
        
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
        ax1.arrow(0.2, 0.5, 0.3, 0, head_width=0.03, head_length=0.05, fc='blue', ec='blue')
        ax1.arrow(0.2, 0.5, 0.3, 0.1, head_width=0.03, head_length=0.05, fc='blue', ec='blue')
        # Точка столкновения
        ax1.plot(0.5, 0.5, 'ro', markersize=8)
        # Выходные частицы
        num_products = len(event['products'])
        angles = np.linspace(-np.pi/3, np.pi/3, num_products)
        for i, angle in enumerate(angles):
            length = 0.4 + 0.1 * np.random.random()
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            particle = event['products'][i]
            color = 'green' if 'boson' in particle['name'] else 'purple' if 'quark' in particle['name'] else 'orange'
            ax1.arrow(0.5, 0.5, dx, dy, head_width=0.03, head_length=0.05, fc=color, ec=color)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Диаграмма столкновения')
        ax1.axis('off')
        
        # 2. Энергетический спектр
        ax2 = plt.subplot(2, 2, 2)
        true_energies = [p['energy']/1e9 for p in event['products']]  # в ГэВ
        reconstructed_energies = []
        for p in event['products']:
            # Имитация детектирования и реконструкции
            if np.random.random() < 0.9:  # 90% детектирования
                if 'photon' in p['name'] or 'electron' in p['name']:
                    resolution = 0.03
                else:
                    resolution = 0.1
                reconstructed_energy = p['energy']/1e9 * (1 + np.random.normal(0, resolution))
                reconstructed_energies.append(reconstructed_energy)
        
        particles = [self.particle_db.get_symbol(p['name']) for p in event['products']]
        y_pos = np.arange(len(true_energies))
        ax2.barh(y_pos, true_energies, align='center', alpha=0.5, color='blue', label='Истинная энергия')
        if reconstructed_energies:
            ax2.barh(y_pos[:len(reconstructed_energies)], reconstructed_energies, 
                    align='center', alpha=0.5, color='red', label='Измеренная энергия')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(particles)
        ax2.set_xlabel('Энергия (ГэВ)')
        ax2.set_title('Энергетический спектр продуктов')
        ax2.legend()
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 3. Распределение по детекторам
        ax3 = plt.subplot(2, 2, 3)
        detector_counts = {}
        for product in event['products']:
            for detector in self.detectors.keys():
                if np.random.random() < 0.9:  # 90% детектирования
                    detector_counts[detector] = detector_counts.get(detector, 0) + 1
        
        if detector_counts:
            ax3.pie(detector_counts.values(), labels=detector_counts.keys(), autopct='%1.1f%%')
            ax3.set_title('Распределение событий по детекторам')
        else:
            ax3.text(0.5, 0.5, 'Нет зарегистрированных событий', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Распределение событий по детекторам')
        ax3.axis('equal')
        
        # 4. Информация о событии
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        # Заголовок
        ax4.text(0.5, 0.95, f"Событие: {event['event_type'].replace('_', ' ').title()}",
                 fontsize=14, ha='center', weight='bold')
        # Энергия
        ax4.text(0.1, 0.85, f"Энергия: {event['energy']/1e12:.1f} ТэВ", fontsize=12)
        # Продукты
        ax4.text(0.1, 0.75, "Основные продукты:", fontsize=12, weight='bold')
        for i, product in enumerate(event['products'][:5]):  # Показываем первые 5 продуктов
            symbol = self.particle_db.get_symbol(product['name'])
            energy = product['energy']/1e9  # в ГэВ
            ax4.text(0.15, 0.7 - i*0.08, f"- {symbol}: {energy:.2f} ГэВ", fontsize=11)
        if len(event['products']) > 5:
            ax4.text(0.15, 0.7 - 5*0.08, f"- и еще {len(event['products'])-5} частиц", fontsize=11)
        
        plt.tight_layout()
        plt.savefig('detector_response.png')
        logger.info("Detector response visualization saved to 'detector_response.png'")

# ===================================================================
# 5. Расчет магнитных полей (OpenParticleLab)
# ===================================================================
class MagneticFieldSolver:
    """Решатель магнитных полей для OpenParticleLab"""
    
    def __init__(self, geometry: List[Dict]):
        """Инициализация решателя магнитных полей"""
        self.geometry = geometry
        self.field_map = {}
        self.magnets = self._initialize_magnets()
        logger.info("Magnetic field solver initialized with experimental geometry")
    
    def _initialize_magnets(self) -> List[Dict]:
        """Инициализация магнитов на основе геометрии"""
        magnets = []
        for component in self.geometry:
            if component['type'] == 'magnet':
                params = component['params']
                magnets.append({
                    'position': component['position'],
                    'orientation': component['orientation'],
                    'strength': params.get('strength', 1.0),  # Тесла
                    'type': params.get('type', 'dipole'),
                    'length': params.get('length', 1.0),
                    'radius': params.get('radius', 0.5)
                })
        return magnets
    
    def calculate_field(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Расчет вектора магнитного поля в заданной точке.
        Параметры:
        position: позиция (x, y, z) в метрах
        Возвращает:
        Вектор магнитного поля (Bx, By, Bz) в Тесла
        """
        total_field = np.array([0.0, 0.0, 0.0])
        
        for magnet in self.magnets:
            # Расчет расстояния от точки до магнита
            magnet_pos = np.array(magnet['position'])
            point_pos = np.array(position)
            r = point_pos - magnet_pos
            
            # Расстояние до магнита
            dist = np.linalg.norm(r)
            
            # Если точка внутри магнита
            if dist < magnet['radius']:
                # Однородное поле внутри дипольного магнита
                if magnet['type'] == 'dipole':
                    # Направление поля зависит от ориентации магнита
                    orientation = np.array(magnet['orientation'])
                    if np.linalg.norm(orientation) == 0:
                        orientation = np.array([0, 0, 1])  # По умолчанию вдоль оси Z
                    else:
                        orientation = orientation / np.linalg.norm(orientation)
                    
                    field = magnet['strength'] * orientation
                    total_field += field
            
            # Вне магнита (упрощенная модель)
            else:
                if magnet['type'] == 'dipole':
                    # Дипольное поле уменьшается как 1/r^3
                    factor = magnet['strength'] * magnet['radius']**3 / dist**3
                    # Дипольное поле: B = (μ₀/4π) * [3(m·r̂)r̂ - m]/r³
                    m = np.array(magnet['orientation']) * magnet['strength']
                    if np.linalg.norm(m) == 0:
                        m = np.array([0, 0, 1]) * magnet['strength']
                    r_hat = r / dist
                    field = 3 * np.dot(m, r_hat) * r_hat - m
                    total_field += field * factor
        
        return tuple(total_field)
    
    def track_particle(self, particle: Dict, start_position: Tuple[float, float, float] = None, 
                      num_steps: int = 100) -> List[Tuple[float, float, float]]:
        """
        Трассировка частицы в магнитном поле.
        Параметры:
        particle: данные о частице
        start_position: начальная позиция (если None, используется позиция из данных частицы)
        num_steps: количество шагов трассировки
        Возвращает:
        Список позиций частицы вдоль траектории
        """
        positions = []
        
        # Получение данных о частице
        particle_name = particle['name']
        energy = particle['energy']
        momentum = particle['momentum']
        
        # Начальная позиция
        if start_position is None:
            start_position = (0, 0, 0)  # По умолчанию начало координат
        
        # Начальная скорость (предполагаем ультрарелятивистскую частицу)
        beta = 1.0  # Для упрощения
        speed = c * beta
        
        # Направление движения (предполагаем движение вдоль оси Z)
        direction = np.array([0, 0, 1])
        if momentum > 0:
            direction = np.array([0, 0, 1])  # Вдоль оси Z
        
        # Начальная позиция и скорость
        pos = np.array(start_position)
        vel = direction * speed
        
        # Шаг по времени
        dt = 1e-12  # 1 пс
        
        # Трассировка
        for _ in range(num_steps):
            # Расчет магнитного поля в текущей позиции
            B = np.array(self.calculate_field(tuple(pos)))
            
            # Заряд частицы
            charge = self.particle_db.get_charge(particle_name)
            
            # Уравнение движения: F = q(v × B)
            force = charge * np.cross(vel, B) * e  # e - элементарный заряд
            
            # Ускорение
            mass = self.particle_db.get_mass_in_gev(particle_name) * 1e9 * e / c**2  # в кг
            acc = force / mass
            
            # Обновление скорости и позиции
            vel += acc * dt
            pos += vel * dt
            
            positions.append(tuple(pos))
        
        return positions
    
    def visualize_field_lines(self, num_lines: int = 20, num_points: int = 50):
        """
        Визуализация линий магнитного поля.
        Параметры:
        num_lines: количество линий поля
        num_points: количество точек на каждой линии
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Falling back to basic visualization.")
            self._basic_field_visualization()
            return
        
        try:
            # Создание фигуры
            fig = go.Figure()
            
            # Генерация линий магнитного поля
            for i in range(num_lines):
                # Начальная позиция (на поверхности магнита)
                magnet = self.magnets[0] if self.magnets else {'position': (0,0,0), 'radius': 1.0}
                theta = 2 * np.pi * i / num_lines
                phi = np.pi / 4  # Угол наклона
                x0 = magnet['position'][0] + magnet['radius'] * np.cos(theta) * np.sin(phi)
                y0 = magnet['position'][1] + magnet['radius'] * np.sin(theta) * np.sin(phi)
                z0 = magnet['position'][2] + magnet['radius'] * np.cos(phi)
                
                # Трассировка линии поля
                pos = np.array([x0, y0, z0])
                points = [pos]
                
                for _ in range(num_points):
                    B = np.array(self.calculate_field(tuple(pos)))
                    if np.linalg.norm(B) == 0:
                        break
                    # Нормализация вектора поля для направления
                    B_dir = B / np.linalg.norm(B)
                    # Следующая точка
                    pos = pos + B_dir * 0.1
                    points.append(pos)
                
                points = np.array(points)
                
                # Добавление линии на график
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='lines',
                    line=dict(width=3, color=f'rgb({i*10 % 255}, {i*5 % 255}, {255-i*5 % 255})'),
                    name=f'Field Line {i+1}'
                ))
            
            # Добавление магнитов
            for magnet in self.magnets:
                x, y, z = magnet['position']
                r = magnet['radius']
                h = magnet['length']
                
                # Цилиндр для магнита
                phi = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(-h/2, h/2, 10)
                X = np.outer(np.cos(phi), np.ones(len(z_cyl))) * r + x
                Y = np.outer(np.sin(phi), np.ones(len(z_cyl))) * r + y
                Z = np.outer(np.ones(len(phi)), z_cyl) + z
                
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.3
                ))
            
            # Настройка макета
            fig.update_layout(
                title='Линии магнитного поля',
                scene=dict(
                    xaxis_title='X (м)',
                    yaxis_title='Y (м)',
                    zaxis_title='Z (м)'
                ),
                width=900,
                height=700
            )
            
            # Сохранение и отображение
            fig.write_html("magnetic_field_lines.html")
            fig.show()
            
            logger.info("Magnetic field visualization saved to 'magnetic_field_lines.html'")
        except Exception as e:
            logger.warning(f"Error in magnetic field visualization: {str(e)}. Falling back to basic visualization.")
            self._basic_field_visualization()
    
    def _basic_field_visualization(self):
        """Базовая визуализация магнитного поля (если Plotly недоступен)"""
        plt.figure(figsize=(12, 10))
        
        # 2D срез магнитного поля
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        
        # Вычисление поля в каждой точке
        for i in range(len(x)):
            for j in range(len(y)):
                B = self.calculate_field((X[i, j], Y[i, j], 0))
                Bx[i, j] = B[0]
                By[i, j] = B[1]
        
        # Нормализация для стрелок
        magnitude = np.sqrt(Bx**2 + By**2)
        Bx_norm = Bx / (magnitude + 1e-10)
        By_norm = By / (magnitude + 1e-10)
        
        # Построение векторного поля
        plt.quiver(X, Y, Bx_norm, By_norm, magnitude, cmap='viridis')
        plt.colorbar(label='|B| (Тл)')
        
        # Рисование магнитов
        for magnet in self.magnets:
            x0, y0, z0 = magnet['position']
            r = magnet['radius']
            circle = plt.Circle((x0, y0), r, fill=False, color='red', linewidth=2)
            plt.gca().add_patch(circle)
        
        plt.title('2D срез магнитного поля')
        plt.xlabel('X (м)')
        plt.ylabel('Y (м)')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig('magnetic_field_2d.png')
        logger.info("Basic magnetic field visualization saved to 'magnetic_field_2d.png'")

# ===================================================================
# 6. Анализатор данных (OpenParticleLab)
# ===================================================================
class DataAnalyzer:
    """Анализатор данных для OpenParticleLab (ROOT-подобный)"""
    
    def __init__(self):
        """Инициализация анализатора данных"""
        self.histograms = {}
        self.roi = {}  # Region of Interest
        logger.info("Data analyzer initialized")
    
    def create_histogram(self, name: str, data: List[float], bins: int = 50, 
                        range: Optional[Tuple[float, float]] = None):
        """
        Создание гистограммы.
        Параметры:
        name: имя гистограммы
         данные для гистограммы
        bins: количество бинов
        range: диапазон значений
        """
        hist, edges = np.histogram(data, bins=bins, range=range)
        self.histograms[name] = {
            'values': hist.tolist(),
            'edges': edges.tolist(),
            'bins': bins,
            'range': range
        }
        logger.info(f"Histogram '{name}' created with {len(data)} entries")
    
    def plot_histogram(self, name: str, log_scale: bool = False, 
                      title: Optional[str] = None, save: bool = True):
        """
        Визуализация гистограммы.
        Параметры:
        name: имя гистограммы
        log_scale: использовать логарифмический масштаб
        title: заголовок графика
        save: сохранить изображение
        """
        if name not in self.histograms:
            logger.error(f"Histogram '{name}' not found")
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
        plt.title(title if title else f"Histogram: {name}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Сохранение
        if save:
            plt.savefig(f"{name}_histogram.png")
            logger.info(f"Histogram '{name}' visualization saved to '{name}_histogram.png'")
        
        plt.show()
    
    def create_2d_histogram(self, name: str, x_data: List[float], y_data: List[float], 
                           x_bins: int = 50, y_bins: int = 50,
                           x_range: Optional[Tuple[float, float]] = None,
                           y_range: Optional[Tuple[float, float]] = None):
        """
        Создание 2D гистограммы.
        Параметры:
        name: имя гистограммы
        x_data: данные по оси X
        y_data: данные по оси Y
        x_bins: количество бинов по X
        y_bins: количество бинов по Y
        x_range: диапазон значений по X
        y_range: диапазон значений по Y
        """
        hist, x_edges, y_edges = np.histogram2d(x_data, y_data, 
                                               bins=[x_bins, y_bins],
                                               range=[x_range, y_range])
        self.histograms[name] = {
            'values': hist.tolist(),
            'x_edges': x_edges.tolist(),
            'y_edges': y_edges.tolist(),
            'x_bins': x_bins,
            'y_bins': y_bins,
            'x_range': x_range,
            'y_range': y_range
        }
        logger.info(f"2D histogram '{name}' created with {len(x_data)} entries")
    
    def plot_2d_histogram(self, name: str, log_scale: bool = False,
                         title: Optional[str] = None, save: bool = True):
        """
        Визуализация 2D гистограммы.
        Параметры:
        name: имя гистограммы
        log_scale: использовать логарифмический масштаб
        title: заголовок графика
        save: сохранить изображение
        """
        if name not in self.histograms:
            logger.error(f"2D histogram '{name}' not found")
            return
        
        hist = self.histograms[name]
        plt.figure(figsize=(10, 8))
        
        # Построение 2D гистограммы
        if log_scale:
            plt.pcolormesh(hist['x_edges'], hist['y_edges'], 
                          np.log10(np.array(hist['values']) + 1),
                          cmap='viridis')
            plt.colorbar(label='log(Count + 1)')
        else:
            plt.pcolormesh(hist['x_edges'], hist['y_edges'], 
                          hist['values'], cmap='viridis')
            plt.colorbar(label='Count')
        
        # Заголовок и метки
        plt.title(title if title else f"2D Histogram: {name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Сохранение
        if save:
            plt.savefig(f"{name}_2d_histogram.png")
            logger.info(f"2D histogram '{name}' visualization saved to '{name}_2d_histogram.png'")
        
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
            logger.error(f"Histogram '{name}' not found")
            return None
        
        hist = self.histograms[name]
        x = (np.array(hist['edges'][:-1]) + np.array(hist['edges'][1:])) / 2
        y = np.array(hist['values'])
        
        # Фильтрация по диапазону
        if range:
            mask = (x >= range[0]) & (x <= range[1])
            x = x[mask]
            y = y[mask]
        
        # Фитирование гауссианой
        if function == 'gaussian':
            from scipy.optimize import curve_fit
            
            # Начальные параметры: [амплитуда, среднее, ширина]
            initial_guess = [max(y), x[np.argmax(y)], (x[-1] - x[0]) / 10]
            
            # Определение функции гауссианы
            def gaussian(x, a, mu, sigma):
                return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
            
            # Фитирование
            try:
                popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess)
                logger.info(f"Gaussian fit for '{name}': A={popt[0]:.2f}, μ={popt[1]:.2f}, σ={popt[2]:.2f}")
                return {
                    'function': 'gaussian',
                    'parameters': {'amplitude': popt[0], 'mean': popt[1], 'sigma': popt[2]},
                    'covariance': pcov.tolist()
                }
            except Exception as e:
                logger.error(f"Error fitting histogram '{name}': {e}")
                return None
        
        # Другие функции фитирования могут быть добавлены здесь
        logger.error(f"Unsupported fit function: {function}")
        return None
    
    def create_roi(self, name: str, x_range: Tuple[float, float], 
                  y_range: Optional[Tuple[float, float]] = None,
                  z_range: Optional[Tuple[float, float]] = None):
        """
        Создание области интереса (Region of Interest).
        Параметры:
        name: имя ROI
        x_range: диапазон по оси X
        y_range: диапазон по оси Y (опционально)
        z_range: диапазон по оси Z (опционально)
        """
        self.roi[name] = {
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range
        }
        logger.info(f"ROI '{name}' created with ranges: X={x_range}, Y={y_range}, Z={z_range}")
    
    def apply_roi(self, name: str, data: List[Dict]) -> List[Dict]:
        """
        Применение области интереса к данным.
        Параметры:
        name: имя ROI
         данные для фильтрации
        Возвращает:
        Отфильтрованные данные
        """
        if name not in self.roi:
            logger.error(f"ROI '{name}' not found")
            return data
        
        roi = self.roi[name]
        filtered_data = []
        
        for item in data:
            # Проверка по оси X
            if 'x' in item and (item['x'] < roi['x_range'][0] or item['x'] > roi['x_range'][1]):
                continue
            
            # Проверка по оси Y (если указана)
            if roi['y_range'] and 'y' in item and \
               (item['y'] < roi['y_range'][0] or item['y'] > roi['y_range'][1]):
                continue
            
            # Проверка по оси Z (если указана)
            if roi['z_range'] and 'z' in item and \
               (item['z'] < roi['z_range'][0] or item['z'] > roi['z_range'][1]):
                continue
            
            filtered_data.append(item)
        
        logger.info(f"ROI '{name}' applied to {len(data)} items, {len(filtered_data)} items remain")
        return filtered_data
    
    def export_to_root(self, filename: str = "analysis_results.root"):
        """
        Экспорт результатов анализа в формат ROOT.
        Параметры:
        filename: имя файла для экспорта
        """
        try:
            import ROOT
            import array
            
            # Создание файла ROOT
            root_file = ROOT.TFile(filename, "RECREATE")
            
            # Создание дерева для гистограмм
            tree = ROOT.TTree("AnalysisResults", "Results of data analysis")
            
            # Определение переменных
            hist_name = ROOT.std.string()
            hist_bins = array.array('i', [0])
            hist_entries = array.array('i', [0])
            hist_mean = array.array('d', [0.0])
            hist_rms = array.array('d', [0.0])
            
            # Создание веток
            tree.Branch("hist_name", hist_name)
            tree.Branch("bins", hist_bins, "bins/I")
            tree.Branch("entries", hist_entries, "entries/I")
            tree.Branch("mean", hist_mean, "mean/D")
            tree.Branch("rms", hist_rms, "rms/D")
            
            # Заполнение дерева данными
            for name, hist in self.histograms.items():
                values = np.array(hist['values'])
                entries = np.sum(values)
                if entries > 0:
                    # Вычисление среднего и RMS
                    x = (np.array(hist['edges'][:-1]) + np.array(hist['edges'][1:])) / 2
                    mean = np.sum(x * values) / entries
                    rms = np.sqrt(np.sum((x - mean)**2 * values) / entries)
                    
                    # Заполнение переменных
                    hist_name = ROOT.std.string(name)
                    hist_bins[0] = len(hist['values'])
                    hist_entries[0] = int(entries)
                    hist_mean[0] = mean
                    hist_rms[0] = rms
                    
                    # Добавление в дерево
                    tree.Fill()
            
            # Сохранение и закрытие файла
            root_file.Write()
            root_file.Close()
            
            logger.info(f"Analysis results exported to ROOT format: {filename}")
            return True
        except ImportError:
            logger.warning("ROOT not installed. Cannot export to ROOT format.")
            return False
        except Exception as e:
            logger.error(f"Error exporting to ROOT: {e}")
            return False

# ===================================================================
# 7. Визуализация (OpenParticleLab)
# ===================================================================
class Visualizer:
    """Интерактивная визуализация для OpenParticleLab"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        logger.info("Visualizer initialized")
    
    def plot_geometry(self, geometry: List[Dict]):
        """
        Визуализация геометрии эксперимента.
        Параметры:
        geometry: геометрия эксперимента
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Falling back to basic visualization.")
            self._basic_geometry_visualization(geometry)
            return
        
        try:
            fig = go.Figure()
            
            # Цвета для разных типов компонентов
            colors = {
                'detector': 'blue',
                'magnet': 'red',
                'target': 'green',
                'beamline': 'purple'
            }
            
            # Рисование компонентов
            for comp in geometry:
                x, y, z = comp['position']
                comp_type = comp['type']
                params = comp['params']
                
                if comp_type == 'detector':
                    size = params.get('size', (1, 1, 1))
                    # Рисуем прямоугольник для детектора
                    fig.add_trace(go.Mesh3d(
                        x=[x - size[0]/2, x - size[0]/2, x + size[0]/2, x + size[0]/2,
                           x - size[0]/2, x - size[0]/2, x + size[0]/2, x + size[0]/2],
                        y=[y - size[1]/2, y + size[1]/2, y + size[1]/2, y - size[1]/2,
                           y - size[1]/2, y + size[1]/2, y + size[1]/2, y - size[1]/2],
                        z=[z - size[2]/2, z - size[2]/2, z - size[2]/2, z - size[2]/2,
                           z + size[2]/2, z + size[2]/2, z + size[2]/2, z + size[2]/2],
                        i=[0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7],
                        j=[1, 2, 4, 2, 5, 6, 7, 5, 6, 1, 2, 3],
                        k=[2, 4, 5, 6, 1, 3, 3, 6, 7, 5, 7, 4],
                        color=colors[comp_type],
                        opacity=0.3,
                        name=comp_type
                    ))
                
                elif comp_type == 'magnet':
                    radius = params.get('radius', 1.0)
                    height = params.get('height', 2.0)
                    # Рисуем цилиндр для магнита
                    phi = np.linspace(0, 2*np.pi, 20)
                    z_cyl = np.linspace(-height/2, height/2, 10)
                    X = np.outer(np.cos(phi), np.ones(len(z_cyl))) * radius + x
                    Y = np.outer(np.sin(phi), np.ones(len(z_cyl))) * radius + y
                    Z = np.outer(np.ones(len(phi)), z_cyl) + z
                    
                    fig.add_trace(go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale='Viridis',
                        showscale=False,
                        opacity=0.3,
                        name=comp_type
                    ))
                
                elif comp_type == 'target':
                    size = params.get('size', (0.1, 0.1, 0.1))
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers',
                        marker=dict(size=size[0]*50, color=colors[comp_type]),
                        name=comp_type
                    ))
                
                elif comp_type == 'beamline':
                    length = params.get('length', 10.0)
                    # Рисуем линию для пучка
                    fig.add_trace(go.Scatter3d(
                        x=[x, x+length], y=[y, y], z=[z, z],
                        mode='lines',
                        line=dict(width=5, color='black'),
                        name=comp_type
                    ))
            
            # Настройка макета
            fig.update_layout(
                title='Экспериментальная геометрия',
                scene=dict(
                    xaxis_title='X (м)',
                    yaxis_title='Y (м)',
                    zaxis_title='Z (м)'
                ),
                width=900,
                height=700
            )
            
            # Сохранение и отображение
            fig.write_html("experiment_geometry_3d.html")
            fig.show()
            
            logger.info("3D experiment geometry visualization saved to 'experiment_geometry_3d.html'")
        except Exception as e:
            logger.warning(f"Error in 3D visualization: {str(e)}. Falling back to basic visualization.")
            self._basic_geometry_visualization(geometry)
    
    def _basic_geometry_visualization(self, geometry: List[Dict]):
        """Базовая визуализация геометрии (если Plotly недоступен)"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Цвета для разных типов компонентов
        colors = {
            'detector': 'blue',
            'magnet': 'red',
            'target': 'green',
            'beamline': 'purple'
        }
        
        # Рисование компонентов
        for comp in geometry:
            x, y, z = comp['position']
            comp_type = comp['type']
            
            if comp_type == 'detector':
                size = comp['params'].get('size', (1, 1, 1))
                # Рисуем прямоугольник для детектора
                r = [-size[0]/2, size[0]/2]
                for s, e in [(s, e) for s in r for e in r]:
                    ax.plot3D([x+s, x+e], [y-r[0], y-r[0]], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+s, x+e], [y+r[0], y+r[0]], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+s, x+e], [y-r[0], y-r[0]], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+s, x+e], [y+r[0], y+r[0]], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y+s, y+e], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y+s, y+e], [z-r[1], z-r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y+s, y+e], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y+s, y+e], [z+r[1], z+r[1]], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y-r[1], y-r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y-r[1], y-r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x-r[0], x-r[0]], [y+r[1], y+r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
                    ax.plot3D([x+r[0], x+r[0]], [y+r[1], y+r[1]], [z+s, z+e], color=colors[comp_type], alpha=0.5)
            elif comp_type == 'magnet':
                radius = comp['params'].get('radius', 1.0)
                height = comp['params'].get('height', 2.0)
                # Рисуем цилиндр для магнита
                phi = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(-height/2, height/2, 10)
                X = np.outer(np.cos(phi), np.ones(len(z_cyl))) * radius + x
                Y = np.outer(np.sin(phi), np.ones(len(z_cyl))) * radius + y
                Z = np.outer(np.ones(len(phi)), z_cyl) + z
                ax.plot_surface(X, Y, Z, color=colors[comp_type], alpha=0.3)
            elif comp_type == 'target':
                size = comp['params'].get('size', (0.1, 0.1, 0.1))
                ax.scatter(x, y, z, s=size[0]*1000, c=colors[comp_type], label=comp_type)
            elif comp_type == 'beamline':
                length = comp['params'].get('length', 10.0)
                # Рисуем линию для пучка
                ax.plot([x, x+length], [y, y], [z, z], 'k-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title('Экспериментальная геометрия')
        plt.legend()
        plt.savefig('experiment_geometry.png')
        logger.info("Experiment geometry visualization saved to 'experiment_geometry.png'")
    
    def plot_particle_tracks(self, tracks: List[List[Tuple]], 
                           event_info: Optional[Dict] = None):
        """
        Визуализация траекторий частиц.
        Параметры:
        tracks: список траекторий частиц
        event_info: информация о событии
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Falling back to basic visualization.")
            self._basic_tracks_visualization(tracks, event_info)
            return
        
        try:
            fig = go.Figure()
            
            # Отображение траекторий частиц
            for i, track in enumerate(tracks):
                x, y, z = zip(*track)
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    name=f'Частица {i+1}',
                    line=dict(width=4)
                ))
            
            # Добавление информации о событии
            if event_info:
                info_text = (
                    f"Событие: {event_info['event_type'].replace('_', ' ').title()}<br>"
                    f"Энергия: {event_info['energy']/1e12:.1f} ТэВ<br>"
                    f"Количество продуктов: {len(event_info['products'])}<br>"
                )
                fig.add_annotation(
                    x=0.05, y=0.95,
                    xref='paper', yref='paper',
                    text=info_text,
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                )
            
            # Настройка макета
            fig.update_layout(
                title='Траектории частиц в магнитном поле',
                scene=dict(
                    xaxis_title='X (м)',
                    yaxis_title='Y (м)',
                    zaxis_title='Z (м)'
                ),
                width=900,
                height=700
            )
            
            # Сохранение и отображение
            fig.write_html("particle_tracks.html")
            fig.show()
            
            logger.info("Particle tracks visualization saved to 'particle_tracks.html'")
        except Exception as e:
            logger.warning(f"Error in particle tracks visualization: {str(e)}. Falling back to basic visualization.")
            self._basic_tracks_visualization(tracks, event_info)
    
    def _basic_tracks_visualization(self, tracks: List[List[Tuple]], 
                                  event_info: Optional[Dict] = None):
        """Базовая визуализация траекторий частиц (если Plotly недоступен)"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Отображение траекторий частиц
        for i, track in enumerate(tracks):
            x, y, z = zip(*track)
            ax.plot(x, y, z, linewidth=2, label=f'Частица {i+1}')
        
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_title('Траектории частиц')
        plt.legend()
        
        # Добавление информации о событии
        if event_info:
            info_text = (
                f"Событие: {event_info['event_type'].replace('_', ' ').title()}\n"
                f"Энергия: {event_info['energy']/1e12:.1f} ТэВ\n"
                f"Количество продуктов: {len(event_info['products'])}"
            )
            plt.gcf().text(0.1, 0.85, info_text, fontsize=10,
                          bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig('particle_tracks.png')
        logger.info("Particle tracks visualization saved to 'particle_tracks.png'")

# ===================================================================
# 8. Основной класс OpenParticleLab
# ===================================================================
class OpenParticleLab:
    """Основной класс фреймворка OpenParticleLab"""
    
    def __init__(self):
        """Инициализация фреймворка"""
        self.particle_db = ParticleDatabase()
        self.geometry = GeometryBuilder()
        self.physics = PhysicsEngine(self.particle_db)
        self.magnetic = None
        self.detector = None
        self.analyzer = DataAnalyzer()
        self.visualizer = Visualizer()
        self.config = {}
        self.results = {
            'events': [],
            'parameters': {},
            'analysis': {}
        }
        logger.info("OpenParticleLab framework initialized")
    
    def load_config(self, config_path: str):
        """
        Загрузка конфигурации из YAML-файла.
        Параметры:
        config_path: путь к конфигурационному файлу
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            self._setup_experiment()
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
    
    def _setup_experiment(self):
        """Настройка эксперимента на основе конфигурации"""
        # Проверка наличия обязательных секций
        required_sections = ['beam', 'target', 'geometry']
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required section '{section}' in configuration")
                raise ValueError(f"Configuration must contain '{section}' section")
        
        # Построение геометрии
        self.geometry.load_from_yaml(self.config['geometry_file']) if 'geometry_file' in self.config \
            else self._build_geometry_from_config()
        
        # Инициализация магнитного поля
        self.magnetic = MagneticFieldSolver(self.geometry.build())
        
        # Инициализация детектора
        self.detector = DetectorSimulator(self.geometry.build(), self.particle_db)
        
        # Сохранение параметров
        self.results['parameters'] = {
            'beam': self.config['beam'],
            'target': self.config['target'],
            'geometry': self.geometry.build()
        }
        
        logger.info("Experiment setup completed")
    
    def _build_geometry_from_config(self):
        """Построение геометрии на основе конфигурации"""
        for comp in self.config['geometry']:
            self.geometry.add_component(comp['type'], comp['params'])
    
    def run_simulation(self, num_events: int = 1000, model: str = 'standard'):
        """
        Запуск симуляции.
        Параметры:
        num_events: количество событий для симуляции
        model: используемая физическая модель
        Возвращает:
        Список событий
        """
        logger.info(f"Starting simulation with {num_events} events using {model} model")
        start_time = time.time()
        
        events = []
        for i in range(num_events):
            if i % 100 == 0:
                logger.info(f"Processing event {i}/{num_events}")
            
            # Генерация первичных частиц
            beam_particle = self.config['beam']['particle']
            target_particle = self.config['target']['particle']
            energy = self.config['beam']['energy']
            
            # Физическое взаимодействие
            products = self.physics.interact(beam_particle, target_particle, energy, model)
            
            # Регистрация детектором
            detected = self.detector.detect(products, i)
            
            # Трассировка в магнитном поле
            tracks = []
            for p in detected:
                particle_data = self.particle_db.get_particle(p['name'])
                if particle_data:
                    # Упрощенное представление частицы для трассировки
                    track_particle = {
                        'name': p['name'],
                        'energy': p['true_energy'],
                        'momentum': p['true_momentum']
                    }
                    # Трассировка частицы
                    track = self.magnetic.track_particle(track_particle)
                    tracks.append(track)
            
            # Сохранение результатов
            event_data = {
                'id': i,
                'event_type': model,
                'beam': beam_particle,
                'target': target_particle,
                'energy': energy,
                'products': products,
                'detected': detected,
                'tracks': tracks,
                'timestamp': time.time()
            }
            events.append(event_data)
        
        # Сохранение результатов
        self.results['events'] = events
        elapsed_time = time.time() - start_time
        logger.info(f"Simulation completed in {elapsed_time:.2f} seconds ({num_events/elapsed_time:.2f} events/sec)")
        
        return events
    
    def analyze_results(self):
        """Анализ результатов симуляции"""
        logger.info("Starting analysis of simulation results")
        
        # Извлечение энергий зарегистрированных частиц
        energies = []
        for event in self.results['events']:
            for p in event['detected']:
                energies.append(p['reconstructed_energy'] / 1e9)  # в ГэВ
        
        # Создание гистограммы энергетического спектра
        self.analyzer.create_histogram("energy_spectrum", energies, bins=100, 
                                     range=(0, max(energies)*1.1))
        
        # Создание 2D гистограммы для энергии и типа частиц
        particle_types = []
        particle_energies = []
        for event in self.results['events']:
            for p in event['detected']:
                particle_types.append(self.particle_db.get_symbol(p['name']))
                particle_energies.append(p['reconstructed_energy'] / 1e9)
        
        # Преобразование типов частиц в числа для 2D гистограммы
        unique_types = list(set(particle_types))
        type_numbers = [unique_types.index(t) for t in particle_types]
        
        self.analyzer.create_2d_histogram("energy_vs_type", 
                                        type_numbers, particle_energies,
                                        x_bins=len(unique_types), y_bins=100,
                                        x_range=(0, len(unique_types)),
                                        y_range=(0, max(particle_energies)*1.1))
        
        # Создание ROI для анализа определенных частиц
        self.analyzer.create_roi("higgs_region", 
                               x_range=(120, 130),  # ГэВ
                               y_range=(0, 1e12))
        
        logger.info("Analysis completed")
        return self.results['analysis']
    
    def visualize_experiment(self):
        """Визуализация эксперимента"""
        # Геометрия
        self.visualizer.plot_geometry(self.geometry.build())
        
        # Траектории частиц (для первого события)
        if self.results['events']:
            tracks = self.results['events'][0]['tracks']
            self.visualizer.plot_particle_tracks(tracks, self.results['events'][0])
    
    def export_results(self, format: str = 'json', path: str = 'results'):
        """
        Экспорт результатов.
        Параметры:
        format: формат экспорта ('json' или 'root')
        path: путь для сохранения
        """
        if format == 'json':
            try:
                # Подготовка данных для экспорта
                export_data = {
                    'parameters': self.results['parameters'],
                    'events': []
                }
                
                # Упрощение данных для JSON
                for event in self.results['events'][:100]:  # Ограничиваем количество событий
                    export_event = {
                        'id': event['id'],
                        'event_type': event['event_type'],
                        'beam': event['beam'],
                        'target': event['target'],
                        'energy': event['energy'],
                        'products': [{
                            'name': p['name'],
                            'energy': p['energy']
                        } for p in event['products']],
                        'detected': [{
                            'name': p['name'],
                            'true_energy': p['true_energy'],
                            'reconstructed_energy': p['reconstructed_energy']
                        } for p in event['detected']]
                    }
                    export_data['events'].append(export_event)
                
                # Сохранение в JSON
                with open(f'{path}.json', 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Results exported to {path}.json")
                return True
            except Exception as e:
                logger.error(f"Error exporting results to JSON: {e}")
                return False
        
        elif format == 'root':
            return self.analyzer.export_to_root(f"{path}.root")
        
        else:
            logger.error(f"Unsupported export format: {format}")
            return False

# ===================================================================
# 9. Модель Большого адронного коллайдера
# ===================================================================
class LHC_Model:
    """Модель Большого адронного коллайдера с интеграцией OpenParticleLab"""
    
    def __init__(self):
        """Инициализация модели БАК с интеграцией OpenParticleLab"""
        # Основные параметры БАК (источники: CERN, LHC Design Report)
        self.circumference = 26658.883  # Длина окружности в метрах
        self.radius = self.circumference / (2 * np.pi)  # Радиус кривизны в метрах
        self.depth = 100  # Глубина в метрах (среднее значение)
        self.tunnel_diameter = 3.8  # Диаметр туннеля в метрах
        # Энергетические параметры
        self.beam_energy = 6.5e12  # Энергия пучка в эВ (6.5 ТэВ)
        self.center_of_mass_energy = 13e12  # Энергия в системе центра масс в эВ
        self.max_energy = 7e12  # Максимальная энергия в эВ (7 ТэВ)
        # Параметры пучка
        self.protons_per_bunch = 2.2e11  # Количество протонов в сгустке
        self.num_bunches = 2808  # Количество сгустков в каждом пучке
        self.bunch_spacing = 25e-9  # Интервал между сгустками в секундах (25 нс)
        self.bunch_length = 7.48  # Длина сгустка в метрах
        # Магнитные параметры
        self.max_magnetic_field = 8.33  # Максимальное магнитное поле в Тл
        self.operating_temperature = 1.9  # Рабочая температура в Кельвинах
        self.dipole_magnets = 1232  # Количество дипольных магнитов
        self.quadrupole_magnets = 392  # Количество квадрупольных магнитов
        # Параметры светимости
        self.beam_size_x = 16.7e-6  # Размер пучка в горизонтальной плоскости (м)
        self.beam_size_y = 16.7e-6  # Размер пучка в вертикальной плоскости (м)
        self.peak_luminosity = 2.0e34  # Пиковая светимость в см⁻²с⁻¹
        # Вакуумная система
        self.vacuum_pressure = 1e-13  # Давление в атмосферах
        self.vacuum_volume = 10000  # Объем вакуумной системы в м³
        # Точки столкновения
        self.collision_points = {
            'ATLAS': {'position': 0.0, 'detector_size': 25, 'energy_resolution': 0.01},
            'CMS': {'position': np.pi * self.radius, 'detector_size': 21, 'energy_resolution': 0.015},
            'ALICE': {'position': np.pi/2 * self.radius, 'detector_size': 16, 'energy_resolution': 0.05},
            'LHCb': {'position': 3*np.pi/2 * self.radius, 'detector_size': 20, 'energy_resolution': 0.02}
        }
        # Инициализация базы данных частиц
        self.particle_db = ParticleDatabase()
        # Инициализация OpenParticleLab
        self.open_lab = OpenParticleLab()
        self._setup_lhc_experiment()
        # Инициализация состояния симуляции
        self.simulation_state = {
            'time': 0.0,
            'beam_energy': 0.0,
            'magnetic_field': 0.0,
            'luminosity': self.peak_luminosity,
            'collisions': [],
            'detected_particles': [],
            'beam_dynamics': {
                'emittance': [],
                'beam_size_x': [],
                'beam_size_y': [],
                'luminosity': [],
                'beam_intensity': []
            }
        }
        logger.info("LHC model initialized with OpenParticleLab integration")
    
    def _setup_lhc_experiment(self):
        """Настройка эксперимента для БАК"""
        # Создание конфигурации БАК
        lhc_config = {
            'beam': {
                'particle': 'proton',
                'energy': 6.5e12  # 6.5 ТэВ
            },
            'target': {
                'particle': 'proton'
            },
            'geometry': [
                {
                    'type': 'detector',
                    'params': {
                        'name': 'ATLAS',
                        'position': [0, 0, 0],
                        'size': [46, 25, 25],
                        'efficiency': 0.99,
                        'resolution': 0.01,
                        'material': 'silicon',
                        'subdetectors': [
                            {'name': 'Inner Detector', 'radius': (0.05, 1.1), 'resolution': 0.01, 'efficiency': 0.99},
                            {'name': 'Calorimeter', 'radius': (1.1, 4.3), 'resolution': 0.05, 'efficiency': 0.95},
                            {'name': 'Muon Spectrometer', 'radius': (5.6, 11.0), 'resolution': 0.02, 'efficiency': 0.90}
                        ],
                        'energy_resolution': {
                            'electromagnetic': lambda E: np.sqrt(0.1**2 + (0.01/E)**2) if E > 0 else 0.1,
                            'hadronic': lambda E: np.sqrt(0.5**2 + (0.05/E)**2) if E > 0 else 0.5
                        },
                        'tracking_resolution': lambda pT: 0.01 * (1 + pT/100)
                    }
                },
                {
                    'type': 'detector',
                    'params': {
                        'name': 'CMS',
                        'position': [np.pi * self.radius, 0, 0],
                        'size': [21, 15, 15],
                        'efficiency': 0.995,
                        'resolution': 0.008,
                        'material': 'silicon',
                        'subdetectors': [
                            {'name': 'Tracker', 'radius': (0.04, 1.1), 'resolution': 0.008, 'efficiency': 0.995},
                            {'name': 'ECAL', 'radius': (1.1, 1.5), 'resolution': 0.03, 'efficiency': 0.97},
                            {'name': 'HCAL', 'radius': (1.5, 3.0), 'resolution': 0.1, 'efficiency': 0.95},
                            {'name': 'Muon System', 'radius': (3.0, 6.5), 'resolution': 0.015, 'efficiency': 0.92}
                        ],
                        'energy_resolution': {
                            'electromagnetic': lambda E: np.sqrt(0.03**2 + (0.005/E)**2) if E > 0 else 0.03,
                            'hadronic': lambda E: np.sqrt(0.1**2 + (0.03/E)**2) if E > 0 else 0.1
                        },
                        'tracking_resolution': lambda pT: 0.007 * (1 + pT/150)
                    }
                },
                {
                    'type': 'magnet',
                    'params': {
                        'position': [0, 0, 0],
                        'orientation': [0, 0, 1],
                        'strength': 2.0,  # Тесла
                        'type': 'dipole',
                        'length': 5.0,
                        'radius': 2.0
                    }
                }
            ]
        }
        
        # Сохранение конфигурации во временный YAML-файл
        with open('lhc_config.yml', 'w') as f:
            yaml.dump(lhc_config, f)
        
        # Загрузка конфигурации в OpenParticleLab
        self.open_lab.load_config('lhc_config.yml')
        logger.info("LHC experiment setup completed")
    
    # ===================================================================
    # 10. Расчетные методы
    # ===================================================================
    def calculate_magnetic_field(self, energy: float) -> float:
        """
        Расчет требуемого магнитного поля для удержания частиц на орбите.
        Используем формулу: B = p/(0.3·q·R)
        где p - импульс в ГэВ/c, q - заряд в единицах e, R - радиус в метрах
        Параметры:
        energy: энергия частицы в эВ
        Возвращает:
        Требуемое магнитное поле в Тл
        """
        # Конвертация энергии в ГэВ
        energy_gev = energy / 1e9
        # Для ультрарелятивистских частиц (E >> mc²) импульс p ≈ E/c
        momentum_gevc = energy_gev
        # Формула: B (Тл) = p (ГэВ/с) / (0.3 * q * R (м))
        magnetic_field = momentum_gevc / (0.3 * 1 * self.radius)
        return magnetic_field
    
    def calculate_relativistic_factor(self, energy: float, particle: str = 'proton') -> float:
        """
        Расчет лоренц-фактора для заданной энергии.
        Параметры:
        energy: энергия частицы в эВ
        particle: тип частицы
        Возвращает:
        Лоренц-фактор γ
        """
        # Масса покоя частицы в эВ
        rest_mass = self.particle_db.get_mass_in_gev(particle)  # ГэВ/c²
        rest_energy = rest_mass * 1e9  # ГэВ -> эВ
        # Лоренц-фактор
        gamma = energy / rest_energy
        return gamma
    
    def calculate_speed(self, energy: float, particle: str = 'proton') -> float:
        """
        Расчет скорости частицы для заданной энергии.
        Параметры:
        energy: энергия частицы в эВ
        particle: тип частицы
        Возвращает:
        Скорость в м/с
        """
        gamma = self.calculate_relativistic_factor(energy, particle)
        beta = np.sqrt(1 - 1/gamma**2)
        speed = beta * c
        return speed
    
    def calculate_luminosity(self) -> float:
        """
        Расчет светимости БАК.
        Формула: L = (N₁·N₂·f·n_b)/(4π·σₓ·σᵧ)
        Возвращает:
        Светимость в см⁻²с⁻¹
        """
        # Частота обращения
        revolution_freq = c / self.circumference
        # Формула светимости
        luminosity = (self.protons_per_bunch**2 * revolution_freq * self.num_bunches) / \
                     (4 * np.pi * self.beam_size_x * self.beam_size_y)
        # Конвертация в см⁻²с⁻¹
        luminosity_cm = luminosity * 1e-4  # м² -> см²
        return luminosity_cm
    
    def calculate_synchrotron_radiation(self, energy: float, particle: str = 'proton') -> float:
        """
        Расчет потерь энергии на синхротронное излучение.
        Параметры:
        energy: энергия частицы в эВ
        particle: тип частицы
        Возвращает:
        Потери энергии в Вт/частицу
        """
        # Масса частицы в кг
        mass_kg = self.particle_db.get_mass_in_gev(particle) * 1e9 * e / (c**2)
        # Лоренц-фактор
        gamma = self.calculate_relativistic_factor(energy, particle)
        # Потери на синхротронное излучение
        power_loss = (e**4 * gamma**4) / (6 * np.pi * epsilon_0 * c * mass_kg**4 * self.radius**2)
        return power_loss
    
    def calculate_bunch_length(self) -> float:
        """
        Расчет длины сгустка.
        Возвращает:
        Длину сгустка в метрах
        """
        return c * self.bunch_spacing
    
    def calculate_revolution_time(self) -> float:
        """
        Расчет времени одного оборота.
        Возвращает:
        Время оборота в секундах
        """
        return self.circumference / c
    
    def calculate_revolution_frequency(self) -> float:
        """
        Расчет частоты обращения.
        Возвращает:
        Частоту обращения в Гц
        """
        return 1 / self.calculate_revolution_time()
    
    def calculate_beta_function(self, s: float) -> float:
        """
        Расчет бета-функции вдоль кольца.
        Параметры:
        s: позиция вдоль кольца в метрах
        Возвращает:
        Бета-функцию в метрах
        """
        # Упрощенная модель для БАК
        return 150 * (1 + 0.1 * np.sin(2 * np.pi * s / self.circumference))
    
    def calculate_emittance(self, energy: float, particle: str = 'proton') -> float:
        """
        Расчет эмиттанса пучка.
        Параметры:
        energy: энергия частицы в эВ
        particle: тип частицы
        Возвращает:
        Эмиттанс в м·рад
        """
        # Нормированный эмиттанс для БАК
        normalized_emittance = 3.75e-6  # м·рад
        # Геометрический эмиттанс
        gamma = self.calculate_relativistic_factor(energy, particle)
        beta = np.sqrt(1 - 1/gamma**2)
        emittance = normalized_emittance / (beta * gamma)
        return emittance
    
    def calculate_beam_size(self, energy: float, particle: str = 'proton') -> Tuple[float, float]:
        """
        Расчет размеров пучка.
        Параметры:
        energy: энергия частицы в эВ
        particle: тип частицы
        Возвращает:
        (горизонтальный размер, вертикальный размер) в метрах
        """
        emittance = self.calculate_emittance(energy, particle)
        beta_x = self.calculate_beta_function(0)
        beta_y = beta_x * 0.2  # Вертикальная бета-функция меньше
        sigma_x = np.sqrt(emittance * beta_x)
        sigma_y = np.sqrt(emittance * beta_y)
        return sigma_x, sigma_y
    
    # ===================================================================
    # 11. Методы симуляции
    # ===================================================================
    def simulate_particle_motion(self, num_turns: int = 10, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        r = self.radius * (1 + 0.001 * np.sin(10 * theta))
        # Координаты в горизонтальной плоскости
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Малые колебания в вертикальной плоскости (из-за квадрупольных магнитов)
        z = 0.1 * np.sin(5 * theta)  # Амплитуда в метрах
        return x, y, z
    
    def simulate_collision(self, energy: Optional[float] = None, num_events: int = 1) -> List[Dict]:
        """
        Симуляция столкновения частиц и генерация продуктов.
        Параметры:
        energy: энергия столкновения в эВ (если None, используется энергия БАК)
        num_events: количество событий для симуляции
        Возвращает:
        Список словарей с информацией о событиях
        """
        if energy is None:
            energy = self.center_of_mass_energy
        events = []
        for _ in range(num_events):
            event = {
                'event_id': len(self.simulation_state['collisions']) + 1,
                'energy': energy,
                'primary_particles': ['proton', 'proton'],
                'products': [],
                'timestamp': time.time()
            }
            # Определение вероятностей различных процессов
            probabilities = self._get_collision_probabilities(energy)
            # Выбор типа события на основе вероятностей
            event_type = self._select_event_type(probabilities)
            # Генерация продуктов столкновения
            products = self._generate_collision_products(event_type, energy)
            # Добавление продуктов в событие
            event['event_type'] = event_type
            event['products'] = products
            # Регистрация события
            self.simulation_state['collisions'].append(event)
            events.append(event)
            # Регистрация обнаруженных частиц
            for product in products:
                if np.random.random() < 0.9:  # 90% детектирования
                    self.simulation_state['detected_particles'].append({
                        'event_id': event['event_id'],
                        'particle': product['name'],
                        'energy': product['energy'],
                        'momentum': product['momentum'],
                        'timestamp': time.time()
                    })
        return events
    
    def _get_collision_probabilities(self, energy: float) -> Dict[str, float]:
        """
        Получение вероятностей различных типов столкновений с использованием реальных данных сечений.
        Параметры:
        energy: энергия столкновения в эВ
        Возвращает:
        Словарь с вероятностями
        """
        # Энергия в ТэВ
        energy_tev = energy / 1e12
        
        # Реальные данные сечений (приблизительные значения для 13 ТэВ)
        cross_sections = {
            'elastic_scattering': 30.7 * (13.0/energy_tev)**0.08,  # мб
            'inelastic_scattering': 73.7 * (13.0/energy_tev)**0.05,  # мб
            'higgs_boson_production': 55.6 * (energy_tev/13.0)**1.5 if energy_tev >= 1.25 else 0.0,  # пб
            'top_quark_production': 832 * (energy_tev/13.0)**1.2 if energy_tev >= 1.0 else 0.0,  # пб
            'z_boson_production': 2000 * (energy_tev/13.0)**0.8 if energy_tev >= 0.091 else 0.0,  # пб
            'w_boson_production': 21000 * (energy_tev/13.0)**0.7 if energy_tev >= 0.08 else 0.0,  # пб
            'dijet_production': 1e8 * (energy_tev/13.0)**0.5,  # пб
            'b_quark_production': 500000 * (energy_tev/13.0)**0.9,  # пб
            'tau_production': 4000 * (energy_tev/13.0)**0.7  # пб
        }
        
        # Конвертация пб в мб для нормализации
        for key in cross_sections:
            if 'pb' in key:
                cross_sections[key] *= 1e-9  # 1 пб = 1e-9 мб
        
        # Нормализация вероятностей
        total = sum(cross_sections.values())
        probabilities = {k: v/total for k, v in cross_sections.items()}
        
        return probabilities
    
    def _select_event_type(self, probabilities: Dict[str, float]) -> str:
        """
        Выбор типа события на основе вероятностей.
        Параметры:
        probabilities: словарь с вероятностями
        Возвращает:
        Тип события
        """
        # Создание массива с типами и кумулятивными вероятностями
        event_types = list(probabilities.keys())
        cum_probs = np.cumsum([probabilities[e] for e in event_types])
        # Генерация случайного числа
        r = np.random.random()
        # Определение типа события
        for i, p in enumerate(cum_probs):
            if r < p:
                return event_types[i]
        return event_types[-1]  # На случай погрешностей
    
    def _generate_collision_products(self, event_type: str, energy: float) -> List[Dict]:
        """
        Генерация продуктов столкновения для заданного типа события.
        Параметры:
        event_type: тип события
        energy: энергия столкновения в эВ
        Возвращает:
        Список продуктов столкновения
        """
        products = []
        
        # Определение, является ли это QCD-процессом
        if event_type in ['dijet_production', 'b_quark_production', 'tau_production']:
            return self._generate_qcd_events(event_type, energy)
        
        if event_type == 'elastic_scattering':
            # Упругое рассеяние - те же частицы с измененными импульсами
            products.append({'name': 'proton', 'energy': energy/2 * 0.95, 'momentum': energy/2 * 0.95 / c})
            products.append({'name': 'proton', 'energy': energy/2 * 0.95, 'momentum': energy/2 * 0.95 / c})
        elif event_type == 'inelastic_scattering':
            # Неупругое рассеяние - генерация вторичных частиц
            num_products = np.random.randint(2, 10)
            for _ in range(num_products):
                particle = np.random.choice(['pion_plus', 'kaon_plus', 'muon', 'electron', 'photon'])
                fraction = np.random.uniform(0.05, 0.3)
                products.append({
                    'name': particle,
                    'energy': energy * fraction,
                    'momentum': energy * fraction / c
                })
        elif event_type == 'higgs_boson_production':
            # Распад бозона Хиггса
            products.append({'name': 'higgs_boson', 'energy': energy * 0.9, 'momentum': energy * 0.9 / c})
            # Основные каналы распада
            decay_products = self.particle_db.get_particle_decay_products('higgs_boson')
            for particle, prob in decay_products:
                if np.random.random() < prob:
                    fraction = np.random.uniform(0.05, 0.3)
                    products.append({
                        'name': particle,
                        'energy': energy * fraction,
                        'momentum': energy * fraction / c
                    })
        elif event_type == 'top_quark_production':
            # Производство пары верхних кварков
            products.append({'name': 'top_quark', 'energy': energy * 0.4, 'momentum': energy * 0.4 / c})
            products.append({'name': 'top_quark', 'energy': energy * 0.4, 'momentum': energy * 0.4 / c})
            # Распад верхних кварков
            decay_products = self.particle_db.get_particle_decay_products('top_quark')
            for particle, prob in decay_products:
                if np.random.random() < prob:
                    fraction = np.random.uniform(0.1, 0.3)
                    products.append({
                        'name': particle,
                        'energy': energy * fraction,
                        'momentum': energy * fraction / c
                    })
        elif event_type == 'z_boson_production':
            # Производство Z-бозона
            products.append({'name': 'z_boson', 'energy': energy * 0.8, 'momentum': energy * 0.8 / c})
            # Распад Z-бозона
            decay_products = self.particle_db.get_particle_decay_products('z_boson')
            for particle, prob in decay_products:
                if np.random.random() < prob:
                    fraction = np.random.uniform(0.05, 0.2)
                    products.append({
                        'name': particle,
                        'energy': energy * fraction,
                        'momentum': energy * fraction / c
                    })
        elif event_type == 'w_boson_production':
            # Производство W-бозона
            products.append({'name': 'w_boson', 'energy': energy * 0.8, 'momentum': energy * 0.8 / c})
            # Распад W-бозона
            decay_products = self.particle_db.get_particle_decay_products('w_boson')
            for particle, prob in decay_products:
                if np.random.random() < prob:
                    fraction = np.random.uniform(0.05, 0.2)
                    products.append({
                        'name': particle,
                        'energy': energy * fraction,
                        'momentum': energy * fraction / c
                    })
        return products
    
    def _generate_qcd_events(self, event_type: str, energy: float) -> List[Dict]:
        """
        Генерация QCD-событий с учетом реальных сечений.
        """
        products = []
        
        # Определение типа QCD-процесса
        if event_type == 'dijet_production':
            # Генерация двух струй
            for _ in range(2):
                # Выбор типа струи
                jet_type = np.random.choice(['light', 'gluon', 'b', 'c'], p=[0.6, 0.3, 0.07, 0.03])
                # Энергия струи (распределение по степенному закону)
                fraction = np.random.power(2.5) * 0.4  # до 40% энергии
                jet_energy = energy * fraction
                
                # Генерация частиц в струе
                num_particles = max(1, int(np.random.normal(10, 5)))
                for _ in range(num_particles):
                    # Выбор частицы в струе
                    if jet_type == 'light':
                        particle = np.random.choice(['pion_plus', 'pion_minus', 'kaon_plus', 'proton'])
                    elif jet_type == 'gluon':
                        particle = np.random.choice(['pion_plus', 'pion_minus', 'gluon'])
                    elif jet_type == 'b':
                        particle = np.random.choice(['pion_plus', 'b_hadron'])
                    else:  # 'c'
                        particle = np.random.choice(['pion_plus', 'c_hadron'])
                    
                    # Доля энергии для частицы
                    particle_fraction = np.random.beta(1, 3)
                    particle_energy = jet_energy * particle_fraction
                    
                    products.append({
                        'name': particle,
                        'energy': particle_energy,
                        'momentum': particle_energy / c,
                        'source': 'jet'
                    })
        
        elif event_type == 'b_quark_production':
            # Производство b-кварков
            products.append({'name': 'bottom_quark', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c, 'source': 'b_jet'})
            products.append({'name': 'antibottom_quark', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c, 'source': 'b_jet'})
            
            # Распад b-кварков
            b_decay_products = self.particle_db.get_particle_decay_products('bottom_quark')
            for particle, prob in b_decay_products:
                if np.random.random() < prob:
                    fraction = np.random.uniform(0.05, 0.2)
                    products.append({
                        'name': particle,
                        'energy': energy * fraction,
                        'momentum': energy * fraction / c,
                        'source': 'b_decay'
                    })
        
        elif event_type == 'tau_production':
            # Производство тау-лептонов
            products.append({'name': 'tau', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c})
            products.append({'name': 'antitau', 'energy': energy * 0.3, 'momentum': energy * 0.3 / c})
            
            # Распад тау-лептонов
            tau_decay_products = self.particle_db.get_particle_decay_products('tau')
            for particle, prob in tau_decay_products:
                if np.random.random() < prob:
                    fraction = np.random.uniform(0.05, 0.2)
                    products.append({
                        'name': particle,
                        'energy': energy * fraction,
                        'momentum': energy * fraction / c,
                        'source': 'tau_decay'
                    })
        
        return products
    
    # ===================================================================
    # НОВЫЙ МЕТОД: Симуляция одного оборота пучка
    # ===================================================================
    def step_simulation(self, include_space_charge: bool = True):
        """
        Симуляция одного оборота пучка частиц в LHC.
        
        Параметры:
        include_space_charge: учитывать ли эффекты пространственного заряда
        """
        # 1. Расчет времени одного оборота
        revolution_time = self.circumference / c
        
        # 2. Обновление общего времени симуляции
        self.simulation_state['time'] += revolution_time
        
        # 3. Расчет количества столкновений за оборот
        current_luminosity = self.simulation_state.get('luminosity', self.peak_luminosity)
        # Перевод светимости из см⁻²с⁻¹ в м⁻²с⁻¹ (умножение на 1e4)
        collision_rate = current_luminosity * 1e4
        num_collisions = int(collision_rate * revolution_time)
        
        # 4. Симуляция столкновений
        if num_collisions > 0:
            logger.debug(f"Simulating {num_collisions} collisions for this revolution")
            collisions = self.simulate_collision(num_events=num_collisions)
            self.simulation_state['collisions'].extend(collisions)
        
        # 5. Обновление динамики пучка
        beam_dynamics = self.simulation_state['beam_dynamics']
        
        # Получение текущих параметров (последние значения или начальные)
        current_emittance = beam_dynamics['emittance'][-1] if beam_dynamics['emittance'] else self.calculate_emittance(self.beam_energy)
        current_intensity = beam_dynamics['beam_intensity'][-1] if beam_dynamics['beam_intensity'] else 1.0
        gamma = self.calculate_relativistic_factor(self.beam_energy)
        
        # Эффекты пространственного заряда
        space_charge_effect = 0.0
        if include_space_charge:
            # Упрощенная модель эффекта пространственного заряда
            bunch_density = self.protons_per_bunch / (np.pi * self.beam_size_x * self.beam_size_y * self.bunch_length)
            # Коэффициент, зависящий от энергии и плотности
            space_charge_effect = 0.001 * bunch_density * (1/gamma**2) * current_intensity
            # Дополнительное увеличение эмиттанса из-за пространственного заряда
            current_emittance *= (1 + space_charge_effect)
        
        # Потери частиц (0.01% потерь за оборот)
        loss_rate = 0.0001
        current_intensity *= (1 - loss_rate)
        
        # Размеры пучка
        beta_function = 150  # м (типичная бета-функция в БАК)
        sigma_x = np.sqrt(current_emittance * beta_function)
        sigma_y = np.sqrt(current_emittance * beta_function * 0.2)  # Вертикальная размерность меньше
        
        # Светимость (обратно пропорциональна квадрату размера пучка и пропорциональна интенсивности)
        current_luminosity = self.peak_luminosity * (self.beam_size_x / sigma_x)**2 * (self.beam_size_y / sigma_y)**2 * current_intensity
        
        # 6. Обновление состояния
        beam_dynamics['emittance'].append(current_emittance)
        beam_dynamics['beam_size_x'].append(sigma_x)
        beam_dynamics['beam_size_y'].append(sigma_y)
        beam_dynamics['luminosity'].append(current_luminosity)
        beam_dynamics['beam_intensity'].append(current_intensity)
        
        # Обновление текущей светимости в основном состоянии
        self.simulation_state['luminosity'] = current_luminosity
        
        # Логирование результатов
        logger.info(f"Completed revolution {len(beam
