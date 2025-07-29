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
# 2. Модель Большого адронного коллайдера
# ===================================================================
class LHC_Model:
    """Улучшенная модель Большого адронного коллайдера на основе реальных данных"""
    def __init__(self):
        """Инициализация модели БАК с реальными параметрами"""
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
        # Инициализация состояния симуляции
        self.simulation_state = {
            'time': 0.0,
            'beam_energy': 0.0,
            'magnetic_field': 0.0,
            'luminosity': 0.0,
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
        # Инициализация детекторов с реалистичными параметрами
        self.initialize_detectors()
        logger.info("LHC model initialized with real parameters from CERN sources")
    def initialize_detectors(self):
        """Инициализация детекторов с реалистичными параметрами"""
        self.detectors = {
            'ATLAS': {
                'position': 0.0,
                'size': {'length': 46, 'diameter': 25},
                'subdetectors': [
                    {'name': 'Inner Detector', 'radius': (0.05, 1.1), 'resolution': 0.01, 'efficiency': 0.99},
                    {'name': 'Calorimeter', 'radius': (1.1, 4.3), 'resolution': 0.05, 'efficiency': 0.95},
                    {'name': 'Muon Spectrometer', 'radius': (5.6, 11.0), 'resolution': 0.02, 'efficiency': 0.90}
                ],
                'magnetic_field': 2.0,  # Тл
                'energy_resolution': {
                    'electromagnetic': lambda E: np.sqrt(0.1**2 + (0.01/E)**2) if E > 0 else 0.1,
                    'hadronic': lambda E: np.sqrt(0.5**2 + (0.05/E)**2) if E > 0 else 0.5
                },
                'tracking_resolution': lambda pT: 0.01 * (1 + pT/100)  # pT в ГэВ
            },
            'CMS': {
                'position': np.pi * self.radius,
                'size': {'length': 21, 'diameter': 15},
                'subdetectors': [
                    {'name': 'Tracker', 'radius': (0.04, 1.1), 'resolution': 0.008, 'efficiency': 0.995},
                    {'name': 'ECAL', 'radius': (1.1, 1.5), 'resolution': 0.03, 'efficiency': 0.97},
                    {'name': 'HCAL', 'radius': (1.5, 3.0), 'resolution': 0.1, 'efficiency': 0.95},
                    {'name': 'Muon System', 'radius': (3.0, 6.5), 'resolution': 0.015, 'efficiency': 0.92}
                ],
                'magnetic_field': 3.8,  # Тл
                'energy_resolution': {
                    'electromagnetic': lambda E: np.sqrt(0.03**2 + (0.005/E)**2) if E > 0 else 0.03,
                    'hadronic': lambda E: np.sqrt(0.1**2 + (0.03/E)**2) if E > 0 else 0.1
                },
                'tracking_resolution': lambda pT: 0.007 * (1 + pT/150)  # pT в ГэВ
            },
            'ALICE': {
                'position': np.pi/2 * self.radius,
                'size': {'length': 16, 'diameter': 16},
                'subdetectors': [
                    {'name': 'ITS', 'radius': (0.03, 0.4), 'resolution': 0.02, 'efficiency': 0.98},
                    {'name': 'TPC', 'radius': (0.4, 2.5), 'resolution': 0.04, 'efficiency': 0.96},
                    {'name': 'TRD', 'radius': (2.5, 3.7), 'resolution': 0.03, 'efficiency': 0.94},
                    {'name': 'TOF', 'radius': (3.7, 4.0), 'resolution': 0.05, 'efficiency': 0.92},
                    {'name': 'EMCal', 'radius': (4.0, 4.3), 'resolution': 0.08, 'efficiency': 0.88}
                ],
                'magnetic_field': 0.5,  # Тл
                'energy_resolution': {
                    'electromagnetic': lambda E: np.sqrt(0.05**2 + (0.02/E)**2) if E > 0 else 0.05,
                    'hadronic': lambda E: np.sqrt(0.15**2 + (0.06/E)**2) if E > 0 else 0.15
                },
                'tracking_resolution': lambda pT: 0.02 * (1 + pT/50)  # pT в ГэВ
            },
            'LHCb': {
                'position': 3*np.pi/2 * self.radius,
                'size': {'length': 20, 'diameter': 13},
                'subdetectors': [
                    {'name': 'Vertex Locator', 'radius': (0.01, 0.08), 'resolution': 0.005, 'efficiency': 0.99},
                    {'name': 'Tracker', 'radius': (0.08, 1.0), 'resolution': 0.01, 'efficiency': 0.98},
                    {'name': 'RICH 1', 'radius': (1.0, 2.0), 'resolution': 0.03, 'efficiency': 0.95},
                    {'name': 'RICH 2', 'radius': (2.0, 4.0), 'resolution': 0.03, 'efficiency': 0.95},
                    {'name': 'Calorimeter', 'radius': (4.0, 5.0), 'resolution': 0.06, 'efficiency': 0.93},
                    {'name': 'Muon System', 'radius': (5.0, 12.0), 'resolution': 0.02, 'efficiency': 0.90}
                ],
                'magnetic_field': 1.0,  # Тл
                'energy_resolution': {
                    'electromagnetic': lambda E: np.sqrt(0.04**2 + (0.015/E)**2) if E > 0 else 0.04,
                    'hadronic': lambda E: np.sqrt(0.12**2 + (0.04/E)**2) if E > 0 else 0.12
                },
                'tracking_resolution': lambda pT: 0.012 * (1 + pT/80)  # pT в ГэВ
            }
        }
    # ===================================================================
    # 3. Расчетные методы
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
    # 4. Методы симуляции
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
    def simulate_beam_dynamics(self, num_turns: int = 1000, include_space_charge: bool = True) -> Dict[str, List[float]]:
        """
        Симуляция динамики пучка с учетом эффектов пространственного заряда.
        """
        # Инициализация параметров
        emittance = self.calculate_emittance(self.beam_energy)
        beta_function = 150  # Бета-функция в м
        revolution_freq = self.calculate_revolution_frequency()
        gamma = self.calculate_relativistic_factor(self.beam_energy)
        
        # Инициализация списков для данных
        emittance_growth = []
        beam_size_x = []
        beam_size_y = []
        luminosity = []
        beam_intensity = []
        
        # Начальные значения
        current_emittance = emittance
        current_intensity = 1.0  # Нормализованная интенсивность
        
        for turn in range(num_turns):
            # 1. Эффекты пространственного заряда
            space_charge_effect = 0.0
            if include_space_charge and turn > 0:
                # Упрощенная модель эффекта пространственного заряда
                bunch_density = self.protons_per_bunch / (np.pi * self.beam_size_x * self.beam_size_y * self.bunch_length)
                # Коэффициент, зависящий от энергии и плотности
                space_charge_effect = 0.001 * bunch_density * (1/gamma**2) * current_intensity
                
                # Дополнительное увеличение эмиттанса из-за пространственного заряда
                current_emittance *= (1 + space_charge_effect)
            
            # 2. Инжекционные эффекты (только в начале)
            injection_effect = 0.0
            if turn < 100:
                injection_effect = 0.005 * (1 - turn/100)
                current_emittance *= (1 + injection_effect)
            
            # 3. Потери частиц
            loss_rate = 0.0001  # 0.01% потерь на оборот
            current_intensity *= (1 - loss_rate)
            
            # Сохранение данных
            emittance_growth.append(current_emittance)
            sigma_x = np.sqrt(current_emittance * beta_function)
            sigma_y = np.sqrt(current_emittance * beta_function * 0.2)
            beam_size_x.append(sigma_x)
            beam_size_y.append(sigma_y)
            
            # Светимость (обратно пропорциональна квадрату размера пучка и пропорциональна интенсивности)
            current_luminosity = self.peak_luminosity * (self.beam_size_x / sigma_x)**2 * (self.beam_size_y / sigma_y)**2 * current_intensity
            luminosity.append(current_luminosity)
            beam_intensity.append(current_intensity)
        
        # Сохранение в состояние симуляции
        self.simulation_state['beam_dynamics'] = {
            'emittance': emittance_growth,
            'beam_size_x': beam_size_x,
            'beam_size_y': beam_size_y,
            'luminosity': luminosity,
            'beam_intensity': beam_intensity
        }
        
        return self.simulation_state['beam_dynamics']
    def accelerate_beam(self, target_energy: float = 6.5e12, steps: int = 100) -> Dict[str, List[float]]:
        """
        Симуляция процесса ускорения пучка от инжекционной энергии до целевой.
        """
        # Инжекционная энергия (450 ГэВ для БАК)
        injection_energy = 450e9
        energies = np.linspace(injection_energy, target_energy, steps)
        
        # Списки для хранения данных
        magnetic_fields = []
        revolution_frequencies = []
        gamma_factors = []
        speeds = []
        
        for energy in energies:
            # Расчет магнитного поля
            magnetic_field = self.calculate_magnetic_field(energy)
            magnetic_fields.append(magnetic_field)
            
            # Расчет частоты обращения (с учетом релятивистского увеличения массы)
            gamma = self.calculate_relativistic_factor(energy)
            beta = np.sqrt(1 - 1/gamma**2)
            revolution_freq = c / (self.circumference * beta)
            revolution_frequencies.append(revolution_freq)
            
            # Сохранение других параметров
            gamma_factors.append(gamma)
            speeds.append(beta * c)
        
        # Сохранение конечного состояния
        self.beam_energy = target_energy
        self.simulation_state['beam_energy'] = target_energy
        
        return {
            'energy': energies,
            'magnetic_field': magnetic_fields,
            'revolution_frequency': revolution_frequencies,
            'gamma': gamma_factors,
            'speed': speeds
        }
    def reconstruct_event(self, event: Dict) -> Dict:
        """
        Реконструкция события с учетом характеристик детектора и шумов.
        """
        reconstructed = {
            'event_id': event['event_id'],
            'original_event': event,
            'reconstructed_products': [],
            'missing_energy': 0.0,
            'confidence': 1.0
        }
        
        # Выбор детектора для реконструкции (случайный выбор из доступных)
        detector_name = np.random.choice(list(self.detectors.keys()))
        detector = self.detectors[detector_name]
        
        # Реконструкция каждой частицы
        for i, product in enumerate(event['products']):
            particle_name = product['name']
            true_energy = product['energy']
            true_momentum = product['momentum']
            
            # Определение, детектируется ли частица
            detected = False
            for subdetector in detector['subdetectors']:
                # Эффективность детектирования зависит от типа частицы и поддетектора
                efficiency = subdetector['efficiency']
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
                resolution_func = detector['energy_resolution']['electromagnetic']
            else:
                resolution_func = detector['energy_resolution']['hadronic']
            
            # Стохастический компонент разрешения
            stochastic = np.random.normal(0, resolution_func(true_energy/1e9))
            reconstructed_energy = true_energy * (1 + stochastic)
            
            # Разрешение по импульсу для заряженных частиц
            if self.particle_db.get_charge(particle_name) != 0:
                pT = true_momentum * np.random.uniform(0.5, 1.0)  # Поперечный импульс
                momentum_resolution = detector['tracking_resolution'](pT/1e9)
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
                'subdetector': subdetector['name']
            })
        
        # Вычисление общей уверенности в реконструкции
        if reconstructed['reconstructed_products']:
            reconstructed['confidence'] = min(1.0, 0.9 * len(reconstructed['reconstructed_products']) / len(event['products']))
        
        return reconstructed
    def simulate_particle_decay(self, particle: str, num_steps: int = 100) -> List[Dict]:
        """
        Симуляция распада частицы.
        Параметры:
        particle: имя частицы
        num_steps: количество шагов симуляции
        Возвращает:
        Список состояний распада
        """
        decay_chain = []
        current_particle = particle
        current_time = 0.0
        decay_lifetime = self.particle_db.get_lifetime(particle)
        # Добавление начальной частицы
        decay_chain.append({
            'particle': current_particle,
            'time': current_time,
            'remaining_lifetime': decay_lifetime
        })
        # Симуляция распада
        for _ in range(num_steps):
            # Если время жизни истекло
            if current_time >= decay_lifetime:
                # Получение продуктов распада
                decay_products = self.particle_db.get_particle_decay_products(current_particle)
                if decay_products:
                    # Выбор продукта распада
                    product, _ = decay_products[np.random.randint(len(decay_products))]
                    # Добавление продукта
                    decay_chain.append({
                        'particle': product,
                        'time': current_time,
                        'remaining_lifetime': self.particle_db.get_lifetime(product)
                    })
                    # Обновление текущей частицы
                    current_particle = product
                    decay_lifetime = self.particle_db.get_lifetime(product)
                else:
                    # Конец цепочки распада
                    break
            # Увеличение времени
            time_step = decay_lifetime / num_steps
            current_time += time_step
        return decay_chain
    # ===================================================================
    # 5. Методы визуализации
    # ===================================================================
    def visualize_ring(self, show_magnets: bool = True, show_collision_points: bool = True):
        """
        Визуализация кольца коллайдера.
        Параметры:
        show_magnets: показывать магниты
        show_collision_points: показывать точки столкновения
        """
        plt.figure(figsize=(12, 12))
        # Рисование кольца
        theta = np.linspace(0, 2*np.pi, 1000)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        plt.plot(x, y, 'b-', linewidth=2)
        # Добавление магнитов (если требуется)
        if show_magnets:
            magnet_angles = np.linspace(0, 2*np.pi, self.dipole_magnets, endpoint=False)
            magnet_x = self.radius * np.cos(magnet_angles)
            magnet_y = self.radius * np.sin(magnet_angles)
            plt.scatter(magnet_x, magnet_y, s=5, c='r', alpha=0.5, label='Дипольные магниты')
            # Квадрупольные магниты
            quad_angles = np.linspace(0, 2*np.pi, self.quadrupole_magnets, endpoint=False) + np.pi/self.quadrupole_magnets
            quad_x = self.radius * np.cos(quad_angles)
            quad_y = self.radius * np.sin(quad_angles)
            plt.scatter(quad_x, quad_y, s=5, c='g', alpha=0.5, label='Квадрупольные магниты')
        # Отметить точки столкновения
        if show_collision_points:
            colors = ['r', 'g', 'b', 'm']
            for i, (detector, data) in enumerate(self.collision_points.items()):
                angle = data['position'] / self.radius
                plt.plot(self.radius * np.cos(angle), 
                         self.radius * np.sin(angle), 
                         'o', markersize=10, color=colors[i], label=detector)
        plt.title('Большой адронный коллайдер (вид сверху)')
        plt.xlabel('X (м)')
        plt.ylabel('Y (м)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.savefig('lhc_ring.png')
        logger.info("Ring visualization saved to 'lhc_ring.png'")
    def visualize_ring_3d(self, show_magnets: bool = True, show_collision_points: bool = True):
        """
        Интерактивная 3D-визуализация кольца коллайдера с использованием Plotly.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Falling back to 2D visualization.")
            self.visualize_ring(show_magnets, show_collision_points)
            return
            
        try:
            # Создание кольца
            theta = np.linspace(0, 2*np.pi, 1000)
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            z = np.zeros_like(theta)
            
            # Создание фигуры
            fig = go.Figure()
            
            # Добавление кольца
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='blue', width=5),
                name='Тоннель'
            ))
            
            # Добавление магнитов
            if show_magnets:
                # Дипольные магниты
                magnet_angles = np.linspace(0, 2*np.pi, self.dipole_magnets, endpoint=False)
                magnet_x = self.radius * np.cos(magnet_angles)
                magnet_y = self.radius * np.sin(magnet_angles)
                magnet_z = np.zeros(self.dipole_magnets)
                
                fig.add_trace(go.Scatter3d(
                    x=magnet_x, y=magnet_y, z=magnet_z,
                    mode='markers',
                    marker=dict(size=4, color='red', opacity=0.7),
                    name='Дипольные магниты'
                ))
                
                # Квадрупольные магниты
                quad_angles = np.linspace(0, 2*np.pi, self.quadrupole_magnets, endpoint=False) + np.pi/self.quadrupole_magnets
                quad_x = self.radius * np.cos(quad_angles)
                quad_y = self.radius * np.sin(quad_angles)
                quad_z = np.zeros(self.quadrupole_magnets)
                
                fig.add_trace(go.Scatter3d(
                    x=quad_x, y=quad_y, z=quad_z,
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.7),
                    name='Квадрупольные магниты'
                ))
            
            # Добавление точек столкновения
            if show_collision_points:
                for i, (detector, data) in enumerate(self.collision_points.items()):
                    angle = data['position'] / self.radius
                    fig.add_trace(go.Scatter3d(
                        x=[self.radius * np.cos(angle)],
                        y=[self.radius * np.sin(angle)],
                        z=[0],
                        mode='markers',
                        marker=dict(size=8, color=['red', 'green', 'blue', 'purple'][i]),
                        name=detector
                    ))
            
            # Настройка макета
            fig.update_layout(
                title='Большой адронный коллайдер (3D-представление)',
                scene=dict(
                    xaxis_title='X (м)',
                    yaxis_title='Y (м)',
                    zaxis_title='Z (м)',
                    aspectmode='data'
                ),
                width=900,
                height=700,
                margin=dict(r=20, l=10, b=10, t=40)
            )
            
            # Сохранение и отображение
            fig.write_html("lhc_ring_3d.html")
            fig.show()
            
            logger.info("3D ring visualization saved to 'lhc_ring_3d.html'")
        except Exception as e:
            logger.warning(f"Error in 3D visualization: {str(e)}. Falling back to 2D.")
            self.visualize_ring(show_magnets, show_collision_points)
    def visualize_collision(self, energy: Optional[float] = None):
        """
        Визуализация столкновения частиц и продуктов.
        Параметры:
        energy: энергия столкновения в эВ
        """
        # Симуляция столкновения
        events = self.simulate_collision(energy=energy, num_events=1)
        event = events[0]
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
        energies = [p['energy']/1e9 for p in event['products']]  # в ГэВ
        particles = [self.particle_db.get_symbol(p['name']) for p in event['products']]
        y_pos = np.arange(len(energies))
        ax2.barh(y_pos, energies, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(particles)
        ax2.set_xlabel('Энергия (ГэВ)')
        ax2.set_title('Энергетический спектр продуктов')
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        # 3. Вероятности процессов
        ax3 = plt.subplot(2, 2, 3)
        probabilities = self._get_collision_probabilities(event['energy'])
        # Фильтрация малых вероятностей
        filtered_probs = {k: v for k, v in probabilities.items() if v > 1e-8}
        ax3.bar(filtered_probs.keys(), filtered_probs.values())
        ax3.set_yscale('log')
        ax3.set_xticklabels(filtered_probs.keys(), rotation=45, ha='right')
        ax3.set_title('Вероятности различных процессов')
        ax3.set_ylabel('Вероятность')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
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
        plt.savefig('collision_visualization.png')
        logger.info("Collision visualization saved to 'collision_visualization.png'")
    def visualize_collision_interactive(self, energy: Optional[float] = None):
        """
        Интерактивная визуализация столкновения частиц и продуктов.
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not installed. Falling back to standard visualization.")
            self.visualize_collision(energy)
            return
            
        try:
            # Симуляция столкновения
            events = self.simulate_collision(energy=energy, num_events=1)
            event = events[0]
            
            # Создание подграфиков
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Диаграмма Фейнмана', 
                    'Энергетический спектр',
                    'Вероятности процессов',
                    'Информация о событии'
                ),
                specs=[
                    [{"type": "scene"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "scene"}]
                ]
            )
            
            # 1. Диаграмма Фейнмана
            # Входные частицы
            fig.add_trace(
                go.Scatter3d(
                    x=[0.2, 0.5], y=[0.5, 0.5], z=[0, 0],
                    mode='lines+markers',
                    line=dict(color='blue', width=5),
                    marker=dict(size=5),
                    name='Входные частицы'
                ),
                row=1, col=1
            )
            
            # Точка столкновения
            fig.add_trace(
                go.Scatter3d(
                    x=[0.5], y=[0.5], z=[0],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name='Столкновение'
                ),
                row=1, col=1
            )
            
            # Выходные частицы
            num_products = len(event['products'])
            angles = np.linspace(-np.pi/3, np.pi/3, num_products)
            for i, angle in enumerate(angles):
                length = 0.4 + 0.1 * np.random.random()
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                particle = event['products'][i]
                color = 'green' if 'boson' in particle['name'] else 'purple' if 'quark' in particle['name'] else 'orange'
                fig.add_trace(
                    go.Scatter3d(
                        x=[0.5, 0.5+dx], y=[0.5, 0.5+dy], z=[0, 0],
                        mode='lines+markers',
                        line=dict(color=color, width=4),
                        marker=dict(size=5),
                        name=self.particle_db.get_symbol(particle['name'])
                    ),
                    row=1, col=1
                )
            
            # 2. Энергетический спектр
            energies = [p['energy']/1e9 for p in event['products']]  # в ГэВ
            particles = [self.particle_db.get_symbol(p['name']) for p in event['products']]
            colors = ['green' if 'boson' in p['name'] else 'purple' if 'quark' in p['name'] else 'orange' 
                     for p in event['products']]
            
            fig.add_trace(
                go.Bar(
                    x=energies,
                    y=particles,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{e:.2f} ГэВ" for e in energies],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Вероятности процессов
            probabilities = self._get_collision_probabilities(event['energy'])
            # Фильтрация малых вероятностей
            filtered_probs = {k: v for k, v in probabilities.items() if v > 1e-8}
            
            fig.add_trace(
                go.Bar(
                    x=list(filtered_probs.keys()),
                    y=list(filtered_probs.values()),
                    marker_color='blue'
                ),
                row=2, col=1
            )
            fig.update_yaxes(type="log", row=2, col=1)
            
            # 4. Информация о событии (текст в 3D для интерактивности)
            info_text = (
                f"Событие: {event['event_type'].replace('_', ' ').title()}<br>"
                f"Энергия: {event['energy']/1e12:.1f} ТэВ<br><br>"
                f"Основные продукты:<br>"
            )
            for i, product in enumerate(event['products'][:5]):
                symbol = self.particle_db.get_symbol(product['name'])
                energy = product['energy']/1e9  # в ГэВ
                info_text += f"- {symbol}: {energy:.2f} ГэВ<br>"
            if len(event['products']) > 5:
                info_text += f"- и еще {len(event['products'])-5} частиц"
            
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='text',
                    text=[info_text],
                    textfont=dict(size=14),
                    hoverinfo='text'
                ),
                row=2, col=2
            )
            
            # Настройка макета
            fig.update_layout(
                height=900,
                width=1200,
                title_text=f"Столкновение: {event['event_type'].replace('_', ' ').title()}",
                showlegend=False
            )
            
            # Сохранение и отображение
            fig.write_html("collision_interactive.html")
            fig.show()
            
            logger.info("Interactive collision visualization saved to 'collision_interactive.html'")
        except Exception as e:
            logger.warning(f"Error in interactive collision visualization: {str(e)}. Falling back to standard visualization.")
            self.visualize_collision(energy)
    def visualize_beam_dynamics(self, num_turns: int = 1000):
        """
        Визуализация динамики пучка.
        Параметры:
        num_turns: количество оборотов для отображения
        """
        dynamics = self.simulate_beam_dynamics(num_turns=num_turns)
        plt.figure(figsize=(14, 10))
        # 1. Рост эмиттанса
        plt.subplot(2, 2, 1)
        plt.plot(dynamics['emittance'])
        plt.title('Рост эмиттанса пучка')
        plt.xlabel('Оборот')
        plt.ylabel('Эмиттанс (м·рад)')
        plt.grid(True)
        # 2. Изменение размеров пучка
        plt.subplot(2, 2, 2)
        plt.plot(dynamics['beam_size_x'], label='Горизонтальный размер')
        plt.plot(dynamics['beam_size_y'], label='Вертикальный размер')
        plt.title('Изменение размеров пучка')
        plt.xlabel('Оборот')
        plt.ylabel('Размер пучка (м)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        # 3. Светимость
        plt.subplot(2, 2, 3)
        plt.plot(dynamics['luminosity'])
        plt.axhline(y=self.peak_luminosity, color='r', linestyle='--', label='Пиковая светимость')
        plt.title('Светимость в зависимости от числа оборотов')
        plt.xlabel('Оборот')
        plt.ylabel('Светимость (см⁻²с⁻¹)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        # 4. Фазовая диаграмма
        plt.subplot(2, 2, 4)
        plt.scatter(dynamics['beam_size_x'], dynamics['beam_size_y'], c=range(num_turns), alpha=0.6)
        plt.colorbar(label='Номер оборота')
        plt.title('Фазовая диаграмма пучка')
        plt.xlabel('Горизонтальный размер (м)')
        plt.ylabel('Вертикальный размер (м)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('beam_dynamics.png')
        logger.info("Beam dynamics visualization saved to 'beam_dynamics.png'")
    def visualize_particle_decay(self, particle: str = 'higgs_boson'):
        """
        Визуализация цепочки распада частицы.
        Параметры:
        particle: имя частицы для симуляции распада
        """
        decay_chain = self.simulate_particle_decay(particle)
        plt.figure(figsize=(12, 8))
        # 1. Цепочка распада
        ax1 = plt.subplot(1, 2, 1)
        # Рисование цепочки
        y_pos = np.arange(len(decay_chain))
        particles = [entry['particle'] for entry in decay_chain]
        symbols = [self.particle_db.get_symbol(p) for p in particles]
        # Цвета по категориям
        colors = []
        for p in particles:
            category = self.particle_db.get_category(p)
            if 'boson' in category:
                colors.append('red')
            elif 'quark' in category:
                colors.append('blue')
            elif 'lepton' in category:
                colors.append('green')
            else:
                colors.append('purple')
        # Рисование
        for i in range(len(decay_chain)-1):
            ax1.plot([0, 1], [y_pos[i], y_pos[i+1]], 'k-', alpha=0.3)
        ax1.scatter(np.zeros(len(decay_chain)), y_pos, s=100, c=colors, zorder=5)
        # Подписи
        for i, (symbol, entry) in enumerate(zip(symbols, decay_chain)):
            ax1.text(0.05, y_pos[i], symbol, ha='left', va='center', fontsize=12, fontweight='bold')
            lifetime = entry['remaining_lifetime']
            if lifetime == float('inf'):
                lifetime_str = "∞"
            elif lifetime > 1:
                lifetime_str = f"{lifetime:.2f} с"
            else:
                lifetime_str = f"{lifetime*1e9:.2f} нс"
            ax1.text(-0.1, y_pos[i], lifetime_str, ha='right', va='center', fontsize=10)
        ax1.set_ylim(-1, len(decay_chain))
        ax1.set_yticks([])
        ax1.set_xlim(-0.3, 1.1)
        ax1.set_title(f'Цепочка распада: {self.particle_db.get_symbol(particle)}')
        ax1.axis('off')
        # 2. Временные характеристики
        ax2 = plt.subplot(1, 2, 2)
        # Время жизни
        lifetimes = []
        for entry in decay_chain:
            lifetime = entry['remaining_lifetime']
            if lifetime == float('inf'):
                lifetimes.append(1e10)  # Большое число для стабильных частиц
            else:
                lifetimes.append(lifetime)
        # Логарифмическая шкала
        lifetimes_log = np.log10(lifetimes)
        # Рисование
        ax2.bar(range(len(decay_chain)), lifetimes_log, color=colors)
        ax2.set_xticks(range(len(decay_chain)))
        ax2.set_xticklabels(symbols)
        ax2.set_yscale('log')
        ax2.set_ylabel('Время жизни (с)')
        ax2.set_title('Время жизни частиц в цепочке')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        # Добавление реальных значений на график
        for i, lifetime in enumerate(lifetimes):
            if lifetime < 1e-9:
                label = f"{lifetime*1e12:.2f} пс"
            elif lifetime < 1e-6:
                label = f"{lifetime*1e9:.2f} нс"
            elif lifetime < 1e-3:
                label = f"{lifetime*1e6:.2f} мкс"
            elif lifetime < 1:
                label = f"{lifetime*1000:.2f} мс"
            else:
                label = f"{lifetime:.2f} с"
            ax2.text(i, lifetimes_log[i] * 1.1, label, ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'particle_decay_{particle}.png')
        logger.info(f"Particle decay visualization saved to 'particle_decay_{particle}.png'")
    def create_animation(self, filename: str = "lhc_animation.mp4", num_frames: int = 100):
        """
        Создание анимации движения частиц в коллайдере.
        Параметры:
        filename: имя файла для сохранения
        num_frames: количество кадров
        """
        logger.info(f"Creating animation with {num_frames} frames...")
        # Подготовка данных
        x, y, z = self.simulate_particle_motion(num_turns=3, num_points=num_frames)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Инициализация графика
        line, = ax.plot([], [], [], 'r-', linewidth=2)
        point, = ax.plot([], [], [], 'ro', markersize=8)
        # Настройка графика
        ax.set_xlim(-self.radius*1.2, self.radius*1.2)
        ax.set_ylim(-self.radius*1.2, self.radius*1.2)
        ax.set_zlim(-self.radius*0.2, self.radius*0.2)
        ax.set_title('Движение частицы в Большом адронном коллайдере')
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')
        ax.set_box_aspect([1,1,0.2])  # Сжатие по Z для лучшей видимости
        # Добавление кольца для ориентации
        theta = np.linspace(0, 2*np.pi, 100)
        ring_x = self.radius * np.cos(theta)
        ring_y = self.radius * np.sin(theta)
        ring_z = np.zeros_like(theta)
        ax.plot(ring_x, ring_y, ring_z, 'b--', alpha=0.3)
        # Функция для обновления кадра
        def update(frame):
            line.set_data(x[:frame+1], y[:frame+1])
            line.set_3d_properties(z[:frame+1])
            point.set_data([x[frame]], [y[frame]])
            point.set_3d_properties([z[frame]])
            return line, point
        # Создание анимации
        anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                      interval=50, blit=True)
        # Сохранение анимации
        try:
            anim.save(filename, writer='ffmpeg', dpi=100)
            logger.info(f"Animation saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving animation: {e}")
            logger.info("Trying to save as GIF instead...")
            try:
                anim.save(filename.replace('.mp4', '.gif'), writer='pillow')
                logger.info(f"Animation saved as GIF to {filename.replace('.mp4', '.gif')}")
            except Exception as e2:
                logger.error(f"Error saving GIF animation: {e2}")
    def visualize_parameter_dependencies(self):
        """
        Визуализация зависимостей ключевых параметров
        """
        # Диапазон энергий от 1 ГэВ до 7 ТэВ
        energies = np.linspace(1e9, 7e12, 100)
        plt.figure(figsize=(14, 12))
        # 1. Магнитное поле vs энергия
        plt.subplot(3, 2, 1)
        fields = [self.calculate_magnetic_field(energy) for energy in energies]
        plt.plot(energies/1e12, fields)
        plt.axhline(y=self.max_magnetic_field, color='r', linestyle='--', label='Макс. поле (8.33 Тл)')
        plt.xlabel('Энергия, ТэВ')
        plt.ylabel('Магнитное поле, Тл')
        plt.title('Зависимость магнитного поля от энергии')
        plt.grid(True)
        plt.legend()
        # 2. Лоренц-фактор vs энергия
        plt.subplot(3, 2, 2)
        gamma = [self.calculate_relativistic_factor(energy) for energy in energies]
        plt.semilogy(energies/1e12, gamma)
        plt.xlabel('Энергия, ТэВ')
        plt.ylabel('Лоренц-фактор (γ)')
        plt.title('Релятивистские эффекты')
        plt.grid(True)
        # 3. Скорость vs энергия
        plt.subplot(3, 2, 3)
        speeds = [self.calculate_speed(energy) for energy in energies]
        speed_diff = [c - speed for speed in speeds]
        plt.semilogy(energies/1e12, speed_diff)
        plt.xlabel('Энергия, ТэВ')
        plt.ylabel('Отличие от скорости света (м/с)')
        plt.title('Скорость протона')
        plt.grid(True)
        # 4. Светимость vs энергия
        plt.subplot(3, 2, 4)
        luminosities = []
        for energy in energies:
            # Упрощенная модель: светимость уменьшается с ростом энергии из-за увеличения размера пучка
            emittance = self.calculate_emittance(energy)
            beam_size = np.sqrt(emittance * 150)  # Упрощенная оценка
            luminosity = self.peak_luminosity * (self.beam_size_x / beam_size)**2
            luminosities.append(luminosity)
        plt.semilogy(energies/1e12, luminosities)
        plt.axhline(y=self.peak_luminosity, color='r', linestyle='--', label='Пиковая светимость')
        plt.xlabel('Энергия, ТэВ')
        plt.ylabel('Светимость (см⁻²с⁻¹)')
        plt.title('Светимость в зависимости от энергии')
        plt.grid(True)
        plt.legend()
        # 5. Потери на синхротронное излучение
        plt.subplot(3, 2, 5)
        radiation_losses = [self.calculate_synchrotron_radiation(energy) for energy in energies]
        plt.semilogy(energies/1e12, radiation_losses)
        plt.xlabel('Энергия, ТэВ')
        plt.ylabel('Потери энергии (Вт/частицу)')
        plt.title('Синхротронное излучение')
        plt.grid(True)
        # 6. Эмиттанс vs энергия
        plt.subplot(3, 2, 6)
        emittances = [self.calculate_emittance(energy) for energy in energies]
        plt.plot(energies/1e12, emittances)
        plt.xlabel('Энергия, ТэВ')
        plt.ylabel('Эмиттанс (м·рад)')
        plt.title('Эмиттанс пучка')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('parameter_dependencies.png')
        logger.info("Parameter dependencies visualization saved to 'parameter_dependencies.png'")
    def export_to_root(self, filename: str = "lhc_simulation.root"):
        """
        Экспорт данных симуляции в формат ROOT.
        """
        try:
            import ROOT
            import array
            
            # Создание файла ROOT
            root_file = ROOT.TFile(filename, "RECREATE")
            
            # Создание дерева для событий
            tree = ROOT.TTree("Events", "Simulated LHC Events")
            
            # Определение переменных
            event_id = array.array('i', [0])
            event_type = ROOT.std.string()
            energy = array.array('d', [0.0])
            num_products = array.array('i', [0])
            
            # Создание веток
            tree.Branch("event_id", event_id, "event_id/I")
            tree.Branch("event_type", event_type)
            tree.Branch("energy", energy, "energy/D")
            tree.Branch("num_products", num_products, "num_products/I")
            
            # Добавление частиц как векторов
            max_products = 50
            product_names = [ROOT.std.string() for _ in range(max_products)]
            product_energies = array.array('d', [0.0] * max_products)
            
            tree.Branch("product_names", product_names, f"product_names[{max_products}]/string")
            tree.Branch("product_energies", product_energies, f"product_energies[{max_products}]/D")
            
            # Заполнение дерева данными
            for event in self.simulation_state['collisions']:
                event_id[0] = event['event_id']
                event_type = ROOT.std.string(event['event_type'])
                energy[0] = event['energy']
                num_products[0] = len(event['products'])
                
                # Заполнение данных о продуктах
                for i, product in enumerate(event['products'][:max_products]):
                    product_names[i] = ROOT.std.string(product['name'])
                    product_energies[i] = product['energy']
                
                tree.Fill()
            
            # Сохранение и закрытие файла
            root_file.Write()
            root_file.Close()
            
            logger.info(f"Simulation data exported to ROOT format: {filename}")
            return True
        except ImportError:
            logger.warning("ROOT not installed. Cannot export to ROOT format.")
            return False
    def generate_events_with_madgraph(self, process: str = "p p > h", num_events: int = 100, 
                                     energy: float = 13e12, output_file: str = "mg_events.lhe"):
        """
        Генерация событий с использованием MadGraph.
        """
        try:
            import subprocess
            import tempfile
            import os
            
            # Создание временного скрипта для MadGraph
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                script = f"""
                generate {process}
                output lhc_simulation
                launch
                set ebeam1 {energy/2e12}
                set ebeam2 {energy/2e12}
                set nevents {num_events}
                set iseed {int(time.time())}
                set ptj 20
                set etaj 5
                shower=OFF
                done
                """
                f.write(script)
                script_path = f.name
            
            # Запуск MadGraph
            logger.info(f"Running MadGraph to generate {num_events} events for process: {process}")
            result = subprocess.run(
                ['mg5_aMC', script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Поиск сгенерированного файла
            event_file = "lhc_simulation/Events/run_01/unweighted_events.lhe.gz"
            if os.path.exists(event_file):
                import shutil
                shutil.copy2(event_file, output_file)
                logger.info(f"MadGraph events saved to {output_file}")
                return output_file
            else:
                logger.error("MadGraph event file not found")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"MadGraph execution failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error interfacing with MadGraph: {str(e)}")
            return None
        finally:
            if 'script_path' in locals() and os.path.exists(script_path):
                os.unlink(script_path)
    # ===================================================================
    # 6. Основной метод симуляции
    # ===================================================================
    def run_full_simulation(self, num_collisions: int = 10):
        """
        Запуск полной симуляции с анализом результатов.
        Параметры:
        num_collisions: количество столкновений для симуляции
        """
        logger.info("\n" + "="*70)
        logger.info("ЗАПУСК ПОЛНОЙ СИМУЛЯЦИИ БОЛЬШОГО АДРОННОГО КОЛЛАЙДЕРА")
        logger.info("Это улучшенная учебная модель, основанная на реальных данных из открытых источников")
        logger.info("ВНИМАНИЕ: Модель не предназначена для профессиональных научных исследований")
        logger.info("="*70)
        # 1. Визуализация кольца
        logger.info("\n1. Визуализация кольца коллайдера")
        self.visualize_ring()
        self.visualize_ring_3d()
        # 2. Визуализация движения частицы
        logger.info("\n2. Визуализация движения частицы")
        self.visualize_particle_motion()
        # 3. Симуляция и визуализация столкновений
        logger.info(f"\n3. Симуляция {num_collisions} столкновений")
        for i in range(num_collisions):
            logger.info(f"   Столкновение {i+1}/{num_collisions}")
            self.visualize_collision()
            self.visualize_collision_interactive()
        # 4. Визуализация динамики пучка
        logger.info("\n4. Визуализация динамики пучка")
        self.visualize_beam_dynamics()
        # 5. Визуализация распада частиц
        logger.info("\n5. Визуализация распада частиц")
        self.visualize_particle_decay('higgs_boson')
        self.visualize_particle_decay('top_quark')
        # 6. Визуализация зависимостей параметров
        logger.info("\n6. Визуализация зависимостей ключевых параметров")
        self.visualize_parameter_dependencies()
        # 7. Создание анимации
        logger.info("\n7. Создание анимации движения частиц")
        self.create_animation()
        # 8. Визуализация свойств частиц
        logger.info("\n8. Визуализация свойств элементарных частиц")
        particles = ['proton', 'electron', 'muon', 'higgs_boson', 'top_quark', 'w_boson']
        for particle in particles:
            self.particle_db.plot_particle_properties(particle)
        # 9. Анализ результатов
        logger.info("\n9. Анализ результатов симуляции")
        # Расчет ключевых параметров
        logger.info(f"\nПараметры магнитного поля:")
        req_field = self.calculate_magnetic_field(self.beam_energy)
        logger.info(f"- Требуемое магнитное поле для 6.5 ТэВ: {req_field:.2f} Тл")
        logger.info(f"- Доступное максимальное поле: {self.max_magnetic_field} Тл")
        logger.info(f"\nРелятивистские параметры:")
        gamma = self.calculate_relativistic_factor(self.beam_energy)
        logger.info(f"- Лоренц-фактор: {gamma:.0f}")
        speed = self.calculate_speed(self.beam_energy)
        logger.info(f"- Скорость протона: {speed:.2f} м/с")
        logger.info(f"- Отличие от скорости света: {c - speed:.4f} м/с")
        logger.info(f"\nПараметры светимости:")
        luminosity = self.calculate_luminosity()
        logger.info(f"- Расчетная светимость: {luminosity:.1e} см^-2 с^-1")
        logger.info(f"- Реальная пиковая светимость: {self.peak_luminosity:.1e} см^-2 с^-1")
        logger.info(f"\nДлина пучка:")
        bunch_length = self.calculate_bunch_length()
        logger.info(f"- Длина пучка: {bunch_length:.2f} м")
        logger.info(f"\nВремя одного оборота:")
        revolution_time = self.calculate_revolution_time()
        revolution_freq = self.calculate_revolution_frequency()
        logger.info(f"- Время оборота: {revolution_time*1e6:.2f} мкс")
        logger.info(f"- Частота обращения: {revolution_freq/1e3:.2f} кГц")
        # Сохранение состояния симуляции
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._save_simulation_state(timestamp)
        # Экспорт в ROOT (если возможно)
        self.export_to_root()
        logger.info("\n" + "="*70)
        logger.info("СИМУЛЯЦИЯ БОЛЬШОГО АДРОННОГО КОЛЛАЙДЕРА ЗАВЕРШЕНА")
        logger.info(f"Все результаты сохранены с меткой времени {timestamp}")
        logger.info("="*70)
    def _save_simulation_state(self, timestamp: str):
        """
        Сохранение состояния симуляции.
        Параметры:
        timestamp: временная метка для имени файла
        """
        try:
            # Создание директории для сохранения
            output_dir = f"lhc_simulation_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            # Сохранение основных параметров
            with open(f"{output_dir}/parameters.txt", "w") as f:
                f.write(f"Время симуляции: {timestamp}\n")
                f.write(f"Длина окружности: {self.circumference} м\n")
                f.write(f"Радиус: {self.radius} м\n")
                f.write(f"Энергия пучка: {self.beam_energy/1e12} ТэВ\n")
                f.write(f"Максимальное магнитное поле: {self.max_magnetic_field} Тл\n")
                f.write(f"Пиковая светимость: {self.peak_luminosity} см^-2 с^-1\n")
            # Сохранение состояния
            state_data = {
                'timestamp': timestamp,
                'parameters': {
                    'circumference': self.circumference,
                    'radius': self.radius,
                    'beam_energy': self.beam_energy,
                    'max_magnetic_field': self.max_magnetic_field,
                    'peak_luminosity': self.peak_luminosity
                },
                'simulation_state': self.simulation_state
            }
            # Сохранение состояния в JSON
            with open(f"{output_dir}/simulation_state.json", "w") as f:
                json.dump(state_data, f, indent=4)
            # Копирование изображений
            for image in ['lhc_ring.png', 'particle_motion.png', 'collision_visualization.png',
                         'beam_dynamics.png', 'parameter_dependencies.png', 'lhc_ring_3d.html',
                         'collision_interactive.html']:
                if os.path.exists(image):
                    shutil.copy2(image, f"{output_dir}/{image}")
            # Сохранение базы данных частиц
            self.particle_db.save_to_json(f"{output_dir}/particle_database.json")
            logger.info(f"Simulation state saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving simulation state: {e}")
# ===================================================================
# 7. Основная функция
# ===================================================================
def main():
    """Основная функция для запуска симуляции"""
    logger.info("\n" + "="*70)
    logger.info("ЗАПУСК МОДЕЛИ БОЛЬШОГО АДРОННОГО КОЛЛАЙДЕРА")
    logger.info("Это улучшенная учебная модель, основанная на реальных данных из открытых источников")
    logger.info("ВНИМАНИЕ: Модель не предназначена для профессиональных научных исследований")
    logger.info("="*70)
    try:
        # Создание модели БАК
        lhc = LHC_Model()
        # Запуск полной симуляции
        lhc.run_full_simulation(num_collisions=5)
        logger.info("\n" + "="*70)
        logger.info("МОДЕЛИРОВАНИЕ БОЛЬШОГО АДРОННОГО КОЛЛАЙДЕРА ЗАВЕРШЕНО УСПЕШНО")
        logger.info("Все результаты сохранены в соответствующих файлах")
        logger.info("="*70)
    except Exception as e:
        logger.exception("КРИТИЧЕСКАЯ ОШИБКА ВО ВРЕМЯ СИМУЛЯЦИИ")
        raise
    finally:
        logger.info("\nЗАВЕРШЕНИЕ РАБОТЫ. Спасибо за использование модели БАК!")
if __name__ == "__main__":
    main()
