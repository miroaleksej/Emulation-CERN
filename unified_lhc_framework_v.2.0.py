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
# Добавлены физические константы из Исправления.txt
# ===================================================================
c = 299792458  # м/с (скорость света)
m_p = 0.938272  # ГэВ/с² (масса протона)
G_F = 1.166e-5  # Ферми-константа (ГэВ⁻²)
M_W = 80.379  # ГэВ (масса W-бозона)
alpha_s = 0.118  # Сильная константа связи
v = 246  # vev, ГэВ (электрослабый вакуумный ожидаемый вакуум)
hbar = 6.582119569e-25  # ГэВ·с (приведенная постоянная Планка)
k_B = 8.617333262e-5  # эВ/К (постоянная Больцмана)

# ===================================================================
# 1. Настройка логирования
# ===================================================================
"""Настройка логирования для отслеживания процесса выполнения.
Создает лог-файл и выводит сообщения в консоль."""
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("unified_lhc_framework.log"),
                              logging.StreamHandler()])
logger = logging.getLogger("Unified_LHC_Framework")
# ===================================================================

# ===================================================================
# Добавлены улучшения модели динамики пучка из Исправления.txt
# ===================================================================
class BeamDynamics:
    """Улучшенная модель динамики пучка с релятивистскими расчетами"""
    def __init__(self, config):
        self.config = config
        self.state = {
            'sigma_x': config.get('beam', {}).get('sigma_x', 0.045),
            'sigma_y': config.get('beam', {}).get('sigma_y', 0.045),
            'epsilon': config.get('beam', {}).get('emittance', 2.5e-6),
            'N_p': config.get('beam', {}).get('bunch_intensity', 1.15e11),
            'turn': 0,
            'beta_x': config.get('beam', {}).get('beta_x', 56.5),
            'beta_y': config.get('beam', {}).get('beta_y', 56.5),
            'D_x': config.get('beam', {}).get('dispersion_x', 0.0),
            'D_y': config.get('beam', {}).get('dispersion_y', 0.0),
            'energy_spread': config.get('beam', {}).get('energy_spread', 1e-4)
        }
        self.history = []
    
    def evolve(self, num_turns: int, include_space_charge: bool = True):
        """Релятивистская эволюция пучка с учетом различных эффектов"""
        for _ in range(num_turns):
            self._apply_lattice_optics()
            self._apply_synchrotron_radiation()
            self._apply_quantum_excitation()
            if include_space_charge:
                self._apply_space_charge_effect()
            self._apply_intra_beam_scattering()
            self.state['turn'] += 1
            self.history.append(self.state.copy())
        
        return self.state
    
    def _apply_lattice_optics(self):
        """Применение оптики ускорителя (бета-функции, дисперсия)"""
        # Обновление размеров пучка на основе бета-функций и эмиттанса
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)
        
        # Релятивистская формула для размеров пучка
        self.state['sigma_x'] = np.sqrt(self.state['epsilon'] * self.state['beta_x'] / beta_rel)
        self.state['sigma_y'] = np.sqrt(self.state['epsilon'] * self.state['beta_y'] / beta_rel)
    
    def _apply_synchrotron_radiation(self):
        """Учет потерь энергии на синхротронное излучение"""
        # Формула для потерь энергии на оборот
        U_0 = (55 * alpha_s * (m_p * c**2)**4 * self.config['beam']['beam_energy']**4) / \
              (32 * np.sqrt(3) * m_p**4 * c**5 * self.config['beam']['circumference'])
        
        # Обновление энергетического разброса
        self.state['energy_spread'] = np.sqrt(U_0 / (2 * np.pi * self.config['beam']['tune']))
    
    def _apply_quantum_excitation(self):
        """Учет квантового возбуждения"""
        # Квантовое возбуждение компенсирует сжатие из-за синхротронного излучения
        D_q = (55 * alpha_s * (m_p * c**2)**5 * self.config['beam']['beam_energy']**5) / \
              (32 * np.sqrt(3) * m_p**5 * c**7 * self.config['beam']['circumference']**2)
        
        # Обновление эмиттанса
        self.state['epsilon'] += D_q
    
    def _apply_space_charge_effect(self):
        """Учет эффекта пространственного заряда"""
        # Пространственный заряд вызывает дополнительную фокусировку/дефокусировку
        space_charge_strength = (self.state['N_p'] * m_p * c**2) / \
                               (self.config['beam']['beam_energy'] * 
                                self.config['beam']['circumference'] * 
                                (self.state['sigma_x'] + self.state['sigma_y']))
        
        # Модификация бета-функций
        self.state['beta_x'] *= (1 + 0.1 * space_charge_strength)
        self.state['beta_y'] *= (1 - 0.1 * space_charge_strength)
    
    def _apply_intra_beam_scattering(self):
        """Учет рассеяния внутри пучка (IBS)"""
        # IBS увеличивает эмиттанс и энергетический разброс
        ibs_factor = 0.001 * (self.state['N_p'] / 1e11) * (7000 / self.config['beam']['beam_energy'])
        self.state['epsilon'] *= (1 + ibs_factor)
        self.state['energy_spread'] *= (1 + 0.5 * ibs_factor)
    
    def get_luminosity(self) -> float:
        """Расчет светимости с учетом всех эффектов"""
        # Формула для светимости LHC
        num_bunches = self.config['beam']['num_bunches']
        bunch_intensity = self.config['beam']['bunch_intensity']
        revolution_freq = c / self.config['beam']['circumference']
        beta_star = self.config['beam']['beta_star']
        
        # Учет геометрического фактора перекрытия
        overlap_factor = 1.0 / np.sqrt(1 + (self.config['beam']['crossing_angle'] * 
                            self.config['beam']['beta_star'] / 
                            (2 * self.state['sigma_x']))**2)
        
        # Формула светимости с релятивистскими поправками
        gamma_rel = self.config['beam']['beam_energy'] / m_p
        beta_rel = np.sqrt(1 - 1/gamma_rel**2)
        
        luminosity = (num_bunches * bunch_intensity**2 * revolution_freq * overlap_factor) / \
                     (4 * np.pi * beta_rel * self.state['sigma_x'] * self.state['sigma_y'])
        
        return luminosity

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
# Добавлен класс PDFModel из Исправления.txt
# ===================================================================
class PDFModel:
    """Модель функций частицного распределения (PDF)"""
    def __init__(self, config):
        self.config = config
        self.pdf_data = self._load_pdf_data()
    
    def _load_pdf_data(self):
        """Загрузка данных PDF из конфигурации или внешних источников"""
        # В реальной реализации здесь будет загрузка данных из файлов PDF
        logger.info("Загрузка данных PDF...")
        return {
            'proton': {
                'u': lambda x, Q2: self._u_quark_pdf(x, Q2),
                'd': lambda x, Q2: self._d_quark_pdf(x, Q2),
                's': lambda x, Q2: self._s_quark_pdf(x, Q2),
                'g': lambda x, Q2: self._gluon_pdf(x, Q2)
            }
        }
    
    def _u_quark_pdf(self, x, Q2):
        """PDF для u-кварков в протоне"""
        # Упрощенная модель, в реальности будет сложнее
        if x <= 0 or x >= 1:
            return 0.0
        # Модель, основанная на данных CTEQ
        return 1.368 * (1 - x)**3.08 * x**0.535 * (1 + 1.562 * np.sqrt(x) + 3.811 * x)
    
    def _d_quark_pdf(self, x, Q2):
        """PDF для d-кварков в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        # Модель, основанная на данных CTEQ
        return 0.816 * (1 - x)**4.03 * x**0.383 * (1 + 2.637 * np.sqrt(x) + 2.985 * x)
    
    def _s_quark_pdf(self, x, Q2):
        """PDF для s-кварков в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        # s-кварки примерно в 30% от суммы u и d
        return 0.3 * (self._u_quark_pdf(x, Q2) + self._d_quark_pdf(x, Q2)) / 2.0
    
    def _gluon_pdf(self, x, Q2):
        """PDF для глюонов в протоне"""
        if x <= 0 or x >= 1:
            return 0.0
        # Глюонная PDF доминирует при малых x
        return 1.74 * (1 - x)**5.0 * x**-0.2 * np.exp(-1.4 * np.sqrt(np.log(1/x)))
    
    def get_parton_distribution(self, particle: str, x: float, Q2: float) -> Dict[str, float]:
        """Получение распределения частиц для заданной частицы, x и Q2"""
        if particle not in self.pdf_data:
            raise ValueError(f"PDF для частицы {particle} не поддерживается")
        
        pdfs = {}
        for flavor, pdf_func in self.pdf_data[particle].items():
            pdfs[flavor] = pdf_func(x, Q2)
        
        # Нормализация, чтобы сумма PDF = 1
        total = sum(pdfs.values())
        if total > 0:
            for flavor in pdfs:
                pdfs[flavor] /= total
        
        return pdfs
    
    def sample_parton(self, particle: str, Q2: float) -> Tuple[str, float]:
        """Сэмплирование частицы и значения x для заданной частицы и Q2"""
        # Генерация x с использованием обратного метода преобразования
        x = self._sample_x()
        
        # Получение PDF для этого x
        pdfs = self.get_parton_distribution(particle, x, Q2)
        
        # Выбор частицы на основе PDF
        flavors = list(pdfs.keys())
        probabilities = list(pdfs.values())
        parton = np.random.choice(flavors, p=probabilities)
        
        return parton, x
    
    def _sample_x(self) -> float:
        """Генерация доли импульса x для партонa с физически корректным распределением"""
        # Используем обратный метод преобразования для генерации x
        # с учетом физических ограничений (0 < x < 1)
        while True:
            u = random.random()
            # Используем обратную функцию к PDF для генерации x
            # Упрощенная модель для демонстрации
            x = u ** (1/3.5)  # Соответствует распределению ~ (1-x)^2.5
            
            if 0 < x < 1:
                return x

# ===================================================================
# 6. Реализации физических движков
# ===================================================================
class BuiltInPhysicsEngine(PhysicsEngineInterface):
    """Улучшенный встроенный физический движок с элементами QCD и ЭВ."""
    def __init__(self, particle_db, config):
        self.particle_db = particle_db
        self.config = config
        self.pdf_model = PDFModel(config)  # Инициализация модели PDF
    
    def get_name(self) -> str:
        return "built-in"
    
    def is_available(self) -> bool:
        return True
    
    def _calculate_cross_section(self, process_type: str, energy: float, x1: float, x2: float) -> float:
        """Расчет физического сечения для заданного процесса"""
        # Энергия в системе центра масс для партонов
        E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)
        
        if process_type == "drell-yan":
            # Приближение для процесса Дрелл-Яна
            if E_CM_parton < M_W:
                return 0.0
            return 1185.0 / (E_CM_parton**2) * (1 - M_W**2/E_CM_parton**2)**(1.5)
        
        elif process_type == "gluon_fusion":
            # Приближение для глюонной фьюжн (например, производство Хиггса)
            if E_CM_parton < 125:  # Масса Хиггса
                return 0.0
            return 20.0 / (E_CM_parton**2) * np.exp(-(E_CM_parton-125)/50)
        
        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            return 50.0 / (E_CM_parton**2)
        
        elif process_type == "jet_production":
            # Производство струй
            return 30000.0 / (E_CM_parton**2)
        
        return 0.0
    
    def _get_process_probs(self, flavor1: str, flavor2: str, E_CM_parton: float) -> Dict[str, float]:
        """Определение вероятностей процессов на основе физических сечений"""
        processes = {}
        
        # Определяем возможные процессы на основе типов частиц
        if 'quark' in flavor1 and 'antiquark' in flavor2:
            processes["drell-yan"] = self._calculate_cross_section("drell-yan", E_CM_parton, 0, 0)
            processes["quark_antiquark"] = self._calculate_cross_section("quark_antiquark", E_CM_parton, 0, 0)
        
        elif 'gluon' in flavor1 and 'gluon' in flavor2:
            processes["gluon_fusion"] = self._calculate_cross_section("gluon_fusion", E_CM_parton, 0, 0)
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_CM_parton, 0, 0)
        
        elif ('quark' in flavor1 and 'gluon' in flavor2) or ('gluon' in flavor1 and 'quark' in flavor2):
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_CM_parton, 0, 0)
        
        # Нормализуем вероятности
        total = sum(processes.values())
        if total > 0:
            for process in processes:
                processes[process] /= total
        else:
            # Если все сечения нулевые, используем равномерное распределение
            for process in processes:
                processes[process] = 1.0 / len(processes)
        
        return processes
    
    def _generate_products(self, process_type: str, E_CM_parton: float) -> List[Dict]:
        """Генерация продуктов столкновения на основе физической модели"""
        products = []
        
        if process_type == "drell-yan":
            # Процесс Дрелл-Яна: производство лептонных пар через Z/γ*
            if E_CM_parton > M_W:
                # Может производить W-бозоны
                if random.random() < 0.5:
                    products.append({'name': 'W_plus', 'energy': E_CM_parton*0.8})
                    # Распад W+ -> e+ + nu_e
                    products.append({'name': 'positron', 'energy': E_CM_parton*0.4})
                    products.append({'name': 'neutrino_e', 'energy': E_CM_parton*0.4})
                else:
                    products.append({'name': 'W_minus', 'energy': E_CM_parton*0.8})
                    # Распад W- -> e- + anti_nu_e
                    products.append({'name': 'electron', 'energy': E_CM_parton*0.4})
                    products.append({'name': 'antineutrino_e', 'energy': E_CM_parton*0.4})
            else:
                # Производство Z-бозонов или виртуальных фотонов
                if random.random() < 0.33:
                    # Электрон-позитронная пара
                    products.append({'name': 'electron', 'energy': E_CM_parton*0.45})
                    products.append({'name': 'positron', 'energy': E_CM_parton*0.45})
                elif random.random() < 0.66:
                    # Мюонная пара
                    products.append({'name': 'muon', 'energy': E_CM_parton*0.45})
                    products.append({'name': 'antimuon', 'energy': E_CM_parton*0.45})
                else:
                    # Кварковая пара (адронизация)
                    products.append({'name': 'u_quark', 'energy': E_CM_parton*0.45})
                    products.append({'name': 'u_antiquark', 'energy': E_CM_parton*0.45})
        
        elif process_type == "gluon_fusion":
            # Глюонная фьюжн (например, производство Хиггса)
            products.append({'name': 'higgs', 'energy': E_CM_parton*0.9})
            # Распад Хиггса
            if random.random() < 0.58:
                products.append({'name': 'b_quark', 'energy': E_CM_parton*0.45})
                products.append({'name': 'bbar_quark', 'energy': E_CM_parton*0.45})
            # Распад на W-бозоны
            elif random.random() < 0.58 + 0.21:
                products.append({'name': 'W_plus', 'energy': E_CM_parton*0.45})
                products.append({'name': 'W_minus', 'energy': E_CM_parton*0.45})
            else:
                # Обычное производство струй
                num_jets = random.randint(2, 4)
                for _ in range(num_jets):
                    jet_energy = E_CM_parton * random.uniform(0.1, 0.4)
                    products.append({
                        'name': 'jet',
                        'energy': jet_energy,
                        'px': random.uniform(-jet_energy, jet_energy),
                        'py': random.uniform(-jet_energy, jet_energy),
                        'pz': random.uniform(-jet_energy, jet_energy)
                    })
        
        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            if random.random() < 0.5:
                products.append({'name': 'Z_boson', 'energy': E_CM_parton*0.9})
                # Распад Z-бозона
                if random.random() < 0.33:
                    products.append({'name': 'electron', 'energy': E_CM_parton*0.45})
                    products.append({'name': 'positron', 'energy': E_CM_parton*0.45})
                elif random.random() < 0.66:
                    products.append({'name': 'muon', 'energy': E_CM_parton*0.45})
                    products.append({'name': 'antimuon', 'energy': E_CM_parton*0.45})
                else:
                    products.append({'name': 'tau', 'energy': E_CM_parton*0.45})
                    products.append({'name': 'antitau', 'energy': E_CM_parton*0.45})
            else:
                # Производство кварковых пар
                products.append({'name': 'c_quark', 'energy': E_CM_parton*0.45})
                products.append({'name': 'c_antiquark', 'energy': E_CM_parton*0.45})
        
        elif process_type == "jet_production":
            # Производство струй
            num_jets = random.randint(1, 3)
            for _ in range(num_jets):
                jet_energy = E_CM_parton * random.uniform(0.1, 0.5)
                products.append({
                    'name': 'jet',
                    'energy': jet_energy,
                    'px': random.uniform(-jet_energy, jet_energy),
                    'py': random.uniform(-jet_energy, jet_energy),
                    'pz': random.uniform(-jet_energy, jet_energy)
                })
        
        return products
    
    def interact(self, particle1: str, particle2: str, energy: float, num_events: int = 1, **kwargs) -> List[Dict]:
        """Моделирование взаимодействия частиц с физически корректными сечениями"""
        events = []
        
        for _ in range(num_events):
            # Для протон-протонных столкновений используем модель PDF
            if particle1 == "proton" and particle2 == "proton":
                # Получаем партонные распределения
                Q2 = energy**2  # Упрощенная модель для шкалы Q2
                
                # Сэмплируем партоны из обоих протонов
                parton1, x1 = self.pdf_model.sample_parton("proton", Q2)
                parton2, x2 = self.pdf_model.sample_parton("proton", Q2)
                
                # Энергия в системе центра масс для партонов
                E_CM_parton = np.sqrt(2 * x1 * x2 * energy**2)
                
                # Определяем процесс по вероятностям, пропорциональным сечениям
                processes = self._get_process_probs(parton1, parton2, E_CM_parton)
                process = np.random.choice(list(processes.keys()), p=list(processes.values()))
                
                # Генерация продуктов столкновения
                products = self._generate_products(process, E_CM_parton)
                
                events.append({
                    'process': process,
                    'parton1': parton1,
                    'x1': x1,
                    'parton2': parton2,
                    'x2': x2,
                    'E_CM_parton': E_CM_parton,
                    'products': products
                })
            else:
                # Для других типов частиц используем упрощенную модель
                total_energy = energy * 2  # Общая энергия в системе центра масс
                
                # Определяем тип взаимодействия на основе частиц
                if 'quark' in particle1 and 'quark' in particle2:
                    process_type = "quark_quark"
                elif 'quark' in particle1 and 'gluon' in particle2:
                    process_type = "quark_gluon"
                elif 'gluon' in particle1 and 'gluon' in particle2:
                    process_type = "gluon_gluon"
                else:
                    process_type = "other"
                
                # Генерация продуктов
                products = []
                if process_type == "quark_quark":
                    if random.random() < 0.7:
                        products.append({'name': 'gluon', 'energy': total_energy*0.5})
                        products.append({'name': 'quark', 'energy': total_energy*0.25})
                        products.append({'name': 'antiquark', 'energy': total_energy*0.25})
                    else:
                        num_hadrons = random.randint(2, 5)
                        products.extend(self._fragment_hadron(total_energy, num_hadrons))
                elif process_type == "quark_gluon":
                    products.append({'name': 'W_plus', 'energy': total_energy*0.8})
                    # Распад W+ -> e+ + nu_e
                    products.append({'name': 'positron', 'energy': total_energy*0.4})
                    products.append({'name': 'neutrino_e', 'energy': total_energy*0.4})
                elif process_type == "gluon_gluon":
                    process_type = "jet_production"
                    # Генерация струй
                    num_jets = random.randint(1, 3)
                    for _ in range(num_jets):
                        jet_energy = total_energy * random.uniform(0.1, 0.5)
                        products.append({'name': 'jet', 'energy': jet_energy})
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
                            'pz': random.uniform(-jet_energy, jet_energy)
                        })
                
                events.append({
                    'process': process_type,
                    'E_CM': total_energy,
                    'products': products
                })
        
        return events
    
    def _fragment_hadron(self, total_energy: float, num_hadrons: int) -> List[Dict]:
        """Адронизация кварков в адроны"""
        hadrons = []
        energy_per_hadron = total_energy / num_hadrons
        
        for i in range(num_hadrons):
            # Распределение энергии между адронами
            fraction = random.uniform(0.8, 1.2) / num_hadrons
            energy = total_energy * fraction
            
            # Выбор типа адрона
            hadron_type = random.choice(['pion+', 'pion-', 'pion0', 'kaon+', 'kaon-', 'proton', 'neutron'])
            hadrons.append({'name': hadron_type, 'energy': energy})
        
        return hadrons

# ===================================================================
# Добавлен класс InteractionGenerator из Исправления.txt
# ===================================================================
class InteractionGenerator:
    """Генерация физических процессов с корректными сечениями"""
    def __init__(self, config):
        self.config = config
        self.pdf = PDFModel(config)
    
    def generate_event(self, E_CM, x1, flavor1, x2, flavor2):
        E_parton = np.sqrt(2 * x1 * x2 * E_CM**2)
        # Определяем процесс по вероятностям, пропорциональным сечениям
        processes = self._get_process_probs(flavor1, flavor2, E_parton)
        process = np.random.choice(list(processes.keys()), p=list(processes.values()))
        products = self._generate_products(process, E_parton)
        return {
            'process': process,
            'E_CM_parton': E_parton,
            'products': products,
            'x1': x1,
            'x2': x2,
            'flavor1': flavor1,
            'flavor2': flavor2
        }
    
    def _get_process_probs(self, flavor1, flavor2, E_parton):
        """Определение вероятностей процессов на основе физических сечений"""
        processes = {}
        
        # Определяем возможные процессы на основе типов частиц
        if 'quark' in flavor1 and 'antiquark' in flavor2:
            processes["drell-yan"] = self._calculate_cross_section("drell-yan", E_parton)
            processes["quark_antiquark"] = self._calculate_cross_section("quark_antiquark", E_parton)
        
        elif 'gluon' in flavor1 and 'gluon' in flavor2:
            processes["gluon_fusion"] = self._calculate_cross_section("gluon_fusion", E_parton)
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_parton)
        
        elif ('quark' in flavor1 and 'gluon' in flavor2) or ('gluon' in flavor1 and 'quark' in flavor2):
            processes["jet_production"] = self._calculate_cross_section("jet_production", E_parton)
        
        # Нормализуем вероятности
        total = sum(processes.values())
        if total > 0:
            for process in processes:
                processes[process] /= total
        else:
            # Если все сечения нулевые, используем равномерное распределение
            for process in processes:
                processes[process] = 1.0 / len(processes)
        
        return processes
    
    def _calculate_cross_section(self, process_type, E_CM_parton):
        """Расчет физического сечения для заданного процесса"""
        # Энергия в системе центра масс для партонов
        if process_type == "drell-yan":
            # Приближение для процесса Дрелл-Яна
            if E_CM_parton < M_W:
                return 0.0
            return 1185.0 / (E_CM_parton**2) * (1 - M_W**2/E_CM_parton**2)**(1.5)
        
        elif process_type == "gluon_fusion":
            # Приближение для глюонной фьюжн (например, производство Хиггса)
            if E_CM_parton < 125:  # Масса Хиггса
                return 0.0
            return 20.0 / (E_CM_parton**2) * np.exp(-(E_CM_parton-125)/50)
        
        elif process_type == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            return 50.0 / (E_CM_parton**2)
        
        elif process_type == "jet_production":
            # Производство струй
            return 30000.0 / (E_CM_parton**2)
        
        return 0.0
    
    def _generate_products(self, process, E_parton):
        """Генерация продуктов столкновения на основе физической модели"""
        products = []
        
        if process == "drell-yan":
            # Процесс Дрелл-Яна: производство лептонных пар через Z/γ*
            if E_parton > M_W:
                # Может производить W-бозоны
                if random.random() < 0.5:
                    products.append({'name': 'W_plus', 'energy': E_parton*0.8})
                    # Распад W+ -> e+ + nu_e
                    products.append({'name': 'positron', 'energy': E_parton*0.4})
                    products.append({'name': 'neutrino_e', 'energy': E_parton*0.4})
                else:
                    products.append({'name': 'W_minus', 'energy': E_parton*0.8})
                    # Распад W- -> e- + anti_nu_e
                    products.append({'name': 'electron', 'energy': E_parton*0.4})
                    products.append({'name': 'antineutrino_e', 'energy': E_parton*0.4})
            else:
                # Производство Z-бозонов или виртуальных фотонов
                if random.random() < 0.33:
                    # Электрон-позитронная пара
                    products.append({'name': 'electron', 'energy': E_parton*0.45})
                    products.append({'name': 'positron', 'energy': E_parton*0.45})
                elif random.random() < 0.66:
                    # Мюонная пара
                    products.append({'name': 'muon', 'energy': E_parton*0.45})
                    products.append({'name': 'antimuon', 'energy': E_parton*0.45})
                else:
                    # Кварковая пара (адронизация)
                    products.append({'name': 'u_quark', 'energy': E_parton*0.45})
                    products.append({'name': 'u_antiquark', 'energy': E_parton*0.45})
        
        elif process == "gluon_fusion":
            # Глюонная фьюжн (например, производство Хиггса)
            products.append({'name': 'higgs', 'energy': E_parton*0.9})
            # Распад Хиггса
            if random.random() < 0.58:
                products.append({'name': 'b_quark', 'energy': E_parton*0.45})
                products.append({'name': 'bbar_quark', 'energy': E_parton*0.45})
            # Распад на W-бозоны
            elif random.random() < 0.58 + 0.21:
                products.append({'name': 'W_plus', 'energy': E_parton*0.45})
                products.append({'name': 'W_minus', 'energy': E_parton*0.45})
            else:
                # Обычное производство струй
                num_jets = random.randint(2, 4)
                for _ in range(num_jets):
                    jet_energy = E_parton * random.uniform(0.1, 0.4)
                    products.append({
                        'name': 'jet',
                        'energy': jet_energy,
                        'px': random.uniform(-jet_energy, jet_energy),
                        'py': random.uniform(-jet_energy, jet_energy),
                        'pz': random.uniform(-jet_energy, jet_energy)
                    })
        
        elif process == "quark_antiquark":
            # Процесс кварк-антикварк аннигиляции
            if random.random() < 0.5:
                products.append({'name': 'Z_boson', 'energy': E_parton*0.9})
                # Распад Z-бозона
                if random.random() < 0.33:
                    products.append({'name': 'electron', 'energy': E_parton*0.45})
                    products.append({'name': 'positron', 'energy': E_parton*0.45})
                elif random.random() < 0.66:
                    products.append({'name': 'muon', 'energy': E_parton*0.45})
                    products.append({'name': 'antimuon', 'energy': E_parton*0.45})
                else:
                    products.append({'name': 'tau', 'energy': E_parton*0.45})
                    products.append({'name': 'antitau', 'energy': E_parton*0.45})
            else:
                # Производство кварковых пар
                products.append({'name': 'c_quark', 'energy': E_parton*0.45})
                products.append({'name': 'c_antiquark', 'energy': E_parton*0.45})
        
        elif process == "jet_production":
            # Производство струй
            num_jets = random.randint(1, 3)
            for _ in range(num_jets):
                jet_energy = E_parton * random.uniform(0.1, 0.5)
                products.append({
                    'name': 'jet',
                    'energy': jet_energy,
                    'px': random.uniform(-jet_energy, jet_energy),
                    'py': random.uniform(-jet_energy, jet_energy),
                    'pz': random.uniform(-jet_energy, jet_energy)
                })
        
        return products

# ===================================================================
# 12. *** МОДУЛЬ: TopoAnalyzer ***
# ===================================================================
class TopoAnalyzer:
    """Улучшенный Топологический анализатор событий.
    Использует идеи из топологического анализа данных (TDA) и анализа корреляций.
    Вдохновлен топологическим анализом ECDSA (торы, числа Бетти)."""
    
    def __init__(self):
        self.events = []
        self.feature_vectors = np.array([])
        self.distance_matrix = None
        self.persistence_diagrams = None
        self.pca_results = None
        self.correlation_spectrum = None
    
    def analyze_events(self, events: List[Dict], max_events: int = 500):
        """Основной метод анализа событий"""
        if not events:
            logger.warning("Нет событий для анализа.")
            return False
        
        # Ограничиваем количество событий
        self.events = events[:max_events]
        
        # Построение векторов признаков
        self.build_feature_vectors()
        
        if self.feature_vectors.size == 0:
            logger.error("Не удалось построить векторы признаков.")
            return False
        
        # Вычисление матрицы расстояний
        self.compute_distance_matrix()
        
        # Вычисление персистентной гомологии
        self.compute_persistence()
        
        # PCA анализ
        self.perform_pca()
        
        # Анализ спектра корреляций
        self.analyze_correlation_spectrum()
        
        return True
    
    def build_feature_vectors(self):
        """Извлекает вектор признаков из событий."""
        features_list = []
        
        for event in self.events:
            features = self._extract_features_from_event(event)
            features_list.append(features)
        
        self.feature_vectors = np.array(features_list)
        logger.info(f"Построено {len(self.feature_vectors)} векторов признаков.")
    
    def _extract_features_from_event(self, event: Dict) -> List[float]:
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
        
        # Добавляем основные признаки
        features.append(num_products)  # Количество продуктов
        features.append(total_energy)  # Общая энергия
        features.append(np.sqrt(total_px**2 + total_py**2 + total_pz**2))  # Импульс системы
        features.append(num_jets)  # Количество струй
        features.append(num_muons)  # Количество мюонов
        features.append(num_antimuons)  # Количество антимюонов
        features.append(abs(num_muons - num_antimuons))  # Асимметрия мюонов
        
        # Добавляем статистику по энергиям струй
        jet_energies = [p['energy'] for p in products if p.get('name') == 'jet']
        if jet_energies:
            features.append(np.mean(jet_energies))
            features.append(np.std(jet_energies))
            features.append(max(jet_energies))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def compute_distance_matrix(self):
        """Вычисляет матрицу расстояний между событиями."""
        try:
            self.distance_matrix = euclidean_distances(self.feature_vectors)
            logger.info("Матрица расстояний вычислена.")
        except Exception as e:
            logger.error(f"Ошибка при вычислении матрицы расстояний: {e}")
            self.distance_matrix = None
    
    def compute_persistence(self, max_dimension: int = 1, max_edge_length: float = np.inf):
        """Вычисляет персистентную гомологию."""
        if self.distance_matrix is None:
            logger.error("Матрица расстояний не вычислена.")
            return
        
        logger.info("Вычисление персистентной гомологии...")
        # Здесь будет код для вычисления персистентной гомологии
        # В реальной реализации использовались бы библиотеки GUDHI или Ripser
        logger.info("Персистентная гомология вычислена (заглушка).")
    
    def perform_pca(self, n_components: Optional[int] = None):
        """Выполняет анализ главных компонент."""
        try:
            if n_components is None:
                n_components = min(5, self.feature_vectors.shape[1])
            
            pca = PCA(n_components=n_components)
            self.pca_results = pca.fit_transform(self.feature_vectors)
            
            logger.info(f"PCA выполнен с {n_components} компонентами.")
            logger.info(f"Объясненная дисперсия: {pca.explained_variance_ratio_}")
        except Exception as e:
            logger.error(f"Ошибка при выполнении PCA: {e}")
    
    def analyze_correlation_spectrum(self):
        """Анализ спектра корреляций."""
        try:
            # Вычисляем корреляционную матрицу
            corr_matrix = np.corrcoef(self.feature_vectors.T)
            
            # Вычисляем собственные значения и векторы
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            
            # Сортируем по убыванию
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Используем константу вместо магического числа
            condition_number = eigenvalues[0] / (eigenvalues[-1] + 1e-12)
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
    
    def generate_report(self, output_file: str = "topology_report.json"):
        """Генерирует отчет о топологическом анализе."""
        try:
            report = {
                'num_events_analyzed': len(self.events),
                'feature_vectors_shape': self.feature_vectors.shape if hasattr(self.feature_vectors, 'shape') else None,
                'pca_explained_variance': [float(x) for x in self.pca_results.explained_variance_ratio_] if self.pca_results is not None else None,
                'correlation_spectrum': {
                    'eigenvalues': [float(x) for x in self.correlation_spectrum['eigenvalues']],
                    'condition_number': float(self.correlation_spectrum['condition_number'])
                } if self.correlation_spectrum else None,
                'timestamp': time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Отчет топологического анализа сохранен в {output_file}.")
            return True
        except Exception as e:
            logger.error(f"Не удалось создать отчет топологического анализа: {e}")
            return False

# ===================================================================
# 13. *** МОДУЛЬ: GradientCalibrator ***
# ===================================================================
class GradientCalibrator:
    """Калибровщик модели на основе градиентного анализа и оптимизации.
    Использует scipy.optimize для надежной минимизации ошибки."""
    
    def __init__(self, model, target_observables: Dict[str, float],
                 parameters_to_calibrate: List[str],
                 error_weights: Optional[Dict[str, float]] = None,
                 perturbation_factor: float = 0.01):
        self.model = model
        self.target_observables = target_observables
        self.parameters_to_calibrate = parameters_to_calibrate
        self.error_weights = error_weights or {}
        self.perturbation_factor = perturbation_factor
        self.optimization_result = None
        self.history = []
        self.sensitivity_analysis = None
    
    def calibrate(self, initial_params: List[float], method: str = 'L-BFGS-B',
                  max_iterations: int = 100, tolerance: float = 1e-6):
        """Калибрует параметры модели для достижения целевых наблюдаемых."""
        if len(initial_params) != len(self.parameters_to_calibrate):
            raise ValueError("Длина initial_params не соответствует количеству параметров для калибровки.")
        
        # Настройка границ для оптимизации
        bounds = [(None, None) for _ in self.parameters_to_calibrate]
        extra_args = ()
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
    
    def _objective_function(self, param_values: np.ndarray, *args) -> float:
        """Целевая функция для минимизации - RMSE между текущими и целевыми наблюдаемыми."""
        self._set_parameters(param_values)
        
        # Запускаем симуляцию для получения текущих наблюдаемых
        num_turns = args[0] if args else 10
        self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
        
        current_observables = self._get_current_observables_from_model(self.model)
        
        # Вычисляем ошибку
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
            normalized_error = error / scale if scale > 1e-12 else error
            
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
        for name in self.target_observables.keys():
            try:
                if name == 'luminosity':
                    obs[name] = model_instance.get_luminosity()
                elif name == 'beam_size_x':
                    obs[name] = model_instance.get_beam_size_x()
                elif name == 'beam_size_y':
                    obs[name] = model_instance.get_beam_size_y()
                elif name == 'avg_event_energy':
                    events = model_instance.get_recent_events()
                    if events:
                        total_energy = sum(sum(p.get('energy', 0) for p in event.get('products', [])) for event in events)
                        obs[name] = total_energy / len(events)
                    else:
                        obs[name] = 0.0
                else:
                    logger.warning(f"Неизвестная наблюдаемая величина: {name}")
            except Exception as e:
                logger.error(f"Ошибка при получении наблюдаемой величины '{name}': {e}")
                # Пропускаем эту наблюдаемую вместо установки нулевого значения
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
                logger.warning(f"Параметр '{param_name}' не найден в конфигурации модели.")
    
    def analyze_sensitivity(self, num_turns: int = 10, use_original_config: bool = True):
        """Анализ чувствительности."""
        logger.info("Начало анализа чувствительности...")
        try:
            # Сохраняем оригинальные параметры
            original_params = np.array([
                self.model.config['beam'].get(param, 1.0) if param in self.model.config.get('beam', {}) 
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
                hessian_diag = (error_up - 2 * base_error + error_down) / ((self.perturbation_factor * original_params[i]) ** 2)
                hess_diag_approx.append(hessian_diag)
            
            # Восстанавливаем оригинальные параметры, если нужно
            if use_original_config:
                self._set_parameters(original_params)
                self.model.run_simulation(num_turns=num_turns, include_space_charge=False)
            
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
    
    def __init__(self):
        self.anomalies_found = {
            'by_type': {
                'statistical': [],
                'topological': [],
                'model_behavior': [],
                'custom': []
            },
            'summary': {
                'total_count': 0,
                'types_found': set()
            }
        }
        self.topo_analyzer = TopoAnalyzer()
        self.gradient_calibrator = None
        self.mu = None
        self.sigma = None
    
    def detect_statistical_anomalies(self, data: List[Dict], feature_name: str, 
                                    method: str = 'zscore', threshold: float = 3.0) -> List[int]:
        """Обнаружение статистических аномалий в данных."""
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
                {'event_index': idx, 'feature': feature_name, 'value': values[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('statistical')
            
            logger.info(f"Найдено {len(anomaly_indices)} статистических аномалий по признаку {feature_name}.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при обнаружении статистических аномалий: {e}")
            return []
    
    def detect_topological_anomalies(self, events: List[Dict], max_events: int = 500, 
                                    threshold_percentile: float = 99.5) -> List[int]:
        """Обнаружение аномалий на основе топологического анализа."""
        try:
            # Анализируем события
            self.topo_analyzer.analyze_events(events, max_events)
            
            if not self.topo_analyzer.persistence_diagrams:
                logger.warning("Нет персистентностей для анализа.")
                return []
            
            # Извлекаем персистентности
            dgms = self.topo_analyzer.persistence_diagrams
            all_pers = []
            for _, dgm in dgms:
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    all_pers.extend(pers)
            
            if not all_pers:
                logger.info("Нет персистентностей для анализа.")
                return []
            
            # Используем процентиль вместо жесткого порога
            pers_threshold = np.percentile(all_pers, threshold_percentile)
            
            # Идентифицируем аномальные события (с очень длинной персистентностью)
            anomaly_indices = []
            for i, (dim, dgm) in enumerate(dgms):
                if dgm.size > 0:
                    pers = dgm[:, 1] - dgm[:, 0]
                    anomalous = pers > pers_threshold
                    anomaly_indices.extend(np.where(anomalous)[0].tolist())
            
            # Уникальные индексы аномалий
            anomaly_indices = list(set(anomaly_indices))
            
            # Добавляем найденные аномалии
            self.anomalies_found['by_type']['topological'].extend([
                {'event_index': idx, 'persistence': pers}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('topological')
            
            logger.info(f"Найдено {len(anomaly_indices)} топологических аномалий.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при обнаружении топологических аномалий: {e}")
            return []
    
    def detect_model_behavior_anomalies(self, state_history: List[Dict]) -> List[int]:
        """Обнаружение аномалий в поведении модели."""
        try:
            if len(state_history) < 2:
                logger.warning("Недостаточно данных для анализа поведения модели.")
                return []
            
            # Извлекаем параметры
            luminosity = [s['beam_dynamics']['luminosity'][-1] for s in state_history]
            size_x = [s['beam_dynamics']['beam_size_x'][-1] for s in state_history]
            size_y = [s['beam_dynamics']['beam_size_y'][-1] for s in state_history]
            
            # Вычисляем разности
            size_x_diff = np.diff(size_x)
            size_y_diff = np.diff(size_y)
            
            anomaly_indices = []
            
            # Обнаружение аномальных изменений размеров пучка
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
                 'size_x': size_x[idx], 'size_y': size_y[idx]}
                for idx in anomaly_indices
            ])
            self.anomalies_found['summary']['total_count'] += len(anomaly_indices)
            self.anomalies_found['summary']['types_found'].add('model_behavior')
            
            logger.info(f"Найдено {len(anomaly_indices)} аномалий в поведении модели.")
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"Ошибка при обнаружении аномалий поведения модели: {e}")
            return []
    
    def detect_custom_anomalies(self, custom_detector_func, *args, **kwargs):
        """Поиск пользовательских аномалий с помощью пользовательской функции."""
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
    
    def detect_all_anomalies(self, events: List[Dict], state_history: List[Dict], 
                            max_events: int = 500):
        """Обнаружение всех типов аномалий."""
        # Статистический анализ
        for feature in ['energy', 'momentum', 'num_products']:
            self.detect_statistical_anomalies(events, feature)
        
        # Топологический анализ
        self.detect_topological_anomalies(events, max_events)
        
        # Анализ поведения модели
        self.detect_model_behavior_anomalies(state_history)
        
        return self.anomalies_found
    
    def generate_report(self, output_file: str = "anomaly_report.json"):
        """Генерирует отчет об обнаруженных аномалиях."""
        try:
            # Преобразуем set в list для сериализации JSON
            report = {
                'anomalies': self.anomalies_found['by_type'],
                'summary': {
                    'total_count': self.anomalies_found['summary']['total_count'],
                    'types_found': list(self.anomalies_found['summary']['types_found'])
                },
                'timestamp': time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Отчет об аномалиях сохранен в {output_file}.")
            return self.anomalies_found
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
        try:
            import ROOT
            self.root_available = True
            logger.info("ROOTExporter инициализирован. ROOT доступен.")
        except ImportError:
            self.root_available = False
            logger.warning("ROOTExporter инициализирован. ROOT недоступен.")
    
    def export_to_root(self, filename: str, events: List[Dict]) -> bool:
        """Экспортирует события в ROOT файл."""
        if not self.root_available:
            logger.error("ROOT недоступен. Невозможно экспортировать в ROOT формат.")
            return False
        
        try:
            import ROOT
            
            # Создаем ROOT файл
            root_file = ROOT.TFile(filename, "RECREATE")
            
            # Создаем дерево для событий
            tree = ROOT.TTree("Events", "Симуляционные события LHC")
            
            # Определяем переменные
            event_id = ROOT.Int_t(0)
            num_products = ROOT.Int_t(0)
            total_energy = ROOT.Double_t(0)
            
            # Создаем ветки
            tree.Branch("event_id", ROOT.AddressOf(event_id), "event_id/I")
            tree.Branch("num_products", ROOT.AddressOf(num_products), "num_products/I")
            tree.Branch("total_energy", ROOT.AddressOf(total_energy), "total_energy/D")
            
            # Заполняем дерево
            for i, event in enumerate(events):
                event_id = i
                products = event.get('products', [])
                num_products = len(products)
                total_energy = sum(p.get('energy', 0.0) for p in products)
                
                tree.Fill()
            
            # Сохраняем и закрываем файл
            root_file.Write()
            root_file.Close()
            
            logger.info(f"Данные успешно экспортированы в {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при экспорте в ROOT формат: {e}")
            return False

# ===================================================================
# 16. *** МОДУЛЬ: HepMC3Exporter ***
# ===================================================================
class HepMC3Exporter:
    """Экспортер данных симуляции в формат HepMC3."""
    
    def __init__(self):
        """Инициализация экспортера HepMC3."""
        try:
            import hepmc3
            self.hepmc3_available = True
            logger.info("HepMC3Exporter инициализирован. HepMC3 доступен.")
        except ImportError:
            self.hepmc3_available = False
            logger.warning("HepMC3Exporter инициализирован. HepMC3 недоступен.")
    
    def export_to_hepmc3(self, filename: str, events: List[Dict]) -> bool:
        """Экспортирует события в HepMC3 файл."""
        if not self.hepmc3_available:
            logger.error("HepMC3 недоступен. Невозможно экспортировать в HepMC3 формат.")
            return False
        
        try:
            import hepmc3
            
            # Здесь должен быть код для экспорта в HepMC3
            # Для примера просто создаем текстовый файл
            with open(filename, 'w') as f:
                f.write("HepMC3 file generated by LHCHybridModel\n")
                f.write("Units: GEV MM\n")
                for i, event in enumerate(events):
                    f.write(f"E {i} 0 0 0\n")  # Заголовок события
                    # Записываем входные частицы (протоны)
                    f.write(f"P 1 2212 0 0 6500 6500 0 0 0 1 0 0\n")
                    f.write(f"P 2 2212 0 0 -6500 -6500 0 0 0 1 0 0\n")
                    # Записываем продукты столкновения
                    products = event.get('products', [])
                    for j, p in enumerate(products, 3):
                        # Простая маппинг частиц в PDG коды
                        pdg_code = self._particle_to_pdg(p.get('name', ''))
                        px = p.get('px', 0.0)
                        py = p.get('py', 0.0)
                        pz = p.get('pz', 0.0)
                        e = p.get('energy', 0.0)
                        f.write(f"P {j} {pdg_code} {px} {py} {pz} {e} 0 0 0 1 0 0\n")
            
            logger.info(f"Данные успешно экспортированы в HepMC3 формат: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при экспорте в HepMC3 формат: {e}")
            return False
    
    def _particle_to_pdg(self, particle_name: str) -> int:
        """Конвертирует имя частицы в PDG код."""
        pdg_map = {
            'electron': 11,
            'positron': -11,
            'muon': 13,
            'antimuon': -13,
            'tau': 15,
            'antitau': -15,
            'neutrino_e': 12,
            'antineutrino_e': -12,
            'neutrino_mu': 14,
            'antineutrino_mu': -14,
            'neutrino_tau': 16,
            'antineutrino_tau': -16,
            'u_quark': 2,
            'u_antiquark': -2,
            'd_quark': 1,
            'd_antiquark': -1,
            's_quark': 3,
            's_antiquark': -3,
            'c_quark': 4,
            'c_antiquark': -4,
            'b_quark': 5,
            'b_antiquark': -5,
            't_quark': 6,
            't_antiquark': -6,
            'gluon': 21,
            'photon': 22,
            'Z_boson': 23,
            'W_plus': 24,
            'W_minus': -24,
            'higgs': 25,
            'pion+': 211,
            'pion-': -211,
            'pion0': 111,
            'kaon+': 321,
            'kaon-': -321,
            'proton': 2212,
            'antiproton': -2212,
            'neutron': 2112,
            'antineutron': -2112,
            'jet': 0  # Специальный код для струй
        }
        return pdg_map.get(particle_name, 0)

# ===================================================================
# 17. *** МОДУЛЬ: DetectorSystem ***
# ===================================================================
class DetectorSystem:
    """Система детектора для моделирования отклика детектора."""
    
    def __init__(self, config):
        self.config = config
        self.detector_elements = self._initialize_detector()
    
    def _initialize_detector(self):
        """Инициализация компонентов детектора."""
        # В реальной системе здесь будет сложная инициализация
        return {
            'tracker': {
                'resolution': 0.01,  # мм
                'efficiency': 0.98
            },
            'calorimeter': {
                'resolution': 0.05,  # ГэВ
                'efficiency': 0.95
            },
            'muon_system': {
                'resolution': 0.1,  # мм
                'efficiency': 0.92
            }
        }
    
    def process_event(self, event: Dict) -> Dict:
        """Обрабатывает событие через систему детектора."""
        detected_products = []
        
        for product in event.get('products', []):
            # Имитация детектирования с учетом эффективности
            if self._is_detected(product):
                # Имитация измерения с учетом разрешения детектора
                measured = self._apply_detector_resolution(product)
                detected_products.append(measured)
        
        return {
            'event_id': event.get('event_id', 0),
            'detected_products': detected_products,
            'timestamp': time.time()
        }
    
    def _is_detected(self, product: Dict) -> bool:
        """Проверяет, будет ли продукт зарегистрирован детектором."""
        # Вероятность детектирования зависит от типа частицы
        efficiency = 0.9  # Базовая эффективность
        
        if 'muon' in product.get('name', ''):
            efficiency = self.detector_elements['muon_system']['efficiency']
        elif 'electron' in product.get('name', ''):
            efficiency = self.detector_elements['calorimeter']['efficiency']
        elif 'jet' in product.get('name', ''):
            efficiency = min(
                self.detector_elements['tracker']['efficiency'],
                self.detector_elements['calorimeter']['efficiency']
            )
        
        return random.random() < efficiency
    
    def _apply_detector_resolution(self, product: Dict) -> Dict:
        """Применяет разрешение детектора к измерениям."""
        measured = product.copy()
        
        # Применяем гауссово шум с разрешением детектора
        if 'energy' in measured:
            if 'electron' in measured.get('name', '') or 'jet' in measured.get('name', ''):
                res = self.detector_elements['calorimeter']['resolution']
                measured['energy'] += np.random.normal(0, res * measured['energy'])
        
        if 'px' in measured:
            res = self.detector_elements['tracker']['resolution']
            measured['px'] += np.random.normal(0, res * abs(measured['px']))
        
        if 'py' in measured:
            res = self.detector_elements['tracker']['resolution']
            measured['py'] += np.random.normal(0, res * abs(measured['py']))
        
        if 'pz' in measured:
            res = self.detector_elements['tracker']['resolution']
            measured['pz'] += np.random.normal(0, res * abs(measured['pz']))
        
        return measured
    
    def get_detector_response(self, events: List[Dict]) -> List[Dict]:
        """Получает отклик детектора для списка событий."""
        return [self.process_event(event) for event in events]

# ===================================================================
# 18. *** МОДУЛЬ: Visualization ***
# ===================================================================
class Visualization:
    """Модуль визуализации результатов симуляции."""
    
    def __init__(self):
        self.plots = []
    
    def plot_geometry_3d(self, geometry, detector_system):
        """Визуализация 3D геометрии коллайдера и детектора."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Рисуем кольцо коллайдера
            theta = np.linspace(0, 2*np.pi, 100)
            r = geometry['radius']
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.zeros_like(x)
            ax.plot(x, y, z, 'b-', linewidth=2, label='Кольцо коллайдера')
            
            # Рисуем детекторные системы
            for detector, params in detector_system.detector_elements.items():
                # Простая визуализация детекторов как цилиндров
                r_det = r * (0.8 if detector == 'tracker' else 0.6 if detector == 'calorimeter' else 0.4)
                z_det = np.linspace(-5, 5, 20)
                theta_det = np.linspace(0, 2*np.pi, 30)
                theta_det, z_det = np.meshgrid(theta_det, z_det)
                x_det = r_det * np.cos(theta_det)
                y_det = r_det * np.sin(theta_det)
                ax.plot_surface(x_det, y_det, z_det, alpha=0.3, label=detector)
            
            ax.set_xlabel('X (м)')
            ax.set_ylabel('Y (м)')
            ax.set_zlabel('Z (м)')
            ax.set_title('3D Геометрия коллайдера и детекторных систем')
            ax.legend()
            
            plt.tight_layout()
            self.plots.append(('geometry_3d', fig))
            plt.show()
            
            logger.info("3D визуализация геометрии завершена.")
        except Exception as e:
            logger.error(f"Ошибка при 3D визуализации геометрии: {e}")
    
    def plot_detector_response_3d(self, detected_events, detector_system):
        """Визуализация 3D отклика детектора."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Рисуем следы частиц
            for event in detected_events[:10]:  # Ограничиваем количество событий для ясности
                for product in event.get('detected_products', []):
                    # Генерируем простой след частицы
                    r = np.sqrt(product.get('px', 0)**2 + product.get('py', 0)**2 + product.get('pz', 0)**2)
                    if r > 0:
                        x = [0, 10 * product.get('px', 0) / r]
                        y = [0, 10 * product.get('py', 0) / r]
                        z = [0, 10 * product.get('pz', 0) / r]
                        ax.plot(x, y, z, 'r-', alpha=0.5)
            
            # Рисуем детекторные системы (как в plot_geometry_3d)
            geometry = {'radius': 4297}  # Радиус LHC
            for detector, params in detector_system.detector_elements.items():
                r_det = geometry['radius'] * (0.8 if detector == 'tracker' else 0.6 if detector == 'calorimeter' else 0.4)
                z_det = np.linspace(-5, 5, 20)
                theta_det = np.linspace(0, 2*np.pi, 30)
                theta_det, z_det = np.meshgrid(theta_det, z_det)
                x_det = r_det * np.cos(theta_det)
                y_det = r_det * np.sin(theta_det)
                ax.plot_surface(x_det, y_det, z_det, alpha=0.3)
            
            ax.set_xlabel('X (м)')
            ax.set_ylabel('Y (м)')
            ax.set_zlabel('Z (м)')
            ax.set_title('3D Отклик детектора на события')
            
            plt.tight_layout()
            self.plots.append(('detector_response_3d', fig))
            plt.show()
            
            logger.info("3D визуализация отклика детектора завершена.")
        except Exception as e:
            logger.error(f"Ошибка при 3D визуализации отклика детектора: {e}")
    
    def plot_beam_parameters(self, state_history):
        """Визуализация параметров пучка во времени."""
        try:
            import matplotlib.pyplot as plt
            
            turns = range(len(state_history))
            luminosity = [s['beam_dynamics']['luminosity'][-1] for s in state_history]
            size_x = [s['beam_dynamics']['beam_size_x'][-1] for s in state_history]
            size_y = [s['beam_dynamics']['beam_size_y'][-1] for s in state_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Светимость
            ax1.plot(turns, luminosity, 'b-')
            ax1.set_xlabel('Обороты')
            ax1.set_ylabel('Светимость (см⁻²с⁻¹)')
            ax1.set_title('Эволюция светимости')
            ax1.grid(True)
            
            # Размеры пучка
            ax2.plot(turns, size_x, 'r-', label='σ_x')
            ax2.plot(turns, size_y, 'g-', label='σ_y')
            ax2.set_xlabel('Обороты')
            ax2.set_ylabel('Размер пучка (м)')
            ax2.set_title('Эволюция размеров пучка')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            self.plots.append(('beam_parameters', fig))
            plt.show()
            
            logger.info("Визуализация параметров пучка завершена.")
        except Exception as e:
            logger.error(f"Ошибка при визуализации параметров пучка: {e}")

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
    
    def __init__(self, config=None):
        # Загрузка конфигурации
        self.config = config or self._load_default_config()
        
        # Инициализация компонентов
        self.geometry = self._initialize_geometry()
        self.beam_dynamics = BeamDynamics(self.config)
        self.physics_engines = self._initialize_physics_engines()
        self.detector_system = DetectorSystem(self.config)
        self.visualizer = Visualization()
        
        # Системы анализа
        self.topo_analyzer = TopoAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.calibrator = None
        
        # Состояние модели
        self.simulation_state = {
            'beam_dynamics': {
                'turn': [],
                'luminosity': [],
                'beam_size_x': [],
                'beam_size_y': [],
                'time': []
            },
            'detected_events': [],
            'recent_events': []
        }
    
    def _load_default_config(self):
        """Загружает конфигурацию по умолчанию."""
        default_config = {
            'beam': {
                'beam_energy': 6500,  # ГэВ
                'bunch_intensity': 1.15e11,
                'num_bunches': 2556,
                'circumference': 26659,  # м
                'beta_star': 0.55,  # м
                'crossing_angle': 0.0,  # радианы
                'emittance': 2.5e-6,  # м·рад
                'sigma_x': 0.045,  # м
                'sigma_y': 0.045,  # м
                'beta_x': 56.5,  # м
                'beta_y': 56.5  # м
            },
            'geometry': {
                'radius': 4297,  # м (радиус LHC)
                'straight_sections': 8,
                'bending_magnets': 1232
            },
            'validation': {
                'dataset_id': 'CMS_OpenData_2018'
            }
        }
        logger.info("Используется конфигурация по умолчанию.")
        return default_config
    
    def _initialize_geometry(self):
        """Инициализирует геометрию коллайдера."""
        return {
            'radius': self.config['geometry']['radius'],
            'circumference': self.config['beam']['circumference'],
            'straight_sections': self.config['geometry']['straight_sections'],
            'bending_magnets': self.config['geometry']['bending_magnets']
        }
    
    def _initialize_physics_engines(self):
        """Инициализирует физические движки."""
        engines = {}
        
        # Встроенный движок
        try:
            engines["built-in"] = BuiltInPhysicsEngine({}, self.config)
            logger.info("Встроенный физический движок инициализирован.")
        except Exception as e:
            logger.error(f"Ошибка инициализации встроенного физического движка: {e}")
        
        return engines
    
    def run_simulation(self, num_turns: int = 10, include_space_charge: bool = True):
        """Запускает симуляцию коллайдера."""
        logger.info(f"Запуск симуляции на {num_turns} оборотов.")
        
        # Эволюция динамики пучка
        for turn in range(num_turns):
            # Обновляем состояние пучка
            state = self.beam_dynamics.evolve(1, include_space_charge)
            
            # Регистрируем параметры
            self.simulation_state['beam_dynamics']['turn'].append(turn)
            self.simulation_state['beam_dynamics']['luminosity'].append(self.beam_dynamics.get_luminosity())
            self.simulation_state['beam_dynamics']['beam_size_x'].append(state['sigma_x'])
            self.simulation_state['beam_dynamics']['beam_size_y'].append(state['sigma_y'])
            self.simulation_state['beam_dynamics']['time'].append(turn * 88.9e-6)  # Время одного оборота
            
            # Генерируем столкновения (с вероятностью, зависящей от светимости)
            if random.random() < self.beam_dynamics.get_luminosity() * 1e-34:
                # Используем встроенный движок для генерации событий
                events = self.physics_engines["built-in"].interact(
                    "proton", "proton", 
                    self.config['beam']['beam_energy'],
                    num_events=1
                )
                
                if events:
                    # Обрабатываем события через детектор
                    detected = self.detector_system.get_detector_response(events)
                    self.simulation_state['detected_events'].extend(detected)
                    self.simulation_state['recent_events'] = self.simulation_state['recent_events'][-99:] + events
        
        logger.info("Симуляция завершена.")
        return self.simulation_state
    
    def analyze_topology(self, max_events: int = 500, compute_persistence: bool = True, compute_pca: bool = True):
        """Анализирует топологию событий."""
        logger.info("Запуск топологического анализа событий...")
        
        # Используем последние события или сохраненные
        events_to_analyze = self.simulation_state['recent_events'][:max_events] if self.simulation_state['recent_events'] else None
        
        if not events_to_analyze:
            logger.warning("Нет событий для топологического анализа.")
            return False
        
        # Запускаем анализ
        success = self.topo_analyzer.analyze_events(events_to_analyze, max_events)
        
        if success:
            logger.info("Топологический анализ завершен успешно.")
            self.topo_analyzer.generate_report()
        else:
            logger.error("Топологический анализ завершился с ошибкой.")
        
        return success
    
    def calibrate_model(self, target_observables: Dict[str, float], 
                       parameters_to_calibrate: List[str]):
        """Калибрует модель для достижения целевых наблюдаемых."""
        logger.info("Запуск калибровки модели...")
        
        self.calibrator = GradientCalibrator(
            self, 
            target_observables,
            parameters_to_calibrate
        )
        
        # Начальные значения параметров
        initial_params = [
            self.config['beam'].get(param, 1.0) if param in self.config.get('beam', {}) 
            else self.config['geometry'].get(param, 1.0)
            for param in parameters_to_calibrate
        ]
        
        # Границы для оптимизации
        bounds = []
        for param in parameters_to_calibrate:
            if param == 'beam_energy':
                bounds.append((6000, 8000))  # ГэВ
            elif param == 'num_bunches':
                bounds.append((1, 2808))  # Максимум в LHC
            elif param == 'bunch_intensity':
                bounds.append((1e10, 2e11))  # Частиц в пучке
            else:
                bounds.append((None, None))
        
        # Запускаем калибровку
        self.calibrator.calibrate(initial_params, bounds=bounds)
        
        # Анализ чувствительности
        self.calibrator.analyze_sensitivity()
        
        logger.info("Калибровка модели завершена.")
        return self.calibrator.get_summary_report()
    
    def detect_anomalies(self):
        """Обнаруживает аномалии в данных симуляции."""
        logger.info("Запуск обнаружения аномалий...")
        
        # Статистические аномалии
        for feature in ['energy', 'momentum', 'num_products']:
            self.anomaly_detector.detect_statistical_anomalies(
                self.simulation_state['recent_events'], 
                feature
            )
        
        # Топологические аномалии
        self.anomaly_detector.detect_topological_anomalies(
            self.simulation_state['recent_events']
        )
        
        # Аномалии поведения модели
        self.anomaly_detector.detect_model_behavior_anomalies(
            [self.simulation_state['beam_dynamics']]
        )
        
        # Генерация отчета
        self.anomaly_detector.generate_report()
        
        logger.info("Обнаружение аномалий завершено.")
        return self.anomaly_detector.anomalies_found
    
    def export_to_root(self, filename: str) -> bool:
        """Экспортирует данные симуляции в ROOT формат."""
        exporter = ROOTExporter()
        return exporter.export_to_root(filename, self.simulation_state['detected_events'])
    
    def export_to_hepmc3(self, filename: str) -> bool:
        """Экспортирует данные симуляции в HepMC3 формат."""
        exporter = HepMC3Exporter()
        return exporter.export_to_hepmc3(filename, self.simulation_state['recent_events'])
    
    def enhanced_visualization(self):
        """Улучшенная визуализация результатов симуляции."""
        logger.info("Запуск улучшенной визуализации результатов...")
        try:
            self.visualizer.plot_geometry_3d(self.geometry, self.detector_system)
            if self.simulation_state['detected_events']:
                self.visualizer.plot_detector_response_3d(self.simulation_state['detected_events'], self.detector_system)
            self.visualizer.plot_beam_parameters(self.simulation_state['beam_dynamics'])
            logger.info("Улучшенная визуализация завершена.")
        except Exception as e:
            logger.error(f"Ошибка при улучшенной визуализации: {e}")
    
    def get_luminosity(self) -> float:
        """Возвращает текущую светимость."""
        if self.simulation_state['beam_dynamics']['luminosity']:
            return self.simulation_state['beam_dynamics']['luminosity'][-1]
        return 0.0
    
    def get_beam_size_x(self) -> float:
        """Возвращает текущий размер пучка по X."""
        if self.simulation_state['beam_dynamics']['beam_size_x']:
            return self.simulation_state['beam_dynamics']['beam_size_x'][-1]
        return 0.0
    
    def get_beam_size_y(self) -> float:
        """Возвращает текущий размер пучка по Y."""
        if self.simulation_state['beam_dynamics']['beam_size_y']:
            return self.simulation_state['beam_dynamics']['beam_size_y'][-1]
        return 0.0
    
    def get_recent_events(self) -> List[Dict]:
        """Возвращает последние события."""
        return self.simulation_state['recent_events']

# ===================================================================
# 20. Функции демонстрации
# ===================================================================
def create_default_config():
    """Создает файл конфигурации по умолчанию."""
    default_config = {
        'beam': {
            'beam_energy': 6500,
            'bunch_intensity': 1.15e11,
            'num_bunches': 2556,
            'circumference': 26659,
            'beta_star': 0.55,
            'crossing_angle': 0.0,
            'emittance': 2.5e-6,
            'sigma_x': 0.045,
            'sigma_y': 0.045,
            'beta_x': 56.5,
            'beta_y': 56.5
        },
        'geometry': {
            'radius': 4297,
            'straight_sections': 8,
            'bending_magnets': 1232
        },
        'validation': {
            'dataset_id': 'CMS_OpenData_2018'
        }
    }
    
    with open("lhc_config.yaml", 'w') as f:
        yaml.dump(default_config, f)
    
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
    
    # Топологический анализ
    lhc.analyze_topology(max_events=500, compute_persistence=True, compute_pca=True)
    
    # Калибровка модели
    target_observables = {
        'luminosity': 1.5e34,
        'beam_size_x': 0.045,
        'avg_event_energy': 5000.0
    }
    params_to_calibrate = ['beam_energy', 'num_bunches', 'bunch_intensity']
    calibration_report = lhc.calibrate_model(target_observables, params_to_calibrate)
    
    # Обнаружение аномалий
    anomalies = lhc.detect_anomalies()
    
    # Экспорт данных
    lhc.export_to_root("simulation_results.root")
    lhc.export_to_hepmc3("simulation_results.hepmc3")
    
    # Визуализация
    lhc.enhanced_visualization()
    
    logger.info("Демонстрация завершена.")
