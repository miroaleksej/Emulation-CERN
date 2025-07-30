# OpenParticleLab Framework

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/169f4c43-2468-4076-9b0d-949999452d0f" />

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=yourusername.qpp" alt="visitors">

 <p align="center">
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://python.org)
[![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%2C%20Matplotlib-orange)](requirements.txt)

**OpenParticleLab** - это модульный фреймворк для моделирования физических процессов в ускорителях частиц и детекторах. Фреймворк объединяет функциональность профессиональных инструментов (Geant4, ROOT и MAD-X) в единой, доступной и научно достоверной реализации, специально разработанной для исследований и обучения.

![LHC Simulation](https://github.com/yourusername/OpenParticleLab/blob/main/examples/lhc_ring.png)

## 🌟 Основные возможности

- **Реалистичная модель Большого адронного коллайдера** с использованием данных CERN
- **Динамика пучка частиц** с симуляцией эффектов пространственного заряда
- **Интегрированная база данных частиц** с актуальными параметрами из Particle Data Group
- **Моделирование столкновений** с поддержкой Стандартной модели, QCD и гипотетических процессов
- **3D-визуализация** геометрии экспериментальных установок и траекторий частиц
- **Анализ данных** с ROOT-подобным функционалом (гистограммы, фитирование, ROI)
- **Симуляция отклика детектора** с учетом реальных характеристик
- **Поддержка YAML-конфигурации** для быстрого создания экспериментов

## 📦 Требования

- Python 3.7+
- Основные зависимости:
  ```bash
  pip install numpy matplotlib scipy plotly pyyaml
  ```
- Для ROOT-экспорта (опционально):
  ```bash
  pip install uproot root-numpy
  ```

## 🚀 Быстрый старт

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/OpenParticleLab.git
   cd OpenParticleLab
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустите пример симуляции:
   ```python
   from open_particle_lab import LHC_Model
   
   # Создание модели БАК
   lhc = LHC_Model()
   
   # Запуск симуляции 1000 оборотов пучка
   lhc.run_simulation(num_turns=1000)
   
   # Визуализация динамики пучка
   lhc.visualize_beam_dynamics()
   ```

## 🛠 Основные компоненты

### 1. ParticleDatabase
Расширенная база данных элементарных частиц с параметрами из Particle Data Group (2023):
```python
from open_particle_lab import ParticleDatabase

db = ParticleDatabase()
proton = db.get_particle('proton')
print(f"Масса протона: {proton['mass']} МэВ/c²")
print(f"Заряд протона: {proton['charge']}e")

# Визуализация свойств частицы
db.plot_particle_properties('higgs_boson')
```

### 2. LHC_Model
Специализированная модель Большого адронного коллайдера:
```python
from open_particle_lab import LHC_Model

lhc = LHC_Model()

# Симуляция одного оборота пучка
lhc.step_simulation()

# Симуляция 500 столкновений
events = lhc.simulate_collision(num_events=500)

# Визуализация столкновения
lhc.visualize_collision()
```

### 3. OpenParticleLab Framework
Интегрированный фреймворк для создания собственных экспериментов:
```python
from open_particle_lab import OpenParticleLab

# Создание фреймворка
lab = OpenParticleLab()

# Загрузка конфигурации
lab.load_config('experiment.yml')

# Запуск симуляции
lab.run_simulation(num_events=1000)

# Анализ и визуализация
lab.analyze_results()
lab.visualize_experiment()
```

## 📐 Пример конфигурации эксперимента (experiment.yml)

```yaml
beam:
  particle: "proton"
  energy: 6.5e12  # 6.5 ТэВ
target:
  particle: "proton"
geometry:
  - type: "detector"
    params:
      name: "ATLAS"
      position: [0, 0, 0]
      size: [46, 25, 25]
      efficiency: 0.99
      resolution: 0.01
      subdetectors:
        - name: "Inner Detector"
          radius: [0.05, 1.1]
          resolution: 0.01
          efficiency: 0.99
        - name: "Calorimeter"
          radius: [1.1, 4.3]
          resolution: 0.05
          efficiency: 0.95
        - name: "Muon Spectrometer"
          radius: [5.6, 11.0]
          resolution: 0.02
          efficiency: 0.90
  - type: "magnet"
    params:
      position: [0, 0, 0]
      orientation: [0, 0, 1]
      strength: 2.0
      type: "dipole"
      length: 5.0
      radius: 2.0
```

## 📚 Примеры использования

### 1. Исследование динамики пучка
```python
lhc = LHC_Model()
lhc.run_simulation(num_turns=500)

# Анализ роста эмиттанса
print(f"Рост эмиттанса: {lhc.simulation_state['beam_dynamics']['emittance'][-1]/lhc.simulation_state['beam_dynamics']['emittance'][0]:.2f}x")
print(f"Потери светимости: {100 * (1 - lhc.simulation_state['beam_dynamics']['luminosity'][-1]/lhc.peak_luminosity):.2f}%")
```

### 2. Поиск бозона Хиггса
```python
lhc = LHC_Model()
events = lhc.simulate_collision(num_events=10000)

# Фильтрация событий с производством бозона Хиггса
higgs_events = [e for e in events if e['event_type'] == 'higgs_boson_production']

# Анализ распадов в b-кварки
b_quark_energies = [p['energy']/1e9 for e in higgs_events 
                   for p in e['products'] 
                   if p['name'] in ['bottom_quark', 'antibottom_quark']]
```

### 3. Создание анимации движения частиц
```python
lhc = LHC_Model()
lhc.create_animation(filename="lhc_motion.mp4", num_frames=200)
```

## 📁 Структура проекта

```
OpenParticleLab/
├── open_particle_lab.py    # Основной модуль фреймворка
├── examples/
│   ├── lhc_simulation.ipynb # Пример Jupyter Notebook
│   ├── experiment.yml       # Пример конфигурации
│   └── results/             # Примеры результатов
├── docs/
│   ├── api_reference.md     # Документация API
│   └── tutorials/           # Учебные материалы
├── requirements.txt         # Зависимости
├── LICENSE                  # Лицензия MIT
└── README.md                # Этот файл
```

## 🌐 Возможные пути развития

- Интеграция с реальными данными из CERN Open Data
- Поддержка GPU-ускорения для сложных симуляций
- Веб-интерфейс на основе Streamlit или Dash
- Расширение базы данных частиц
- Добавление поддержки других коллайдеров (FCC, ILC)
- Интеграция с Geant4 через pyg4ometry

## 📝 Лицензия

Этот проект распространяется под лицензией MIT. Подробнее см. в файле [LICENSE](LICENSE).

## 🤝 Вклад в проект

Приветствуются pull request'ы! Пожалуйста, следуйте этим шагам:
1. Форкните репозиторий
2. Создайте новую ветку (`git checkout -b feature/AmazingFeature`)
3. Сделайте коммит изменений (`git commit -m 'Add some AmazingFeature'`)
4. Запушьте ветку (`git push origin feature/AmazingFeature`)
5. Создайте новый Pull Request

## 📬 Контакты

Для вопросов и предложений:
- Email: miro-aleksej@yandex.ru
- Issue tracker: [GitHub Issues](https://github.com/miroaleksej/OpenParticleLab/issues)

---

*OpenParticleLab не является официальным продуктом CERN. Все данные и модели основаны на открытых источниках и учебных материалах.*
---

# ⚠️ Предупреждение: Ограничения использования системы OpenParticleLab

## ВАЖНОЕ ЮРИДИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ

![Warning Icon](https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Warning.svg/1200px-Warning.svg.png)

**ДАННАЯ СИСТЕМА ПРЕДНАЗНАЧЕНА ИСКЛЮЧИТЕЛЬНО ДЛЯ НАУЧНЫХ И ОБРАЗОВАТЕЛЬНЫХ ЦЕЛЕЙ.**

**КОММЕРЧЕСКОЕ ИСПОЛЬЗОВАНИЕ ВСЕГО КОДА ИЛИ ЕГО ЧАСТИ - СТРОГО ЗАПРЕЩЕНО!**

---

## Подробное объяснение ограничений

**НАУЧНОЕ ИСПОЛЬЗОВАНИЕ РАЗРЕШЕНО**

- Исследовательские работы в области физики
- Публикации в научных журналах (с обязательной ссылкой на источник)
- Академические исследования и анализ
- Обучение студентов 

**ОБРАЗОВАТЕЛЬНОЕ ИСПОЛЬЗОВАНИЕ РАЗРЕШЕНО**

- Использование в учебных заведениях для демонстрации принципов физики высоких энергий
- Практические занятия 
- Научные семинары и конференции

**КОММЕРЧЕСКОЕ ИСПОЛЬЗОВАНИЕ СТРОГО ЗАПРЕЩЕНО**

- Внедрение в коммерческие продукты или сервисы
- Использование в коммерческих системах безопасности
- Интеграция в коммерческие библиотеки
- Любое использование с целью получения финансовой выгоды

---

## Последствия нарушения

Нарушение данных условий использования может привести к:

- Юридическим последствиям в соответствии с международным законодательством об авторских правах
- Уголовной ответственности за неправомерное использование исследовательских материалов
- Отзыву прав на использование материалов без предварительного уведомления

---

> "Если бы природа хотела, чтобы мы изучали Вселенную на больших энергиях, она бы сделала её больше." - Дэвид Гросс (Нобелевская премия по физике, 2004)
