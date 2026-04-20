# OptiCat

OptiCat - интерактивный симулятор оптоэлектронных схем.

Проект поддерживает два режима:
- `SuperMan` - интерактивный редактор графа в Jupyter (ipywidgets + matplotlib).
- `SuperCat` - desktop GUI на `tkinter` с историей графиков, логами и экспортом схемы в `.py`.

## Что моделируется

- Электрические сигналы: `ElectricalSignal`
- Оптические сигналы: `OpticalSignal`
- Сигналы детектора: `DetectorSignal`
- Компоненты тракта: AWG, лазер, MZM (интенсивностный модулятор), волокно, сплиттер, фотодетектор, когерентный/некогерентный детектор, RC/полосовой фильтр, генератор шума, осциллограф

## Структура проекта

```text
OptiCat/
  main.ipynb              # исходный ноутбук (исторически основной файл)
  requirements.txt
  pyproject.toml
  run_supercat.py
  run_notebook_mode.py
  src/
    opticat/
      __init__.py
      __main__.py         # python -m opticat
      signals.py          # dataclass-модели сигналов
      components.py       # физические/логические блоки
      core.py             # SuperMan (граф, выполнение, экспорт)
      gui.py              # SuperCat (tkinter GUI)
  Tmp/                    # временные/экспортные артефакты
```

## Установка

Требуется Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Запуск

### 1) Desktop GUI (SuperCat)

```bash
python3 -m opticat
```

или

```bash
python3 run_supercat.py
```

### 2) В ноутбуке (SuperMan)

```python
from opticat import SuperMan

superman = SuperMan()
superman.ui()
```

### 3) Ноутбук-совместимость

`main.ipynb` можно продолжать использовать как раньше. Теперь код также доступен в модульной структуре `src/opticat`.

### 4) Safe notebook-mode launcher (без tkinter)

Если desktop GUI (`tkinter`) недоступен или падает в текущем окружении:

```bash
python3 run_notebook_mode.py
```

## Экспорт схемы

В GUI доступна кнопка `Export.py`.

Она генерирует `scheme_export.py`, который:
- восстанавливает узлы и рёбра,
- выполняет схему в топологическом порядке,
- строит соответствующие графики на выходах.

## Полезные замечания

- Для desktop GUI нужен `tkinter` (обычно встроен в Python на macOS/Windows).
- Для интерактивного drag-and-drop в Jupyter нужен backend `ipympl`.
- Для превью Plotly в PNG используется `kaleido`.

## Типичный workflow

1. Собрать схему из блоков.
2. Настроить параметры узлов.
3. Нажать `Run` и проверить графики/логи.
4. При необходимости экспортировать схему в `.py`.

## Лицензия

MIT, см. файл `LICENSE`.
