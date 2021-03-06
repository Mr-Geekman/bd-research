# Bachelor's Degree Research

## Описание

Это репозиторий с проектом по дипломной работе "Исправление опечаток и грамматических ошибок в русскоязычных текстах".

Подробности можно найти в директории reports/.

## Текущие результаты

| Spell Checker        | Precision | Recall    | F-measure |
| -------------------- | --------- | --------- | --------- |
| Yandex.Speller       | 83.09     | 59.86     | 69.59     |
| SpellRuEval Baseline | 55.91     | 46.41     | 50.72     |
| SpellRuEval Winner   | 81.98     | 69.25     | 75.07     |
| Our solution (best)  | **87.70** | **73.23** | **79.81** |
| Our solution (fast)  | 86.04     | 70.80     | 77.68     |


## Разработка

Для установки зависимостей воспользуейтесь командой:
```bash
conda create --name <env> --file requirements.txt
```

Для создания директорий с данными и их частичной загрузки:
```bash
snakemake data --cores 1
```

Все данные можно найти на [гугл-диске](https://drive.google.com/drive/folders/1iUR-SiMzbZkuHAAzJZLfiO9q7rIze2GI?usp=sharing).

## Демонстрация

Демонстрация работы модели есть в [ноутбуке](https://colab.research.google.com/drive/12DL7M491FX8C1YACRIuZoP-9OstghCA7?usp=sharing) на Google Colab.


## Языковые модели

Можно взять уже обученные в ноутбуке `2.0-db-training_lm.ipynb` языковые модели по [ссылке](https://drive.google.com/drive/folders/1iUR-SiMzbZkuHAAzJZLfiO9q7rIze2GI?usp=sharing): в директории models/kenlm.

Если требуется обучить языковые модели на своем корпусе, то потребуется склонировать [kenlm](https://github.com/kpu/kenlm) в src/kenlm и скомпилировать библиотеку, а потом повторить то, что было сделано в ноутбуке `2.0-db-training_lm.ipynb`.

## Текст работы и презентация

С итоговым текстом работы можно ознакомиться в директории `reports/thesis`. Презентация находится в `reports/presentation`.

## Структура проекта

------------

    ├── Snakefile           <- Snakefile with commands like `snakemake data` or `snakemake train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentation for the project
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `conda list -e > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── models         <- Models classes
        │
        ├── configs        <- Configurations of pipelines 
        │
        └── utils          <- Useful functions and classes for all project
