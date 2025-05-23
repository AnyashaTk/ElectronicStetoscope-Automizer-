# ElectronicStetoscopeAutomizer

Подробнее о деталях проекта можно узнать в [System Design](/docs/ml_system_design_doc.md) документе

## Оглавление
1. [О проекте](#о-проекте)
2. [Структура](#структура)
3. [Технологии](#технологии)
4. [Установка](#установка)
5. [Использование](#использование)

## О проекте
Основной целью проекта является внедрение технологий искуственного интеллекта в процесс анализа цифровой кардиофонограммы в процессе аускультации сердца
> Аускультация сердца — это метод диагностики, при котором с помощью стетоскопа слушают звуки, издаваемые сердцем, для оценки его работы, выявления заболеваний и аномалий

Качество выполнения данного исследования значительно зависит от персональных особенностей врача, таких как квалификация, опыт и слух. В виду этого создание интеллектуальной экспертной системы поддержки принятия врачебных решений при аускускультации сердца сможет повысить информативность достоверность и доказательность метода

## Структура

На данный момент проект соответствует структуре:
> * artefacts # Директория со всеми артефактами для описания проекта
> * docs
> * * ml_system_design_doc.md # Дизайн документ проекта
> * app
> * * main.py
> * * models.py
> * notebooks
> * * \< eda and trainong notebooks \>
> * .pre-commit-config.yaml # pre-commit файл проекта
> * poetry.lock # poetry файл о проекте
> * pyproject.toml # poetry файл о проекте
> * README.md # Основное описание проекта


## Технологии

В качестве основного Языка программирования выступает python, модели работают с помощью skikit-learn, torch, сервер использует streamlit

Все требуемые зависимости указаны в pyproject.toml

## Установка

>  git clone https://github.com/AnyashaTk/ElectronicStetoscope-Automizer-.git

>  poetry install

## Использование

#### Для проверки перед коммитом можно отдельно руками запустить процесс изменений pre-commit:

>  pre-commit run --show-diff-on-failure --color=always --all-files

black можно использовать с помощью команды

> black <file_for_check>

#### Для запуска приложения 

>  streamlit run app/main.py

На данный момент приложение является скорее демонстрацией создаваемого продукта с примером интерфейса

На семплах, находящихся в директории data можно проверить разные выводы, если будете использовать свои файлы, будем рады получить обратную связь (telegram @AnTkDm)
