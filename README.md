
#  Автоматизированный ML-пайплайн для диагностики рака груди с Airflow и MinIO

## 📌 Обзор проекта
**Автоматизированный ETL-пайплайн** для прогнозирования диагноза рака молочной железы с использованием:
- **Apache Airflow 2.7.1** для оркестрации
- **MinIO** как S3-совместимого хранилища
- **Логистической регрессии** (scikit-learn)
- **Датасета Breast Cancer Wisconsin** из UCI ML репозитория


## 🧠 Постановка ML-задачи
**Бинарная классификация** опухолей на злокачественные/доброкачественные по 30 признакам ядер клеток

## 🏗️ Архитектура системы
### Диаграмма компонентов
```mermaid
graph TD
    A[Источник данных UCI] --> B[Airflow DAG]
    B --> C{Хранилище MinIO}
    C --> D[Сырые данные]
    C --> E[Обработанные данные]
    C --> F[Модели]
    C --> G[Метрики]
    B --> H[PostgreSQL]
    B --> I[Redis]
```

### Структура DAG
```mermaid
flowchart LR
    загрузка --> предобработка --> обучение --> оценка --> уведомление
```

## 🛠️ Структура проекта
```
ml-pipeline/
├── dags/
│   └── ml_pipeline_dag.py
├── etl/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── results/
├── logs/
├── scripts/
│   └── setup_minio.sh
├── config/
│   └── config.yaml
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

## 🚀 Быстрый старт
### 1. Предварительные требования
```bash
docker-compose v2.20+
docker 24.0+
```

### 2. Развертывание
```bash
# Клонировать репозиторий
git clone https://github.com/data_eng_exam
cd data_eng_exam

# Запустить сервисы
docker-compose up -d --build

# Инициализация (подождите 2 минуты)
open http://localhost:8080  # Airflow (логин: admin/admin)
open http://localhost:9001  # MinIO (логин: minioadmin/minioadmin)
```

### 3. Запуск пайплайна
Через браузер:
зайдите http://localhost:8080 логин и пароль admin:admin
Далее нажмите на запуск
![image](https://github.com/user-attachments/assets/c0b7e208-f8ef-4108-8b6a-8c88f8427d9f)

```bash
# Ручной запуск DAG
curl -X POST "http://localhost:8080/api/v1/dags/breast_cancer_pipeline/dagRuns" \
  -H "Content-Type: application/json" \
  -d '{"conf": {}}'
```

## 🔍 Где найти результаты
| Местоположение | Путь | Способ доступа |
|----------------|------|----------------|
| **MinIO** | `ml-pipeline/results/` | Веб-интерфейс или `mc ls local/ml-pipeline` |
| **Локальная ФС** | `./results/` | Прямой доступ к файлам |
| **Airflow** | Логи задач | Веб-интерфейс → DAG Runs → Task Instance |

## 🛡️ Анализ отказов и устойчивость
### Критические точки отказа
| Компонент | Тип отказа | Стратегия обработки |
|-----------|------------|---------------------|
| Источник данных | API недоступен | Повторные попытки (3 раза) |
| MinIO | Бакет отсутствует | Автосоздание в init-скрипте |
| Обучение модели | Не сходится | Ранняя остановка + алерты |
| Airflow | Потеря соединения с БД | Health checks + ретраи |

### Механизмы устойчивости
- ✅ Автоматические повторы для всех задач
- ✅ Изолированные шаги обработки
- ✅ Дублирование хранения (MinIO + локально)
- ✅ Health checks для всех сервисов
- ✅ Подробное логирование

## 💡 Идеи для улучшения
### Ближайшие планы
1. Добавить валидацию данных с Great Expectations
2. Реализовать версионирование моделей через MLflow
3. Настроить автоматические тесты (pytest)

### Перспективные
```mermaid
graph LR
    A[Текущая версия] --> B[Feature Store]
    A --> C[Реалтайм-инференс]
    A --> D[Мониторинг дрейфа данных]
    D --> E[Авторетренинг]
```

## 📊 Пример результатов
**metrics.json**
```json
{"accuracy": 0.9736842105263158, "precision": 0.9761904761904762, "recall": 0.9534883720930233, "f1_score": 0.9647058823529412}
```

## 🖼️ Скриншоты системы
![image](https://github.com/user-attachments/assets/0ba54d5b-203c-48c0-8f18-2cc90c672947)

*Выполнение пайплайна в интерфейсе Airflow*

*Артефакты в MinIO хранилище*
![image](https://github.com/user-attachments/assets/22e7a0d7-f2e0-4db7-b012-c727270495d3)
![image](https://github.com/user-attachments/assets/487b9834-7fe2-4f1b-ae2c-b21a4107f5fe)
![image](https://github.com/user-attachments/assets/2f2736f7-4939-4d9a-8ebd-b1b32f723e94)
![image](https://github.com/user-attachments/assets/47f3d3ac-df0c-4f22-81fc-230306f33ced)
![image](https://github.com/user-attachments/assets/4d8427da-599f-4f0e-805c-189b93e951ba)

## 📚 Документация
- [Официальная документация Airflow](https://airflow.apache.org/docs/)
- [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- [Описание датасета](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

