# InboxAI 📬
Учебный проект, выполненный в рамках 1 семестра.

## 💡 Возможности

- **Парсинг** публичных Telegram-каналов
- **Обогащение** постов через локальную LLM
- **Векторизация** текстов моделью BGE-M3 (1024 измерения)
- **Хранение** векторов и метаданных в Qdrant
- **Семантический поиск** с реранкингом (BGE-Reranker-v2-m3)
- **Два режима дайджестов:**
  - `search` — поисковый дайджест по конкретному запросу
  - `sub` — обзорный дайджест по категориям, источникам и периоду
- **Дедупликация** с динамическим порогом и MMR для разнообразия
- **Генерация** читаемых аннотаций через LLM
- **Локальное исполнение** — все модели работают через LM Studio, данные не покидают контур

## 📋 Требования

- Python 3.11+
- Docker (для Qdrant)
- LM Studio с загруженными моделями:
  - `bge-m3` (эмбеддинги)
  - `Qwen3.6 27B` (обогащение и генерация дайджестов)
- Модель реранкер `BAAI/bge-reranker-v2-m3` (из `sentence-transformers`)
- Данные от Telegram UserBot (app_id, api_hash)

## 📦 Установка

```bash
git clone https://github.com/grigao/semestroviy-project-v1
cd semestroviy-project-v1
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
pip install -r requirements.txt
```

## ⚙️ Настройка

- **`.env`** — переменные окружения
- **`docker-compose.qdrant.yml`** — запуск Qdrant: `docker compose up -d`
- **`workers.json`** — список каналов и системные настройки для парсера
- **LM Studio** — загрузить модели: `bge-m3`, `Qwen3.6 27B`

## ▶️ Запуск

1. `docker compose -f docker-compose.qdrant.yml up -d` — Qdrant
2. `python -m modules.parser.call_worker` — парсинг каналов
3. `python -m modules.enrichment.enricher` — обогащение (категории, keywords)
4. `python -m modules.embedding.embedder` — векторизация и загрузка в Qdrant
5. Варианты генерации дайджеста:
   - `python scripts/digest_cli.py search "запрос"` — поисковый
   - `python scripts/digest_cli.py search "запрос" 5 --sources habr_com --period 48h` — поисковый с фильтрами
   - `python scripts/digest_cli.py sub --sources habr_com --period 24h` — обзорный
   - `python scripts/digest_cli.py sub technology --sources habr_com` — тематический
