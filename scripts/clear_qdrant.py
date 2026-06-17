"""Очистка коллекции blocks в Qdrant"""
from modules.qdrant.client import client
from config import settings

collection = settings.qdrant_collection

# Удаляем коллекцию
if client.collection_exists(collection):
    client.delete_collection(collection)
    print(f"Коллекция '{collection}' удалена")
else:
    print(f"Коллекция '{collection}' не существует")

# Пересоздаём
from modules.qdrant.collections import init_collections
init_collections()
print(f"Коллекция '{collection}' создана заново")