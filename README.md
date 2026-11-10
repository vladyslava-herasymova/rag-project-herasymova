# RAG-система для пошуку інформації про заїзди Формулу Один 

## Стислий опис сценарію використання

**Цільовий користувач:** фанати Формули Один, які бажають отримати швидку відповідь на запитання щодо результатів та ходу заїздів Формули Один.

**Приклад типового запитання:** Who won the 1978 Monaco Grand Prix?

**Критерії гарної відповіді:** гарна відповідь має відповідати дійсності (першоджерелу), бути лаконічною та, у випадку відповіді на відкриті питання, вагомою (казати лише найважливіші речі). В залежності від види помилки відповідь, що не є коректною, може бути частково гарною.

## Існтрукції із встановлення обраного стеку

### Векторне сховище Chroma

**Посилання:** https://docs.trychroma.com/docs/overview/getting-started

**Короткий план встановлення та початку роботи:**
1. Встановлення через pip: pip install chromadb
2. Імпорт: import chromadb
3. Створення клієнту: chroma_client = chromadb.Client()

### Модель ембедингів all-MiniLM-L6-v2

Встановлюється через модуль sentence-transformers.

**Посилання:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**Короткий план встановлення та початку роботи:**
1. Встановлення sentence-transformers через pip: pip install -U sentence-transformers
2. Імпорт: from sentence_transformers import SentenceTransformer
3. Підваження моделі: model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

### Модель генерації Llama 3.2 3B

Встановлюється через Ollama.

**Посилання:** https://ollama.com/library/llama3.2

**Короткий план встановлення та початку роботи:**
1. Скачування Ollama за посиланням https://ollama.com/download 
2. У випадку мануального скачування: запуск скачуваного .exe файлу та встановка Ollama
3. Підваження моделі через термінал: ollama pull llama3.2
4. Імпорт функції chat в коді: from ollama import chat
5. Запуск функції чат зі специфікацією моделі: response = chaе(model='llama3.2', messages=\[...\],)

### Мова програмування Python

**Посилання:** https://python.org/downloads/

**Короткий план встановлення та початку роботи:**
1. Скачування та запуск .exe файлу бажаної версії