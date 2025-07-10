# Текущие промпты JoyCaption

## Используемые промпты (версия 2.0):

### 1. Descriptive (Описание)
```
Write a medium-length descriptive caption for this image in a casual tone.
```
- Генерирует описание средней длины в неформальном стиле
- Результат сохраняется в поле `description`

### 2. Danbooru Tags (Теги)
```
Write a medium-length list of Booru-like tags for this image.
```
- Генерирует список тегов в стиле Booru
- Теги разделены запятыми
- Используется для автоматической категоризации

### 3. Analysis (Структурированный анализ)
```
Analyze this image and provide structured information about:
- Characters or subjects (people, animals, creatures)
- Objects (items, props, accessories)
- Colors (primary and secondary colors)
- Materials (textures, surfaces, fabric types)
- Environment/setting (location, background)
- Lighting (type, direction, quality)
- Pose or position of subjects
- Art style (cartoon, realistic, anime, etc)
- Mood or atmosphere
- Any text or symbols visible
- Genre or time period (modern, medieval, sci-fi, fantasy)
- What each character or subject is doing
```
- Генерирует детальный структурированный анализ
- Результаты парсятся в категории

## Режимы по умолчанию:
```python
modes = ['descriptive', 'danbooru_tags', 'analysis']
```

## Обработка результатов:

1. **Из тегов извлекаются категории:**
   - Characters (персонажи)
   - Objects (объекты) 
   - Colors (цвета)
   - Materials (материалы)
   - Clothing (одежда)
   - Style (стиль)
   - Mood (настроение)
   - Environment (окружение)
   - Lighting (освещение)
   - Pose (поза)

2. **Из анализа извлекаются дополнительные данные:**
   - Genre (жанр)
   - Subject Actions (действия персонажей)
   - Text/Symbols (текст и символы)

3. **Интеллектуальная обработка:**
   - Извлечение цветов из составных тегов ("blue skin" → color: blue)
   - Фильтрация общих слов и дескрипторов
   - Объединение данных из разных источников

## Почему может не работать как ожидается:

1. JoyCaption может не всегда следовать структуре промпта для анализа
2. Модель обучена на определенном формате и может давать ответы в свободной форме
3. Парсинг зависит от качества ответа модели

## Рекомендации:

- Если структурированный анализ не работает хорошо, полагайтесь на теги
- Используйте "Show Detailed Results" для просмотра полного вывода
- Категории из тегов обычно более надежны, чем из структурированного анализа