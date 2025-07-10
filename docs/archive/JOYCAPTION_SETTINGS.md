# JoyCaption Settings & Configuration

## Текущие параметры генерации:

```python
temperature = 0.6      # Контролирует случайность (0.0 = детерминировано, 1.0 = максимально случайно)
top_p = 0.9           # Nucleus sampling - рассматривает только топ 90% вероятных токенов
max_new_tokens = 512  # Максимальная длина генерации
do_sample = True      # Использовать вероятностную выборку
```

## Обновленные промпты (v3.0):

### 1. Descriptive (Описание)
```
Write a descriptive caption for this image in a casual tone.
```

### 2. Danbooru Tags (Структурированные теги)
```
Generate only comma-separated Danbooru tags. Strict order: character tags (1girl, 1boy), appearance, clothing, pose, expression, actions, objects, background, style. Use lowercase_underscores. No extra text.
```

### 3. Structured Tags (Категоризированные теги)
```
Write comma-separated tags. Order: 
character: (characters/species)
objects: (items, props, accessories)  
colors: (specific colors visible)
materials: (textures, fabrics)
environment: (setting, background)
lighting: (light quality and direction)
pose: (body position, gestures)
style: (art style, medium)
mood: (atmosphere, emotion)
```

## Различия локально vs онлайн:

### Возможные причины различий:
1. **Разные параметры**: Онлайн демо может использовать другие temperature/top_p
2. **Seed**: Разные seed дают разные результаты даже с одинаковыми параметрами
3. **Версия модели**: Возможно незначительные различия в загрузке модели
4. **Промпты**: Онлайн версия может использовать дополнительные инструкции

### Рекомендации для более консистентных результатов:
- Снизить temperature до 0.4-0.5 для более предсказуемых результатов
- Увеличить top_p до 0.95 для большего разнообразия
- Использовать seed для воспроизводимости (если поддерживается)

## Настройка параметров:

При создании анализатора можно указать свои параметры:
```python
analyzer = JoyCaptionAnalyzer(
    temperature=0.5,      # Более консистентные результаты
    top_p=0.95,          # Больше вариативности
    max_new_tokens=768   # Больше токенов для детальных описаний
)
```

## Оптимизация для разных задач:

### Для точных тегов:
- temperature = 0.3-0.4
- top_p = 0.8
- Использовать danbooru_tags промпт

### Для креативных описаний:
- temperature = 0.7-0.8
- top_p = 0.95
- Использовать descriptive промпт

### Для структурированного анализа:
- temperature = 0.5
- top_p = 0.9
- Использовать structured_tags промпт