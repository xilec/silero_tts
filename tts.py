import torch
import soundfile as sf
import numpy as np
import re
from datetime import datetime
import unicodedata
from num2words import num2words

def convert_number_to_words(number_str):
    """Конвертация числа в слова"""
    try:
        # Удаляем пробелы и разделители
        number_str = number_str.replace(' ', '').replace(',', '.')
        
        # Проверяем, является ли число дробным
        if '.' in number_str:
            integer_part, decimal_part = number_str.split('.')
            # Конвертируем целую часть
            words = num2words(int(integer_part), lang='ru')
            # Добавляем дробную часть
            words += f" целых {num2words(int(decimal_part), lang='ru')} сотых"
        else:
            words = num2words(int(number_str), lang='ru')
        
        return words
    except ValueError:
        return number_str

def process_numbers(text):
    """Обработка чисел"""
    def num_to_text(match):
        num = match.group(0)
        # Обработка дат
        if re.match(r'\d{2}\.\d{2}\.\d{4}', num):
            try:
                date = datetime.strptime(num, '%d.%m.%Y')
                day = num2words(int(date.day), lang='ru')
                month = date.strftime('%B')  # TODO: перевести название месяца
                year = num2words(int(date.year), lang='ru')
                return f"{day} {month} {year} года"
            except ValueError:
                return num
        # Обработка времени
        elif re.match(r'\d{2}:\d{2}', num):
            hours, minutes = num.split(':')
            hours_text = num2words(int(hours), lang='ru')
            minutes_text = num2words(int(minutes), lang='ru')
            return f"{hours_text} часов {minutes_text} минут"
        # Обработка обычных чисел
        else:
            return convert_number_to_words(num)
    
    # Находим все числа (целые, дробные, даты, время)
    text = re.sub(r'\d+(?:[.,]\d+)?(?::\d{2})?', num_to_text, text)
    return text

def normalize_text(text):
    """Нормализация текста"""
    # Замена специальных символов
    text = text.replace('…', '...')
    text = text.replace('\\', ' ')
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    
    # Нормализация пробелов
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Нормализация пунктуации
    text = re.sub(r'[,.!?;:][\s]*', lambda m: m.group(0).strip() + ' ', text)
    
    return text

def process_abbreviations(text):
    """Обработка аббревиатур"""
    # Добавление пробелов между буквами в аббревиатурах
    text = re.sub(r'([А-ЯA-Z]{2,})', lambda m: ' '.join(list(m.group(1))), text)
    return text

def improve_pronunciation(text):
    """Улучшение произношения"""
    # Замена английских букв на русское произношение
    en_to_ru = {
        'Ph': 'Ф',
        'Th': 'З',
        'ch': 'ч',
        'sh': 'ш',
        'tion': 'шн',
        'x': 'кс'
    }
    
    for en, ru in en_to_ru.items():
        text = text.replace(en, ru)
    
    return text

def process_special_text(text):
    """Обработка специального текста"""
    # Замена URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                 'ссылка', text)
    
    # Замена email
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', 'электронная почта', text)
    
    return text

def add_pauses(text):
    """Добавление пауз"""
    text = text.replace('.', '... ')
    text = text.replace('!', '! ... ')
    text = text.replace('?', '? ... ')
    text = text.replace(',', ', ')
    return text

def preprocess_text(text):
    """Полная предобработка текста"""
    text = normalize_text(text)
    text = process_numbers(text)
    text = process_abbreviations(text)
    text = process_special_text(text)
    text = improve_pronunciation(text)
    text = add_pauses(text)
    return text

def is_russian(text):
    """Проверяет, содержит ли текст русские буквы"""
    return bool(re.search('[а-яА-Я]', text))

def is_english(text):
    """Проверяет, содержит ли текст английские буквы"""
    return bool(re.search('[a-zA-Z]', text))

def split_by_language(text, max_length=900):
    """Разделяет текст на части по языку и длине"""
    # Предварительная обработка текста
    text = preprocess_text(text)
    
    # Разделяем текст на предложения
    sentences = re.split('[.!?]', text)
    chunks = []
    current_chunk = []
    current_length = 0
    current_lang = None
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Добавляем знак препинания обратно
        sentence = sentence + '.'
        
        # Определяем язык предложения
        is_rus = is_russian(sentence)
        is_eng = is_english(sentence)
        
        # Если предложение содержит оба языка, разбиваем его на части
        if is_rus and is_eng:
            parts = re.split('([а-яА-Я]+[^a-zA-Z]*|[a-zA-Z]+[^а-яА-Я]*)', sentence)
            for part in parts:
                if not part.strip():
                    continue
                if is_russian(part):
                    chunks.append(('ru', part))
                elif is_english(part):
                    chunks.append(('en', part))
        else:
            # Если предложение на одном языке
            if is_rus:
                chunks.append(('ru', sentence))
            elif is_eng:
                chunks.append(('en', sentence))
    
    return chunks

# Загрузка русской модели
device = torch.device('cpu')
torch.set_num_threads(4)

print("Загрузка русской модели...")
model_ru, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                            model='silero_tts',
                            language='ru',
                            speaker='v4_ru')
model_ru.to(device)

print("Загрузка английской модели...")
model_en, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                            model='silero_tts',
                            language='en',
                            speaker='v3_en')
model_en.to(device)

# Чтение текста из файла
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

# Разделяем текст на части по языкам
text_chunks = split_by_language(text)

# Синтез речи для каждой части
audio_chunks = []
for i, (lang, chunk) in enumerate(text_chunks, 1):
    print(f"Обработка части {i} из {len(text_chunks)} ({lang})...")
    try:
        if lang == 'ru':
            audio = model_ru.apply_tts(text=chunk,
                                     speaker='xenia',
                                     sample_rate=48000)
        else:
            audio = model_en.apply_tts(text=chunk,
                                     speaker='en_1',
                                     sample_rate=48000)
        
        # Добавляем небольшую паузу между частями (0.2 секунды тишины)
        pause = np.zeros(int(48000 * 0.2))
        
        audio_chunks.append(audio.numpy())
        audio_chunks.append(pause)
        
    except Exception as e:
        print(f"Ошибка при обработке части: {chunk}")
        print(f"Ошибка: {str(e)}")
        continue

# Объединяем все части
if audio_chunks:
    combined_audio = np.concatenate(audio_chunks)

    # Сохранение аудио в файл
    sf.write('output.wav', combined_audio, 48000)
    print("Аудио файл сохранен как 'output.wav'")
else:
    print("Не удалось создать аудио: нет обработанных частей текста")