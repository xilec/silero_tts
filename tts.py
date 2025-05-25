import torch
import soundfile as sf
import numpy as np
import re

def is_russian(text):
    """Проверяет, содержит ли текст русские буквы"""
    return bool(re.search('[а-яА-Я]', text))

def is_english(text):
    """Проверяет, содержит ли текст английские буквы"""
    return bool(re.search('[a-zA-Z]', text))

def split_by_language(text, max_length=900):
    """Разделяет текст на части по языку и длине"""
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
                                     speaker='en_0',
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