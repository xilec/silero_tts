import torch
import soundfile as sf
import numpy as np
import re
from datetime import datetime
import unicodedata
from num2words import num2words
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Словарь фонетического произношения английских букв
ENGLISH_PHONETIC = {
    'A': 'ay',
    'B': 'bee',
    'C': 'see',
    'D': 'dee',
    'E': 'ee',
    'F': 'ef',
    'G': 'gee',
    'H': 'aitch',
    'I': 'eye',
    'J': 'jay',
    'K': 'kay',
    'L': 'el',
    'M': 'em',
    'N': 'en',
    'O': 'oh',
    'P': 'pee',
    'Q': 'queue',
    'R': 'ar',
    'S': 'es',
    'T': 'tee',
    'U': 'you',
    'V': 'vee',
    'W': 'double you',
    'X': 'ex',
    'Y': 'why',
    'Z': 'zee'
}

def validate_file_extension(filename):
    """Проверка расширения файла"""
    if not filename.lower().endswith('.txt'):
        raise argparse.ArgumentTypeError('Файл должен иметь расширение .txt')
    return filename

def parse_arguments():
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Преобразование текста в речь')
    parser.add_argument('--input', '-i', 
                       type=validate_file_extension,
                       default='text.txt',
                       help='Путь к входному текстовому файлу (по умолчанию: text.txt)')
    parser.add_argument('--test_eng', 
                       action='store_true',
                       help='Тестировать все английские голоса (en_0 до en_117)')
    parser.add_argument('--dump_chunks',
                       action='store_true',
                       help='Сохранять чанки в отдельный файл для отладки')
    
    try:
        args = parser.parse_args()
        return args
    except argparse.ArgumentTypeError as e:
        print(f"Ошибка: {str(e)}")
        sys.exit(1)

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
    def replace_eng_abbr(match):
        """Заменяет английские буквы на их фонетическое произношение"""
        abbr = match.group(1)
        # Проверяем, что это английская аббревиатура (содержит только английские буквы)
        if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in abbr):
            return ' '.join(ENGLISH_PHONETIC.get(c, c) for c in abbr)
        # Для русских аббревиатур оставляем пробелы между буквами
        return ' '.join(list(abbr))
    
    # Находим все последовательности заглавных букв (2 и более)
    text = re.sub(r'([А-ЯA-Z]{2,})', replace_eng_abbr, text)
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

def get_audio_duration(sample_rate, num_samples):
    """Вычисляет длительность аудио в секундах"""
    return num_samples / sample_rate

def format_time(seconds):
    """Форматирует время в читаемый вид"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    if minutes > 0:
        return f"{minutes} мин. {seconds:.1f} сек."
    return f"{seconds:.1f} сек."

def process_chunk(chunk_info):
    """Обработка одного текстового фрагмента"""
    i, (lang, chunk), models, speaker_override = chunk_info
    model_ru, model_en = models
    
    chunk_start = time.time()
    try:
        if lang == 'ru':
            audio = model_ru.apply_tts(text=chunk,
                                     speaker='xenia',
                                     sample_rate=48000)
        else:
            speaker = speaker_override if speaker_override else 'en_9'
            audio = model_en.apply_tts(text=chunk,
                                     speaker=speaker,
                                     sample_rate=48000)
        
        # Добавляем небольшую паузу между частями (0.2 секунды тишины)
        pause = np.zeros(int(48000 * 0.2))
        
        chunk_time = time.time() - chunk_start
        return {
            'index': i,
            'audio': audio.numpy(),
            'pause': pause,
            'time': chunk_time,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'index': i,
            'audio': None,
            'pause': None,
            'time': time.time() - chunk_start,
            'success': False,
            'error': str(e)
        }

def process_text(text, input_file, models, eng_speaker=None, show_stats=True, dump_chunks=False):
    """
    Обработка текста и генерация аудио
    
    Args:
        text (str): Исходный текст
        input_file (str): Путь к входному файлу
        models (tuple): Кортеж (model_ru, model_en)
        eng_speaker (str, optional): Идентификатор английского спикера
        show_stats (bool): Показывать ли статистику выполнения
        dump_chunks (bool): Сохранять ли чанки в файл для отладки
    """
    start_time = time.time()
    model_ru, model_en = models
    
    # Формируем имя выходного файла
    filename, _ = os.path.splitext(input_file)
    output_file = f"{filename}_{eng_speaker}.wav" if eng_speaker else f"{filename}.wav"
    
    # Выводим информацию о размере текста
    text_size = len(text)
    if show_stats:
        print(f"Размер текста: {text_size:,} символов")
    
    # Разделяем текст на части по языкам
    text_chunks = split_by_language(text)
    num_chunks = len(text_chunks)
    
    # Сохраняем чанки в файл, если указан флаг
    if dump_chunks:
        chunks_file = f"{filename}_chunks.txt"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for i, (lang, chunk) in enumerate(text_chunks, 1):
                f.write(f"Чанк {i}/{num_chunks} [{lang}]:\n{chunk}\n{'='*50}\n")
        print(f"Чанки сохранены в файл: {chunks_file}")
    
    if show_stats:
        print(f"Количество частей для обработки: {num_chunks}")
    
    # Определяем оптимальное количество потоков
    num_threads = min(num_chunks, multiprocessing.cpu_count())
    if show_stats:
        print(f"Используется потоков: {num_threads}")
    
    # Подготовка данных для параллельной обработки
    chunk_data = [
        (i, chunk, (model_ru, model_en), eng_speaker)
        for i, chunk in enumerate(text_chunks, 1)
    ]
    
    # Синтез речи для каждой части в параллельном режиме
    audio_chunks = [None] * num_chunks
    processing_start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk_info): chunk_info[0]
            for chunk_info in chunk_data
        }
        
        for future in as_completed(future_to_chunk):
            result = future.result()
            chunk_index = result['index']
            
            if result['success']:
                msg = f"Часть {chunk_index} из {num_chunks} обработана"
                if show_stats:
                    msg += f" за {format_time(result['time'])}"
                print(msg)
                audio_chunks[chunk_index - 1] = [result['audio'], result['pause']]
            else:
                print(f"Ошибка при обработке части {chunk_index}: {result['error']}")
    
    processing_time = time.time() - processing_start
    if show_stats:
        print(f"\nВремя обработки текста: {format_time(processing_time)}")
    
    # Объединяем все части
    if any(chunk is not None for chunk in audio_chunks):
        combined_chunks = []
        for chunk_pair in audio_chunks:
            if chunk_pair is not None:
                combined_chunks.extend(chunk_pair)
        
        combined_audio = np.concatenate(combined_chunks)
        
        # Сохранение аудио в файл
        sf.write(output_file, combined_audio, 48000)
        
        if show_stats:
            # Вычисляем длительность аудио
            audio_duration = get_audio_duration(48000, len(combined_audio))
            
            # Выводим статистику
            total_time = time.time() - start_time
            print("\nСтатистика:")
            print(f"Размер исходного текста: {text_size:,} символов")
            print(f"Длительность аудио: {format_time(audio_duration)}")
            print(f"Общее время выполнения: {format_time(total_time)}")
            print(f"  - Обработка текста: {format_time(processing_time)}")
            print(f"  - Накладные расходы: {format_time(total_time - processing_time)}")
        
        print(f"Аудио файл сохранен как '{output_file}'")
        return True
    else:
        print("Не удалось создать аудио: нет обработанных частей текста")
        return False

def test_english_speakers(text, models, input_file):
    """Тестирование всех английских спикеров"""
    for speaker_idx in range(118):  # от 0 до 117
        speaker = f'en_{speaker_idx}'
        print(f"\nТестирование спикера {speaker}")
        process_text(text, input_file, models, eng_speaker=speaker, show_stats=False)

def main():
    # Получаем аргументы командной строки
    args = parse_arguments()
    input_file = args.input
    
    # Проверяем существование входного файла
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден")
        return
    
    # Загрузка моделей
    device = torch.device('cpu')
    torch.set_num_threads(4)
    
    model_loading_start = time.time()
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
    
    model_loading_time = time.time() - model_loading_start
    print(f"Время загрузки моделей: {format_time(model_loading_time)}")
    
    # Чтение текста из файла
    print(f"\nЧтение текста из файла {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    if args.test_eng:
        # Запускаем тестирование всех английских спикеров
        test_english_speakers(text, (model_ru, model_en), input_file)
    else:
        # Стандартная обработка одним спикером
        process_text(text, input_file, (model_ru, model_en), dump_chunks=args.dump_chunks)

if __name__ == '__main__':
    main()