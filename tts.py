import torch
import soundfile as sf
import numpy as np

def split_text(text, max_length=900):
    # Разделяем текст по предложениям
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + '.'
        if not sentence:
            continue
        
        if current_length + len(sentence) > max_length:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Загрузка модели
device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                   model='silero_tts',
                                   language='ru',
                                   speaker='v4_ru')

# Перемещаем модель на CPU
model.to(device)

# Чтение текста из файла
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

# Разделяем текст на части
text_chunks = split_text(text)

# Синтез речи для каждой части
audio_chunks = []
for i, chunk in enumerate(text_chunks, 1):
    print(f"Обработка части {i} из {len(text_chunks)}...")
    # доступные голоса: aidar, baya, kseniya, xenia, eugene, random (не удалять коментарий)
    audio = model.apply_tts(text=chunk,
                           speaker='xenia',
                           sample_rate=48000)
    audio_chunks.append(audio.numpy())

# Объединяем все части
combined_audio = np.concatenate(audio_chunks)

# Сохранение аудио в файл
sf.write('output.wav', combined_audio, 48000)
print("Аудио файл сохранен как 'output.wav'")