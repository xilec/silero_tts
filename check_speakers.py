import torch

print("Загрузка русской модели...")
model_ru, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                            model='silero_tts',
                            language='ru',
                            speaker='v4_ru')

print("\nЗагрузка английской модели...")
model_en, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                            model='silero_tts',
                            language='en',
                            speaker='v3_en')

print("\nДоступные голоса для русского языка:")
print(model_ru.speakers)

print("\nДоступные голоса для английского языка:")
print(model_en.speakers) 