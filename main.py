import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import pickle
import pandas as pd
import string
import re
import random
from layers import TransformerEncoder, TransformerDecoder, PositionalEmbedding

@register_keras_serializable()
def custom_standardization(input_string):
    import re
    import string
    lowercase = tf.strings.lower(input_string)
    strip_chars = string.punctuation + "." + "," + "!"
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

# Пути к файлам
DATA_PATH = 'dataset/rus.txt'
MODEL_EN_RU_PATH = 'model/model1.h5'
VECTORIZE_EN_PATH = 'model/source_vectorization1.pkl'
TARGET_VECTORIZE_EN_PATH = 'model/target_vectorization1.pkl'

MODEL_RU_EN_PATH = 'model/model2.h5'
VECTORIZE_RU_PATH = 'model/source_vectorization2.pkl'
TARGET_VECTORIZE_RU_PATH = 'model/target_vectorization2.pkl'

max_decoded_sentence_length = 30

# Загрузка данных
with open(DATA_PATH, encoding='utf-8') as file:
    lines = file.read().split("\n")

pairs_en_ru = []
pairs_ru_en = []

for line in lines:
    parts = line.split("\t")
    if len(parts) >= 2:
        en, ru = parts[0], parts[1]
        # Для en->ru
        pairs_en_ru.append((en, "[start] " + ru + " [end]"))
        # Для ru->en
        pairs_ru_en.append((ru, "[start] " + en + " [end]"))

# Перемешивание и разделение на train/val/test
def split_data(pairs):
    random.shuffle(pairs)
    n = len(pairs)
    val_size = int(0.2 * n)
    train_size = n - 2 * val_size
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]
    return train_pairs, val_pairs, test_pairs

train_pairs_en_ru, val_pairs_en_ru, test_pairs_en_ru = split_data(pairs_en_ru)
train_pairs_ru_en, val_pairs_ru_en, test_pairs_ru_en = split_data(pairs_ru_en)

# Векторизация для en->ru
with open(VECTORIZE_EN_PATH, 'rb') as f:
    source_vectorization_en = pickle.load(f)
with open(TARGET_VECTORIZE_EN_PATH, 'rb') as f:
    target_vectorization_en = pickle.load(f)

# Векторизация для ru->en
with open(VECTORIZE_RU_PATH, 'rb') as f:
    source_vectorization_ru = pickle.load(f)
with open(TARGET_VECTORIZE_RU_PATH, 'rb') as f:
    target_vectorization_ru = pickle.load(f)

# Загрузка моделей
model_en_ru = load_model(MODEL_EN_RU_PATH, custom_objects={
    "TransformerEncoder": TransformerEncoder,
    "TransformerDecoder": TransformerDecoder,
    "PositionalEmbedding": PositionalEmbedding
})
model_ru_en = load_model(MODEL_RU_EN_PATH, custom_objects={
    "TransformerEncoder": TransformerEncoder,
    "TransformerDecoder": TransformerDecoder,
    "PositionalEmbedding": PositionalEmbedding
})

# Обратные словари для декодирования
def get_index_lookup(target_vectorization):
    vocab = target_vectorization.get_vocabulary()
    return dict(zip(range(len(vocab)), vocab))

index_lookup_en = get_index_lookup(target_vectorization_en)
index_lookup_ru = get_index_lookup(target_vectorization_ru)

# Транслитерация
def simple_translit(word):
    translit_map = {
        'a':'а','b':'б','v':'в','g':'г','d':'д','e':'е','yo':'ё',
        'zh':'ж','z':'з','i':'и','j':'й','k':'к','l':'л','m':'м',
        'n':'н','o':'о','p':'п','r':'р','s':'с','t':'т','u':'у',
        'f':'ф','h':'х','ts':'ц','ch':'ч','sh':'ш','shch':'щ','y':'ы',
        'ye':'е','yu':'ю','ya':'я'
    }
    return ''.join([translit_map.get(ch, ch) for ch in word])

# Декодирование
def decode_sequence(input_sentence, model, source_vectorization, target_vectorization, index_lookup):
    tokenized_input = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input, tokenized_target])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "[end]":
            break
        if sampled_token == "[UNK]":
            last_word = input_sentence.split()[-1]
            translit_word = simple_translit(last_word)
            decoded_sentence += " " + translit_word
        else:
            decoded_sentence += " " + sampled_token
    return decoded_sentence

# Глобальный режим
direction = 'en-ru'  # или 'ru-en'

def switch_direction():
    global direction
    if direction == 'en-ru':
        direction = 'ru-en'
        btn_switch.config(text='Русский → Английский')
        lbl_input.config(text='Введите текст на русском:')
        lbl_output.config(text='Перевод на английский:')
    else:
        direction = 'en-ru'
        btn_switch.config(text='Английский → Русский')
        lbl_input.config(text='Введите текст на английском:')
        lbl_output.config(text='Перевод на русский:')
    txt_input.delete('1.0', tk.END)
    txt_output.delete('1.0', tk.END)

def translate():
    input_text = txt_input.get('1.0', tk.END).strip()
    if not input_text:
        messagebox.showwarning("Внимание", "Пожалуйста, введите текст для перевода.")
        return
    try:
        if direction == 'en-ru':
            result = decode_sequence(input_text, model_en_ru, source_vectorization_en, target_vectorization_en, index_lookup_en)
        else:
            result = decode_sequence(input_text, model_ru_en, source_vectorization_ru, target_vectorization_ru, index_lookup_ru)
        txt_output.delete('1.0', tk.END)
        txt_output.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

# Создаем GUI
root = tk.Tk()
root.title("Переводчик: Английский <-> Русский")
root.geometry("700x600")

lbl_input = ttk.Label(root, text='Введите текст на английском:')
lbl_input.pack(pady=5)

txt_input = tk.Text(root, height=5, width=80)
txt_input.pack(pady=5)

lbl_output = ttk.Label(root, text='Перевод на русский:')
lbl_output.pack(pady=5)

txt_output = tk.Text(root, height=15, width=80)
txt_output.pack(pady=5)

btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)

btn_translate = ttk.Button(btn_frame, text='Перевести', command=translate)
btn_translate.pack(side='left', padx=5)

btn_switch = ttk.Button(btn_frame, text='Английский → Русский', command=switch_direction)
btn_switch.pack(side='left', padx=5)

root.mainloop()