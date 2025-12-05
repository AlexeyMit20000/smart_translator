import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
        pairs_en_ru.append((en, "[start] " + ru + " [end]"))
        pairs_ru_en.append((ru, "[start] " + en + " [end]"))

def split_data(pairs):
    random.shuffle(pairs)
    n = len(pairs)
    val_size = int(0.2 * n)
    train_size = n - 2 * val_size
    return pairs[:train_size], pairs[train_size:train_size + val_size], pairs[train_size + val_size:]

train_pairs_en_ru, val_pairs_en_ru, test_pairs_en_ru = split_data(pairs_en_ru)
train_pairs_ru_en, val_pairs_ru_en, test_pairs_ru_en = split_data(pairs_ru_en)

with open(VECTORIZE_EN_PATH, 'rb') as f:
    source_vectorization_en = pickle.load(f)
with open(TARGET_VECTORIZE_EN_PATH, 'rb') as f:
    target_vectorization_en = pickle.load(f)
with open(VECTORIZE_RU_PATH, 'rb') as f:
    source_vectorization_ru = pickle.load(f)
with open(TARGET_VECTORIZE_RU_PATH, 'rb') as f:
    target_vectorization_ru = pickle.load(f)

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

def get_index_lookup(target_vectorization):
    vocab = target_vectorization.get_vocabulary()
    return dict(zip(range(len(vocab)), vocab))

index_lookup_en = get_index_lookup(target_vectorization_en)
index_lookup_ru = get_index_lookup(target_vectorization_ru)

def simple_translit(word):
    translit_map = {
        'a':'а','b':'б','v':'в','g':'г','d':'д','e':'е','yo':'ё',
        'zh':'ж','z':'з','i':'и','j':'й','k':'к','l':'л','m':'м',
        'n':'н','o':'о','p':'п','r':'р','s':'с','t':'т','u':'у',
        'f':'ф','h':'х','ts':'ц','ch':'ч','sh':'ш','shch':'щ','y':'ы',
        'ye':'е','yu':'ю','ya':'я'
    }
    return ''.join([translit_map.get(ch, ch) for ch in word])

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
    # Удаляем [start] и [end]
    decoded_sentence = re.sub(r'\[start\]', '', decoded_sentence)
    decoded_sentence = re.sub(r'\[end\]', '', decoded_sentence)
    # Удаляем пробелы
    decoded_sentence = re.sub(r'\s+', ' ', decoded_sentence).strip()
    return decoded_sentence


# Новая функция для перевода текста по предложениям
def translate_full_text(text):
    # Разделяем текст на предложения по точкам, вопросительным и восклицательным знакам
    sentences = re.split(r'(?<=[.!?])\s+', text)
    translated_sentences = []

    for sentence in sentences:
        # Переводим каждое предложение отдельно
        if direction == 'en-ru':
            translated_sentence = decode_sequence(sentence.strip(), model_en_ru, source_vectorization_en, target_vectorization_en, index_lookup_en)
        else:
            translated_sentence = decode_sequence(sentence.strip(), model_ru_en, source_vectorization_ru, target_vectorization_ru, index_lookup_ru)
        translated_sentences.append(translated_sentence)

    # Объединяем переведённые предложения
    return ' '.join(translated_sentences)

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

def on_translate_click():
    input_text = txt_input.get('1.0', tk.END).strip()
    if not input_text:
        messagebox.showwarning("Внимание", "Пожалуйста, введите текст для перевода.")
        return
    try:
        # Используем новую функцию
        result = translate_full_text(input_text)
        txt_output.delete('1.0', tk.END)
        txt_output.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))

def translate_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            # Используем новую функцию
            translation = translate_full_text(content)
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(translation)
                messagebox.showinfo("Готово", f"Перевод сохранен в {save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", "Ошибка при переводе файла: " + str(e))

# Создаем GUI
root = tk.Tk()
root.title("Офлайн переводчик")
root.geometry("900x700")
root.configure(background='#f0f4f8')

style = ttk.Style()
style.theme_use('clam')
style.configure('BlueButton.TButton', font=('Arial', 11), padding=6, relief='flat', background='#66b3ff', foreground='white')
style.map('BlueButton.TButton', background=[('active', '#3399ff')])

# Верхняя панель для переключения
top_frame = ttk.Frame(root)
top_frame.pack(pady=10)

btn_switch = ttk.Button(top_frame, text='сменить язык', command=switch_direction, style='BlueButton.TButton')
btn_switch.pack()

# Основной интерфейс
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Левая часть - оригинальный текст
left_frame = ttk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
lbl_input = ttk.Label(left_frame, text='Введите текст на английском:')
lbl_input.pack(anchor='w')
txt_input = tk.Text(left_frame, height=15, width=40, font=("Arial", 12), bd=0, relief=tk.FLAT)
txt_input.pack(fill=tk.BOTH, expand=True, pady=5)

# Правая часть - перевод
right_frame = ttk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
lbl_output = ttk.Label(right_frame, text='Перевод на русский:')
lbl_output.pack(anchor='w')
txt_output = tk.Text(right_frame, height=15, width=40, font=("Arial", 12), bd=0, relief=tk.FLAT)
txt_output.pack(fill=tk.BOTH, expand=True, pady=5)

# Кнопки
buttons_frame = ttk.Frame(root)
buttons_frame.pack(pady=10)

# Добавляем кнопки
translate_button = ttk.Button(buttons_frame, text='Перевести', style='BlueButton.TButton', command=on_translate_click)
translate_button.grid(row=0, column=0, padx=10)

translate_file_button = ttk.Button(buttons_frame, text='Перевести файл', style='BlueButton.TButton', command=translate_file)
translate_file_button.grid(row=0, column=1, padx=10)

# Добавляем кнопку Очистить
def clear_text():
    txt_input.delete("1.0", tk.END)
    txt_output.delete("1.0", tk.END)

clear_button = ttk.Button(buttons_frame, text='Очистить', style='BlueButton.TButton', command=clear_text)
clear_button.grid(row=0, column=2, padx=10)

# Стиль и оформление
border_color = "#a7c7e7"
txt_input.config(highlightbackground=border_color, highlightcolor=border_color, highlightthickness=1)
txt_output.config(highlightbackground=border_color, highlightcolor=border_color, highlightthickness=1)

style.configure('.', background="#f0f4f8")
style.map('TNotebook.Tab', background=[('selected', '#66b3ff')])

# Запуск
root.mainloop()