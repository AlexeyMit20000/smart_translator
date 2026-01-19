import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import pickle
import sqlite3
import io
import os
import re
import shutil
import string
import random
from openpyxl import Workbook
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from layers import TransformerEncoder, TransformerDecoder, PositionalEmbedding


# --- 1. ЛОГИКА ПЕРЕВОДЧИКА ---

@register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    strip_chars = string.punctuation + "." + "," + "!"
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

# Пути к файлам
MODEL_EN_RU_PATH = 'model/model1.h5'
VECTORIZE_EN_PATH = 'model/source_vectorization1.pkl'
TARGET_VECTORIZE_EN_PATH = 'model/target_vectorization1.pkl'
MODEL_RU_EN_PATH = 'model/model2.h5'
VECTORIZE_RU_PATH = 'model/source_vectorization2.pkl'
TARGET_VECTORIZE_RU_PATH = 'model/target_vectorization2.pkl'

# Загрузка векторизаций
with open(VECTORIZE_EN_PATH, 'rb') as f:
    source_vectorization_en = pickle.load(f)
with open(TARGET_VECTORIZE_EN_PATH, 'rb') as f:
    target_vectorization_en = pickle.load(f)
with open(VECTORIZE_RU_PATH, 'rb') as f:
    source_vectorization_ru = pickle.load(f)
with open(TARGET_VECTORIZE_RU_PATH, 'rb') as f:
    target_vectorization_ru = pickle.load(f)

custom_objects = {
    "TransformerEncoder": TransformerEncoder,
    "TransformerDecoder": TransformerDecoder,
    "PositionalEmbedding": PositionalEmbedding
}

model_en_ru = load_model(MODEL_EN_RU_PATH, custom_objects=custom_objects)
model_ru_en = load_model(MODEL_RU_EN_PATH, custom_objects=custom_objects)


def get_index_lookup(target_vectorization):
    vocab = target_vectorization.get_vocabulary()
    return dict(zip(range(len(vocab)), vocab))

index_lookup_en = get_index_lookup(target_vectorization_en)
index_lookup_ru = get_index_lookup(target_vectorization_ru)

def simple_translit(word):
    translit_map = {
        'a':'а','b':'б','v':'в','g':'г','d':'д','e':'е','yo':'ё','zh':'ж','z':'з',
        'i':'и','j':'й','k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','r':'р',
        's':'с','t':'т','u':'у','f':'ф','h':'х','ts':'ц','ch':'ч','sh':'ш','shch':'щ',
        'y':'ы','ye':'е','yu':'ю','ya':'я'
    }
    return ''.join([translit_map.get(ch, ch) for ch in word.lower()])


def decode_sequence(input_sentence, model, source_vectorization, target_vectorization, index_lookup):
    tokenized_input = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(30):
        tokenized_target = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input, tokenized_target])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "[end]":
            break
        if sampled_token == "[UNK]":
            last_word = input_sentence.split()[-1]
            decoded_sentence += " " + simple_translit(last_word)
        else:
            decoded_sentence += " " + sampled_token
    decoded_sentence = decoded_sentence.replace("[start]", "").replace("[end]", "").strip()
    decoded_sentence = re.sub(r'\bend\s*$', '', decoded_sentence)
    return decoded_sentence.strip()

def translate_full_text_logic(text, direction_mode):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    translated_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        if direction_mode == 'en-ru':
            res = decode_sequence(sentence.strip(), model_en_ru, source_vectorization_en, target_vectorization_en, index_lookup_en)
        else:
            res = decode_sequence(sentence.strip(), model_ru_en, source_vectorization_ru, target_vectorization_ru, index_lookup_ru)
        res = res.capitalize()
        if res and not res.endswith(('.', '!', '?')):
            res += '.'
        translated_sentences.append(res)
    return ' '.join(translated_sentences)


# --- 2. БАЗА ДАННЫХ ---
conn = sqlite3.connect('library.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    annotation TEXT,
    code TEXT,
    author TEXT,
    location TEXT,
    photo BLOB
)
''')
conn.commit()


# --- 3. ГЛАВНОЕ ПРИЛОЖЕНИЕ ---

class CombinedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Библиотека и Переводчик")
        self.root.geometry("900x600")
        self.direction = 'en-ru'
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.translator_tab = ttk.Frame(self.notebook)
        self.library_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.translator_tab, text="Переводчик")
        self.notebook.add(self.library_tab, text="Каталогизатор")

        self.setup_translator_ui()
        self.library_module = LibraryModule(self.library_tab, self.root)

    def add_context_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_separator()
        menu.add_command(label="Выбрать всё", command=lambda: widget.tag_add("sel", "1.0", "end"))
        def show_menu(event):
            menu.post(event.x_root, event.y_root)
        # <Button-3> для Windows/Linux, <Button-2> для macOS
        widget.bind("<Button-3>", show_menu)

    def setup_translator_ui(self):
        top_frame = ttk.Frame(self.translator_tab)
        top_frame.pack(pady=10)
        self.btn_switch = ttk.Button(top_frame, text='Английский → Русский', command=self.switch_translation_direction)
        self.btn_switch.pack()

        middle_frame = ttk.Frame(self.translator_tab)
        middle_frame.pack(fill='both', expand=True, padx=10)
        middle_frame.columnconfigure(0, weight=1, uniform='group1')
        middle_frame.columnconfigure(1, weight=1, uniform='group1')

    # Левая часть
        left_frame = ttk.Frame(middle_frame)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5)
        self.label_input = ttk.Label(left_frame, text="Введите текст на английском:")
        self.label_input.pack(anchor='w')
        self.txt_input = tk.Text(left_frame, font=("Arial", 11), undo=True)
        self.txt_input.pack(fill='both', expand=True)
        self.add_context_menu(self.txt_input)

    # Правая часть
        right_frame = ttk.Frame(middle_frame)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5)
        self.label_output = ttk.Label(right_frame, text="Перевод на русский:")
        self.label_output.pack(anchor='w')
        self.txt_output = tk.Text(right_frame, font=("Arial", 11), bg="#f8f9fa")
        self.txt_output.pack(fill='both', expand=True)
        self.add_context_menu(self.txt_output)

    # Нижняя панель с кнопками
        bottom_frame = ttk.Frame(self.translator_tab)
        bottom_frame.pack(pady=15)
        ttk.Button(bottom_frame, text="Перевести", command=self.on_translate).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="Перевести файл", command=self.on_translate_file).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="Очистить", command=self.clear_translation_fields).pack(side='left', padx=5)
    
    def switch_translation_direction(self):
        if self.direction == 'en-ru':
           self.direction = 'ru-en'
           self.btn_switch.config(text='Русский → Английский')
           self.label_input.config(text="Введите текст на русском:")
           self.label_output.config(text="Перевод на английский:")
        else:
           self.direction = 'en-ru'
           self.btn_switch.config(text='Английский → Русский')
           self.label_input.config(text="Введите текст на английском:")
           self.label_output.config(text="Перевод на русский:")
        self.clear_translation_fields()

    def on_translate(self):
        text = self.txt_input.get('1.0', 'end').strip()
        if text:
            self.txt_output.delete('1.0', 'end')
            self.txt_output.insert('end', translate_full_text_logic(text, self.direction))

#    def on_translate_file(self):
#        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
#        if file_path:
#            with open(file_path, 'r', encoding='utf-8') as f:
#                content = f.read()
#            self.txt_output.delete('1.0', 'end')
#            self.txt_output.insert('end', translate_full_text_logic(content, self.direction))

    def on_translate_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Получаем перевод
                translation = translate_full_text_logic(content, self.direction)
                # Запрашиваем путь для сохранения
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".txt", 
                    filetypes=[("Text files", "*.txt")],
                    title="Сохранить перевод как"
                )
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(translation)
                    messagebox.showinfo("Готово", f"Перевод сохранен в {save_path}")  
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при обработке файла: {e}")

    def clear_translation_fields(self):
        self.txt_input.delete('1.0', 'end')
        self.txt_output.delete('1.0', 'end')


# --- 4. МОДУЛЬ КАТАЛОГИЗАТОРА ---

class LibraryModule:
    def __init__(self, container, main_root):
        self.root = container
        self.main_root = main_root
        self.col_configs = {
            'Title': ('Название', 100),
            'Annotation': ('Аннотация', 150),
            'Code': ('Код', 40),
            'Author': ('Автор', 100),
            'Location': ('Расположение', 100)
        }
        self.cur_blob = None
        self.editing_id = None
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        # Верхняя панель с управлением
        top_frame = tk.Frame(self.root)
        top_frame.pack(side='top', fill='x')
        control_frame = tk.Frame(top_frame)
        control_frame.pack(fill='x')

        # Поиск
        search_frame = tk.Frame(control_frame)
        search_frame.pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(search_frame, text="Поиск:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', self.update_filter)
        self.entry_search = tk.Entry(search_frame, textvariable=self.search_var, width=32)
        self.entry_search.pack(side=tk.LEFT, padx=5)

        tk.Button(search_frame, text="x", command=lambda: self.entry_search.delete(0, tk.END)).pack(side=tk.LEFT)

        # Кнопки управления
        self.btn_view_toggle = tk.Button(control_frame, text="Вид ▼", command=self.toggle_column_visibility)
        self.btn_view_toggle.pack(side='left', padx=5)

        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side='left', padx=10)
        tk.Button(btn_frame, text="Добавить", command=self.open_add_book_window).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Редактировать", command=self.open_edit_book_window).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Удалить", command=self.delete_selected_book).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Копировать", command=self.copy_selected_book).pack(side='left', padx=2)
        tk.Button(btn_frame, text="Дополнительно", command=self.open_options_window).pack(side='left', padx=2)

        # Панель отображения колонок
        self.columns_frame = tk.Frame(top_frame, bg="#eee", bd=1, relief='groove')
        self.column_vars = {}
        for col_id, (col_name, col_width) in self.col_configs.items():
            var = tk.BooleanVar(value=True)
            self.column_vars[col_id] = var
            tk.Checkbutton(self.columns_frame, text=col_name, variable=var, bg="#eee", command=self.apply_column_visibility).pack(side='left', padx=5)

        # Основной контейнер
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)

        # Левая часть — дерево
        left_frame = tk.Frame(main_frame, width=350)
        left_frame.pack(side='left', fill='y')
        left_frame.pack_propagate(False)

        tree_frame = tk.Frame(left_frame)
        tree_frame.pack(fill='both', expand=True)

        self.vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        self.hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        self.tree = ttk.Treeview(
            tree_frame, columns=list(self.col_configs.keys()), show='headings',
            yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set
        )
        self.vsb.pack(side='right', fill='y')
        self.hsb.pack(side='bottom', fill='x')
        self.tree.pack(fill='both', expand=True)

        self.vsb.config(command=self.tree.yview)
        self.hsb.config(command=self.tree.xview)

        for col_id, (col_name, col_width) in self.col_configs.items():
            self.tree.heading(col_id, text=col_name)
            self.tree.column(col_id, width=col_width)

        self.tree.bind('<<TreeviewSelect>>', self.on_tree_selection)

        # Правая часть — детали
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        detail_inner_frame = tk.Frame(right_frame)
        detail_inner_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.photo_label = tk.Label(detail_inner_frame)
        self.photo_label.pack(side=tk.LEFT, padx=1, anchor=tk.N)

        info_frame = tk.Frame(detail_inner_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.title_var = tk.StringVar()
        self.annotation_var = tk.StringVar()
        self.code_var = tk.StringVar()
        self.author_var = tk.StringVar()
        self.location_var = tk.StringVar()

        # Поля отображения
        labels = [("Название:", self.title_var), ("Аннотация:", self.annotation_var), 
                  ("Код:", self.code_var), ("Автор:", self.author_var), ("Расположение:", self.location_var)]
        
        for i, (txt, var) in enumerate(labels):
            tk.Label(info_frame, text=txt, font=('Arial', 10, 'bold')).pack(anchor=tk.W)
            if txt == "Аннотация:":
                tk.Label(info_frame, textvariable=var, wraplength=350, justify=tk.LEFT).pack(anchor=tk.W, pady=(0,10))
            else:
                tk.Label(info_frame, textvariable=var).pack(anchor=tk.W, pady=(0,10))

        #кнопка перевода анотации
        tk.Button(right_frame, text="Перевести аннотацию", bg="#66b3ff", command=self.pop_translate_annotation).pack(pady=20)

    def toggle_column_visibility(self):
        if self.columns_frame.winfo_viewable():
            self.columns_frame.pack_forget()
            self.btn_view_toggle.config(text="Вид ▼")
        else:
            self.columns_frame.pack(side='top', fill='x')
            self.btn_view_toggle.config(text="Вид ▲")

    def apply_column_visibility(self):
        visible_columns = [c for c in self.col_configs if self.column_vars[c].get()]
        self.tree["displaycolumns"] = visible_columns

    def load_data(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        cursor.execute("SELECT id, title, annotation, code, author, location FROM books")
        for r in cursor.fetchall():
            self.tree.insert('', 'end', iid=r[0], values=r[1:])

    def on_tree_selection(self, event):
        selected_item = self.tree.selection()
        if not selected_item:
            self.clear_details()
            return
        book_id = selected_item[0]
        cursor.execute("SELECT title, annotation, code, photo, author, location FROM books WHERE id=?", (book_id,))
        row = cursor.fetchone()
        if row:
            title, annotation, code, photo_blob, author, location = row
            self.title_var.set(title)
            self.annotation_var.set(annotation)
            self.code_var.set(code)
            self.author_var.set(author if author else "")
            self.location_var.set(location if location else "")
            if photo_blob:
                try:
                    img = Image.open(io.BytesIO(photo_blob))
                    img = img.resize((150, 200), resample=Image.Resampling.LANCZOS)
                    self.photo_image = ImageTk.PhotoImage(img)
                    self.photo_label.config(image=self.photo_image)
                except Exception:
                    self.photo_label.config(image='')
            else:
                self.photo_label.config(image='')

    def clear_details(self):
        for v in [self.title_var, self.annotation_var, self.code_var, self.author_var, self.location_var]:
            v.set('')
        self.photo_label.config(image='')

    def open_add_book_window(self):
        self.editing_id = None
        self.show_edit_book_window()

    def open_edit_book_window(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Редактировать", "Выберите книгу для редактирования")
            return
        self.editing_id = sel[0]
        cursor.execute("SELECT title, annotation, code, photo, author, location FROM books WHERE id=?", (self.editing_id,))
        data = cursor.fetchone()
        self.show_edit_book_window(data)

    def show_edit_book_window(self, data=None):
        w = tk.Toplevel(self.root)
        w.title("Книга")
        w.grab_set()

        fields = [("Название", 0), ("Код", 2), ("Автор", 4), ("Расположение", 5)]
        self.entries = {}
        for txt, idx in fields:
            tk.Label(w, text=txt).pack(anchor='w')
            e = tk.Entry(w, width=45)
            e.pack(padx=5, pady=2)
            if data:
                e.insert(0, data[idx] if data[idx] else "")
            self.entries[txt] = e

        tk.Label(w, text="Аннотация").pack(anchor='w')
        self.e_ann = tk.Text(w, height=4, width=45)
        self.e_ann.pack(padx=5, pady=2)
        if data:
            self.e_ann.insert('1.0', data[1] if data[1] else "")

        self.cur_blob = data[3] if data else None
        self.p_prev = tk.Label(w)
        self.p_prev.pack(pady=5)
        tk.Button(w, text="Фото", command=self.load_photo).pack()
        if self.cur_blob:
            self._update_photo_preview()

        def save():
            title = self.entries["Название"].get().strip()
            annotation = self.e_ann.get('1.0', 'end').strip()
            code = self.entries["Код"].get().strip()
            author = self.entries["Автор"].get().strip()
            location = self.entries["Расположение"].get().strip()
            photo_blob = self.cur_blob

            if not title:
                messagebox.showwarning("Ошибка", "Введите название книги")
                return

            if self.editing_id:
                cursor.execute(
                    "UPDATE books SET title=?, annotation=?, code=?, author=?, location=?, photo=? WHERE id=?",
                    (title, annotation, code, author, location, photo_blob, self.editing_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO books (title, annotation, code, author, location, photo) VALUES (?,?,?,?,?,?)",
                    (title, annotation, code, author, location, photo_blob)
                )
            conn.commit()
            self.load_data()
            w.destroy()

        tk.Button(w, text="Сохранить", command=save).pack(side='left', padx=20, pady=10)
        tk.Button(w, text="Отмена", command=w.destroy).pack(side='left', pady=10)

    def load_photo(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            with open(path, 'rb') as f:
                self.cur_blob = f.read()
            self._update_photo_preview()

    def _update_photo_preview(self):
        img = Image.open(io.BytesIO(self.cur_blob)).resize((100, 130), Image.Resampling.LANCZOS)
        self.photo_preview_img = ImageTk.PhotoImage(img)
        self.p_prev.config(image=self.photo_preview_img)

    def delete_selected_book(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Удалить", "Выберите книгу для удаления")
            return
        book_id = selected_item[0]
        answer = messagebox.askyesno("Подтверждение", "Вы действительно хотите удалить выбранную книгу?")
        if answer:
            cursor.execute("DELETE FROM books WHERE id=?", (book_id,))
            conn.commit()
            self.load_data()
            self.clear_details()

    def update_filter(self, *args):
        search_text = self.search_var.get().lower()
        for row in self.tree.get_children():
            self.tree.delete(row)
        cursor.execute("SELECT id, title, annotation, code, author, location FROM books")
        for row in cursor.fetchall():
            vals = [str(x).lower() for x in row[1:]]
            if any(search_text in v for v in vals):
                self.tree.insert('', tk.END, iid=row[0], values=row[1:])

    def copy_selected_book(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Копировать", "Выберите книгу для копирования")
            return
        cursor.execute("SELECT title, author, annotation, code, location FROM books WHERE id=?", (sel[0],))
        r = cursor.fetchone()
        if r:
            data_str = (
                f"Название: {r[0]}\n"
                f"Автор: {r[1]}\n"
                f"Анотация: {r[2]}\n"
                f"Код: {r[3]}\n"
                f"Полка: {r[4]}"
            )
            self.main_root.clipboard_clear()
            self.main_root.clipboard_append(data_str)
            messagebox.showinfo("Копировать", "Данные книги скопированы в буфер обмена")

    def open_options_window(self):
        o = tk.Toplevel(self.root)
        tk.Button(o, text="Статистика", width=25, command=self.show_stats).pack(pady=5, padx=20)
        tk.Button(o, text="Excel", width=25, command=self.export_to_excel).pack(pady=5)
        tk.Button(o, text="Бекап", width=25, command=self.create_backup).pack(pady=5)

    #def show_stats(self):
    #    cursor.execute("SELECT author, COUNT(*) FROM books GROUP BY author ")
    #    a_d = cursor.fetchall()
    #    cursor.execute("SELECT location, COUNT(*) FROM books GROUP BY location")
    #    l_d = cursor.fetchall()
    #    st = tk.Toplevel(self.root)
    #    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    #    if a_d:
    #        x, y = zip(*a_d)
    #        ax[0].bar(x, y)
    #        ax[0].set_title("Авторы")
    #    if l_d:
    #        x, y = zip(*l_d)
    #        ax[1].pie(y, labels=x, autopct='%1.1f%%')
    #        ax[1].set_title("Полки")
    #    plt.tight_layout()
    #    Canvas = FigureCanvasTkAgg(fig, master=st)
    #    Canvas.get_tk_widget().pack()

    def show_stats(self):
    # Берем только 5 самых частых авторов и 5 самых заполненных полок
        cursor.execute("SELECT author, COUNT(*) as c FROM books GROUP BY author ORDER BY c DESC LIMIT 5")
        a_d = cursor.fetchall()
        cursor.execute("SELECT location, COUNT(*) as c FROM books GROUP BY location ORDER BY c DESC LIMIT 5")
        l_d = cursor.fetchall()
        st = tk.Toplevel(self.root)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        if a_d:
           x, y = zip(*a_d)
           ax[0].barh(x, y, color='skyblue') # barh (горизонтальный) лучше для длинных имен
           ax[0].set_title("Топ-5 авторов")
        if l_d:
           x, y = zip(*l_d)
           ax[1].pie(y, labels=x, autopct='%1.0f%%')
           ax[1].set_title("Топ-5 полок")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=st)
        canvas.get_tk_widget().pack()

    def export_to_excel(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx")
        if filepath:
            wb = Workbook()
            ws = wb.active
            ws.append(["Название", "Автор", "Аннотация", "Код", "Полка"])
            cursor.execute("SELECT title, author, annotation, code, location FROM books")
            for r in cursor.fetchall():
                ws.append(r)
            wb.save(filepath)
            messagebox.showinfo("Экспорт", "Экспорт завершен")

    def create_backup(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            shutil.copy2('library.db', os.path.join(dir_path, 'library_backup.db'))
            messagebox.showinfo("Бекап", "Успешно")

    def pop_translate_annotation(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Перевод аннотации", "Выберите книгу для перевода аннотации")
            return
        bid = sel[0]

        window = tk.Toplevel(self.root)
        window.title("AI Перевод аннотации")
        window.geometry("850x550")
        window.grab_set()

        ctrl_frame = tk.Frame(window)
        ctrl_frame.pack(pady=15)
        txt_frame = tk.Frame(window)
        txt_frame.pack(fill='both', expand=True, padx=15)

        text_orig = tk.Text(txt_frame, height=15, width=40, font=("Arial", 10))
        text_orig.pack(side='left', fill='both', expand=True, padx=5)
        text_orig.insert('1.0', self.annotation_var.get())

        text_translated = tk.Text(txt_frame, height=15, width=40, font=("Arial", 10), bg="#f0f7ff")
        text_translated.pack(side='right', fill='both', expand=True, padx=5)

        def run_translation():
            src_text = text_orig.get('1.0', 'end').strip()
            mode = 'ru-en' if any('а' <= c <= 'я' for c in src_text.lower()) else 'en-ru'
            translated_text = translate_full_text_logic(src_text, mode)
            text_translated.delete('1.0', 'end')
            text_translated.insert('end', translated_text)

        def apply_translation():
            new_text = text_translated.get('1.0', 'end').strip()
            if new_text:
                cursor.execute("UPDATE books SET annotation=? WHERE id=?", (new_text, bid))
                conn.commit()
                self.load_data()
                self.annotation_var.set(new_text)
                window.destroy()

        tk.Button(ctrl_frame, text="Перевести", width=15, bg="#4caf50", fg="white", command=run_translation).pack(side='left', padx=10)
        tk.Button(ctrl_frame, text="Применить перевод", width=18, bg="#2196f3", fg="white", command=apply_translation).pack(side='left', padx=10)
        tk.Button(ctrl_frame, text="Закрыть", width=12, command=window.destroy).pack(side='left', padx=10)

# --- запуск ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = CombinedApp(main_root)
    main_root.mainloop()
    conn.close()