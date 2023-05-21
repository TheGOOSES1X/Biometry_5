import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Функция для загрузки изображений из базы данных лиц ORL
def load_images(directory):
    images = []
    labels = []
    for foldername in os.listdir(directory):
        folderpath = os.path.join(directory, foldername)
        if not os.path.isdir(folderpath):
            continue
        label = int(foldername[1:])
        for filename in os.listdir(folderpath):
            if filename.endswith('.pgm'):
                imagepath = os.path.join(folderpath, filename)
                image = Image.open(imagepath).convert('L')  # Преобразование в оттенки серого
                images.append(np.array(image, dtype='uint8'))
                labels.append(label)
    return images, labels

# Функция для выполнения метода главных компонент (PCA) на базе данных лиц
def perform_pca(images, labels, n_components):
    X = np.array(images)
    X = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return pca, X_pca

# Функция для распознавания лица на основе метода главных компонент (PCA)
def recognize_face(test_image, pca, X_pca, images, labels, test_label):
    test_image = np.array(test_image, dtype='uint8').flatten()
    test_image_pca = pca.transform([test_image])

    min_distance = float('inf')
    min_index = -1
    for i, image_pca in enumerate(X_pca):
        distance = np.linalg.norm(test_image_pca - image_pca)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    recognized_image = images[min_index]
    recognized_label = labels[min_index]

    # Запись точности распознавания в список accuracy и обновление графика точности
    accuracy.append(1 if recognized_label == test_label else 0)
    update_accuracy_plot()

    return recognized_image, recognized_label

# Функция для выбора случайного тестового изображения
def choose_random_test_image():
    if len(test_images) == 0:
        return

    index = random.randint(0, len(test_images) - 1)
    test_image = test_images[index]
    test_label = test_labels[index]

    recognized_image, recognized_label = recognize_face(test_image, pca, X_pca, images, labels, test_label)
    show_results(test_image, recognized_image, recognized_label)

    # Удаление использованного тестового изображения
    test_images.pop(index)
    test_labels.pop(index)

    # Проверка наличия тестовых изображений
    if len(test_images) == 0:
        random_button.config(state='disabled')
        start_button.config(state='disabled')

# Функция для обновления графика точности
def update_accuracy_plot():
    total_accuracy = accuracy + manual_accuracy
    accuracy_plot.clear()
    accuracy_plot.plot(range(len(total_accuracy)), total_accuracy)
    accuracy_plot.set_xlabel('Тестовое изображение')
    accuracy_plot.set_ylabel('Точность')
    accuracy_plot.set_title('Точность распознавания\nСредняя точность: {:.2f}%'.format(np.mean(total_accuracy) * 100))
    fig_canvas.draw()

# Функция для отображения результатов
def show_results(test_image, recognized_image, recognized_label):
    selected_photo_ax.clear()
    recognized_photo_ax.clear()

    selected_photo_ax.imshow(test_image, cmap='gray')
    selected_photo_ax.set_title('Тестовое изображение')
    selected_photo_ax.axis('off')  # Удаление осей

    recognized_photo_ax.imshow(recognized_image, cmap='gray')
    recognized_photo_ax.set_title(f'Распознано как s{recognized_label}')
    recognized_photo_ax.axis('off')  # Удаление осей

    fig_canvas.draw()

# Функция для выбора изображения из проводника
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.pgm")])
    if file_path:
        test_label = int(os.path.basename(os.path.dirname(file_path))[1:])
        image = Image.open(file_path).convert('L')
        test_image = np.array(image, dtype='uint8').flatten()
        test_image_pca = pca.transform([test_image])

        recognized_image, recognized_label = perform_recognition_for_choosen_image(test_image_pca, images, labels, test_label)
        show_results(image, recognized_image, recognized_label)

        accuracy.append(1 if recognized_label == test_label else 0)
        update_accuracy_plot()


# Функция для выполнения распознавания лица на основе PCA для выбранного вручную изображения
def perform_recognition_for_choosen_image(test_image_pca, images, labels, test_label):
    min_distance = float('inf')
    min_index = -1
    for i, image_pca in enumerate(X_pca):
        distance = np.linalg.norm(test_image_pca - image_pca)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    recognized_image = images[min_index]
    recognized_label = labels[min_index]

    return recognized_image, recognized_label

# Создание главного окна Tkinter
root = tk.Tk()
root.title('PCA распознавание лиц')

# Загрузка базы данных лиц ORL
image_directory = 'Faces'
images, labels = load_images(image_directory)

# Выбор случайных изображений для тестирования
test_images = []
test_labels = []
for i in range(51):
    index = random.randint(0, len(images) - 1)
    test_images.append(images.pop(index))
    test_labels.append(labels.pop(index))

# Выполнение метода главных компонент (PCA) на базе данных лиц
n_components_pca = 40  # Количество главных компонент для PCA
pca, X_pca = perform_pca(images, labels, n_components_pca)

# Создание фигуры для отображения изображений и графика точности
fig = plt.figure(figsize=(12, 4))  # Увеличение ширины фигуры
fig.subplots_adjust(wspace=0.3)  # Увеличение расстояния между окнами
selected_photo_ax = fig.add_subplot(131)
recognized_photo_ax = fig.add_subplot(132)
accuracy_plot = fig.add_subplot(133)

# Создание холста для отображения фигуры
fig_canvas = FigureCanvasTkAgg(fig, master=root)
fig_canvas.get_tk_widget().pack()

# Создание переменной accuracy для записи точности распознавания
accuracy = []
manual_accuracy = []

# Функция для запуска распознавания с заданными паузами
def start_recognition():
    random_button.config(state='disabled')
    for i in range(51):
        root.after(i * 500, choose_random_test_image)
    total_accuracy = accuracy + manual_accuracy
    average_accuracy = np.mean(total_accuracy) * 100

# Создание кнопки для выбора случайного изображения
random_button = tk.Button(root, text='Случайное изображение', command=choose_random_test_image)
random_button.pack(pady=10)

# Создание кнопки для выбора изображения из проводника
choose_button = tk.Button(root, text='Выбор изображения вручную', command=choose_image)
choose_button.pack(pady=10)

# Создание кнопки для автоматического запуска распознавания
start_button = tk.Button(root, text='Автоматическое случайное распознавание', command=start_recognition)
start_button.pack(pady=10)

# Запуск главного цикла событий Tkinter
root.mainloop()
