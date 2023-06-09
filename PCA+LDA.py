import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

# Функция для разделения изображений на тестовые и распознаваемые изображения
def split_images(images, labels, test_ratio):
    test_images = []
    test_labels = []
    recognized_images = []
    recognized_labels = []
    for i in range(len(images)):
        if i % 10 < test_ratio:
            test_images.append(images[i])
            test_labels.append(labels[i])
        else:
            recognized_images.append(images[i])
            recognized_labels.append(labels[i])
    return test_images, test_labels, recognized_images, recognized_labels

# Функция для выполнения методов главных компонент (PCA) и линейного дискриминантного анализа (LDA) на базе данных лиц
def perform_pca_lda(images, labels, n_components_pca, n_components_lda):
    X = np.array(images)
    X = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=n_components_pca)
    X_pca = pca.fit_transform(X)

    lda = LDA(n_components=n_components_lda)
    X_lda = lda.fit_transform(X_pca, labels)

    return pca, lda, X_pca, X_lda

# Функция для распознавания лица на основе методов главных компонент (PCA) и линейного дискриминантного анализа (LDA)
def recognize_face(test_image, pca, lda, X_pca, X_lda, images, labels, test_label):
    test_image = np.array(test_image, dtype='uint8').flatten()
    test_image_pca = pca.transform([test_image])
    test_image_lda = lda.transform(test_image_pca)

    min_distance = float('inf')
    min_index = -1
    for i, image_lda in enumerate(X_lda):
        distance = np.linalg.norm(test_image_lda - image_lda)
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

    recognized_image, recognized_label = recognize_face(test_image, pca, lda, X_pca, X_lda, recognized_images, recognized_labels, test_label)
    show_results(test_image, recognized_image, recognized_label)

    # Удаление использованного тестового изображения
    test_images.pop(index)
    test_labels.pop(index)

    # Проверка наличия тестовых изображений
    if len(test_images) == 0:
        random_button.config(state='disabled')

# Функция для обновления графика точности распознавания
def update_accuracy_plot():
    accuracy_plot.clear()
    accuracy_plot.plot(range(len(accuracy)), accuracy)
    accuracy_plot.set_xlabel('Тестовое изображение')
    accuracy_plot.set_ylabel('Точность')
    accuracy_plot.set_title('Точность распознавания')

    # Вычисление среднего значения точности в процентах
    average_accuracy = calculate_average_accuracy()
    accuracy_plot.text(0, 1.1, f'Средняя точность: {average_accuracy:.2f}%', transform=accuracy_plot.transAxes, fontsize=12)

    fig_canvas.draw()

# Функция для вычисления среднего значения точности в процентах
def calculate_average_accuracy():
    total_tests = len(accuracy)
    correct_predictions = sum(accuracy)
    average_accuracy = (correct_predictions / total_tests) * 100
    return average_accuracy

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

# Создание главного окна Tkinter
root = tk.Tk()
root.title('PCA+LDA распознавание лиц')

# Загрузка базы данных лиц ORL
image_directory = 'Faces'
images, labels = load_images(image_directory)

# Получение от пользователя количества тестовых изображений
test_ratio = int(input('Введите количество тестовых изображений (от 1 до 8): '))
test_images, test_labels, recognized_images, recognized_labels = split_images(images, labels, test_ratio)

# Выполнение методов главных компонент (PCA) и линейного дискриминантного анализа (LDA) на базе данных лиц
n_components_pca = 40  # Количество главных компонент для PCA
n_components_lda = 39  # Количество главных компонент для LDA
pca, lda, X_pca, X_lda = perform_pca_lda(recognized_images, recognized_labels, n_components_pca, n_components_lda)

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

# Функция для выбора изображения из проводника
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.pgm")])
    if file_path:
        image = Image.open(file_path).convert('L')  # Преобразование в оттенки серого
        test_label = int(os.path.basename(os.path.dirname(file_path))[1:])
        test_image = np.array(image, dtype='uint8').flatten()
        test_image_pca = pca.transform([test_image])
        test_image_lda = lda.transform(test_image_pca)

        recognized_image, recognized_label = perform_recognition_for_choosen_image(test_image_lda, X_lda, recognized_images, recognized_labels, -1)
        show_results(image, recognized_image, recognized_label)
        accuracy.append(1 if recognized_label == test_label else 0)
        update_accuracy_plot()

# Функция для распознавания лица на основе методов PCA и LDA для выбранного изображения (24/8 - демострация повышения точности)
def perform_recognition_for_choosen_image(test_image_lda, X_lda, images, labels, test_label):
    min_distance = float('inf')
    min_index = -1
    for i, image_lda in enumerate(X_lda):
        distance = np.linalg.norm(test_image_lda - image_lda)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    recognized_image = images[min_index]
    recognized_label = labels[min_index]

    return recognized_image, recognized_label

# Функция для запуска распознавания с заданными паузами
def start_recognition():
    random_button.config(state='disabled')
    for i in range(len(test_images)):
        root.after(i * 500, choose_random_test_image)

# Создание кнопки для выбора случайного изображения
random_button = tk.Button(root, text='Случайное изображение', command=choose_random_test_image)
random_button.pack(pady=10)

# Создание кнопки для выбора изображения
choose_image_button = tk.Button(root, text='Выбор изображения вручную', command=choose_image)
choose_image_button.pack(pady=10)

# Создание кнопки для автоматического запуска распознавания
start_button = tk.Button(root, text='Автоматическое случайное распознавание', command=start_recognition)
start_button.pack(pady=10)

# Запуск главного цикла Tkinter
root.mainloop()
