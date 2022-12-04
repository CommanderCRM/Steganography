import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters


def messageToBinary(message):  # преобразование сообщения в битовую строку
    if type(message) == str:
        return ''.join([format(ord(i), "08b") for i in message])
    elif type(message) == bytes or type(message) == np.ndarray:
        return [format(i, "08b") for i in message]
    elif type(message) == int or type(message) == np.uint8:
        return format(message, "08b")
    else:
        raise TypeError("Тип ввода не поддерживается")


def hideData(image, secret_message):
    n_bytes = image.shape[0] * image.shape[1] * 3 // 8
    encode_type = input(
        "Размер контейнера \n 1. 20% контейнера \n 2. 40% контейнера \n 3. 60% контейнера \n 4. 80% контейнера \n 5. 100% контейнера \n Ваш выбор: ")
    userinput = int(encode_type)
    if (userinput == 1):
        n_bytes = round(n_bytes * 0.2)
    elif (userinput == 2):
        n_bytes = round(n_bytes * 0.4)
    elif (userinput == 3):
        n_bytes = round(n_bytes * 0.6)
    elif (userinput == 4):
        n_bytes = round(n_bytes * 0.8)
    elif (userinput == 5):
        n_bytes = n_bytes * 1
    print("Максимальное количество битов для встраивания:", n_bytes)
    print("Количество битов встраиваемой информации:", len(secret_message))
    if len(secret_message) > n_bytes:
        raise ValueError(
            "Нужно большее по размеру изображение или меньшее сообщение")
    else:
        print("\nВстраивание....")

    secret_message += "#####"

    data_index = 0

    binary_secret_msg = messageToBinary(secret_message)

    data_len = len(binary_secret_msg)
    for values in image:  # встраивание строки в младшие биты пикселей
        for pixel in values:
            r, g, b = messageToBinary(pixel)
            if data_index < data_len:
                pixel[0] = int(r[:-1] + binary_secret_msg[data_index], 2)
                data_index += 1
            if data_index < data_len:
                pixel[1] = int(g[:-1] + binary_secret_msg[data_index], 2)
                data_index += 1
            if data_index < data_len:
                pixel[2] = int(b[:-1] + binary_secret_msg[data_index], 2)
                data_index += 1
            if data_index >= data_len:
                break

    return image


def showData(image):  # расшифровка битовой строки

    binary_data = ""
    for values in image:
        for pixel in values:
            r, g, b = messageToBinary(pixel)
            binary_data += r[-1]
            binary_data += g[-1]
            binary_data += b[-1]
    all_bytes = [binary_data[i: i+8] for i in range(0, len(binary_data), 8)]
    decoded_data = ""
    for byte in all_bytes:
        decoded_data += chr(int(byte, 2))
        if decoded_data[-5:] == "#####":
            break
    return decoded_data[:-5]


def encode_text():
    image_name = input(
        "Введите название исходного изображения (с расширением): ")
    image = cv2.imread(image_name)

    print("Размер изображения: ", image.shape)

    data = input("Введите информацию для встраивания: ")
    if (len(data) == 0):
        raise ValueError('Информации нет')

    filename = input("Введите название нового изображения (с расширением): ")
    encoded_image = hideData(image, data)
    cv2.imwrite(filename, encoded_image)


def decode_text():
    image_name = input(
        "Введите название изображения, из которого вы хотите извлечь информацию (с расширением): ")
    print("\nИзвлечение....")
    image = cv2.imread(image_name)

    text = showData(image)
    return text


def PSNR():
    image_name_1 = input(
        "Введите название первого изображения (с расширением): ")
    image_name_2 = input(
        "Введите название второго изображения (с расширением): ")
    im1 = cv2.imread(image_name_1)
    im2 = cv2.imread(image_name_2)
    psnr = cv2.PSNR(im1, im2)  # расчет PSNR функцией библиотеки OpenCV
    print("Значение PSNR: ", psnr, "dB")


def Canny():
    image_name = input("Введите название изображения (с расширением): ")
    img = cv2.imread(image_name, 0)

    edges = cv2.Canny(img, 100, 200)  # детектор границ Кэнни через OpenCV

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Края'), plt.xticks([]), plt.yticks([])

    plt.show()

    save_case = input("Сохранить изображение с краями? (Y/N): ")
    userinput = str(save_case)
    if (userinput == 'Y'):
        image_for_save = input(
            "Введите название изображения для сохранения (с расширением): ")
        cv2.imwrite(image_for_save, edges)
        print("Изображение сохранено")
    elif (userinput == 'N'):
        return ()
    else:
        raise Exception("Выберите правильный вариант")


def Laplace():
    ddepth = cv2.CV_16S
    kernel_size = 3

    image_name = input("Введите название изображения (с расширением): ")
    img = cv2.imread(image_name)

    # применение гауссова размытия
    img_blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # перевод оригинального трехканального изображения в одноканальное серое
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    # применение оператора Лапласа
    img_dst = cv2.Laplacian(img_gray, ddepth, ksize=kernel_size)
    # перевод изображения из одноканального серого в оригинальное трехканальное
    img_abs_dst = cv2.convertScaleAbs(img_dst)

    plt.subplot(121), plt.imshow(img_gray, cmap='gray')
    plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_abs_dst, cmap='gray')
    plt.title('Края'), plt.xticks([]), plt.yticks([])

    plt.show()

    save_case = input("Сохранить изображение с краями? (Y/N): ")
    userinput = str(save_case)
    if (userinput == 'Y'):
        image_for_save = input(
            "Введите название изображения для сохранения (с расширением): ")
        cv2.imwrite(image_for_save, img_abs_dst)
        print("Изображение сохранено")
    elif (userinput == 'N'):
        return ()
    else:
        raise Exception("Выберите правильный вариант")


def Prewitt():
    image_name = input("Введите название изображения (с расширением): ")
    img = cv2.imread(image_name, 0)

    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)  # гауссово размытие
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # матрица по x
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # матрица по y
    img_prewittx = cv2.filter2D(
        img_gaussian, -1, kernelx)  # линейный фильтр по x
    img_prewitty = cv2.filter2D(
        img_gaussian, -1, kernely)  # линейный фильтр по y

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_prewittx + img_prewitty, cmap='gray')
    plt.title('Края'), plt.xticks([]), plt.yticks([])

    plt.show()

    save_case = input("Сохранить изображение с краями? (Y/N): ")
    userinput = str(save_case)
    if (userinput == 'Y'):
        image_for_save = input(
            "Введите название изображения для сохранения (с расширением): ")
        cv2.imwrite(image_for_save, img_prewittx + img_prewitty)
        print("Изображение сохранено")
    elif (userinput == 'N'):
        return ()
    else:
        raise Exception("Выберите правильный вариант")


def Dilation():

    image_name = input("Введите название изображения (с расширением): ")
    img = cv2.imread(image_name, 0)

    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dilated_img, cmap='gray')
    plt.title('Наращивание'), plt.xticks([]), plt.yticks([])

    plt.show()

    save_case = input("Сохранить изображение после наращивания? (Y/N): ")
    userinput = str(save_case)
    if (userinput == 'Y'):
        image_for_save = input(
            "Введите название изображения для сохранения (с расширением): ")
        cv2.imwrite(image_for_save, dilated_img)
        print("Изображение сохранено")
    elif (userinput == 'N'):
        return ()
    else:
        raise Exception("Выберите правильный вариант")


def Roberts():

    image_name = input("Введите название изображения (с расширением): ")
    img = cv2.imread(image_name, 0)

    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    img_robertsx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_robertsy = cv2.filter2D(img_gaussian, -1, kernely)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Оригинал'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_robertsx + img_robertsy, cmap='gray')
    plt.title('Края'), plt.xticks([]), plt.yticks([])

    plt.show()

    save_case = input("Сохранить изображение после наращивания? (Y/N): ")
    userinput = str(save_case)
    if (userinput == 'Y'):
        image_for_save = input(
            "Введите название изображения для сохранения (с расширением): ")
        cv2.imwrite(image_for_save, img_robertsx + img_robertsy)
        print("Изображение сохранено")
    elif (userinput == 'N'):
        return ()
    else:
        raise Exception("Выберите правильный вариант")


def Steganography():
    stego_type = input("Метод LSB \n 1. Встроить данные \n 2. Извлечь данные \n 3. Рассчитать PSNR \n 4. Детектор границ Кэнни \n 5. Детектор границ оператором Лапласа \n 6. Детектор границ оператором Прюитт \n 7. Наращивание матрицей 3х3 \n 8. Детектор границ оператором Робертса \n Ваш выбор: ")
    userinput = int(stego_type)
    if (userinput == 1):
        encode_text()

    elif (userinput == 2):
        print("Извлеченное сообщение:  " + decode_text())

    elif (userinput == 3):
        PSNR()

    elif (userinput == 4):
        Canny()

    elif (userinput == 5):
        Laplace()

    elif (userinput == 6):
        Prewitt()

    elif (userinput == 7):
        Dilation()

    elif (userinput == 8):
        Roberts()

    else:
        raise Exception("Выберите правильный вариант")


Steganography()
