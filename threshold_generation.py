import cv2
import numpy as np
import sympy
import statistics

image_name = input("Введите название исходного изображения (с расширением): ") # ввод пути к изображению
image = cv2.imread(image_name) # считываем изображение
image_size = image.size # присваиваем переменной значение в количество пикселей изображения
mean_rows = cv2.reduce(image, 0, cv2.REDUCE_AVG) # преобразуем матрицу изображения в вектор средних значений пикселей каждого ряда
mean_key = np.mean(mean_rows) # находим числовое среднее значение вектора, средний ключ
lower_bound = 0 # нижняя граница подсчета простых значений пикселей
upper_bound = image_size # верхняя граница и полное количество пикселей
prime_list = list(sympy.primerange(lower_bound, upper_bound + 1)) # список простых чисел в промежутке от lower до upper включительно
std_key = statistics.stdev(prime_list) # СКО на базе простых чисел, стандартный ключ
key_xor = int(round(mean_key)) ^ int(round(std_key)) # применение побитового исключающего ИЛИ к двум полученным ключам
p = 100 # идеальные значения
q = 50 # p и q
th_key = p + (key_xor % q)
print ('Пороговый ключ:', th_key)