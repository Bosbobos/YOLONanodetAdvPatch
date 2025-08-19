import os
import json


def convert_bbox(points, img_width, img_height):
    # Извлекаем координаты углов
    x1, y1 = points[0]
    x2, y2 = points[1]

    # Рассчитываем границы bounding box
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)

    # Рассчитываем центр и размеры (нормализованные)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    center_x = (x_min + x_max) / (2 * img_width)
    center_y = (y_min + y_max) / (2 * img_height)

    return center_x, center_y, width, height


# Укажите путь к папке с JSON файлами
folder_path = "./second_dataset/ann"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        json_path = os.path.join(folder_path, filename)

        with open(json_path, 'r') as f:
            data = json.load(f)

        gun_found = False
        output_lines = []

        # Извлекаем размеры изображения
        img_width = data['size']['width']
        img_height = data['size']['height']

        # Ищем первый объект класса "gun"
        for obj in data['objects']:
            if obj['classTitle'] == "gun":
                # Извлекаем координаты
                exterior = obj['points']['exterior']
                center_x, center_y, width, height = convert_bbox(
                    exterior,
                    img_width,
                    img_height
                )

                # Форматируем в строку YOLO
                output_lines.append(f"0 {center_x} {center_y} {width} {height}")
                gun_found = True
                break  # Прерываем после первого найденного gun

        # Создаем TXT файл
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join('./labels/', txt_filename)

        with open(txt_path, 'w') as txt_file:
            if gun_found:
                txt_file.write("\n".join(output_lines))