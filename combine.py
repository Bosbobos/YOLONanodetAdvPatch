from PIL import Image
import os


def combine_images_vertically(folder_path, output_path):
    # Получаем список всех файлов изображений в папке
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print("В папке нет изображений")
        return

    # Открываем все изображения
    images = [Image.open(os.path.join(folder_path, f)) for f in image_files]

    # Проверяем, что все изображения одного размера
    widths, heights = zip(*(i.size for i in images))

    if len(set(widths)) != 1:
        print("Предупреждение: не все изображения имеют одинаковую ширину. Результат может быть неожиданным.")

    # Вычисляем общую высоту и максимальную ширину
    total_height = sum(heights)
    max_width = max(widths)

    # Создаем новое изображение
    combined_image = Image.new('RGB', (max_width, total_height))

    # Вставляем изображения одно под другим
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    # Сохраняем результат
    combined_image.save(output_path)
    print(f"Изображения успешно объединены и сохранены в {output_path}")


# Пример использования
folder_path = 'imgs'  # Укажите путь к папке с изображениями
output_path = 'combined_patch_500.png'  # Укажите имя выходного файла
combine_images_vertically(folder_path, output_path)