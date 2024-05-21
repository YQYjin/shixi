import os
from get_image import fetch_images

# 读取txt文件并解析内容
def read_txt(txt_file):
    mapping = {}
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('：')
            if len(parts) == 4:
                page_num = parts[1].split(',')[0]
                identifier = parts[2].split(',')[0]
                result = parts[3]
                mapping[identifier] = result
    return mapping

# 重命名文件
def rename_files(mapping, folder):
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            file_id = filename.split('.')[0]
            if file_id in mapping:
                new_filename = f"{file_id}_{mapping[file_id]}.png"
                os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
                print(f"Renamed {filename} to {new_filename}")

# 主函数
def main():
    txt_file = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_clustered.txt'  # 你的txt文件路径
    folder = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_test'  # 你的文件夹路径
    folder_result = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_result'  # 你的文件夹路径
    mapping = read_txt(txt_file)
    rename_files(mapping, folder)
    fetch_images(txt_file,folder_result)

if __name__ == "__main__":
    main()