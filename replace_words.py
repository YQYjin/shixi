import os

def replace_special_chars(filename):
    # 定义特殊字符替换规则
    replacements = {'ū': 'v', 'š': 'x', 'ž': 'z'}
    for old_char, new_char in replacements.items():
        filename = filename.replace(old_char, new_char)
    return filename

def rename_files(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print("文件夹路径不存在！")
        return

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        old_filepath = os.path.join(folder_path, filename)
        if os.path.isfile(old_filepath):
            # 获取新文件名
            new_filename = replace_special_chars(filename)
            new_filepath = os.path.join(folder_path, new_filename)

            # 如果新文件名不同于旧文件名，则重命名文件
            if new_filename != filename:
                os.rename(old_filepath, new_filepath)
                print(f"重命名文件: {filename} -> {new_filename}")

if __name__ == "__main__":
    # 输入需要重命名的文件夹路径
    folder_path = "G:\\DAY4.5计算机\\计算机\\DAY4\\yourname\\Image_140"
    rename_files(folder_path)