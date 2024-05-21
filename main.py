import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from get_image import fetch_images


class ImageComparer:
    def __init__(self, root, data_dict, left_folder, right_folder,page_number):
        self.root = root
        # 不符合要求文件个数
        self.error_num = 0
        self.data_dict = data_dict
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.page_number = page_number
        # self.left_images = sorted(os.listdir(left_folder), key=lambda x: int(os.path.splitext(x)[0]))
        # self.right_images = sorted(os.listdir(right_folder), key=lambda x: int(os.path.splitext(x)[0]))

        # 获取左侧文件夹中符合要求的图片文件名
        self.left_images = self.filter_images(os.listdir(left_folder))
        # 获取右侧文件夹中符合要求的图片文件名
        self.right_images = self.filter_images(os.listdir(right_folder))

        # 移除左侧文件夹中与右侧不匹配的文件,即已经命名的文件
        self.left_images = self.left_images[self.error_num:]

        self.index = 0
        self.rename_flag=True;  #记录是正确还是错误,在点击下一个后进行重命名

        self.init_gui()

    def filter_images(self, filenames):
        # 过滤不符合要求的图片文件名
        filtered_images = []
        for filename in filenames:
            # 检查文件名是否符合“{数字}.png”格式
            if filename[:-4].isdigit() and filename.endswith(".png"):
                filtered_images.append(filename)
            else:
                self.error_num += 1
        # 按照文件名中的数字进行排序
        filtered_images.sort(key=lambda x: int(os.path.splitext(x)[0]))
        return filtered_images

    def show_correct_message(self):
        self.correct_window = tk.Toplevel(self.root)
        self.correct_window.geometry("200x100")
        self.correct_window.title("提示")

        label = tk.Label(self.correct_window, text="设置为正确")
        label.pack(pady=20)

        self.correct_window.after(500, self.correct_window.destroy)  # 一秒后关闭窗口

    def show_wrong_message(self):
        self.correct_window = tk.Toplevel(self.root)
        self.correct_window.geometry("200x100")
        self.correct_window.title("提示")

        label = tk.Label(self.correct_window, text="设置为错误")
        label.pack(pady=20)

        self.correct_window.after(500, self.correct_window.destroy)  # 一秒后关闭窗口

    def init_gui(self):
        self.root.title("YQYjin's 实习小帮手")
        # self.root.geometry("700x200")
        # self.root.resizable(False, False)

        # Frame for left image
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)
        self.left_image_label = ttk.Label(self.left_frame)
        self.left_image_label.pack()

        # Label for current image number on the left
        self.left_image_number_label = ttk.Label(self.left_frame, text="当前图片编号: 0")
        self.left_image_number_label.pack()

        # Frame for right image
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10)
        self.right_image_label = ttk.Label(self.right_frame)
        self.right_image_label.pack()

        # Label for current image number on the right
        self.right_image_number_label = ttk.Label(self.right_frame, text="当前图片编号: 0")
        self.right_image_number_label.pack()

        # Controls frame
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=10)

        # 设置行列初始值
        self.row_var = tk.StringVar(value="1")  # 设置初始值为 "1"
        self.col_var = tk.StringVar(value="1")  # 设置初始值为 "1"

        self.column_label = ttk.Label(self.controls_frame, text="列号")
        self.column_label.grid(row=0, column=1, padx=5)
        self.column_entry = ttk.Entry(self.controls_frame, textvariable=self.col_var)
        self.column_entry.grid(row=0, column=2, padx=5)

        self.row_label = ttk.Label(self.controls_frame, text="行号")
        self.row_label.grid(row=0, column=3, padx=5)
        self.row_entry = ttk.Entry(self.controls_frame, textvariable=self.row_var)
        self.row_entry.grid(row=0, column=4, padx=5)

        self.next_column_button = ttk.Button(self.controls_frame, text="下一列", command=self.next_column)
        self.next_column_button.grid(row=0, column=8, padx=5)

        # Controls frame 2
        self.controls_frame2 = ttk.Frame(self.root)
        self.controls_frame2.grid(row=2, column=0, columnspan=2, pady=10)

        self.correct_button = ttk.Button(self.controls_frame2, text="正确", command=self.mark_correct)
        self.correct_button.grid(row=0, column=1, padx=5)

        self.incorrect_button = ttk.Button(self.controls_frame2, text="错误", command=self.mark_incorrect)
        self.incorrect_button.grid(row=0, column=2, padx=5)

        self.next_button = ttk.Button(self.controls_frame2, text="下一个", command=self.next_image)
        self.next_button.grid(row=0, column=3, padx=5)

        # Display the first image
        self.display_image()

    def display_image(self):
        if self.index >= len(self.right_images):
            return

        while self.index < len(self.right_images):
            left_image_name = self.left_images[self.index]
            right_image_name = self.right_images[self.index]

            # 检查两个文件名是否都符合“{数字}.png”格式
            if left_image_name[:-4].isdigit() and right_image_name[:-4].isdigit():
                break
            else:
                self.index += 1

        left_image_path = os.path.join(self.left_folder, self.left_images[self.index])
        right_image_path = os.path.join(self.right_folder, self.right_images[self.index])

        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)

        left_image.thumbnail((200, 200))
        right_image.thumbnail((200, 200))

        left_photo = ImageTk.PhotoImage(left_image)
        right_photo = ImageTk.PhotoImage(right_image)

        self.left_image_label.config(image=left_photo)
        self.left_image_label.image = left_photo

        self.right_image_label.config(image=right_photo)
        self.right_image_label.image = right_photo

        # 获取左侧图片的编号并更新标签文本
        current_left_image_name = self.left_images[self.index]
        current_left_image_number = int(os.path.splitext(current_left_image_name)[0])
        self.left_image_number_label.config(text=f"当前图片编号: {current_left_image_number}")

        # 获取右侧图片的编号并更新标签文本
        current_right_image_name = self.right_images[self.index]
        current_right_image_number = int(os.path.splitext(current_right_image_name)[0])
        self.right_image_number_label.config(text=f"当前图片编号: {current_right_image_number}")

    def next_image(self):
        self.rename_image(self.rename_flag)
        self.index = (self.index + 1) % len(self.right_images)
        self.display_image()
        self.next_row()
        self.rename_flag=True
        

    def mark_correct(self):
        self.rename_flag=True
        self.show_correct_message()
        #self.rename_image(correct=True)

    def mark_incorrect(self):
        self.rename_flag=False
        self.show_wrong_message()
        #self.rename_image(correct=False)

    def rename_image(self, correct):
        print("已重命名")
        row = self.row_entry.get()
        column = self.column_entry.get()
        if row and column:
            current_image = self.right_images[self.index]
            number = int(os.path.splitext(current_image)[0])
            recognition_result = self.data_dict.get(number, '')
            prefix = '0' if correct else '1'
            new_name = f"{prefix}_{self.page_number}_{column}_{row}_{recognition_result}.png"
            new_image_path = os.path.join(self.right_folder, new_name)
            current_image_path = os.path.join(self.right_folder, current_image)
            os.rename(current_image_path, new_image_path)
            self.right_images[self.index] = new_name


    def next_row(self):
        current_row = int(self.row_entry.get())
        current_row += 1
        self.row_entry.delete(0, tk.END)
        self.row_entry.insert(0, str(current_row))

    def next_column(self):
        # 列号加一,行号重设为1
        current_column = int(self.column_entry.get())
        current_column += 1
        self.row_entry.delete(0, tk.END)
        self.row_entry.insert(0, str(1))
        self.column_entry.delete(0, tk.END)
        self.column_entry.insert(0, str(current_column))

def read_txt_file(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            parts = line.split(',')
            number = int(parts[1].split('：')[1])
            recognition_result = parts[2].split('：')[1]
            data_dict[number] = recognition_result
    return data_dict


def main():
    root = tk.Tk()

    txt_file_path = filedialog.askopenfilename(title="选择txt文件", filetypes=[("Text files", "*.txt")])
    if not txt_file_path:
        return

    data_dict = read_txt_file(txt_file_path)

    # 提取页号部分
    page_number_str = txt_file_path.split('_')[1]

    # 去掉前导零并转换为整数
    page_number = int(page_number_str.lstrip('0'))

    # 解析txt文件名，生成保存爬取图片的文件夹名称
    base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    folder_result = os.path.join(os.path.dirname(txt_file_path),
                                 f"{base_name.split('_')[0]}_{base_name.split('_')[1]}_result")

    fetch_images(txt_file_path, folder_result)

    # image_folder = filedialog.askdirectory(title="选择右侧图片文件夹")
    #
    # if not image_folder:
    #     return

    image_folder = os.path.join(os.path.dirname(txt_file_path),
                                 f"{base_name.split('_')[0]}_{base_name.split('_')[1]}")

    app = ImageComparer(root, data_dict, folder_result, image_folder,page_number)
    root.mainloop()

def test():
    txt_file = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_clustered.txt'  # 你的txt文件路径
    folder_result = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_result'  # 你的文件夹路径
    folder = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419'  # 你的文件夹路径
    root = tk.Tk()


    data_dict = read_txt_file(txt_file)


    app = ImageComparer(root, data_dict, folder_result, folder)
    root.mainloop()

if __name__ == "__main__":
    #test()
    main()
