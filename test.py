import traceback

import requests
from lxml import etree
import os
import shutil
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_images(file_path, dir_name='results', base_url='http://anakv.com/', max_workers=5):
    # 如果结果文件夹存在，删除它
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    # 初始化一个空字典来存储编号和识别结果
    data_dict = {}

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行末的换行符和多余的空格
            line = line.strip()

            # 按照指定格式解析行内容
            parts = line.split(',')
            # 获取编号
            index = int(parts[1].split('：')[1])

            # 获取识别结果
            result = -parts[2].split('：')[1]
            # print("编号：", index, "识别结果：", result)
            # 将编号和识别结果存入字典
            data_dict[index] = result

    # 多线程爬取图片
    def fetch_image(key, result):
        # 替换字符
        temp_result = result.replace('ū', 'v').replace('š', 'x').replace('ž', 'z')

        ua = UserAgent()
        # 定义头信息，包括User-Agent
        headers = {
            'User-Agent': ua.random,
            'Referer': base_url,
            'Origin': base_url
        }
        # 请求参数
        data = {
            'input': temp_result,
            'font': 1
        }
        # 发送 POST 请求获取图片 URL
        response = requests.post(url=base_url, data=data, headers=headers)
        response.encoding = 'utf-8'
        img_src = response.json().get('img', '')  # 假设返回的 JSON 中有个 'img' 键存放图片 URL

        if img_src:
            # 创建存储图片的文件夹
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            # 爬取图片
            image = requests.get(url=base_url + img_src).content
            with open(os.path.join(dir_name, f'{key}.png'), 'wb') as fp:
                fp.write(image)
            print(f"编号：{key}, 识别结果：{result}, 爬取成功！")
        else:
            print(f"编号：{key}, 识别结果：{result}, 未找到图片链接！")

    # 使用线程池进行并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key, result in data_dict.items():
            future = executor.submit(fetch_image, key, result)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                # traceback.print_exc()
                print(f"爬取出错：{e}")


# 示例调用
if __name__ == "__main__":
    txt_file = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_clustered.txt'  # 你的txt文件路径
    folder = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_test'  # 你的文件夹路径
    folder_result = 'G:\\Homeworks\\实习\\day2\\21011642-吴锦耀\\Image_419_result'  # 你的文件夹路径
    fetch_images(txt_file, folder_result)
