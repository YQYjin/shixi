import random
import time
import traceback

import requests
from docker.models.services import Service
from lxml import etree
import os
import shutil
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def fetch_images(file_path, dir_name='results', base_url='http://anakv.com/', max_workers=10):
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
            result = parts[2].split('：')[1]
            if result==None or result=="":
                continue
            # print("编号：", index, "识别结果：", result)
            # 将编号和识别结果存入字典
            data_dict[index] = result


    def fetch_image(key, result):
        # 替换字符
        temp_result = result.replace('ū', 'v').replace('š', 'x').replace('ž', 'z')

        ua = UserAgent()
        # 定义头信息，包括User-Agent

        headers = {
            'User-Agent': ua.random,
            #'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            'Referer': base_url,
            'Origin': base_url
        }

        # 请求参数
        data = {
            'input': temp_result,
            'font': 1
        }
        # print(data)
        # 获取图片src
        try:
            response = requests.post(url=base_url, data=data, headers=headers)
            # for i in range(2):
            #     response = requests.post(url=base_url, data=data, headers=headers)
            #     time.sleep(2)
            # print(response.text)
        except Exception as e:
            traceback.print_exc()
            print(f"编号：{key}, 识别结果：{result}, 获取失败！")
        response.encoding = 'utf-8'
        web_text = response.text
        tree = etree.HTML(web_text)
        #print(tree.xpath('/html/body/div/div[3]/center[1]/a/img/@src'))
        img_src = tree.xpath('/html/body/div/div[3]/center[1]/a/img/@src')[0]
        #print(img_src)
        # 创建存储图片的文件夹
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        # 获取图片
        image = requests.get(url=base_url + img_src).content
        with open(os.path.join(dir_name, f'{key}.png'), 'wb') as fp:
            fp.write(image)
        print(f"编号：{key}, 识别结果：{result}, 获取成功！")

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
                #traceback.print_exc()
                print(f"获取出错：{e}")
                continue


def fetch_images2(file_path, dir_name='results', base_url='http://anakv.com/', max_workers=10):
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
            result = parts[2].split('：')[1]
            if result==None or result=="":
                continue
            # print("编号：", index, "识别结果：", result)
            # 将编号和识别结果存入字典
            data_dict[index] = result


    def fetch_image(key, result):
        # 替换字符
        temp_result = result.replace('ū', 'v').replace('š', 'x').replace('ž', 'z')

        ua = UserAgent()
        # 定义头信息，包括User-Agent

        headers = {
            'User-Agent': ua.random,
            #'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            'Referer': base_url,
            'Origin': base_url
        }

        # 请求参数
        params = {
            'input': temp_result,
            'font': 1
        }

        # 创建存储图片的文件夹
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        # 获取图片
        image = requests.get(url=base_url + 'msc.php', params=params).content

        with open(os.path.join(dir_name, f'{key}.png'), 'wb') as fp:
            fp.write(image)
        print(f"编号：{key}, 识别结果：{result}, 获取成功！")




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
                #traceback.print_exc()
                print(f"获取出错：{e}")
                continue


# 示例调用
if __name__ == "__main__":
    txt_file = ''  # 你的txt文件路径
    folder = ''  # 你的文件夹路径
    folder_result = ''  # 你的文件夹路径
    fetch_images(txt_file, folder_result)
