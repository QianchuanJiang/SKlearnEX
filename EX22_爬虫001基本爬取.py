import requests
# 导入bs库
from bs4 import BeautifulSoup


user_agent='Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
# 定制爬虫的header字典；
headers = {'User-Agent': user_agent}
r = requests.get('http://www.gov.cn/zhengce/content/2019-01/17/content_5358604.htm',headers=headers)
r.encoding='utf-8'
soup = BeautifulSoup(r.text, 'lxml', from_encoding='utf-8')
texts = soup.find_all('p')
for text in texts:
    print(text.string)