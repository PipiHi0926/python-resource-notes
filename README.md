# python-resource-notes
Record some tools or packages that I find useful and helpful for my development.

There are many related tools, but I’m only listing the ones I use most frequently or that I personally prefer ❤️

If you want to learn more, I recommend checking out the following link:
https://github.com/vinta/awesome-python

(我也會放入一些並不直接跟python相關的內容)


## 🔻Deep Processing 
### ▶︎[ Dask](https://www.dask.org/)
- 提供多核心和分散式+並行執行功能
- 若資料龐大(大型向量、資料矩陣)，Dask會將其分成區塊，並將這些區塊分佈到電腦上的所有可用核心上
- 擴展了 pandas、NumPy 和 Spark 等傳統工具的功能，特別是當要處理巨量資料時!!!
- Dask的儀表板可以幫助你了解你工作程序的狀態
```
import dask.dataframe as dd

df = dd.read_csv(path_to_original_data)
```


## 🔻Deep Learning
### ▶︎ [Pytorch](https://github.com/pytorch/pytorch)
- 個人認為pytorch相對tensorflow更容易上手

### ▶︎ [Pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning?source=post_page-----81af12de9bb7--------------------------------)
- 基於Pytorch的高級深度學習框架,旨在簡化Pytorch的使用,讓研究人員能夠更專注於核心的研究代碼,而不是重複的樣板代碼
- 將訓練的各個步驟(初始化、訓練、驗證、測試)封裝成固定的流程,使用者只需要實現這些步驟對應的方法,而不需要關心訓練的細節 (懶人福音QQ)]
- 加快debug麻煩，免除cpu, gpu那些設定，不在需要處理變數與硬體之間的關係

### ▶︎ [Lime](https://github.com/marcotcr/lime) (local interpretable model-agnostic explanations)
- Develop explainable, interpretable deep learning models.

## 🔻Data Visualization
### ▶︎ [Plotly](https://plotly.com/python)
- 建議可結合 [Dash ](https://dash.plotly.com/)去實踐 Dashboard儀表板的網頁應用程式框架
- 結合Dash不用Javascript就能創造出互動性高的動態圖表
### ▶︎ [PyVis](https://towardsdatascience.com/pyvis-visualize-interactive-network-graphs-in-python-77e059791f01)
- Visualize interactive network graphs


## 🔻 OCR
### ▶︎ [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- 支持中文、數字辨別效果好(經驗)
- 簡單、效果好
```
# pip install easyocr

import easyocr

reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)
image_path = '路徑'
result = reader.readtext(image_path)

for (bbox, text, prob) in result:
    print(bbox)
    print(text)
    print(prob)
```
### ▶︎ [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- 純英文辨識能力佳(經驗)、但建議使用EasyOCR即可

## 🔻 NLP

### ▶︎ [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- based on Bert, transformer model
- 可以簡單實踐字串相似度比對(similarity compare)
- [model list](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
```python=
# pip install -U sentence-transformers

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("model name")

# 2. Prepare sentences to encode
sentences = [
    "This is sentence 1 example.",
    "This is sentence 2 example.",
]

# 3. Make prompt if you need
your_prompt = ""

# 4.  Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences, prompt=your_prompt)

# 5. Calculate the embedding similarities
model.similarity(query_embedding, passage_embeddings)
```



### ▶︎ [Hugging Face transformers ](https://github.com/huggingface/transformers)
- You can instantiate ```AutoModelForCausalLM``` model and  ```AutoTokenizer``` 
- https://huggingface.co/docs/transformers/llm_tutorial
- Make transformers pipeline easy





## 🔻LLM agent flow
### ▶︎ [Flowise](https://github.com/FlowiseAI/Flowise)
### ▶︎ [LangFlow](https://github.com/langflow-ai/langflow)

## 🔻API
### ▶︎ FastAPI
- 被認為是當前最快的 Python 框架之一，[易用且簡潔](https://medium.com/seaniap/%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B-%E7%B0%A1%E5%96%AE%E6%98%93%E6%87%82-python%E6%96%B0%E6%89%8B%E7%9A%84fastapi%E4%B9%8B%E6%97%85-ebd09dc0167b)
- 會自動生成互動式 API 文檔
- 提供了強大的類型檢查功能，可以與其他工具/套件整合 (待補)
```
# pip  install fastapi 
# pip install uvicorn # ASGI伺服器

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "FastAPI"}
```
- 使用uvicorn來啟動伺服器
```bash
# main 是您Python檔案名，app是的FastAPI實例

uvicorn main:app --reload
```

## 🔻 UI (for analysis)
### ▶︎ [Gradio](https://github.com/gradio-app/gradio)

### ▶︎ [Streamlit](https://github.com/streamlit/streamlit)


## 🔻 UI Dashboard
### ▶︎ [Metabase](https://www.metabase.com/)

## 🔻 Simple and handy tools
### ▶︎ tqdm 
- 印出迴圈執行進度
```
# pip install tqdm

from tqdm import tqdm
import time

# 把要跑的list用tqdm()包起來
for i in tqdm(range(100)):
    time.sleep(0.1)
```

### ▶︎ breakpoint() / pdb.set_trace()
- 在程式中加入後，當程式運行到此處就會暫停，並提供幾個輸入操作指令方便檢查
```
# Python 3.7 以前:
import pdb
pdb.set_trace()

# Python 3.7 之後:
breakpoint() # 加在你想斷點的地方
```

### ▶︎ pprint
- pretty-print，美化dict list, tuple的印出結果(不會擠在一起)
- 直接把print換成pprint即可實踐
- 跟print相關的還有可愛的冰淇淋([icecream](https://github.com/gruns/icecream))可以玩玩看XD

### ▶︎ mypy
- 輔助實踐type hint，可以命令執行檢查所有的n文件中的類型問題，提早報錯和強化類型檢查
- 雖然現在可以用copilot之類輔助，但還是建議可搭配進行靜態類型檢查
```
在命令提示字元使用mypy指令執行.py檔即可
# pip install mypy
mypy your_script.py
```
### ▶︎ pickle (or joblib)
- 保存各種模型、物件、自定義class的工具，讓對象能夠實踐序列化和反序列化
- 你也可以用來儲存、加載訓練好的機器學習模型
```
# 儲存方法
import pickle

data = [1,{2}] # 各種資料類型

# write with binary (wb) 到 data.pkl檔案
with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)

```
```
# 讀取方法
import pickle

# Open the file in binary read mode and load the data
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)

```


------------
## 🔻 VS Code Extension 
### ▶︎ GitHub Copilot
- 偉大，無須多言
### ▶︎ Black... (Formatter)
- 讓code撰寫風格符合PEP 8風格
- 可參考下面 Ruff


### ▶︎ pylint / [Ruff](https://github.com/astral-sh/ruff)
- 當你的程式有問題、不符合期望風格時，底下就會有毛毛蟲(標色波浪底線)輔助提醒
- 另外推薦Ruff，速度更快、是Python linter + formatter，且已整合更多跟程式風格規範所需的工具(Flake8, Black, isort, autoflake...)
- 可參考其他[網路文章](https://blog.kyomind.tw/ruff/)的Ruff介紹

### ▶︎ GitLens 
- 查看 git 紀錄的工具
- 多人協作專案時，可在 code 上直接看到這行最後的修改是誰改的

### ▶︎ Git管理系列: [Git Graph](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph)、[Git History](https://marketplace.visualstudio.com/items?itemName=donjayamanne.githistory)

### ▶︎ [Python Debugger](https://code.visualstudio.com/docs/python/debugging)




## 🔻 Proxy (代理工具)
### ▶︎[mitmproxy](https://mitmproxy.org/)
- 開源抓包工具
- 支持反向代理，將流量轉發到指定的服務器
- 可以與python進行交互，可以使用Python編寫腳本來自動化流量處理
- 自訂HTTP響應
```
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    flow.response = http.Response.make(
        204,
        '{"foo":"bar"}',
        {"Content-Type": "application/json"}
    )
```
------------
# 其他專案開發工具
## 🔻 與前端溝通
### ▶︎ Figma
- 我主要使用的工具，可以設計、拉出簡單的前端介面，並建立簡單的按鍵、互動、連結等
- 用來跟前端工程師說明預期內容
- 可以生產簡單樣板給客戶了解預期介面長相

### ▶︎ [vaadin](https://start.vaadin.com/app/p)
- java / React
- 建構網路應用程式和網站的Java Web 框架
