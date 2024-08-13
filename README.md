# python-resource-notes
Record some tools or packages that I find useful and helpful for my development.

There are many related tools, but I’m only listing the ones I use most frequently or that I personally prefer ❤️

If you want to learn more, I recommend checking out the following link:
https://github.com/vinta/awesome-python

(我也放入一些並不直接跟python相關的內容)

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
### pickle (or joblib)
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

