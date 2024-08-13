# python-resource-notes
Record some tools or packages that I find useful and helpful for my development.

There are many related tools, but I’m only listing the ones I use most frequently or that I personally prefer ❤️

If you want to learn more, I recommend checking out the following link:
https://github.com/vinta/awesome-python

## 🔻Deep Learning
### ▶︎ [Lime](https://github.com/marcotcr/lime) (local interpretable model-agnostic explanations)
- Develop explainable, interpretable deep learning models.

## 🔻Data Visualization
### ▶︎ [Plotly](https://plotly.com/python)
### ▶︎ [PyVis](https://towardsdatascience.com/pyvis-visualize-interactive-network-graphs-in-python-77e059791f01)
- Visualize interactive network graphs


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

