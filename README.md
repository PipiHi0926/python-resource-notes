# python-resource-notes
Record some tools or packages that I find useful and helpful for my development.





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


## 🔻 UI (for analysis)
### ▶︎ [Gradio](https://github.com/gradio-app/gradio)

### ▶︎ [Streamlit](https://github.com/streamlit/streamlit)


##  UI Dashboard
### ▶︎ [Metabase](https://www.metabase.com/)

