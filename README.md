# AI Chatbot using Langchain Pinecone

<img width="1051" height="708" alt="image" src="https://github.com/user-attachments/assets/5a9ac747-747c-4957-af67-60b956a58d3c" />

## Deployment

#### 1. Clone Repository 

```bash
  git clone https://github.com/farzanaanjum/AI-Chatbot-Langchain-Pinecone/
```
```bash
  cd ai-chatbot-using-Langchain-Pinecone
```
#### 2. Create Virtual Environment
```bash
  python -m venv env
```
 - For Windows:
```bash
  .\env\Scripts\activate
```
 - For macOS/Linux:
```bash
  source env/bin/activate
```

#### 3. To install require packages 

```bash
  pip install -r requirements.txt
```
#### 4. Replace your own document in **data** folder

#### 5. Replace your own OpenAI, Pinecone API Key and Pinecone environment in indexing.py, main.py & utils.py
 - [OpenAI API Key](https://platform.openai.com)
 - [Pinecone](app.pinecone.io)

#### 6. When you are creating the pinecone index make sure,
   - **index_name = "langchain-chatbot"**
   - **Dimensions of the index is 384**
 
#### 7. Run the web app
```bash
  streamlit run main.py
```
