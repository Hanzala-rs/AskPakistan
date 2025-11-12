import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pyngrok import ngrok

logging.disable(logging.CRITICAL)

os.environ["GOOGLE_API_KEY"] = "AIzaSyDlJvbl6wsyawY_MPIXaiOSUCmpZUZT9WM"

Settings.llm = GoogleGenAI(model="gemini-2.5-flash")

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AskPakistan</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" href="https://img.icons8.com/emoji/48/robot-emoji.png" type="image/png">
  <style>
    :root{
      --brand:#3C0A6D;
      --brand-dark:#3e9b43;
      --bg:#f7f9fc;
      --bubble-user:#3C0A6D;
      --bubble-bot:#f1f0f0;
      --text:#222;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background:var(--bg);
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:16px;
    }
    .chat-wrap{
      width:100%;
      max-width:780px;
    }
    .chat-container{
      height:75vh;
      min-height:520px;
      display:flex;
      flex-direction:column;
      background:#fff;
      border-radius:16px;
      box-shadow:0 10px 30px rgba(0,0,0,.08);
      overflow:hidden;
    }
    .chat-header{
      background:var(--brand);
      color:#fff;
      display:flex;
      align-items:center;
      gap:10px;
      padding:14px 18px;
      font-weight:700;
      letter-spacing:.3px;
    }
    .chat-header .title{
      display:flex;align-items:center;gap:10px;font-size:18px
    }
    .chat-header img{height:28px}
    #messages{
      flex:1;
      padding:16px;
      overflow-y:auto;
      background:linear-gradient(180deg,#f8fafc 0,#fff 35%);
    }


    .row{display:flex;gap:10px;margin:10px 0;align-items:flex-end;}
    .row.user{justify-content:flex-end}
    .row.bot{justify-content:flex-start}

 
    .avatar{
      width:34px;height:34px;border-radius:50%;flex:0 0 34px;
      display:flex;align-items:center;justify-content:center;
      background:#e8f5e9;border:1px solid #e0e0e0;
      box-shadow:0 1px 2px rgba(0,0,0,.04);
      overflow:hidden;
    }
    .avatar img{width:100%;height:100%;object-fit:cover}
    .avatar.user{background:#c8e6c9}
    .avatar.bot{background:#e3f2fd}


    .bubble{
      max-width:min(78%, 560px);
      padding:11px 14px;
      border-radius:16px;
      line-height:1.45;
      font-size:15px;
      color: var(--text);
      box-shadow:0 2px 6px rgba(0,0,0,.06);
      animation:pop .14s ease-out;
      white-space:pre-wrap; word-break:break-word;
    }
    .user .bubble{
      color: #fff;
      background:var(--bubble-user);
      border-bottom-right-radius:6px;
    }
    .bot .bubble{
      background:var(--bubble-bot);
      border-bottom-left-radius:6px;
    }
    @keyframes pop{from{transform:translateY(4px);opacity:0}to{transform:none;opacity:1}}

  
    .composer{
      display:flex;gap:10px;align-items:center;
      padding:12px; border-top:1px solid #eef0f3; background:#fafafa;
    }
    #messageInput{
      flex:1;padding:12px 14px;border:1px solid #e2e6eb;border-radius:24px;
      outline:none;font-size:16px;background:#fff;
    }
    #sendBtn{
      border:none;border-radius:24px;padding:10px 18px;cursor:pointer;
      background:var(--brand);color:#fff;font-weight:600;transition:filter .15s ease;
    }
    #sendBtn:disabled{filter:grayscale(0.3) brightness(0.9);cursor:not-allowed}
    #sendBtn:hover:not(:disabled){filter:brightness(0.95)}


    .typing{display:none;align-items:center;gap:10px;padding:0 16px 14px}
    .dots{display:flex;gap:6px;align-items:center}
    .dot{width:7px;height:7px;border-radius:50%;background:var(--brand);animation:blink 1s infinite ease-in-out}
    .dot:nth-child(2){animation-delay:.15s}
    .dot:nth-child(3){animation-delay:.3s}
    @keyframes blink{0%,80%,100%{opacity:.35;transform:translateY(0)}40%{opacity:1;transform:translateY(-2px)}}

  
    @media (max-width: 640px){
      .chat-container{height:90vh;min-height:520px}
      .bubble{max-width:85%}
    }
  </style>
</head>
<body>
  <div class="chat-wrap">
    <div class="chat-container">
      <div class="chat-header">
        <div class="title">
          AskPakistan
        </div>
      </div>

      <div id="messages" aria-live="polite"></div>

      <div class="typing" id="typing">
        <div class="avatar bot">
          <img src="https://img.icons8.com/emoji/48/robot-emoji.png" alt="bot">
        </div>
        <div class="dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
      </div>

      <form class="composer" id="messageForm">
        <input id="messageInput" type="text" placeholder="Type a message…" autocomplete="off" />
        <button id="sendBtn" type="submit">Send</button>
      </form>
    </div>
  </div>

  <script>
 
    const messagesEl = document.getElementById('messages');
    const typingEl   = document.getElementById('typing');
    const inputEl    = document.getElementById('messageInput');
    const sendBtn    = document.getElementById('sendBtn');

   
    const STORAGE_KEY = 'otto-chat-history';
    function saveHistory(){
      const items = [...messagesEl.querySelectorAll('.row')].map(r => ({
        who: r.classList.contains('user') ? 'user' : 'bot',
        html: r.querySelector('.bubble').innerHTML
      }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    }
    function restoreHistory(){
      const data = localStorage.getItem(STORAGE_KEY);
      if(!data) return;
      try{
        JSON.parse(data).forEach(item => addMessageHTML(item.html, item.who));
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }catch{}
    }

    function mdToHtml(text){
      let t = text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      t = t.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>')
           .replace(/\\*(.+?)\\*/g,'<em>$1</em>');
   
      t = t.replace(/^(?:\\s*[-*]\\s+.+\\n?)+/gm, m => {
        const items = m.trim().split(/\\n/).map(l => l.replace(/^\\s*[-*]\\s+/,'').trim());
        return '<ul>' + items.map(i=>`<li>${i}</li>`).join('') + '</ul>';
      });

      t = t.replace(/\\n/g,'<br>');
      return t;
    }

    function rowTemplate(who, html){
      const isUser = who === 'user';
      const avatarSrc = isUser
        ? 'https://img.icons8.com/ios-glyphs/30/user--v1.png'
        : 'https://img.icons8.com/emoji/48/robot-emoji.png';
      return `
        <div class="row ${who}">
          ${!isUser ? `<div class="avatar bot"><img src="${avatarSrc}" alt="bot"></div>` : ''}
          <div class="bubble">${html}</div>
          ${ isUser ? `<div class="avatar user"><img src="${avatarSrc}" alt="you"></div>` : '' }
        </div>`;
    }

    function addMessageHTML(html, who){
      const wrapper = document.createElement('div');
      wrapper.innerHTML = rowTemplate(who, html);
      messagesEl.appendChild(wrapper.firstElementChild);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      saveHistory();
    }

    function addMessageText(text, who){
      addMessageHTML(mdToHtml(text), who);
    }

 
    const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${wsProtocol}://${location.host}/ws`);

    ws.addEventListener('open', () => {
      restoreHistory();
      if(messagesEl.children.length === 0){
        addMessageText('Hello! I\\'m **AskPakistan**. Ask me anything about Pakistan.', 'bot');
      }
    });

    ws.addEventListener('message', (e) => {
      typingEl.style.display = 'none';
      sendBtn.disabled = false;
      addMessageText(e.data, 'bot');
    });

    ws.addEventListener('close', () => {
      typingEl.style.display = 'none';
      sendBtn.disabled = false;
      addMessageText('⚠️ Connection closed.', 'bot');
    });

 
    document.getElementById('messageForm').addEventListener('submit', (ev) => {
      ev.preventDefault();
      const text = inputEl.value.trim();
      if(!text || ws.readyState !== WebSocket.OPEN) return;
      addMessageText(text, 'user');
      inputEl.value = '';
      typingEl.style.display = 'flex';
      sendBtn.disabled = true;
      ws.send(text);
    });

    inputEl.addEventListener('keydown', (e) => {
      if(e.key === 'Enter' && !e.shiftKey){
        e.preventDefault();
        document.getElementById('messageForm').dispatchEvent(new Event('submit'));
      }
    });
  </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

executor = ThreadPoolExecutor()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data.strip() == "":
                await websocket.send_text("Please type something to ask. 🙂")
                continue
            if data.lower() in ["exit", "quit"]:
                await websocket.send_text("Chat session ended.")
                await websocket.send_text("Thank you for using AskPakistan. Have a great day! 👋")
                break
            try:
               
                response = await asyncio.get_event_loop().run_in_executor(
                    executor, query_engine.query, data
                )
                await websocket.send_text(response.response)

            except Exception as e:
                await websocket.send_text(f"⚠️ Error: {str(e)}")
                print("Error during query:", e)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

if __name__ == "__main__":
    ngrok.set_auth_token("31rgFhp3bgn7jB10vtqCTq3hKUq_5MuXVHJYekdC7mzBxNZBZ")
    tunnel = ngrok.connect(8000)
    print("ngrok tunnel opened at:", tunnel.public_url)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


