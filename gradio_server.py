import time
import gradio as gr
from vectors_retrieval import save_vectors_db, init_chain

bot = None

def do_user(user_message, history):
    if history is None:
        history = []
    if not isinstance(user_message, str):
        user_message = str(user_message)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})
    return '', history

def do_it(history):
    if not history or len(history) < 2:
        return []

    ai_message = history[-1]
    question = history[-2].get("content", "")
    if not isinstance(question, str):
        question = str(question)
    if not question.strip():
        ai_message["content"] = "请输入有效的问题！"
        yield history
        return

    try:
        res = bot.invoke({'input': question})
        resp = res.get('answer', '这个问题，我建议你直接问人工！')
    except Exception as e:
        ai_message["content"] = f"AI处理出错：{str(e)}"
        yield history
        return

    # 流式输出
    for i in range(0, len(resp), 5):
        ai_message["content"] = resp[:i + 5]
        time.sleep(0.05)
        yield history

def run_gradio():
    css = """
    /* 页面背景 */
    body {background-color: #f0f4f8; font-family: 'Helvetica Neue', Arial, sans-serif;}

    /* 聊天框 */
    .gradio-chatbot {border-radius: 12px; border: 1px solid #d0d0d0; padding: 10px; background-color: #ffffff;}
    .gradio-chatbot .message {font-size: 16px; line-height: 1.5; margin-bottom: 8px;}

    /* 用户输入框 */
    .feedback textarea {font-size: 18px !important; border-radius: 8px; border: 1px solid #ccc; padding: 10px;}
    #bgc {background-color: #e0f7f5 !important;}

    /* 清除按钮 */
    .gr-button {border-radius: 8px; background-color: #007acc; color: white; font-weight: bold;}
    .gr-button:hover {background-color: #005fa3;}
    """

    with gr.Blocks(title='房产销售AI机器人') as instance:
        gr.Label('房产销售AI机器人', container=False)
        chatbot = gr.Chatbot(label='聊天记录', height=350,
                             placeholder='AI机器人: 你可以问任何房产相关问题')
        msg = gr.Textbox(label='请输入问题', placeholder='输入你的问题…',
                         elem_classes='feedback', elem_id='bgc')
        clear = gr.ClearButton(value='清除聊天记录', components=[msg, chatbot])

        msg.submit(do_user, [msg, chatbot], [msg, chatbot], queue=False)\
           .then(do_it, [chatbot], [chatbot])

    instance.queue()
    instance.launch(server_name='127.0.0.1', server_port=8080, css=css)

def init():
    save_vectors_db()
    global bot
    bot = init_chain()

if __name__ == '__main__':
    init()
    run_gradio()