from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage,ImageMessage
import os
app = Flask(__name__)


LINE_CHANNEL_ACCESS_TOKEN='xxxx'
LINE_CHANNEL_SECRET='xxxx'
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
line_handler = WebhookHandler(LINE_CHANNEL_SECRET)


@app.route('/')
def home():
    return 'Hello World'

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@line_handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)

    # 建立一個資料夾來儲存圖片（如果不存在）
    image_dir = 'received_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 儲存圖片到本地
    image_path = os.path.join(image_dir, f"{message_id}.jpg")
    with open(image_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)


@line_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
 
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))      


if __name__ == "__main__":
    app.run()