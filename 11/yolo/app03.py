from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage,ImageMessage,ImageSendMessage
import os
import cv2
from ultralytics import YOLO


app = Flask(__name__)
os.makedirs("static", exist_ok=True)

LINE_CHANNEL_ACCESS_TOKEN='xxxx'
LINE_CHANNEL_SECRET='xxxx'
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
line_handler = WebhookHandler(LINE_CHANNEL_SECRET)


OKimg=''

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
    global OKimg
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)


    image_dir = 'received_images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


    image_path = os.path.join(image_dir, f"{message_id}.jpg")
    with open(image_path, 'wb') as fd:
        
        for chunk in message_content.iter_content():
            fd.write(chunk)
            
    imgP='received_images/'+message_id+'.jpg'
    print(imgP)   
   
    model = YOLO("last150.pt")
    OKimg = model(imgP)
   
    for i, result in enumerate(OKimg):
        annotated_img = result.plot()
        save_path = f"static/result_{i}.jpg"
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved: {save_path}")
    
    reply_arr=[
    TextMessage(text='辨識結果:'),
    ImageSendMessage(
        original_content_url= 'https://XXXXXXX.ngrok-free.app/'+'static/result_0.jpg',
        preview_image_url= 'https://XXXXXXX.ngrok-free.app/'+'static/result_0.jpg'),
    ]
    
    line_bot_api.reply_message(event.reply_token,reply_arr) 

@line_handler.add(MessageEvent, message=TextMessage)
def handle_message(event): 
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))      


if __name__ == "__main__":
    app.run()