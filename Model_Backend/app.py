import pandas as pd
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import base64
img_pth=""
app = Flask(__name__)
# model1 = YOLO("yolo/mainv3.pt")
# model2 = YOLO("yolo/car_status.pt")
# model3 = YOLO("yolo/tire_status.pt")
# model4 = YOLO("yolo/tire_crack.pt")

def main_model(img_pth):
  model1=YOLO("mainv3.pt")
  results = model1(img_pth)
  for r in results:
    for c in r.boxes.cls:
        word=str((model1.names[int(c)]))
        break
  return(word)

def car_status(img_pth):
  model2=YOLO("car_status.pt")
  results = model2(img_pth)
  for r in results:
    for c in r.boxes.cls:
        word=str((model2.names[int(c)]))
        break
  return(word)

def tire_status(img_pth):
  model3=YOLO("tire_status.pt")
  results = model3(img_pth)
  for r in results:
    for c in r.boxes.cls:
        word=str((model3.names[int(c)]))
        break
  return(word)

def tire_crack(img_pth):
  model4=YOLO("tire_crack.pt")
  results = model4(img_pth)
  for r in results:
    for c in r.boxes.cls:
        word=str((model4.names[int(c)]))
        break
  return(word)

from car_dam import predict_cond
def car_damage(img_pth):
    return(predict_cond(img_pth))

from oil_det import det
def oil_check(img_pth):
    return(det(img_pth))

@app.route('/predict', methods = ["POST"])
def predict():
    json_ = request.get_json()
    file = json_
    decoded_data=base64.b64decode((file))
    img_file = open('image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()
    img_pth="./image.jpeg"
    prediction=""
    # query_df = pd.DataFrame(json_)
    # prediction1 = model1.predict(query_df)
    print("model started")
    out1=main_model(img_pth)
    print("detected object : ",out1)
    prediction+="detected object : "+out1
    if(out1=="car"):
      out2=car_status(img_pth)
      print("car_status : ",out2)
      prediction+="car status : "+out2
      out3=str(car_damage(img_pth))
      print(out3)
      prediction+="car damage: "+out3
    elif(out1=="TIRE"):
      out4=tire_status(img_pth)
      print("tire_status : ",out4)
      prediction+="tire_status : "+out4
      out5=tire_crack(img_pth)
      if(out5=="0"):
        print("tire_crack : Not Cracked")
        prediction+="tire crack : Not Cracked"
      else:
        print("tire_crack : Cracked")
        prediction+="tire crack : Cracked"
    else:
      out6=oil_check(img_pth)
      print("oil condition : ",out6)
      prediction+="oil condition : "+out6
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=8888)