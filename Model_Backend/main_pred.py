from ultralytics import YOLO

def main_model(img_pth):
  word="model"
  model1=YOLO("mainv3.pt")
  results = model1(img_pth)
  for r in results:
    for c in r.boxes.cls:
        word=str((model1.names[int(c)]))
        break
  if(word!="model"):
    return(word)
  else:
     return("wrong input")

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

def main_pred(img_pth):
    prediction=""
    print("model started")
    out1="model"
    out1=main_model(img_pth)
    if(out1!="model"):
      print("detected object : ",out1)
    else:
       print("wrong input")
       return("wrong input")
    
    prediction+="detected object : "+out1+"\n"
    if(out1=="car"):
      out2=car_status(img_pth)
      print("car_status : ",out2)
      prediction+="car status : "+out2+"\n"
      out3=str(car_damage(img_pth))
      print(out3)
      prediction+="car damage: "+out3+"\n"
    elif(out1=="TIRE"):
      out4=tire_status(img_pth)
      out4=float(out4)*100
      print("tire_status : ",str(out4)+" % of life span left")
      prediction+="tire_status : "+str(out4)+" % of life span left "+"\n"
      out5=tire_crack(img_pth)
      if(out5=="0"):
        print("tire_crack : Not Cracked")
        prediction+="tire crack : Not Cracked"+"\n"
      else:
        print("tire_crack : Cracked")
        prediction+="tire crack : Cracked"+"\n"
    elif(out1=="oil"):
      out6=oil_check(img_pth)
      print("oil condition : ",out6)
      prediction+="oil condition : "+out6+"\n"
    else:
       print("wrong input")
    return (prediction,out1)