import cv2
import numpy as np
import time
import tensorflow as tf

isimler = ["ruya", "meral", "safa", "sezer", "cise", "evren", "sevval"]
model_path = "C:/Users/muhammedsezer/Desktop/yapay-zeka/inference_graph/saved_model"

cam = cv2.VideoCapture("2020-12-03 22-44-27.mp4")
sezer = cv2.VideoCapture("sezer.mp4")
sevval = cv2.VideoCapture("sevval.mp4")
meral = cv2.VideoCapture("meral.mp4")
ruya = cv2.VideoCapture("ruya.mp4")
evren = cv2.VideoCapture("evren.mp4")
cise = cv2.VideoCapture("cise.mp4")
safa_frame = cv2.imread("ok.jpg")
safa_frame = cv2.resize(safa_frame, (160, 144))
sezerkoorx = [0, 0, 0, 0, 0]
sezerkoory = [0, 0, 0, 0, 0]
sezersay = 0
sezerguven = 3
sezergordu = 1

sevvalkoorx = [0, 0, 0, 0, 0]
sevvalkoory = [0, 0, 0, 0, 0]
sevvalsay = 0
sevvalguven = 3
sevvalgordu = 1

meralkoorx = [0, 0, 0, 0, 0]
meralkoory = [0, 0, 0, 0, 0]
meralsay = 0
meralguven = 3
meralgordu = 1

evrenkoorx = [0, 0, 0, 0, 0]
evrenkoory = [0, 0, 0, 0, 0]
evrensay = 0
evrenguven =3
evrengordu = 1

ruyasay = 0
ruyaguven = 3
ruyakoorx = [0, 0, 0, 0, 0]
ruyakoory = [0, 0, 0, 0, 0]
ruyagordu = 1

cisegordu = 1
cisesay = 0
ciseguven = 3
cisekoorx = [0, 0, 0, 0, 0]
cisekoory = [0, 0, 0, 0, 0]

safagordu = 1
safasay = 0
safaguven = 3
safakoorx = [0, 0, 0, 0, 0]
safakoory = [0, 0, 0, 0, 0]


for i in range(60*60*7 + 60 * 59):
    c, b = sezer.read()
    c, b = sevval.read()
    c, b = meral.read()
    c, b = ruya.read()
    c, b = cise.read()
    c, b = cam.read()
    print(i)

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

out = cv2.VideoWriter('amungg.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (1920, 1080))

model = tf.saved_model.load(model_path)
model_fn = model.signatures['serving_default']

def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    return output_dict
a = 0
while True:
    try:
        evrentimer = 0
        ret, frame = cam.read()
        black = np.zeros((1480, 2320, 3), np.uint8)
        black[200:1280, 200:2120] = frame

        ret, sezer_frame = sezer.read()
        sezer_frame = cv2.resize(sezer_frame, (192+38, 108+36))
        sezer_frame = sezer_frame[0:144, 40:200]

        ret, sevval_frame = sevval.read()
        sevval_frame = cv2.resize(sevval_frame, (192+38, 108+36))
        sevval_frame = sevval_frame[0:144, 40:200]

        ret, meral_frame = meral.read()
        meral_frame = cv2.resize(meral_frame, (192+38, 108+36))
        meral_frame = meral_frame[0:144, 40:200]

        ret, cise_frame = cise.read()
        cise_frame = cv2.resize(cise_frame, (192+38, 108+36))
        cise_frame = cise_frame[0:144, 40:200]

        if evrentimer == 0:
            ret, evren_frame = evren.read()
            evren_frame = cv2.resize(evren_frame, (192+38, 108+36))
            evren_frame = evren_frame[0:144, 40:200]
            evrentimer = evrentimer + 1
            evrentimer = evrentimer % 2

        ret, ruya_frame = ruya.read()
        ruya_frame = cv2.resize(ruya_frame, (192+38, 108+36))
        ruya_frame = ruya_frame[0:144, 40:200]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output = run_inference_for_single_image(model_fn, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        sezergordu = -1
        sevvalgordu = -1
        meralgordu = -1
        safagordu = -1
        cisegordu = -1
        evrengordu = -1
        ruyagordu = -1
        for i in range(7):
            if np.max(output["detection_multiclass_scores"][i]) > .6:
                box = output["detection_boxes"][i]
                (topY, topX, bottomY, bottomX) = box
                topY = int(topY * 1080)
                topX = int(topX * 1920)
                bottomY = int(bottomY * 1080)
                bottomX = int(bottomX * 1920)
                if "sezer" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and sezergordu == -1:
                    sezerkoorx[sezersay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    sezerkoory[sezersay] = YY
                    ortalamax = int(sum(sezerkoorx) / 5)
                    ortalamay = int(sum(sezerkoory) / 5)
                    sezersay = sezersay + 1
                    sezersay = sezersay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = sezer_frame
                    sezergordu = 1
                if "evren" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and evrengordu == -1:
                    evrenkoorx[evrensay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    evrenkoory[evrensay] = YY
                    ortalamax = int(sum(evrenkoorx) / 5)
                    ortalamay = int(sum(evrenkoory) / 5)
                    evrensay = evrensay + 1
                    evrensay = evrensay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = evren_frame
                    evrengordu = 1
                if "meral" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and meralgordu == -1:
                    meralkoorx[meralsay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    meralkoory[meralsay] = YY
                    ortalamax = int(sum(meralkoorx) / 5)
                    ortalamay = int(sum(meralkoory) / 5)
                    meralsay = meralsay + 1
                    meralsay = meralsay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = meral_frame
                    meralgordu = 1
                if "cise" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and cisegordu == -1:
                    cisekoorx[cisesay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    cisekoory[cisesay] = YY
                    ortalamax = int(sum(cisekoorx) / 5)
                    ortalamay = int(sum(cisekoory) / 5)
                    cisesay = cisesay + 1
                    cisesay = cisesay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = cise_frame
                    cisegordu = 1
                if "safa" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and safagordu == -1:
                    safakoorx[safasay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    safakoory[safasay] = YY
                    ortalamax = int(sum(safakoorx) / 5)
                    ortalamay = int(sum(safakoory) / 5)
                    safasay = safasay + 1
                    safasay = safasay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = safa_frame
                    safagordu = 1
                if "ruya" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and ruyagordu == -1:
                    ruyakoorx[ruyasay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    ruyakoory[ruyasay] = YY
                    ortalamax = int(sum(ruyakoorx) / 5)
                    ortalamay = int(sum(ruyakoory) / 5)
                    ruyasay = ruyasay + 1
                    ruyasay = ruyasay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = ruya_frame
                    ruyagordu = 1
                if "sevval" == isimler[np.argmax(output["detection_multiclass_scores"][i])] and sevvalgordu == -1:
                    sevvalkoorx[sevvalsay] = topX
                    YY = topY - 140
                    if YY < 0:
                        YY = 0
                    sevvalkoory[sevvalsay] = YY
                    ortalamax = int(sum(sevvalkoorx) / 5)
                    ortalamay = int(sum(sevvalkoory) / 5)
                    sevvalsay = sevvalsay + 1
                    sevvalsay = sevvalsay % 5
                    black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = sevval_frame
                    sevvalgordu = 1

        if sezergordu == -1:
            if sezerguven:
                ortalamax = int(sum(sezerkoorx) / 5)
                ortalamay = int(sum(sezerkoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = sezer_frame


        if sezerguven < 3 and sezergordu == 1:
            sezerguven = sezerguven + 1
        if sezerguven > 0 and sezergordu == -1:
            sezerguven = sezerguven - 1


        if evrengordu == -1:
            if evrenguven:
                ortalamax = int(sum(evrenkoorx) / 5)
                ortalamay = int(sum(evrenkoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = evren_frame


        if evrenguven < 3 and evrengordu == 1:
            evrenguven = evrenguven + 1
        if evrenguven > 0 and evrengordu == -1:
            evrenguven = evrenguven - 1


        if meralgordu == -1:
            if meralguven:
                ortalamax = int(sum(meralkoorx) / 5)
                ortalamay = int(sum(meralkoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = meral_frame


        if meralguven < 3 and meralgordu == 1:
            meralguven = meralguven + 1
        if meralguven > 0 and meralgordu == -1:
            meralguven = meralguven - 1


        if cisegordu == -1:
            if ciseguven:
                ortalamax = int(sum(cisekoorx) / 5)
                ortalamay = int(sum(cisekoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = cise_frame


        if ciseguven < 3 and cisegordu == 1:
            ciseguven = ciseguven + 1
        if ciseguven > 0 and cisegordu == -1:
            ciseguven = ciseguven - 1


        if safagordu == -1:
            if safaguven:
                ortalamax = int(sum(safakoorx) / 5)
                ortalamay = int(sum(safakoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = safa_frame


        if safaguven < 3 and safagordu == 1:
            safaguven = safaguven + 1
        if safaguven > 0 and safagordu == -1:
            safaguven = safaguven - 1


        if ruyagordu == -1:
            if ruyaguven:
                ortalamax = int(sum(ruyakoorx) / 5)
                ortalamay = int(sum(ruyakoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = ruya_frame


        if ruyaguven < 3 and ruyagordu == 1:
            ruyaguven = ruyaguven + 1
        if ruyaguven > 0 and ruyagordu == -1:
            ruyaguven = ruyaguven - 1


        if sevvalgordu == -1:
            if sevvalguven:
                ortalamax = int(sum(sevvalkoorx) / 5)
                ortalamay = int(sum(sevvalkoory) / 5)
                black[(200 + ortalamay):(200+ortalamay + 144), (200+ortalamax):(200+ortalamax+160)] = sevval_frame


        if sevvalguven < 3 and sevvalgordu == 1:
            sevvalguven = sevvalguven + 1
        if sevvalguven > 0 and sevvalgordu == -1:
            sevvalguven = sevvalguven - 1

        a = a + 1
        out.write(black[200:1280, 200:2120])
        cv2.imshow("frame", cv2.resize(black[200:1280, 200:2120], (900, 600)))
        cv2.waitKey(1)
    except Exception as err:
        print(err)
