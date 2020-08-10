# 語音辨識實作

# 情境 - 辨識客服對話
要將大量的客服對話內容轉為文字資料，大致步驟有：
- 語音檔案轉為文字
- 分句進行語音解析
- 辨識說話者

## 資料集
本次資料集使用科技政策研究與資訊中心中，科技大擂台競賽的第一輪的測試資料集：
https://scidm.nchc.org.tw/dataset/grandchallenge

語音資料集中包含了文章、問題與選項的音檔以及對應的文字資料，
本次僅針對**文章的部分進行轉換**

| 資料屬性 | 描述                                |
| -------- | ----------------------------------- |
| 音檔數量 | 1297 個                             |
| 音檔格式 | .wav                                |
| 總大小   | 1.69GB                              |
| 平均大小 | 1.3MB                               |
| 音檔長度 | 長度不一，大部分音檔約在10~30秒之間 |


setup dataset
```sh=bash
wget -O dataset.zip https://scidm.nchc.org.tw/dataset/grandchallenge/resource/1ca6d0aa-8af5-4e97-8922-76fc877d13e5/nchcproxy
unzip dataset.zip
cd GrandChallenge\ 1st\ round/data/wav/A
ls
```
```sh=bash
A0000001.wav  A0000547.wav  A0000742.wav  A0000929.wav  A0001119.wav  A0001305.wav  A0001494.wav
A0000002.wav  A0000548.wav  A0000743.wav  A0000930.wav  A0001120.wav  A0001306.wav  A0001495.wav
A0000003.wav  A0000549.wav  A0000744.wav  A0000931.wav  A0001121.wav  A0001307.wav  A0001496.wav
A0000004.wav  A0000550.wav  A0000745.wav  A0000932.wav  A0001122.wav  A0001308.wav  A0001497.wav
A0000005.wav  A0000551.wav  A0000746.wav  A0000933.wav  A0001123.wav  A0001309.wav  A0001498.wav
A0000006.wav  A0000552.wav  A0000747.wav  A0000934.wav  A0001124.wav  A0001310.wav  A0001499.wav
...
```



## 語音檔案轉文字
將語音檔案轉為文字資料，現行市面上已經有許多公司提供該服務，並且技術相當成熟
在python中實作
```python
import speech_recognition as sr

r = sr.Recognizer()
def recoginze_wav_file(wav_file):
    """
    wav_file: file path to the .wav file
    """
    try:
        with sr.WavFile(wav_file) as source:
            audio = r.record(source)
        s = r.recognize_google(audio, language="zh-TW")
        return s
    except:
        return None
```
使用python中的 ***speech_recognition*** package 來幫助我們進行語音識別，該package提供了完善的語音識別api的接口，包括：
- CMU Sphinx (works offline)
- Google Speech Recognition
- Google Cloud Speech API
- Wit.ai
- Microsoft Azure Speech
- Microsoft Bing Voice Recognition (Deprecated)
- Houndify API
- IBM Speech to Text
- Snowboy Hotword Detection (works offline)

上述的服務大部分需要收費或需要額外的設定，而我們這次主要使用recognize_google，
為google提供使用者進行語音辨識服務的測試用途，不用進行其他設定。

**辨識目錄下的所有檔案**
```python
import os
wav_path = os.getcwd()
wav_list = [x for x in os.listdir(wav_path) if x.endswith(".wav")]

result = []
for wav_file in wav_list:
    wav_file = os.path.join(wav_path, wav_file)
    s = recoginze_wav_file(wav_file)
    result.append(s)
print(result[0:6])
```
```jsonld=
['明年元旦起交通部新規定汽車輪胎胎紋深度將納入定期檢驗項目之一一旦深度未達1.6公里吃一個月內沒換胎將會被吊銷牌照民眾除了定期檢驗胎紋也可以自己用10元硬幣檢測只要看得見國父像衣領下緣表示該換輪胎了',
 '塑化劑是一種環境荷爾蒙又稱內分泌幹擾物只會幹擾生物體內分泌之外因性化學物質若長時間高劑量使用dehp江島之男嬰生殖器發育不良女童性早熟一集成年男性精蟲減少等問題常見塑化劑可分成8種其中的HP在人體代謝快速大部分代謝物會於24到48小時內排出體外dinp者是在72小時內有糞便和尿液排出因此若非大量使用其實並沒有立即的安全問題',
 '紐西蘭的海岸邊今日出現數百具小藍企鵝的屍體專家指出這些企鵝都是被活活餓死的會有企鵝大量20的現象是因為氣候變遷導致他們的食物短缺企鵝杯20本來不是韓式但今年死亡的數量卻多到異常世界上最小的企鵝小藍企鵝近日被發現大量的因為食物匱乏與營養不良死亡目前已有數百具屍體被發現專家指出全鈕西蘭受到影響的企鵝可能高達數千隻企鵝專家泰勒指出這是企鵝大量死亡應該是受到今年反聖嬰現象的關係海水變暖強冷的海水與食物都被退相聲海讓這些小藍企鵝們找不到食物但太熱也表示儘管企鵝們找不到食物20是很正常的事但這次企鵝屍體數目多到相當不尋常4G1998年近3500隻企鵝死亡之後死亡規模最大的一次',
 '綜合報導科學期刊當代生物學昨天看燈1份美澳聯合研究研究指出一年澳洲大堡礁北部的新出生的綠蠵龜有90%以上是詞性雄性出生率劇減雄性海龜的出生溫度為攝氏29.3度以下攝氏29.3度以上摺會生出磁性研究員觀察發現90年代後沙灘溫度經常維持高於生出雄性海龜的溫度丁認為氣候暖化是這種現象產生的關鍵因素',
 '花蓮強震發生後是否還可能發生更大的地震中央大學地科系教授馬國鳳警告昨晚花蓮強震應該是主震關鍵是米崙斷層引起只斷層南延伸到台東縱谷連動性如何還無法完全掌握',
 '牛頓第一運動定律是慣性定律除非物體有受到外力要不然保持靜止的物體會一直保持靜止沿一直線作等速度運動的物體也會一直保持等速度運動牛頓第二運動定律也稱為運動定律當物體受外力作用時會在力的方向產生加速度其大小與外力成正比與質量成反比牛頓第三運動定律也稱為作用與反作用定律當斯加力於物體時會同時產生1個大小相等且方向相反的反作用力作用力與反作用力大小相等方向相反且作用在同一直線上因為受力對像不同所以不能互相抵銷兩者同時發生同時消失']
```
**檢視語音辨識結果**
可以透過字錯計算來檢驗語音辨識結果，計算公式如下圖(*資料來源: https://www.itread01.com/content/1554303541.html*)
![](https://i.imgur.com/XVxGK4s.png)
![](https://i.imgur.com/FOK1kLv.png)

python code
```python
# 計算 CERs
import nltk
def calculate_CER(s1, s2):
    # 先去除標點符號
    s1 = re.sub("[，,。、「」；《》\n 『』（）、〈〉()？!！?\[\]]", "", s1)
    s2 = re.sub("[，,。、「」；《》\n 『』（）、〈〉()？!！?\[\]]", "", s2)
    S_D_I = nltk.edit_distance(s1, s2)
    CER = S_D_I / max(len(s1), len(s2))
    return CER
```

檢視語音辨識的辨識率
```python
import pandas as pd
import numpy as np
import re
# 讀取labeled text data
dataset = pd.read_excel('../../../文本總整理Beta 2.xlsx', index_col=0)

CERs = []
for s1, s2 in zip(result, dataset['文章']):
    CER = calculate_CER(s1, s2)
    CERs.append(CER)
print("mean error rate", sum(CERs)/len(CERs))
print("error rate std.", np.std(CERs))
```
```
mean error rate 0.16618959589217042
error rate std. 0.15332470055160605
```

## 分句進行語音解析
```python
import pydub
from pydub import AudioSegment
from pydub.playback import play
import time

# 將音檔音量正規化
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
```

```python
result = {}
for filename in wav_list:
    # print(filename)
    wav_file = os.path.join(wav_path, filename)
    # 讀取音檔
    audio = AudioSegment.from_wav(wav_file)
    # 將所有音檔的音量正規化為-20db
    audio = match_target_amplitude(audio, -20)
    # 根據音檔中的停頓進行斷句
    audio_seg = pydub.silence.split_on_silence(audio, min_silence_len=400, silence_thresh=-35, keep_silence=200 )

    sentences = []
    audio = ""
    for seg in audio_seg:
        seg.export("temp.wav", format = 'wav') # 輸出音檔
        text = recoginze_wav_file("temp.wav")
        if text is not None:
            sentences.append(text)
    result[filename] = {"sentences": sentences, "dialog": " ".join(sentences)}
    
result["A0000001.wav"]
```
```jsonld=
{'sentences': ['明年元旦起',
  '交通部新規定',
  '汽車輪胎胎紋深度',
  '將納入定期檢驗項目之一',
  '一旦深度未達1.6公里',
  '吃一個月內沒換胎',
  '將會被吊銷牌照',
  '民眾除了定期檢驗胎紋',
  '也可以自己用10元硬幣檢測',
  '只要看得見國父像衣領下緣',
  '表示該換輪胎了'],
 'dialog': '明年元旦起 交通部新規定 汽車輪胎胎紋深度 將納入定期檢驗項目之一 一旦深度未達1.6公里 吃一個月內沒換胎 將會被吊銷牌照 民眾除了定期檢驗胎紋 也可以自己用10元硬幣檢測 只要看得見國父像衣領下緣 表示該換輪胎了'}
```
比較原始文章，發現斷句結果還算不錯
```python
print(dataset["文章"][1])
明年元旦起，交通部新規定，汽車輪胎胎紋深度將納入定期檢驗項目之一，一旦深度未達1.6公厘，及1個月內沒換胎，將會被吊銷牌照。民眾除了定期檢測胎紋，也可以自已用10元硬幣檢測，只要看的見國父像衣領下緣，表示該換輪胎了。
```
## 辨識說話者
在對話錄音中，知道是誰說話是很重要的一個標籤，現行有許多辨識對話的方法，
本次使用在網路上看到的一個特征提取的方法，mfcc來針對說話者進行特征提取，並且進行分群。

**本次的資料集中，每個音檔都只有一個人說話，因此我們提取每個音檔的第一句話進行辨識，並且對每個音檔的說話者進行分類**
```python
from python_speech_features import mfcc
import scipy.io.wavfile as wav

# 提取語音特征
def get_feature(filepath):
    try:
        if os.path.exists(filepath):
            (rate,sig) = wav.read(filepath)
            mfcc_feat = mfcc(sig,rate)
            return mfcc_feat.mean(0) # 取平均壓縮成一個維度
        else:
            return None
    except:
        return None
```

提取每個音檔的第一句話
```python
first_sent_path = os.path.join(wav_path, "first_sent")
if not os.path.exists(first_sent_path):
    os.makedirs(first_sent_path)
    
for filename in wav_list:
    wav_file = os.path.join(wav_path, filename)
    audio = AudioSegment.from_wav(wav_file)
    audio = match_target_amplitude(audio, -20)
    audio_seg = pydub.silence.split_on_silence(audio, min_silence_len=400, silence_thresh=-35, keep_silence=200 )    
    audio_seg[0].export(os.path.join(first_sent_path, "%s_1.wav"% filename.replace(".wav", "")), format = 'wav')
```

提取每個音檔說話者的特征
```python
features = []
for f in wav_list:
    filename = "%s_1.wav" % str(f.replace(".wav", ""))
    feature = get_feature(os.path.join(first_sent_path, filename))
    if feature is not None:
        features.append(feature)

audio_list = pd.concat([pd.DataFrame({"filename": wav_list}), pd.DataFrame(features)], axis=1)
audio_list = audio_list.dropna()
audio_list.head()
```
![](https://i.imgur.com/IRLDQrI.png)

進行階層式分群
```python
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(audio_list[audio_list.columns[1:]], method  = "ward"))
plt.xlabel('audio ID')
plt.ylabel('Euclidean distances')
plt.axhline(y=250, color='r', linestyle='-')
plt.show()
```
![](https://i.imgur.com/GvBbLi3.png)

觀察分佈，大致可以分為6群
```python
from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(audio_list[audio_list.columns[1:]])

audio_list = pd.concat([pd.DataFrame({"speaker":y_hc}), audio_list], axis=1)
audio_list
```
![](https://i.imgur.com/5sRBrPn.png)

本次實作因為資料集並非對話錄音，每個錄音檔中只會有一位說話者，因此我們比較每個音檔之間的說話者。比較每個音檔之間的特征差異，發覺可以大致將該資料集中所有的音檔分為6群。
**實際檢查資料集：**
1 ~ 25個音檔為同一位錄音者
26 ~ 50個音檔後換為另一位錄音者
第50個音檔後又換為好幾個人交錯錄音

**在本次的分群結果中：**
1 ~ 25 基本上被分為一群
26 ~ 50 被分為一群
51 ~ 在分成剩餘四

而詳細檢查第51個音檔後的樣本，還是可以發現許多錯誤的分群。本次方法在音色差異較大(特別是不同性別)時，能夠順利的區分講著，但若相似度不夠高時則無法順利分類。之後可以後可以用更完整與嚴謹的方法，以提升人聲識別的準確度。

## 結論
**在實際的假想情境中，我們可以先對一整個對話的音檔進項上述的機箱操作，包括：1）進行分句、 2）辨識講者、 3）語音轉文字 後將結果儲存下來**














