**Paddle-OCR on Webcam (Win 10)**

The doc describes the necessary ToDo steps for various experiments ( development environment, libs , etc ) concerning paddle-OCR via webCam

![ref1]

**My Environment:**

|Acer Nitro 5 / GTX 1660 Ti ( GPU available )|
| :- |
|Anaconda 3|
|Paython 3.8|
|Spyder Editor|
|opencv                    4.6.0            |
|paddleocr                 2.7.3|
|paddlepaddle-gpu          2.6.1|
|python                    3.8.19|
|spyder                    5.5.1|
||
|CUDA 11.2 and cuDNN 8|
|VS 2019|
||
|Word-file Translation into english with ASPOSE|

GTX 1660TI






**Paddle OCR / Webcam on Win 10**

|<p><https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/environment_en.md></p><p></p><p>Die dort angegebene kombi ( CUDA 10.2 /  cuDNN 7.6 ) hat in meiner umgebung NICHT funktioniert.</p><p></p><p>*If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install*</p><p>*python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple*</p><p></p><p></p>|<p>**Auspassen** bei den Versionen!</p><p>Verwirrung in den Cuda- Versionsangaben in den Paddle-Docs</p><p></p><p>Die 10er hat bei mir NICHT funktioniert.</p><p>Erst die nach der 10er installation aufpoppenden Fehler haben mich auf die korrekte Cuda Version hingewiesen…</p>|
| :- | :- |
|***CUDA 11.2 and cuDNN 8***|<p>**In short:**</p><p>**Musste diese hier installieren   - OK**</p>|

**Es folgen ein paar Notizen die ich parallel zur Installation gemacht hatte**

|<p>- Visual Studio 2019</p><p>&emsp;- Anaconda py38 env anlegen</p><p>&emsp;- CUDA 11.2 and cuDNN 8 auf Win installieren<br>&emsp;  <https://developer.nvidia.com/rdp/cudnn-archive></p><p>&emsp;- copy some cuDNN libs to conda toolkit on Win<br>&emsp;  <https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows><br>&emsp;  (aber vorsicht ..siehe noch den kommentar bzgl. environment variablen…)</p><p>&emsp;- Install in py38 env:<br>&emsp;  pip install paddleocr ( 2.7.3 )<br>&emsp;  ` `<https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p>&emsp;- Install in py38 env:   <br>&emsp;  ` `python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple</p><p>&emsp;&emsp;</p><p>&emsp;- set win env variables für  nvidia</p><p>&emsp;- test gpu</p><p></p>|Einzelne Steps – Die Reihenfolge ist teilweise wichtig.<br>Z.B. muss VS 2019 vor CUDA da sein!||
| :- | :- | :- |
|![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.003.png)|<p>Win10 environment variables</p><p></p>||
|<p><https://developer.nvidia.com/rdp/cudnn-archive></p><p><https://developer.nvidia.com/cuda-11.2.0-download-archive></p><p><https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p></p><p></p>|<p>cuDNN architecture download</p><p>CUDA 11.2 and cuDNN 8</p><p>Toolkit</p>||
|<p><https://www.techspot.com/downloads/7241-visual-studio-2019.html></p><p></p><p></p><p></p>|![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.004.png)||
|<p><https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p>· Copy the following files from the unzipped package into the NVIDIA cuDNN directory.<a name="installwindows__substeps_zcd_xzm_s1b"></a> </p><p>a. Copy bin\cudnn\*.dll to C:\Program Files\NVIDIA\CUDNN\v8.x\bin .</p><p>b. Copy include\ cudnn \*.h to C:\Program Files\NVIDIA\CUDNN\v8.x\include .</p><p>c. Copy lib\x64\cudnn\*.lib to C:\Program Files\NVIDIA\CUDNN\v8.x\lib\x64 .</p><p>Yt how2:</p><p><https://www.youtube.com/watch?v=ctQi9mU7t9o></p><p></p><p></p>|<p>NVIDIA CUDNN doc / install on C ??</p><p></p><p>Hat so bei mir nicht funktioniert…</p><p>siehe screenshot von meinem environment oben</p>||
|<p>python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"</p><p></p>|<p>Compatible test - ok</p><p>64bit</p><p>AMD64</p>||
|<p>nvidia-smi</p><p></p><p>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.005.png)</p><p>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.006.png)</p><p></p><p>nvcc -- version</p><p>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.007.png)</p>|<p>Test CUDA / envidia Etc</p><p></p><p>nvidia-smi</p><p>nvcc -- version</p><p></p><p>Issue:</p><p></p><p>Invidia-smi zeigt immer noch die 11.6er an. Viel diskussion in GitHub darüber. Angeblich ist das „egal“ so lange nvcc das richtige anzeigt!?</p><p>Mag sein..erst einmal weiter machen …steht bis heute 11.6 drin obwohl alles mit 11.2 läuft!?</p><p></p><p>Diese issues hatten wir schon vor </p><p>5 Jahren ( <https://github.com/iCounterBOX/TensorFlow_CarOccupancyDetector_V2> )</p><p></p>||
|<p>Fix problems that block programs from being installed or removed</p><p></p><p><https://support.microsoft.com/en-gb/topic/fix-problems-that-block-programs-from-being-installed-or-removed-cca7d1b6-65a9-3d98-426b-e9f927e1eb4d></p><p></p><p>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.008.png)</p><p></p><p>*not the device driver ...but the things that came in for 2022 (e.g.) have to get deleted*</p>|<p>Was tun bei falscher CUDA (nvidia) version/installation?</p><p></p><p>Die apps müssen einzeln deinstalliert werden!</p><p>Manche sind geblockt – lassen sich nicht einfach deinstallieren.</p><p>Das microsoft-Programm hat das erledigt.</p>||
|<p><h3>**addleocr --image\_dir ./imgs\_en/img\_12.jpg --use\_angle\_cls true --lang en --use\_gpu false**</h3></p><p><h3></h3></p><p><h3></h3></p><p>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.009.png)</p><p></p>|<p>**Test WITHOUT GPU okJ**</p><p></p><p>[**https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/quickstart_en.md#21-use-by-command-line**](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/quickstart_en.md#21-use-by-command-line)</p><p></p>||
|<p><h3>**RuntimeError : ( PreconditionNotMet ) The third-party dynamic library (cudnn64\_8.dll) that Paddle depends on is not configured correctly. ( error code is 126)**</h3></p><p><h3>**Suggestions:**</h3></p><p><h3>**1. Check if the third-party dynamic library (eg CUDA, CUDNN) is installed correctly and its version is matched with paddlepaddle you installed.**</h3></p><p><h3>**2. Configure third-party dynamic library environment variables as follows:**</h3></p><p><h3>**- Linux: set LD\_LIBRARY\_PATH by `export LD\_LIBRARY\_PATH=...`**</h3></p><p><h3>**- Windows: set PATH by `set PATH=XXX; ( at .. \paddle\phi\ backends \ dynload \dynamic\_loader.cc:312)**</h3></p><p></p><p>We check:</p><p></p><p>conda list</p><p>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.010.png)</p><p></p><p>We search for compatibility ..</p><p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html></p><p></p><p>*paddlepaddle-gpu ==[version code], such as paddlepaddle-gpu ==2.6.1 	The default installation supports the PaddlePaddle installation package corresponding to [version number] of CUDA 11.2 and cuDNN 8*</p><p></p><p></p>|<p>**Der Test mit GPU ergab error. Da war die CUDA version noch 10.. Über diese Fehlermeldung kamen wir dann auf die richtige/compatible Cuda version!**</p><p></p><p>**paddleocr -- image\_dir ./drug1.jpg -- use\_angle\_cls true -- lang en -- use\_gpu true**</p><p></p><p></p><p></p><p>**SO the doc above in paddle OCR did not work for me…following the error message I will now install CUDA 11.2 and cuDNN 8**</p>||
|<h3>![](Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.011.png)</h3>|<p>Cuda 11.2   ..looks better</p><p></p><p>..so sollte die meldung aussehen</p><p></p>||




**First SUCCESSFUL TEST:**  

(drug1.jpg ist irgend ein image mit Schrift!)


(py38a) D:\ALL\_PROJECT\pyQT5\_experimental\ai\paddle> **paddleocr -- image\_dir ./drug1.jpg -- use\_angle\_cls true -- long en -- use\_gpu false**

[2024/05/11 16:19:29] ppocr DEBUG: Namespace(alpha=1.0, alphacolor =(255, 255, 255), benchmark=False, beta=1.0, binarize =False, cls\_batch\_num =6, cls\_image\_shape ='3, 48, 192', cls\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\cls\\ch\_ppocr\_mobile\_v2.0\_cls\_infer', cls\_thresh =0.9, cpu\_threads =10, crop\_res\_save\_dir ='./output', det =True, det\_algorithm ='DB', det\_box\_type ='quad', det\_db\_box\_thresh =0.6, det\_db\_score\_mode ='fast', det\_db\_thresh =0.3, det\_db\_unclip\_ratio =1.5, det\_east\_cover\_thresh =0.1, det\_east\_nms\_thresh =0.2, det\_east\_score\_thresh =0.8, det\_limit\_side\_len =960, det\_limit\_type ='max', det\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\det\\en\\en\_PP-OCRv3\_det\_infer', det\_pse\_box\_thresh =0.85, det\_pse\_min\_area =16, det\_pse\_scale =1, det\_pse\_thresh =0, det\_sast\_nms\_thresh =0.2, det\_sast\_score\_thresh =0.5, draw\_img\_save\_dir ='./ inference\_results ', drop\_score =0.5, e2e\_algorithm=' PGNet ', e2e\_char\_dict\_path='./ ppocr / utils / ic15\_dict.txt', e2e\_limit\_side\_len=768, e2e\_limit\_type='max', e2e\_model\_dir=None, e2e\_pgnet\_mode='fast', e2e\_pgnet\_score\_thresh=0.5, e2e\_pgnet\_valid\_set=' totaltext ', enable\_mkldnn =False, fourier\_degree =5, gpu\_id =0, gpu\_mem =500, help='==SUPPRESS==', image\_dir ='./drug1.jpg', image\_orientation =False, invert=False, ir\_optim =True, kie\_algorithm =' LayoutXLM ', label\_list =['0', '180'], lang =' en ', layout=True, layout\_dict\_path =None, layout\_model\_dir =None, layout\_nms\_threshold =0.5, layout\_score\_threshold =0.5, max\_batch\_size =10, max\_text\_length =25, merge\_no\_span\_structure =True, min\_subgraph\_size =15, mode='structure', ocr =True, ocr\_order\_method =None, ocr\_version ='PP-OCRv4', output='./output', page\_num =0, precision='fp32', process\_id =0, re\_model\_dir =None, rec=True, rec\_algorithm =' SVTR\_LCNet ', rec\_batch\_num =6, rec\_char\_dict\_path='C:\\Users\\kristina\\anaconda3\\envs\\py38a\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en\_dict.txt', rec\_image\_inverse =True, rec\_image\_shape ='3, 48, 320', rec\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\rec\\en\\en\_PP-OCRv4\_rec\_infer', recovery=False, save\_crop\_res =False, save\_log\_path ='./ log\_output /', scales=[8, 16, 32], ser\_dict\_path ='../ train\_data /XFUND/class\_list\_xfun.txt', ser\_model\_dir =None, show\_log =True, sr\_batch\_num =1, sr\_image\_shape ='3, 32, 128', sr\_model\_dir =None, structure\_version ='PP-StructureV2', table=True, table\_algorithm =' TableAttn ', table\_char\_dict\_path =None, table\_max\_len =488, table\_model\_dir =None, total\_process\_num =1, type=' ocr ', use\_angle\_cls =True, use\_dilation =False, use\_gpu =False, use\_mp =False, use\_npu =False, use\_onnx =False, use\_pdf2docx\_api=False, use\_pdserving =False, use\_space\_char =True, use\_tensorrt =False, use\_visual\_backbone =True, use\_xpu =False, vis\_font\_path ='./doc/fonts/simfang.ttf', warmup=False)

[2024/05/11 16:19:30] ppocr INFO: \*\*\*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\*\*

[2024/05/11 16:19:31] ppocr DEBUG: dt\_boxes num : 6, elapsed : 0.5120675563812256

[2024/05/11 16:19:31] ppocr DEBUG: cls num : 6, elapsed : 0.10241532325744629

[2024/05/11 16:19:32] ppocr DEBUG: rec\_res num : 6, elapsed : 0.38327598571777344

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500203728675842)]

[2024/05/11 16:19:32] ppocr INFO: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE ONE TABLET BY', 0.9259032011032104)]

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345)]

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY90', 0.9742363691329956)]

[2024/05/11 16:19:32] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], (' Fied12-01-2019', 0.7921862006187439)]



(py38a) D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle> **paddleocr -- image\_dir ./drug1.jpg -- use\_angle\_cls true -- lang en -- use\_gpu true**

[2024/05/11 16:20:19] ppocr DEBUG: Namespace(alpha=1.0, alphacolor =(255, 255, 255), benchmark=False, beta=1.0, binarize =False, cls\_batch\_num =6, cls\_image\_shape ='3, 48, 192', cls\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\cls\\ch\_ppocr\_mobile\_v2.0\_cls\_infer', cls\_thresh =0.9, cpu\_threads =10, crop\_res\_save\_dir ='./output', det =True, det\_algorithm ='DB', det\_box\_type ='quad', det\_db\_box\_thresh =0.6, det\_db\_score\_mode ='fast', det\_db\_thresh =0.3, det\_db\_unclip\_ratio =1.5, det\_east\_cover\_thresh =0.1, det\_east\_nms\_thresh =0.2, det\_east\_score\_thresh =0.8, det\_limit\_side\_len =960, det\_limit\_type ='max', det\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\det\\en\\en\_PP-OCRv3\_det\_infer', det\_pse\_box\_thresh =0.85, det\_pse\_min\_area =16, det\_pse\_scale =1, det\_pse\_thresh =0, det\_sast\_nms\_thresh =0.2, det\_sast\_score\_thresh =0.5, draw\_img\_save\_dir ='./ inference\_results ', drop\_score =0.5, e2e\_algorithm=' PGNet ', e2e\_char\_dict\_path='./ ppocr / utils / ic15\_dict.txt', e2e\_limit\_side\_len=768, e2e\_limit\_type='max', e2e\_model\_dir=None, e2e\_pgnet\_mode='fast', e2e\_pgnet\_score\_thresh=0.5, e2e\_pgnet\_valid\_set=' totaltext ', enable\_mkldnn =False, fourier\_degree =5, gpu\_id =0, gpu\_mem =500, help='==SUPPRESS==', image\_dir ='./drug1.jpg', image\_orientation =False, invert=False, ir\_optim =True, kie\_algorithm =' LayoutXLM ', label\_list =['0', '180'], lang =' en ', layout=True, layout\_dict\_path =None, layout\_model\_dir =None, layout\_nms\_threshold =0.5, layout\_score\_threshold =0.5, max\_batch\_size =10, max\_text\_length =25, merge\_no\_span\_structure =True, min\_subgraph\_size =15, mode='structure', ocr =True, ocr\_order\_method =None, ocr\_version ='PP-OCRv4', output='./output', page\_num =0, precision='fp32', process\_id =0, re\_model\_dir =None, rec=True, rec\_algorithm =' SVTR\_LCNet ', rec\_batch\_num =6, rec\_char\_dict\_path='C:\\Users\\kristina\\anaconda3\\envs\\py38a\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en\_dict.txt', rec\_image\_inverse =True, rec\_image\_shape ='3, 48, 320', rec\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\rec\\en\\en\_PP-OCRv4\_rec\_infer', recovery=False, save\_crop\_res =False, save\_log\_path ='./ log\_output /', scales=[8, 16, 32], ser\_dict\_path ='../ train\_data /XFUND/class\_list\_xfun.txt', ser\_model\_dir =None, show\_log =True, sr\_batch\_num =1, sr\_image\_shape ='3, 32, 128', sr\_model\_dir =None, structure\_version ='PP-StructureV2', table=True, table\_algorithm =' TableAttn ', table\_char\_dict\_path =None, table\_max\_len =488, table\_model\_dir =None, total\_process\_num =1, type=' ocr ', use\_angle\_cls =True, use\_dilation =False, use\_gpu =True, use\_mp =False, use\_npu =False, use\_onnx =False, use\_pdf2docx\_api=False, use\_pdserving =False, use\_space\_char =True, use\_tensorrt =False, use\_visual\_backbone =True, use\_xpu =False, vis\_font\_path ='./doc/fonts/simfang.ttf', warmup=False)

[2024/05/11 16:20:27] ppocr INFO: \*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\*

[2024/05/11 16:20:28] ppocr DEBUG: dt\_boxes num : 6, elapsed : 1.0897586345672607

[2024/05/11 16:20:29] ppocr DEBUG: cls num : 6, elapsed : 0.5022566318511963

[2024/05/11 16:20:29] ppocr DEBUG: rec\_res num : 6, elapsed : 0.022243261337280273

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500204920768738) ]

[2024/05/11 16:20:29] ppocr INFO: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE ONE TABLET BY', 0.9259033203125)]

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345) ]

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY90', 0.9742364883422852)]

[2024/05/11 16:20:29] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], (' Fied12-01-2019', 0.7921867370605469)]


**Result - OK**

|D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle|Code folder ( .. für mein eigenes remember-me..)|
| :- | :- |
|||
|||
|![ref2]|<p>Result</p><p></p><p></p>|

**Conclusion:**

Bis auf die (üblichen) Verwirrungen bzgl. der richtigen CUDA-Version ist es  eine  straight-forward installation.
Haben mit den richtigen links und Versionen am ende ein gutes LIVE-CAM-OCR Ergebnis erzielt.

Reference YT videos :

|<p><https://www.youtube.com/watch?v=t5xwQguk9XU> </p><p></p>|Rip out Drug Labels using Deep Learning with PaddleOCR & Python|
| :- | :- |
|||
|<p><https://www.techspot.com/downloads/7241-visual-studio-2019.html></p><p></p>|Visual Studio is urgently needed BEFORE installing NVIDIA ( CUDA )|
|<p><https://github.com/PaddlePaddle/PaddleOCR></p><p></p>|Paddle OCR|

[ref1]: Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.001.png
[ref2]: Aspose.Words.98b36e84-dc3a-4782-ab5a-a3e12814d7ff.012.png
