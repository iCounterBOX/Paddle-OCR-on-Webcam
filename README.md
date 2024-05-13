**How2  	Anomaly Detection in Pre-Series  (ADPS) – AI Model LAB**

Date: 09.05.24

Basis für dieses LAB ist die doc:

D:\ALL\_DEVEL\_VIP\DOC\Anaconda\_PY\_ObjectDetect\_Tensor\_ORANGE\How2  Anomaly Detection in Pre\_2.docx

Die doc beschreibt für verschiedene Experimente die notwendigen ToDo-Schritte ( development-environment, Libs, etc ).



**Milestone(s)**

28\.04.25



**Pro:**

- Die Gui ….. Gui-unterstützt vereinfacht.

**Con:**

- SSIM …
  (MSE: 'Mean Squared Error' …….. is the -  sum of the squared difference between the two images )

**Conclusion**:

Wir …. der Anomaly-Detection..zB: **AutoEncoders**
*In the context of anomaly detection, AutoEncoders are particularly useful. **They are trained on normal data to learn the representation of the normal state**. During inference, if an input significantly deviates from this learned representation, the AutoEncoder will likely reconstruct it poorly. (<https://medium.com/@weidagang/demystifying-anomaly-detection-with-autoencoder-neural-networks-1e235840d879>)*

Referenz YT videos:

|<p><https://www.youtube.com/watch?v=t5xwQguk9XU></p><p></p>||
| :- | :- |
||Feedforward neural network|
||Generic alogithm|
|<p><https://www.youtube.com/watch?v=VnN5MYQnGak></p><p></p>|2d unity simulation|
||Train in live environment|
|<p><https://www.youtube.com/watch?v=Aut32pR5PQA></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.001.png)</p><p><https://www.youtube.com/watch?v=cO5g5qLrLSo&list=PLgNJO2hghbmjlE6cuKMws2ejC54BTAaWV></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.002.png)</p>|Deep reinforcement learning|
|||

Other sources:

|<p><https://github.com/nicknochnack/DrugLabelExtraction-></p><p></p>|Github jpyter paddle|
| :- | :- |
|||
|||





screen Capture & Recording: captura



Development environment:

` `**Jupyter notebook** in anaconda environment

|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.003.png)|<p>In anaconda 3 ist jupiter notebook gleich mitinstalliert!</p><p></p><p></p>|
| :- | :- |
|<p><https://www.youtube.com/watch?v=WUeBzT43JyY></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.004.png)</p>|Notebook basics|
|||



**Paddle OCR**

<https://www.youtube.com/watch?v=t5xwQguk9XU>

Rip out Drug Labels using Deep Learning with PaddleOCR & Python

|<p><https://www.paddlepaddle.org.cn/en></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.005.png)</p>|Dependency Matrix for paddle…also for paddleOCR?|
| :- | :- |
|<p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html></p><p></p><p>? important updates Add support for python3.12, and no longer</p><p>`    `supports python3.7</p><p>`    `Add support for CUDA 12.0, and no longer supports CUDA 10.2</p><p></p><p>Besser den?:</p><p></p><p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/conda/windows-conda_en.html></p><p></p><p><h4>***GPU Version of PaddlePaddle***</h4></p><p>- If you are using CUDA 11.2，cuDNN 8.2.1:</p><p>conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge</p><p>- If you are using CUDA 11.6，cuDNN 8.4.0:</p><p>conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge</p><p>- If you are using CUDA 11.7，cuDNN 8.4.1:</p><p>conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge</p><p></p><p></p>|<p>Environment</p><p></p><p></p><p></p><p></p><p>Hab py38</p><p></p><p>Werden mit cuda 11.6 starten </p>|
|<p><https://developer.nvidia.com/rdp/cudnn-archive></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.006.png)</p><p></p>|<p>cuDNN architechture download</p><p></p><p>11er runtergeladen</p>|
|<p><https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html></p><p></p>|Install cuda / nvidia /gpu|
|<p><https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p>·  Copy the following files from the unzipped package into the NVIDIA cuDNN directory.<a name="installwindows__substeps_zcd_xzm_s1b"></a> </p><p>a. Copy bin\cudnn\*.dll to C:\Program Files\NVIDIA\CUDNN\v8.x\bin.</p><p>b. Copy include\cudnn\*.h to C:\Program Files\NVIDIA\CUDNN\v8.x\include.</p><p>c. Copy lib\x64\cudnn\*.lib to C:\Program Files\NVIDIA\CUDNN\v8.x\lib\x64.</p><p>Yt how2:</p><p><https://www.youtube.com/watch?v=ctQi9mU7t9o></p><p></p><p></p>|NVIDIA CUDNN doc /install auf C|
|<p></p><p>1:        </p><p>conda install paddlepaddle-gpu==2.6.1 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge</p><p></p><p>2:</p><p>(py38) D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle>**conda install esri::paddleocr**</p><p></p>|<p>2 paddle things installed:</p><p></p><p>Paddleocr with conda</p><p><https://anaconda.org/esri/paddleocr></p><p></p>|
|<p>TEST:</p><p>(py38) P:\a\_Bosch\uc\_imageDifferences\py38\companyTasks></p><p>python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"</p><p></p>|<p>Compatible test - ok</p><p>64bit</p><p>AMD64</p>|
|||
|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.007.png) vs ![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.008.png)|<p>Version ok?..werden sehen…in der paddleocr doc  steht eine ältere version von cudnn als in der doc für die paddle installation ???</p><p></p><p></p><p>Bleibe mal bei der 11er</p>|

**Install CUDA cuDNN & VS !**

|<p><https://www.techspot.com/downloads/7241-visual-studio-2019.html></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.009.png)</p><p></p>|<p>Nvidia braucht vs 2019</p><p></p><p>*Issue..hatte erst vs2022…nvidia braucht aber 19…jetzt ist 19 installiert…nvidia neu aufspielen?...ich warte maf fehlermeldungen…*</p>|
| :- | :- |
|<p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.010.png)</p><p></p><p>https://www.youtube.com/watch?v=ctQi9mU7t9o</p>|<p>Install CUDA cuDNN & VS !</p><p></p><p></p>|
|||


|<p>nvidia-smi</p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.011.png)</p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.012.png)</p><p></p><p>nvcc --version</p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.013.png)</p>|Test CUDA / envidia etc|
| :- | :- |

**PROBLEM: 2  Anlauf**

Paddle test app zeigt fehler bei ocr vom image in np int!?  Vermutlich falsche CUDA toolkit?

|<p><https://support.microsoft.com/en-gb/topic/fix-problems-that-block-programs-from-being-installed-or-removed-cca7d1b6-65a9-3d98-426b-e9f927e1eb4d></p><p></p><p>darum gehts:</p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.014.png) </p><p></p><p>nicht die device driver…aber die sachen die fpr 2022 kamen müssen raus</p><p></p>|<p>Ms tool um nvidia treiber zu entfernen die NICHT mit uninstall entfernt werden können </p><p></p><p>Fix problems that block programs from being installed or removed</p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.015.png)</p>|
| :- | :- |
|||
|||
|<p>C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin</p><p>C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\libnvvp</p>|<p>Environment variablen erst mal raus</p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.016.png)</p>|
|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.017.png)|Nvidia ist wieder raus|
|||
|||
|<p><https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/environment_en.md></p><p></p><p>`    `PaddlePaddle >= 2.1.2</p><p>`    `Python 3.7</p><p>`    `CUDA 10.1 / CUDA 10.2</p><p>`    `cuDNN 7.6</p>|<p>Strikteres vorgehen!</p><p></p><p>**Vs 2019** ist jetzt schon installiert</p>|
|<p>- VS 2019</p><p>&emsp;- Anaconda py38a env</p><p>&emsp;- Cuda toolkit CUDA 10.1</p><p>&emsp;- cuDNN 7.6  <br>&emsp;  ` `<https://developer.nvidia.com/rdp/cudnn-archive></p><p>&emsp;- copy some cuDNN to conda <br>&emsp;  ` `<https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p>&emsp;- set win env variables – nvidia</p><p>&emsp;- test gpu</p><p>&emsp;&emsp;</p>|7 PUNKTE PLAN|
|<p><https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/quickstart_en.md></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.018.png)</p><p></p><p></p>|**DAS soll jetzt die Vorgabe sein**|
|<p><h3>**1.1 Install PaddlePaddle**</h3></p><p>If you do not have a Python environment, please refer to [Environment Preparation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/environment_en.md).</p><p>- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install</p><p>&emsp;**python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple**</p><p>If you have no available GPU on your machine, please run the following command to install the CPU version</p><p>python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple</p><p>For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/install/quick) for operation.</p><p><h3><a name="user-content-12-install-paddleocr-whl-pa"></a>**1.2 Install PaddleOCR Whl Package**</h3></p><p>pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+</p><p>- **For windows users:** If you getting this error OSError: [WinError 126] The specified module could not be found when you install shapely on windows. Please try to download Shapely whl file [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely).</p><p>&emsp;Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)</p><p></p>|**Todo**|
|<p><h3>**C:\Windows\System32>nvcc -V**</h3></p><p><h3>**nvcc: NVIDIA (R) Cuda compiler driver**</h3></p><p><h3>**Copyright (c) 2005-2019 NVIDIA Corporation**</h3></p><p><h3>**Built on Sun\_Jul\_28\_19:12:52\_Pacific\_Daylight\_Time\_2019**</h3></p><p><h3>**Cuda compilation tools, release 10.1, V10.1.243**</h3></p><p></p><p></p><p>nvidia-smi</p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.019.png)  WTF 11.6</p><p></p><p></p><p></p><p></p>|<p>**7 point plan done!  Es geht schon wieder los L**</p><p></p><p></p><p>**Für den moment ignorieren…???**</p>|

|<p><h3>**[2024/05/11 14:57:46] ppocr INFO: \*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\***</h3></p><p><h3>**[2024/05/11 14:57:46] ppocr DEBUG: dt\_boxes num : 6, elapsed : 0.3679347038269043**</h3></p><p><h3>**[2024/05/11 14:57:46] ppocr DEBUG: cls num  : 6, elapsed : 0.09413385391235352**</h3></p><p><h3>**[2024/05/11 14:57:47] ppocr DEBUG: rec\_res num  : 6, elapsed : 0.32153964042663574**</h3></p><p><h3>**[2024/05/11 14:57:47] ppocr INFO: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500203728675842)]**</h3></p><p><h3>**[2024/05/11 14:57:47] ppocr INFO: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE ONE TABLET BY', 0.9259032011032104)]**</h3></p><p><h3>**[2024/05/11 14:57:47] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345)]**</h3></p><p><h3>**[2024/05/11 14:57:47] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY90', 0.9742363691329956)]**</h3></p><p><h3>**[2024/05/11 14:57:47] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], (' Fied12-01-2019', 0.7921862006187439)]**</h3></p><p></p><p></p><p></p>|**Test OHNE gpu ok  J**|
| :- | :- |
|<p><h3>**RuntimeError: (PreconditionNotMet) The third-party dynamic library (cudnn64\_8.dll) that Paddle depends on is not configured correctly. (error code is 126)**</h3></p><p><h3>`  `**Suggestions:**</h3></p><p><h3>`  `**1. Check if the third-party dynamic library (e.g. CUDA, CUDNN) is installed correctly and its version is matched with paddlepaddle you installed.**</h3></p><p><h3>`  `**2. Configure third-party dynamic library environment variables as follows:**</h3></p><p><h3>`  `**- Linux: set LD\_LIBRARY\_PATH by `export LD\_LIBRARY\_PATH=...`**</h3></p><p><h3>`  `**- Windows: set PATH by `set PATH=XXX; (at ..\paddle\phi\backends\dynload\dynamic\_loader.cc:312)**</h3></p><p></p><p>We check:</p><p></p><p>conda list</p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.020.png)</p><p></p><p>We search for compatibility..</p><p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html></p><p></p><p>*paddlepaddle-gpu==[version code], such as paddlepaddle-gpu==2.6.1 	The default installation supports the PaddlePaddle installation package corresponding to [version number] of CUDA 11.2 and cuDNN 8*</p><p></p><p>**SO the doc above in paddle OCR is wrong…following the error mrssage I will now install  *CUDA 11.2 and cuDNN 8***</p>|<p>**Test MIT gpu ERROR**</p><p></p><p>**paddleocr --image\_dir ./drug1.jpg --use\_angle\_cls true --lang en --use\_gpu true**</p><p></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.021.png)</p><p>**<br></p>|
|<h3></h3>||


**PROBLEM: 3 Anlauf**

|<p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html></p><p></p><p>![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.022.png)</p><p></p>|<p>**GTX 1660 TI**</p><p></p><p>**„Tuning sm\_75“**</p>|
| :- | :- |
|Cuda 11.2  und cuDNN8|Holen uns ..|
|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.023.png)|Entfernen|
|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.024.png)|<p>**Cuda 11.2**</p><p>**..sieht besser aus**</p><p></p><p></p><p>Beim 10er war der kasten leer!?  Ohne meldung ohne fehler??</p>|
|<p></p><p>**CUDA 11.2 and cuDNN 8**</p>|<p>**Environment wie oben aber für 11.2 eingerichtet**</p><p></p><p>**PASST**</p>|




Erster ERFOLGREICHER TEST:


(py38a) D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle>**paddleocr --image\_dir ./drug1.jpg --use\_angle\_cls true --lang en --use\_gpu false**

[2024/05/11 16:19:29] ppocr DEBUG: Namespace(alpha=1.0, alphacolor=(255, 255, 255), benchmark=False, beta=1.0, binarize=False, cls\_batch\_num=6, cls\_image\_shape='3, 48, 192', cls\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\cls\\ch\_ppocr\_mobile\_v2.0\_cls\_infer', cls\_thresh=0.9, cpu\_threads=10, crop\_res\_save\_dir='./output', det=True, det\_algorithm='DB', det\_box\_type='quad', det\_db\_box\_thresh=0.6, det\_db\_score\_mode='fast', det\_db\_thresh=0.3, det\_db\_unclip\_ratio=1.5, det\_east\_cover\_thresh=0.1, det\_east\_nms\_thresh=0.2, det\_east\_score\_thresh=0.8, det\_limit\_side\_len=960, det\_limit\_type='max', det\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\det\\en\\en\_PP-OCRv3\_det\_infer', det\_pse\_box\_thresh=0.85, det\_pse\_min\_area=16, det\_pse\_scale=1, det\_pse\_thresh=0, det\_sast\_nms\_thresh=0.2, det\_sast\_score\_thresh=0.5, draw\_img\_save\_dir='./inference\_results', drop\_score=0.5, e2e\_algorithm='PGNet', e2e\_char\_dict\_path='./ppocr/utils/ic15\_dict.txt', e2e\_limit\_side\_len=768, e2e\_limit\_type='max', e2e\_model\_dir=None, e2e\_pgnet\_mode='fast', e2e\_pgnet\_score\_thresh=0.5, e2e\_pgnet\_valid\_set='totaltext', enable\_mkldnn=False, fourier\_degree=5, gpu\_id=0, gpu\_mem=500, help='==SUPPRESS==', image\_dir='./drug1.jpg', image\_orientation=False, invert=False, ir\_optim=True, kie\_algorithm='LayoutXLM', label\_list=['0', '180'], lang='en', layout=True, layout\_dict\_path=None, layout\_model\_dir=None, layout\_nms\_threshold=0.5, layout\_score\_threshold=0.5, max\_batch\_size=10, max\_text\_length=25, merge\_no\_span\_structure=True, min\_subgraph\_size=15, mode='structure', ocr=True, ocr\_order\_method=None, ocr\_version='PP-OCRv4', output='./output', page\_num=0, precision='fp32', process\_id=0, re\_model\_dir=None, rec=True, rec\_algorithm='SVTR\_LCNet', rec\_batch\_num=6, rec\_char\_dict\_path='C:\\Users\\kristina\\anaconda3\\envs\\py38a\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en\_dict.txt', rec\_image\_inverse=True, rec\_image\_shape='3, 48, 320', rec\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\rec\\en\\en\_PP-OCRv4\_rec\_infer', recovery=False, save\_crop\_res=False, save\_log\_path='./log\_output/', scales=[8, 16, 32], ser\_dict\_path='../train\_data/XFUND/class\_list\_xfun.txt', ser\_model\_dir=None, show\_log=True, sr\_batch\_num=1, sr\_image\_shape='3, 32, 128', sr\_model\_dir=None, structure\_version='PP-StructureV2', table=True, table\_algorithm='TableAttn', table\_char\_dict\_path=None, table\_max\_len=488, table\_model\_dir=None, total\_process\_num=1, type='ocr', use\_angle\_cls=True, use\_dilation=False, use\_gpu=False, use\_mp=False, use\_npu=False, use\_onnx=False, use\_pdf2docx\_api=False, use\_pdserving=False, use\_space\_char=True, use\_tensorrt=False, use\_visual\_backbone=True, use\_xpu=False, vis\_font\_path='./doc/fonts/simfang.ttf', warmup=False)

[2024/05/11 16:19:30] ppocr INFO: \*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\*

[2024/05/11 16:19:31] ppocr DEBUG: dt\_boxes num : 6, elapsed : 0.5120675563812256

[2024/05/11 16:19:31] ppocr DEBUG: cls num  : 6, elapsed : 0.10241532325744629

[2024/05/11 16:19:32] ppocr DEBUG: rec\_res num  : 6, elapsed : 0.38327598571777344

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500203728675842)]

[2024/05/11 16:19:32] ppocr INFO: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE ONE TABLET BY', 0.9259032011032104)]

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345)]

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY90', 0.9742363691329956)]

[2024/05/11 16:19:32] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], (' Fied12-01-2019', 0.7921862006187439)]








(py38a) D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle>**paddleocr --image\_dir ./drug1.jpg --use\_angle\_cls true --lang en --use\_gpu true**

[2024/05/11 16:20:19] ppocr DEBUG: Namespace(alpha=1.0, alphacolor=(255, 255, 255), benchmark=False, beta=1.0, binarize=False, cls\_batch\_num=6, cls\_image\_shape='3, 48, 192', cls\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\cls\\ch\_ppocr\_mobile\_v2.0\_cls\_infer', cls\_thresh=0.9, cpu\_threads=10, crop\_res\_save\_dir='./output', det=True, det\_algorithm='DB', det\_box\_type='quad', det\_db\_box\_thresh=0.6, det\_db\_score\_mode='fast', det\_db\_thresh=0.3, det\_db\_unclip\_ratio=1.5, det\_east\_cover\_thresh=0.1, det\_east\_nms\_thresh=0.2, det\_east\_score\_thresh=0.8, det\_limit\_side\_len=960, det\_limit\_type='max', det\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\det\\en\\en\_PP-OCRv3\_det\_infer', det\_pse\_box\_thresh=0.85, det\_pse\_min\_area=16, det\_pse\_scale=1, det\_pse\_thresh=0, det\_sast\_nms\_thresh=0.2, det\_sast\_score\_thresh=0.5, draw\_img\_save\_dir='./inference\_results', drop\_score=0.5, e2e\_algorithm='PGNet', e2e\_char\_dict\_path='./ppocr/utils/ic15\_dict.txt', e2e\_limit\_side\_len=768, e2e\_limit\_type='max', e2e\_model\_dir=None, e2e\_pgnet\_mode='fast', e2e\_pgnet\_score\_thresh=0.5, e2e\_pgnet\_valid\_set='totaltext', enable\_mkldnn=False, fourier\_degree=5, gpu\_id=0, gpu\_mem=500, help='==SUPPRESS==', image\_dir='./drug1.jpg', image\_orientation=False, invert=False, ir\_optim=True, kie\_algorithm='LayoutXLM', label\_list=['0', '180'], lang='en', layout=True, layout\_dict\_path=None, layout\_model\_dir=None, layout\_nms\_threshold=0.5, layout\_score\_threshold=0.5, max\_batch\_size=10, max\_text\_length=25, merge\_no\_span\_structure=True, min\_subgraph\_size=15, mode='structure', ocr=True, ocr\_order\_method=None, ocr\_version='PP-OCRv4', output='./output', page\_num=0, precision='fp32', process\_id=0, re\_model\_dir=None, rec=True, rec\_algorithm='SVTR\_LCNet', rec\_batch\_num=6, rec\_char\_dict\_path='C:\\Users\\kristina\\anaconda3\\envs\\py38a\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en\_dict.txt', rec\_image\_inverse=True, rec\_image\_shape='3, 48, 320', rec\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\rec\\en\\en\_PP-OCRv4\_rec\_infer', recovery=False, save\_crop\_res=False, save\_log\_path='./log\_output/', scales=[8, 16, 32], ser\_dict\_path='../train\_data/XFUND/class\_list\_xfun.txt', ser\_model\_dir=None, show\_log=True, sr\_batch\_num=1, sr\_image\_shape='3, 32, 128', sr\_model\_dir=None, structure\_version='PP-StructureV2', table=True, table\_algorithm='TableAttn', table\_char\_dict\_path=None, table\_max\_len=488, table\_model\_dir=None, total\_process\_num=1, type='ocr', use\_angle\_cls=True, use\_dilation=False, use\_gpu=True, use\_mp=False, use\_npu=False, use\_onnx=False, use\_pdf2docx\_api=False, use\_pdserving=False, use\_space\_char=True, use\_tensorrt=False, use\_visual\_backbone=True, use\_xpu=False, vis\_font\_path='./doc/fonts/simfang.ttf', warmup=False)

[2024/05/11 16:20:27] ppocr INFO: \*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\*

[2024/05/11 16:20:28] ppocr DEBUG: dt\_boxes num : 6, elapsed : 1.0897586345672607

[2024/05/11 16:20:29] ppocr DEBUG: cls num  : 6, elapsed : 0.5022566318511963

[2024/05/11 16:20:29] ppocr DEBUG: rec\_res num  : 6, elapsed : 0.022243261337280273

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500204920768738)]

[2024/05/11 16:20:29] ppocr INFO: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE ONE TABLET BY', 0.9259033203125)]

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345)]

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY90', 0.9742364883422852)]

[2024/05/11 16:20:29] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], (' Fied12-01-2019', 0.7921867370605469)]



**OK – erkennung läuft mit PaddleOCR – ERSTE model basierende erkennung**

|D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle|Code folder|
| :- | :- |
|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.025.png)||
|<p># -\*- coding: utf-8 -\*-</p><p>"""</p><p>Created on Fri May 10 16:27:06 2024</p><p></p><p>@author: kristina</p><p></p><p>source:</p><p>`    `https://www.youtube.com/watch?v=t5xwQguk9XU</p><p>    </p><p></p><p>paddle OCR - läuft perfekt auf normalen Text</p><p>POOR bei curved Text</p><p></p><p>np guru  https://www.youtube.com/watch?v=oyqNdcbKhew</p><p></p><p></p><p></p><p>"""</p><p></p><p></p><p></p><p>from paddleocr import PaddleOCR, draw\_ocr</p><p>import paddle</p><p>import cv2</p><p>import os</p><p>import pkg\_resources</p><p>import sys</p><p></p><p></p><p></p><p></p><p>def showInMovedWindow(  winname, img, x, y):</p><p>`    `cv2.namedWindow(winname)        # Create a named window</p><p>`    `cv2.moveWindow(winname, x, y)   # Move it to (x,y)...THis way the image ma appear on TOP of other screens!</p><p>`    `cv2.imshow(winname,img)</p><p></p><p>#some add-info</p><p>os.environ["KMP\_DUPLICATE\_LIB\_OK"]="TRUE"</p><p>gpu\_available  = paddle.device.is\_compiled\_with\_cuda()</p><p>print("GPU available:", gpu\_available)</p><p></p><p>print("cv2.\_\_file\_\_: " + cv2.\_\_file\_\_) </p><p>print ( "cv2 Version: " +  cv2. \_\_version\_\_ )</p><p>print ( "paddleOCR  Version: " +  pkg\_resources.get\_distribution("paddleocr").version )</p><p></p><p>#TEST-Image</p><p></p><p>img\_path =  os.getcwd() + '\drug1.jpg'</p><p>imgCV = cv2.imread(img\_path, cv2.IMREAD\_UNCHANGED )</p><p></p><p></p><p>#erkennt nur die hälfte vom purLabel - schlecht bei curved text!</p><p># https://learnopencv.com/optical-character-recognition-using-paddleocr/</p><p></p><p>ocr = PaddleOCR( use\_angle\_cls =True, lang = 'en', use\_gpu=True, det\_limit\_side\_len=3456) </p><p>result = ocr.ocr(img\_path, cls=True)</p><p></p><p></p><p></p><p></p><p>#pprint(result)</p><p></p><p>#extract  single arrays from result-list</p><p></p><p>scores = []</p><p>boxes = []</p><p>texts = []</p><p>for xx in result:     </p><p>`    `#print(xx)</p><p>`    `i=0</p><p>`    `try:</p><p>`        `while xx[i]:</p><p>`            `box, textScore = xx[i]</p><p>`            `#print(box)</p><p>`            `#print(textScore)            </p><p>`            `txt, scr = textScore</p><p>`            `print(txt + " - " + str(scr) )</p><p>`            `scores.append(scr)</p><p>`            `boxes.append(box)</p><p>`            `texts.append(txt)</p><p>`            `i+=1                    </p><p>`    `except:</p><p>`        `pass    </p><p>    </p><p>  </p><p></p><p></p><p># Specifying font path for draw\_ocr method</p><p>font = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')</p><p>  </p><p></p><p># draw annotations on image</p><p>im\_show= draw\_ocr(imgCV, boxes, texts, scores, font\_path=font) </p><p>showInMovedWindow(  "paddle OCR", im\_show, 100, 200)</p><p>cv2.waitKey(0)</p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p></p><p>print ("ende")</p><p></p><p>sys.exit()</p><p></p><p></p><p></p>|Simpler run auf ein image..curved..ergebnis nicht |
|![](Aspose.Words.b46135cb-24dc-48b0-b26d-721ad9951d2a.026.png)|Result|
|<p></p><p>#extract  single arrays from result-list</p><p></p><p>scores = []</p><p>boxes = []</p><p>texts = []</p><p>for xx in result:     </p><p>`    `#print(xx)</p><p>`    `i=0</p><p>`    `try:</p><p>`        `while xx[i]:</p><p>`            `box, textScore = xx[i]</p><p>`            `#print(box)</p><p>`            `#print(textScore)            </p><p>`            `txt, scr = textScore</p><p>`            `print(txt + " - " + str(scr) )</p><p>`            `scores.append(scr)</p><p>`            `boxes.append(box)</p><p>`            `texts.append(txt)</p><p>`            `i+=1                    </p><p>`    `except:</p><p>`        `pass    </p><p>    </p><p></p><p>der rest war ok</p><p>  </p>|<p>Issues – </p><p><https://www.youtube.com/watch?v=t5xwQguk9XU></p><p></p><p>die codes in den videobeispielen funktionieren nicht mehr? Anderes result array?  Musste das anpassen</p>|


