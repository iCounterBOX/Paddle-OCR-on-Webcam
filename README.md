**Paddle OCR on webcam (Win 10)**

The document describes the backlog steps required for various experiments ( development environment, libraries, etc. ) related to paddle-OCR via webCam.

![ref1]

**My environment:**

|Acer Nitro 5 / GTX 1660 Ti (GPU available)|
| :- |
|Anaconda 3|
|Payton 3.8|
|Spyder Editor|
|<p>opencv 4.6.0        ( problem with conda install..mus use pip !! )<br><br>pip install opencv-contrib-python    / <https://stackoverflow.com/questions/23119413/how-do-i-install-python-opencv-through-conda></p><p><br><br></p>|
|padelocr                 2.7.3|
|rowing gpu          2.6.1|
|python 3.8.19|
|spy 5.5.1|
||
|CUDA 11.2 and cuDNN 8|
|against 2019|
||
|Word File English Translation with ASPOSE|

|<p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html></p><p></p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.002.png)</p><p></p>|<p>GTX 1660TI</p><p></p><p>“Tuning sm\_75”</p>|
| :- | :- |

**

**Last minute Extra issues after all was running fine  ( 13.05.24 ):** 

**After running auto-py-to\_exe ..it requires lots of packages extra for the conversion. This was not alerted during the conversion process.. it was dropwise fired when exe was running ( crashed ).**

**This try and error destroyed my nice running py38…so I was forced to rebuild a fresh environment ..also py38**

**My devel environment is Anaconda3. ..As we well know it is not good to mix conda and pip setups I tried again to “conda” as often as possible…but some packs only offer pip..**

**After 2 not working py38 env´s I tried to setup ALL in a fresh anaconda py38 env…This works fine!?**

**Here is my last py38 setup with only pip installs:**   

**pip install opencv-contrib-python**

**pip install scikit-image**

**pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple**

**pip install paddleocr**  

**pip install pytesseract**

**pip install auto-py-to-exe**

**auto-py-to\_exe is now also ok and running:**

**command:**

**pyinstaller --noconfirm --onedir --console --collect-all "paddleocr" --hidden-import "scipy.io" --hidden-import "pyclipper" --collect-data "paddle" --hidden-import "scipy" --collect-all "scipy" --hidden-import "skimage" --collect-all "skimage" --hidden-import "imgaug" --collect-all "imgaug" --hidden-import "lmdb" --collect-all "lbdb"  "D:/ALL\_PROJECT/pyQT5\_experimental/ai/paddle4github/ppOCR\_webcam.py"**

![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.003.png)

![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.004.png)

**Now App is working in Python anaconda spyder and/or as a EXE-file**

**Don’t forget to copy paddleOCR (githubClone) into the EXE folder:**

![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.005.png)

**


**OCR/Webcam palette in Win 10**

|<p><https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/environment_en.md></p><p></p><p>combination specified there. ( CUDA 10.2/ cuDNN 7.6) did NOT work in my environment </p><p>*f you have CUDA 9 or CUDA 10 installed on your machine, run the following command to install*</p><p>*python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple*</p><p></p><p>*…this is installing paddlepaddle-gpu          2.6.1*</p><p></p>|<p>**Adjust** versions!</p><p>Confusion in Cuda - version information at Paddle Docs</p><p></p><p>10 did NOT work for me.</p><p>Only the errors that appeared after installation 10 told me the correct version of Cuda ...</p>|
| :- | :- |

##### **In shorts :  setup in CUDA 11.2 and cuDNN 8 was OK on my side**

**Below are some notes I made during installation.**

|<p>- Visual studio 2019</p><p>&emsp;- Anaconda py38 environment invest</p><p>&emsp;- Install <br>&emsp;  CUDA 11.2 and cuDNN 8 on Win <https://developer.nvidia.com/rdp/cudnn-archive></p><p>&emsp;- copy some cuDNN libraries into conda toolkit in Win <br>&emsp;  [https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows ](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows)<br>&emsp;  ( but be careful .. look nor the comment relating to . atmosphere variables …)</p><p>&emsp;- Install in py38 env : <br>&emsp;  pip install paddleocr    # install  (2.7.3)<br>&emsp; </p><p>&emsp;- Install in py38 env : <br>&emsp;  python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple</p><p>&emsp;&emsp;</p><p>&emsp;- set win environment variables for  nvidia</p><p>&emsp;- gpu test</p><p></p>|Individual steps – order is sometimes important. <br>For example, VS 2019 should be there before CUDA!||
| :- | :- | :- |
|![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.006.png)|<p>win10 environment variables</p><p></p>||
|<p><https://developer.nvidia.com/rdp/cudnn-archive></p><p><https://developer.nvidia.com/cuda-11.2.0-download-archive></p><p><https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p></p><p></p>|<p>cuDNN architecture download</p><p>CUDA 11.2 and cuDNN 8</p><p>Tools kit</p>||
|<p><https://www.techspot.com/downloads/7241-visual-studio-2019.html></p><p></p><p></p><p></p>|![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.007.png)||
|<p><https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-840/install-guide/index.html#install-windows></p><p>· Copy the following files from the unzipped package to the NVIDIA cuDNN directory .<a name="installwindows__substeps_zcd_xzm_s1b"></a> </p><p>a. Copy bin\cudnn\*.dll to C:\Program Files\NVIDIA\CUDNN\v8.x\bin .</p><p>b. Copy include\ cudnn \*.h to C:\Program Files\NVIDIA\CUDNN\v8.x\include .</p><p>c. Copy lib\x64\cudnn\*.lib to C:\Program Files\NVIDIA\CUDNN\v8.x\lib\x64 .</p><p>Yt how2:</p><p><https://www.youtube.com/watch?v=ctQi9mU7t9o></p><p></p><p></p>|<p>NVIDIA CUDNN doc/installation in C?</p><p></p><p>That didn't work for me ...</p><p>see screenshot of my environment above</p>||
|<p>python -c "import platform; print (platform.architecture()[0]); print(platform.machine())"</p><p></p>|<p>Test compatible - ok</p><p>64 bit</p><p>AMD64</p>||
|<p>nvidia-smi</p><p></p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.008.png)</p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.009.png)</p><p></p><p>nvcc - version</p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.010.png)</p>|<p>CUDA/ envy test Etc</p><p></p><p>nvidia-smi</p><p>nvcc - version</p><p></p><p>Affair :</p><p></p><p>Invidia-smi still shows 11.6. Lots of discussion on GitHub about it. Apparently it doesn't matter as long as nvcc shows the right thing!</p><p>Maybe ...continue for now... it still says 11.6 even though everything works in 11.2!?</p><p></p><p>We already had these problems.</p><p>5 years ( <https://github.com/iCounterBOX/TensorFlow_CarOccupancyDetector_V2> )</p><p></p>||
|<p>Fix issues that block programs from installing or removing</p><p></p><p><https://support.microsoft.com/en-gb/topic/fix-problems-that-block-programs-from-being-installed-or-removed-cca7d1b6-65a9-3d98-426b-e9f927e1eb4d></p><p></p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.011.png)</p><p></p><p>*not the device driver... but things that arrived for 2022 (for example) should be removed*</p>|<p>What to do if CUDA ( nvidia ) version / installation is wrong ?</p><p></p><p>Apps must be uninstalled individually!</p><p>Some are locked and cannot be easily uninstalled.</p><p>Microsoft's program did that.</p>||
|<p><h3>**addleocr -- image\_dir ./ imgs\_en /img\_12.jpg -- use\_angle\_cls true --lang is --use\_gpu false.false**</h3></p><p><h3></h3></p><p><h3></h3></p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.012.png)</p><p></p>|<p>**Try WITHOUT GPU okJ**</p><p></p><p>[**https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/quickstart_en.md#21-use-by-command-line**](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/quickstart_en.md#21-use-by-command-line)</p><p></p>||
|<p><h3>**RuntimeError : ( PreconditionNotMet ) The third-party dynamic library (cudnn64\_8.dll) that Paddle depends on is not configured correctly. ( error code is 126)**</h3></p><p><h3>**Suggestions:**</h3></p><p><h3>**1. Check whether the third-party dynamic library (e.g. CUDA, CUDNN) is installed correctly and whether its version matches the paddlepaddle you installed.**</h3></p><p><h3>**2. Configure the third-party dynamic library environment variables as follows:**</h3></p><p><h3>**- Linux: set LD\_LIBRARY\_PATH using `export LD\_LIBRARY\_PATH=...`**</h3></p><p><h3>**- Windows: set PATH using `set PATH=XXX; (in .. \paddle\phi\ backends \ dynload \dynamic\_loader.cc:312)**</h3></p><p></p><p>We verify:</p><p></p><p>count list</p><p>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.013.png)</p><p></p><p>We are looking for compatibility.</p><p><https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html></p><p></p><p>*paddlepaddle-gpu ==[version code], such as paddlepaddle-gpu ==2.6.1 The default installation supports the 	PaddlePaddle installation package corresponding to [version number] of CUDA 11.2 and cuDNN 8*</p><p></p><p></p>|<p>**Testing with GPU showed an error . The CUDA version was still 10. This error message gave us the correct/compatible version cuda version !**</p><p></p><p>**paddleocr -- image\_dir ./drug1.jpg -- use\_angle\_cls true -- long es -- use\_gpu TRUE**</p><p></p><p></p><p></p><p>**So, the above document in paddle OCR didn't work for me... following the error message, now I will install CUDA 11.2 and cuDNN 8**</p>||
|<h3>![](Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.014.png)</h3>|<p>Cuda 11.2.. it seems better</p><p></p><p>...this is what the message should look like</p><p></p>||




**First SUCCESSFUL TEST:**

(droga1.jpg is any image with source!)


(py38a) D:\ALL\_PROJECT\pyQT5\_experimental\ai\paddle> **paddleocr -- image\_dir ./drug1.jpg -- use\_angle\_cls true -- long en -- use\_gpu false**

[2024/05/11 16:19:29] ppocr DEBUG: Namespace (alpha=1.0, alphacolor =(255, 255, 255), benchmark=False, beta=1.0, binarize =False, cls\_batch\_num =6 , cls\_image\_shape =' 3, 48, 192', cls\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\cls\\ch\_ppocr\_mobile\_v2.0\_cls\_infer', cls\_thresh =0.9, cpu\_threads =10, crop\_res\_save\_dir ='./ output', det = True, det\_algorithm ='DB', det\_box\_type ='quad', det\_db\_box\_thresh =0.6, det\_db\_score\_mode ='fast', det\_db\_thresh =0.3 , det\_db\_unclip\_ratio =1.5, det\_east\_cover\_thresh =0.1, det\_east\_nms\_thresh = 0.2, sh =0.8, det\_limit\_side\_len =960, det\_limit\_type ='max', det\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\det\\en\\en\_PP-OCRv3\_det\_infer', det\_pse\_box\_thresh =0.85, det\_pse\_min\_area =16, det\_pse\_scale =1, det\_pse\_thresh =0, det\_sast\_nms\_thresh =0.2, det\_sast\_score\_thresh =0.5, draw\_img\_save\_dir ='./ inference\_results ', drop\_score =0.5, e2e\_algorithm=' PGNet ', e2e\_char\_dict\_path='./ ppocr / utils /ic15\_dict.txt' , e2e\_limit\_side\_len =7 68, e2e\_limit\_type ='max', e2e\_model\_dir=None, e2e\_pgnet\_mode='fast', e2e\_pgnet\_score\_thresh=0.5, e2e\_pgnet\_valid\_set=' totaltext ', enable\_mkldnn =False, fourier\_grade =5, gpu\_id =0, gpu\_mem =500, help='=SUPPRESS==' , image\_dir ='. /drug1.jpg', image\_orientation =False, invert=False, ir\_optim =True, kie\_algorithm =' LayoutXLM ', label\_list =['0', '180'], lang =' en ', layout=True, layout\_dict\_path =None, layout\_model\_dir =None, layout\_nms\_threshold =0.5, layout\_score\_threshold =0.5, max\_batch\_size =10, max\_text\_length =25, merge\_no\_span\_structure =True, min\_subgraph\_size =15, mode='structure', ocr =True, ocr\_order\_method =None, ocr\_version ='PP-OCRv4', output='./output', page\_num =0, precision='fp32', process\_id =0, re\_model\_dir =None, rec=True, rec\_algorithm =' SVTR\_LCNet ', rec\_batch\_num =6, rec\_char\_dict\_path='C:\\Users\\ kristina\\anaconda3\\envs\\py38a\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en\_dict.txt', rec\_image\_inverse =True, rec\_image\_shape ='3, 48, 320', rec\_model\_dir= 'C:\\Users\\kristina/.paddleocr/whl\\rec\\en\\en\_PP-OCRv4\_rec\_infer', recovery=False, save\_crop\_res =False, save\_log\_path ='./ log\_output /', scales=[8, 16 , 32], ser\_dict\_path ='../ train\_data /XFUND/class\_list\_xfun.txt', ser\_model\_dir =None, show\_log =True, sr\_batch\_num =1, sr\_image\_shape ='3, 32, 128', sr\_model\_dir =None, structure\_version ='PP- StructureV2', table=True, table\_algorithm =' TableAttn ', table\_char\_dict\_path =None, table\_max\_len =488, table\_model\_dir =None, total\_process\_num =1, type=' ocr ', use\_angle\_cls =True, use\_dilation =False, use\_gpu =False, use\_mp =False , use\_npu =False, use\_onnx =False, use\_pdf2docx\_api=False, use\_pdserving =False, use\_space\_char =True, use\_tensorrt =False, use\_visual\_backbone =True, use\_xpu =False, vis\_font\_path ='./doc/fonts/simfang.ttf', warmup =False )

[2024/05/11 16:19:30] ppocr INFORMATION: \*\*\*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\*\*\*

[2024/05/11 16:19:31] ppocr DEBUG: dt\_boxes number : 6, elapsed : 0.5120675563812256

[2024/05/11 16:19:31] ppocr DEBUG: cls number : 6, elapsed : 0.10241532325744629

[2024/05/11 16:19:32] ppocr DEBUG: rec\_res number : 6, elapsed: 0.38327598571777344

[2024/05/11 16:19:32] ppocr INFORMATION: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500203728675842) ]

[2024/05/11 16:19:32] ppocr INFORMATION: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE A TABLET BY', 0.9259032011032104)]

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345) ]

[2024/05/11 16:19:32] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY.90', 0.9742363691329956 )]

[2024/05/11 16:19:32] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], ('Fied12-01-2019' , 0.7921862006187439)]



(py38a) D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle> **paddleocr -- image\_dir ./drug1.jpg -- use\_angle\_cls true -- long is -- use\_gpu true**

[2024/05/11 16:20:19] ppocr DEBUG: Namespace (alpha=1.0, alphacolor =(255, 255, 255), benchmark=False, beta=1.0, binarize =False, cls\_batch\_num =6 , cls\_image\_shape =' 3, 48, 192', cls\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\cls\\ch\_ppocr\_mobile\_v2.0\_cls\_infer', cls\_thresh =0.9, cpu\_threads =10, crop\_res\_save\_dir ='./ output', det = True, det\_algorithm ='DB', det\_box\_type ='quad', det\_db\_box\_thresh =0.6, det\_db\_score\_mode ='fast', det\_db\_thresh =0.3 , det\_db\_unclip\_ratio =1.5, det\_east\_cover\_thresh =0.1, det\_east\_nms\_thresh = 0.2, sh =0.8, det\_limit\_side\_len =960, det\_limit\_type ='max', det\_model\_dir='C:\\Users\\kristina/.paddleocr/whl\\det\\en\\en\_PP-OCRv3\_det\_infer', det\_pse\_box\_thresh =0.85, det\_pse\_min\_area =16, det\_pse\_scale =1, det\_pse\_thresh =0, det\_sast\_nms\_thresh =0.2, det\_sast\_score\_thresh =0.5, draw\_img\_save\_dir ='./ inference\_results ', drop\_score =0.5, e2e\_algorithm=' PGNet ', e2e\_char\_dict\_path='./ ppocr / utils /ic15\_dict.txt' , e2e\_limit\_side\_len =7 68, e2e\_limit\_type ='max', e2e\_model\_dir=None, e2e\_pgnet\_mode='fast', e2e\_pgnet\_score\_thresh=0.5, e2e\_pgnet\_valid\_set=' totaltext ', enable\_mkldnn =False, fourier\_grade =5, gpu\_id =0, gpu\_mem =500, help='=SUPPRESS==' , image\_dir ='. /drug1.jpg', image\_orientation =False, invert=False, ir\_optim =True, kie\_algorithm =' LayoutXLM ', label\_list =['0', '180'], lang =' en ', layout=True, layout\_dict\_path =None, layout\_model\_dir =None, layout\_nms\_threshold =0.5, layout\_score\_threshold =0.5, max\_batch\_size =10, max\_text\_length =25, merge\_no\_span\_structure =True, min\_subgraph\_size =15, mode='structure', ocr =True, ocr\_order\_method =None, ocr\_version ='PP-OCRv4', output='./output', page\_num =0, precision='fp32', process\_id =0, re\_model\_dir =None, rec=True, rec\_algorithm =' SVTR\_LCNet ', rec\_batch\_num =6, rec\_char\_dict\_path='C:\\Users\\ kristina\\anaconda3\\envs\\py38a\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en\_dict.txt', rec\_image\_inverse =True, rec\_image\_shape ='3, 48, 320', rec\_model\_dir= 'C:\\Users\\kristina/.paddleocr/whl\\rec\\en\\en\_PP-OCRv4\_rec\_infer', recovery=False, save\_crop\_res =False, save\_log\_path ='./ log\_output /', scales=[8, 16 , 32], ser\_dict\_path ='../ train\_data /XFUND/class\_list\_xfun.txt', ser\_model\_dir =None, show\_log =True, sr\_batch\_num =1, sr\_image\_shape ='3, 32, 128', sr\_model\_dir =None, structure\_version ='PP- StructureV2', table=True, table\_algorithm =' TableAttn ', table\_char\_dict\_path =None, table\_max\_len =488, table\_model\_dir =None, total\_process\_num =1, type=' ocr ', use\_angle\_cls =True, use\_dilation =False, use\_gpu =True, use\_mp =False , use\_npu =False, use\_onnx =False, use\_pdf2docx\_api=False, use\_pdserving =False, use\_space\_char =True, use\_tensorrt =False, use\_visual\_backbone =True, use\_xpu =False, vis\_font\_path ='./doc/fonts/simfang.ttf', warmup =False )

[2024/05/11 16:20:27] ppocr INFORMATION: \*\*\*\*\*\*\*\*\*\*\*./drug1.jpg\*\*\*\*\*\*\*\*\*\*\*

[2024/05/11 16:20:28] ppocr DEBUG: dt\_boxes number : 6, elapsed : 1.0897586345672607

[2024/05/11 16:20:29] ppocr DEBUG: cls number : 6, elapsed : 0.5022566318511963

[2024/05/11 16:20:29] ppocr DEBUG: rec\_res number : 6, elapsed: 0.022243261337280273

[2024/05/11 16:20:29] ppocr INFORMATION: [[[219.0, 199.0], [288.0, 202.0], [288.0, 216.0], [218.0, 214.0]], ('50MCG TABLETS', 0.9500204920768738) ]

[2024/05/11 16:20:29] ppocr INFORMATION: [[[217.0, 218.0], [337.0, 215.0], [338.0, 232.0], [218.0, 235.0]], ('TAKE A TABLET BY', 0.9259033203125)]

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 232.0], [285.0, 234.0], [285.0, 249.0], [218.0, 246.0]], ('EVERY DAY', 0.9418787360191345) ]

[2024/05/11 16:20:29] ppocr INFO: [[[219.0, 257.0], [254.0, 260.0], [253.0, 274.0], [218.0, 272.0]], ('QTY.90', 0.9742364883422852 )]

[2024/05/11 16:20:29] ppocr INFO: [[[218.0, 289.0], [293.0, 293.0], [292.0, 306.0], [217.0, 303.0]], ('Fied12-01-2019' , 0.7921867370605469)]


**Result: correct**

|D:\ALL\_PROJECT\a\_Bosch\pyQT5\_experimental\ai\paddle|code folder (..for my own remember me ..)|
| :- | :- |
|||
|||
|![ref2]|<p>Result</p><p></p><p></p>|

**Conclusion :**

The installation is simple . 
With the right links and versions, we end up with a good LIVE CAM OCR result.

References :

|<p><https://www.youtube.com/watch?v=t5xwQguk9XU> </p><p></p>|Extract drug labels using deep learning with PaddleOCR and Python|
| :- | :- |
|||
|<p><https://www.techspot.com/downloads/7241-visual-studio-2019.html></p><p></p>|Visual Studio is urgently needed BEFORE installing NVIDIA (CUDA)|
|<p><https://github.com/PaddlePaddle/PaddleOCR></p><p></p>|rowing OCR|
|||
|<p><https://nitratine.net/blog/post/issues-when-using-auto-py-to-exe/#google_vignette></p><p></p>|Issues When Using auto-py-to-exe|

[ref1]: Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.001.png
[ref2]: Aspose.Words.dded1f58-1b26-4a7f-9eaf-5dd43d292893.015.png
