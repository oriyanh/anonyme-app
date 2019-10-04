♪ README README README a man of the midnight! ♪<br>
Amit Segal<br>
Oriyan Hermoni<br>
<br><b>anonyME<sup>©</sup> Android Application</b><br>
This app's purpose is anonymizing facial images so that they are unrecognizable to face recognition ML algorithms (as part of our Hebrew University Comp. Eng. final engineering project MVP alpha). Currently the app runs AdvBox (using the Chaquopy library) behind the scenes to create adversarial attacks of 160x160, but in the future this should change.<br>
Currently our app runs on Android API levels 28 (Android Pie) and 29 (Android 10).<br><br><br>
<b>Examples:</b><br>
We ran a hands-on test on a pre-trained advbox white-bo attack on FaceNet face recognition, on pictures of Bill Gates:<br>
<b>Original</b><br>
![alt_text](AdvBox/applications/face_recognition_attack/Bill_Gates_0001.png)<br>
<b>Two different corrupted versions:</b><br>
![alt text](Bill_Gates_0001_2_007.png)<br>
![alt text](Bill_Gates_0001_2_Michael_Jordan_0002.png)
