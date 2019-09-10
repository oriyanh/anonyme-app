package com.mashehu.anonyme;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.mashehu.anonyme.common.Constants;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        Python py = Python.getInstance();
//        PyObject res = py.getModule("AdvBox.applications.face_recognition_attack.facenet_fr").
//                        callAttr("main", Constants.FILES_PATH, Constants.CACHE_PATH);
//        Log.d("ANONYME", "result file path = " + res.toString());
    }

//    public static File getOutputDirectory(Context context)
//    {
//        Context applicationContext = context.getApplicationContext();
//        File mediaDir;
//        if (context.getExternalMediaDirs().length != 0)
//        {
//            mediaDir = context.getExternalMediaDirs()[0];
//        }
//        else
//        {
//            mediaDir = null;
//        }
//        File outputDir = new File(mediaDir,
//                applicationContext.getResources().getString(R.string.app_name));
//        outputDir.mkdirs();
//        if ((mediaDir != null) && (mediaDir.exists()))
//        {
//            return mediaDir;
//        }
//        return applicationContext.getFilesDir();
//    }
}
