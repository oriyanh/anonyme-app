package com.mashehu.anonyme;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Python py = Python.getInstance();
        py.getModule("AdvBox.applications.face_recognition_attack.facenet_fr").
                callAttr("main");
    }
}
