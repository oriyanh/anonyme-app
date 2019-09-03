package com.mashehu.anonyme.services;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.mashehu.anonyme.common.Constants;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;

public class EngineService extends Service {
    final static String TAG = "anonyme.EngineService";
    public EngineService() {
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForeground(1, new Notification());

        // Retrieves Python instance
        Python py = Python.getInstance();
        String images = intent.getStringExtra(EXTRA_ENGINE_INPUT_PATH); // or arraylist of strings
        // TODO generate single string to represent list of images to process, or start different python process per image

        ArrayList<String> out_dir = intent.getStringArrayListExtra(EXTRA_ENGINE_OUT_DIR);
        PyObject res = py.getModule("AdvBox.applications.face_recognition_attack.facenet_fr").
                callAttr("main", Constants.FILES_PATH, Constants.CACHE_PATH);
        Log.d(TAG, "result file path = " + res.toString());
        return super.onStartCommand(intent, flags, startId);
    }
}
