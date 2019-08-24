package com.mashehu.anonyme;
import android.app.Application;
import com.chaquo.python.*;
import com.chaquo.python.android.AndroidPlatform;

public class AnonyME extends Application {
    @Override
    public void onCreate() {
        super.onCreate();

        // Starts python engine when app starts
        Python.start(new AndroidPlatform(this));
    }
}
