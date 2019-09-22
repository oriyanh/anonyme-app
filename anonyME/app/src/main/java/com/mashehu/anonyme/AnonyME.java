package com.mashehu.anonyme;
import android.app.Application;
import android.content.res.AssetManager;
import android.util.Log;

import com.chaquo.python.*;
import com.chaquo.python.android.AndroidPlatform;
import com.mashehu.anonyme.common.Constants;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class AnonyME extends Application {
    public static final String TAG = "anonyme.Application.";
    static boolean COPY_ASSETS = true;

    @Override
    public void onCreate() {
        super.onCreate();

        // Starts python engine when app starts
        Constants.ASSETS_PATH = getFilesDir().toString();
        Constants.CACHE_PATH = getCacheDir().toString();
        if (COPY_ASSETS) {  // Copy resources from assets dir (in APK) to local storage
            Log.d(TAG + "onCreate", "Copying assets");
            try {
                copyAssets(Constants.ASSETS_PATH);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        Python.start(new AndroidPlatform(this));
    }


    public void copyAssets(String dest) throws IOException{
        AssetManager assetManager = getApplicationContext().getAssets();
        String[] files = assetManager.list("");
        assert files != null;
        for (String f : files) {
            if (f.endsWith(".png") || f.endsWith(".pb")) {
                // Constants.ASSETS_PATH.toString()
                OutputStream myOutput = new FileOutputStream(dest + "/" + f);
                byte[] buffer = new byte[1024];
                int length;
                InputStream myInput = assetManager.open(f);
                while ((length = myInput.read(buffer)) > 0) {
                    myOutput.write(buffer, 0, length);
                }
                myInput.close();
                myOutput.flush();
                myOutput.close();
            }
        }
    }

}
