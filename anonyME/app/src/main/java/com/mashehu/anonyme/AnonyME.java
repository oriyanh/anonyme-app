package com.mashehu.anonyme;

import android.app.Application;
import android.os.Environment;
import android.util.Log;

//import com.chaquo.python.*;
//import com.chaquo.python.android.AndroidPlatform;

import static com.mashehu.anonyme.common.Constants.*;

import java.io.File;

import static android.os.Environment.DIRECTORY_DCIM;

public class AnonyME extends Application {
	public static final String TAG = "anonyme.Application.";

	@Override
	public void onCreate() {
		super.onCreate();
		ASSETS_PATH = getFilesDir();
		CACHE_PATH = getExternalCacheDir();
		Log.d(TAG, "cache path: " + CACHE_PATH);
		CAMERA_ROLL_PATH = new File(
				Environment.getExternalStoragePublicDirectory(DIRECTORY_DCIM), "Camera");

		// Starts python engine when app starts
//		Python.start(new AndroidPlatform(this));
	}

}
