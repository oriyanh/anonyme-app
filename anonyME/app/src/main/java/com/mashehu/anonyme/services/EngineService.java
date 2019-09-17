package com.mashehu.anonyme.services;

import android.app.Service;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.IBinder;
import android.util.Log;

import java.util.ArrayList;
import java.util.concurrent.ExecutionException;

import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PICS;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_NUM_IMAGES;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;
import static com.mashehu.anonyme.common.Constants.NOTIFICATION_CH_ID_PROGRESS;
import static com.mashehu.anonyme.common.Utilities.createNotification;

public class EngineService extends Service implements ImageMover{
	final static String TAG = "anonyme.EngineService.";

	public EngineService() {
	}

	@Override
	public IBinder onBind(Intent intent) {
		// TODO: Return the communication channel to the service.
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		startForeground(1, createNotification("bla", // TODO send meaningful notifications to demonstrate progress
								getApplicationContext(),
								NOTIFICATION_CH_ID_PROGRESS));

		int num_images = intent.getIntExtra(EXTRA_ENGINE_NUM_IMAGES, 1); // to be used for demonstrating progress?
		String assets_dir = intent.getStringExtra(EXTRA_ENGINE_ASSETS_PATH);
		String out_dir = intent.getStringExtra(EXTRA_ENGINE_OUT_DIR);

		// TODO generate single string to represent list of images to process, or start different python process per image
		ArrayList<String> images = intent.getStringArrayListExtra(EXTRA_ENGINE_INPUT_PICS); // ArrayList containing list of images to process
		ArrayList<String> results = new ArrayList<>(num_images); // ArrayList containing list of images to process
		Log.d(TAG + "onStartCommand", "Processing " + num_images + " images");
		int progress = 0;
		ArrayList<AsyncTask<String, Void, String>> futures = new ArrayList<>(num_images);
		for (String img : images) {
			progress++;
			Log.d(TAG + "onStartCommand", "Processing image #" + progress + ": " + img);
			futures.add(processImage(assets_dir, out_dir, img));
//			Log.d(TAG + "onStartCommand", "result file path = " + res);
			//TODO start worker thread to move resulting image to camera roll album, using `res`
			// OR start THREAD that runs `processImage()` , then this thread moves the resulting image to the camera roll
			// OR start a DIFFERENT SERVICE that does the same thing. Then the CURRENT SERVICE can be a background service, maybe?
		}

		return super.onStartCommand(intent, flags, startId);
	}


	public AsyncTask<String, Void, String> processImage(String assets_dir, String out_dir, String img) {
		EngineAsyncTask task = new EngineAsyncTask(assets_dir, out_dir);
		task.delegate = this;
		task.execute(img);

//		Python py = Python.getInstance();
//		PyObject res = py.getModule(ENGINE_MODULE_NAME).
//				callAttr("main", assets_dir, out_dir, img);
//
//		return res.toString();
		return task;
	}

	public void moveToGallery(String img) {
		Log.d(TAG + "moveToGallery", "result file path = " + img);

		MoveToGalleryAsyncTask task = new MoveToGalleryAsyncTask();
		task.execute(img); //todo implement actual mechanism
	}
}

interface ImageMover {
	public void moveToGallery(String img);
}