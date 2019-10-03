package com.mashehu.anonyme.services;

import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.net.Uri;
import android.os.IBinder;
import android.preference.PreferenceManager;
import android.util.Log;

import androidx.core.app.NotificationManagerCompat;
import androidx.core.content.FileProvider;

import com.chaquo.python.Python;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.mashehu.anonyme.common.Constants.CACHE_PATH;
import static com.mashehu.anonyme.common.Constants.ENGINE_MODULE_NAME;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PICS;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_NUM_IMAGES;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;
import static com.mashehu.anonyme.common.Constants.NOTIFICATION_CH_ID_PROGRESS;
import static com.mashehu.anonyme.common.Constants.SP_IS_PROCESSING_KEY;
import static com.mashehu.anonyme.common.Utilities.createNotification;

public class EngineService extends Service {
	final static String TAG = "anonyme.EngineService.";
	final static String ACTION_STOP = "com.mashehu.anonyme.action.STOP";

	private ExecutorService singleThreadExecutor;

	public EngineService() {
	}

	@Override
	public IBinder onBind(Intent intent) {
		// TODO: Return the communication channel to the service.
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {

	    if (ACTION_STOP.equals(intent.getAction()))
        {
            Log.d(TAG, "Called to stop service");
            stopForeground(true);
			NotificationManagerCompat.from(this).notify(1, createNotification("Canceling...", null, getApplicationContext(), null, NOTIFICATION_CH_ID_PROGRESS));
            stopSelf();
            return START_NOT_STICKY;
        }
	    else
		{
			singleThreadExecutor = Executors.newSingleThreadExecutor();

			Intent stopService = new Intent(this, EngineService.class);
			stopService.setAction(ACTION_STOP);
			PendingIntent pendingStopService = PendingIntent.getService(
					this, 0, stopService, PendingIntent.FLAG_CANCEL_CURRENT);
			startForeground(1, createNotification("Anonymization in progress...", "Tap to cancel", // TODO send meaningful notifications to demonstrate progress
					getApplicationContext(), pendingStopService,
					NOTIFICATION_CH_ID_PROGRESS));

			int num_images = intent.getIntExtra(EXTRA_ENGINE_NUM_IMAGES, 1); // to be used for demonstrating progress?
			String assetsDir = intent.getStringExtra(EXTRA_ENGINE_ASSETS_PATH);
			String outputDir = intent.getStringExtra(EXTRA_ENGINE_OUT_DIR);

			// TODO generate single string to represent list of images to process, or start different python process per image
			// ArrayList containing list of images to process
			ArrayList<String> images = intent.getStringArrayListExtra(EXTRA_ENGINE_INPUT_PICS);
			Log.d(TAG + "onStartCommand", "Processing " + num_images + " images");

			PreferenceManager.getDefaultSharedPreferences(getApplicationContext()).edit().putBoolean(SP_IS_PROCESSING_KEY, true).apply();

			int progress = 0;
			singleThreadExecutor.execute(() -> {
					try
					{
						String res = null;
						Python py = Python.getInstance();
						for (String image: images)
						{
							if (Thread.interrupted())
							{
								throw new InterruptedException();
							}

							Log.d(TAG + "onStartCommand", "Processing image #" + progress + ": " + image);
							res = py.getModule(ENGINE_MODULE_NAME).callAttr(
									"main", assetsDir, outputDir, image).toString();
							Log.d(TAG, "Finished processing - output located in " + res);
						}

						Intent showImages;

						if (images.size() == 1)
						{
							assert res != null;
							File outputFile = new File(res);
							Uri cameraRollUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".provider", outputFile);
							showImages = new Intent(Intent.ACTION_VIEW);
							showImages.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
							showImages.setDataAndType(cameraRollUri, "image/*");
							showImages.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
						}
						else
						{
							showImages = new Intent(Intent.ACTION_VIEW, Uri.parse("content://media/internal/images/media"));
						}


						PendingIntent pShowImages = PendingIntent.getActivity(getApplicationContext(), 0, showImages, PendingIntent.FLAG_ONE_SHOT);
						NotificationManagerCompat.from(this).notify(1, createNotification("Anonymization complete", "Tap to view image", this, pShowImages, NOTIFICATION_CH_ID_PROGRESS));
					}
					catch (InterruptedException e)
					{
						Log.d(TAG, "Task cancelled");
						NotificationManagerCompat.from(this).cancel(1);
					}
					finally {
						PreferenceManager.getDefaultSharedPreferences(getApplicationContext()).edit().putBoolean(SP_IS_PROCESSING_KEY, false).apply();
					}
				});

//			ArrayList<AsyncTask<String, Void, String>> futures = new ArrayList<>(num_images);
//			for (String imageView : images) {
//				progress++;

//				futures.add(processImage(assets_dir, out_dir, imageView));
//			Log.d(TAG + "onStartCommand", "result file path = " + res);
			super.onStartCommand(intent, flags, startId);
			return START_NOT_STICKY;
		}
	}

	@Override
	public void onDestroy() {
		singleThreadExecutor.shutdownNow();
		File[] cache_files = CACHE_PATH.listFiles();
		if (cache_files != null) {
			for (File f : cache_files) {
				Log.d("EngineService", "removing file " + f.getAbsolutePath());
				f.deleteOnExit();
			}
		}
		PreferenceManager.getDefaultSharedPreferences(getApplicationContext()).edit().putBoolean(SP_IS_PROCESSING_KEY, false).apply();
		super.onDestroy();
	}

//	private class PythonAnonymizer implements Callable<String>
//	{
//
//		private String inputDir;
//		private String outputDir;
//		private String imageFile;
//
//		PythonAnonymizer(String inputDir, String outputDir, String imageFile)
//		{
//			this.inputDir = inputDir;
//			this.outputDir = outputDir;
//			this.imageFile = imageFile;
//		}
//
//		@Override
//		public String call() {
//
//			try
//			{
//				Python py = Python.getInstance();
//				return py.getModule(ENGINE_MODULE_NAME).callAttr(
//						"main", inputDir, outputDir, imageFile).toString();
//			}
//			catch (InterruptedException e)
//			{
//				Log.d(TAG, "Task cancelled.");
//			}
//		}
//	}
}


//	public AsyncTask<String, Void, String> processImage(String assets_dir, String out_dir, String imageView) {
//		EngineAsyncTask task = new EngineAsyncTask(assets_dir, out_dir);
//		task.delegate = this;
//		task.execute(imageView);
//
//		Python py = Python.getInstance();
//		PyObject res = py.getModule(ENGINE_MODULE_NAME).
//				callAttr("main", assets_dir, out_dir, imageView);
//
//		return res.toString();
//		return task;
//	}
//
//	public void moveToGallery(String imageView) {
//		Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(imageView));
//		PendingIntent pendingIntent = PendingIntent.getActivity(
//				this.getApplicationContext(), 0, intent, PendingIntent.FLAG_ONE_SHOT);
//		Notification viewImageNotification = createNotification(
//				"Anonymization complete", "Tap to view result", this, pendingIntent, NOTIFICATION_CH_ID_PROGRESS);
//
//		Log.d(TAG + "moveToGallery", "result file path = " + imageView);
//
//		MoveToGalleryAsyncTask task = new MoveToGalleryAsyncTask();
//		task.execute(imageView); //todo implement actual mechanism
//	}