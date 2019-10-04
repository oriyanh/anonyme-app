package com.mashehu.anonyme.services;

import android.app.Notification;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.net.Uri;
import android.os.IBinder;
import android.util.Log;

import androidx.core.app.NotificationManagerCompat;
import androidx.core.content.FileProvider;

import com.chaquo.python.Python;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.mashehu.anonyme.common.Constants.CACHE_PATH;
import static com.mashehu.anonyme.common.Constants.ENGINE_MODULE_NAME;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PICS;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;
import static com.mashehu.anonyme.common.Constants.NOTIFICATION_CH_ID_PROGRESS;
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
            int notificationId = intent.getIntExtra("notificationId", 1);
			NotificationManagerCompat.from(this).notify(notificationId, createNotification(
					getApplicationContext(), NOTIFICATION_CH_ID_PROGRESS,
					"Canceling...", null, null, true,
					false, false, 0, 0, true));
			stopSelf();
            return START_NOT_STICKY;
        }

	    else
		{

//			PreferenceManager
//					.getDefaultSharedPreferences(getApplicationContext())
//					.edit()
//					.putBoolean(SP_IS_PROCESSING_KEY, true)
//					.commit();

			Date now = new Date();
			int notificationId  = Integer.parseInt(
					new SimpleDateFormat("ddHHmmss", Locale.US).format(now));

			// ArrayList containing list of images to process
			ArrayList<String> images = intent.getStringArrayListExtra(EXTRA_ENGINE_INPUT_PICS);
			String assetsDir = intent.getStringExtra(EXTRA_ENGINE_ASSETS_PATH);
			String outputDir = intent.getStringExtra(EXTRA_ENGINE_OUT_DIR);

			singleThreadExecutor = Executors.newSingleThreadExecutor();

			Notification initialNotification;

			Intent stopService = new Intent(this, EngineService.class);
			stopService.putExtra("notificationId", notificationId);
			stopService.setAction(ACTION_STOP);
			PendingIntent pendingStopService = PendingIntent.getService(
					this, 0, stopService, PendingIntent.FLAG_CANCEL_CURRENT);

			assert images != null;
			if (images.size() == 1)
			{
				initialNotification = createNotification(
						getApplicationContext(), NOTIFICATION_CH_ID_PROGRESS,
						"Anonymization in progress...",null, // TODO send meaningful notifications to demonstrate progress
						null, true, false,
						false, 0, 0, false);
			}
			else
			{

				initialNotification = createNotification(
						getApplicationContext(), NOTIFICATION_CH_ID_PROGRESS,
						"Anonymization in progress...","Tap to cancel", // TODO send meaningful notifications to demonstrate progress
						pendingStopService, true, false, true,
						images.size(), 0, false);
			}
			startForeground(notificationId, initialNotification);

			Log.d(TAG + "onStartCommand", "Processing " + images.size() + " images");

			singleThreadExecutor.execute(() -> {
				int progress = 1;
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
						if (images.size() > 1)
						{
							Notification progressNotification;

							if (progress == images.size())
							{
								progressNotification = createNotification(
										getApplicationContext(), NOTIFICATION_CH_ID_PROGRESS,
										"Anonymization in progress...",
										"Tap to cancel", // TODO send meaningful notifications to demonstrate progress
										pendingStopService, true,
										false, false,
										images.size(), progress, false);
							}
							else
							{
								progressNotification = createNotification(
										getApplicationContext(), NOTIFICATION_CH_ID_PROGRESS,
										"Anonymization in progress...",
										"Tap to cancel", // TODO send meaningful notifications to demonstrate progress
										pendingStopService, true,
										false, true,
										images.size(), progress, false);
							}
							NotificationManagerCompat.from(this).notify(notificationId,
									progressNotification);
						}

						Log.d(TAG, "Finished processing - output located in " + res);
						progress += 1;
					}
					Intent showImages;

					if (images.size() == 1)
					{
						File outputFile = new File(res);
						Uri cameraRollUri = FileProvider.getUriForFile(this,
								getApplicationContext().getPackageName() + ".provider",
								outputFile);
						showImages = new Intent(Intent.ACTION_VIEW);
						showImages.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_NEW_TASK);
						showImages.setDataAndType(cameraRollUri, "image/*");
						showImages.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
					}
					else
					{
						showImages = new Intent(Intent.ACTION_VIEW, Uri.parse("content://media/internal/images/media"));
					}

					PendingIntent pShowImages = PendingIntent.getActivity(getApplicationContext(),
							0, showImages, PendingIntent.FLAG_ONE_SHOT);
					NotificationManagerCompat.from(this).notify(notificationId,
							createNotification(this, NOTIFICATION_CH_ID_PROGRESS,
									"Anonymization complete", "Tap to view results",
									pShowImages, false, true, false, 0,
									0, false));
				}
				catch (InterruptedException e)
				{
					Log.d(TAG, "Task cancelled");
					NotificationManagerCompat.from(this).cancel(notificationId);
				}
				finally {
					stopForeground(false);
				}
//				finally {
//					PreferenceManager
//							.getDefaultSharedPreferences(getApplicationContext())
//							.edit()
//							.putBoolean(SP_IS_PROCESSING_KEY, false)
//							.commit();
//				}
			});

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
//		PreferenceManager
//				.getDefaultSharedPreferences(getApplicationContext())
//				.edit()
//				.putBoolean(SP_IS_PROCESSING_KEY, false)
//				.apply();

		super.onDestroy();
	}
}