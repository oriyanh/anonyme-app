package com.mashehu.anonyme.common;

import android.app.ActivityManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.preference.PreferenceManager;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.RecyclerUtils;
import com.mashehu.anonyme.services.EngineStartReceiver;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static android.content.Context.ACTIVITY_SERVICE;
import static com.mashehu.anonyme.common.Constants.*;

public class Utilities {
	public static final String TAG = "anonyme.Utilities.";

	public static boolean isProcessing(Context context)
	{
		ActivityManager am = (ActivityManager) context.getSystemService(ACTIVITY_SERVICE);
		List<ActivityManager.RunningServiceInfo> l = am.getRunningServices(50);
		for (ActivityManager.RunningServiceInfo runningServiceInfo: l)
		{
			if ((runningServiceInfo.service.getClassName().equals(
					"com.mashehu.anonyme.services.EngineService")) &&
					(runningServiceInfo.foreground))
			{
				return true;
			}
		}
		return false;
	}

	/**
	 * Creates notification channels for app if they don't already exist
	 */
	public static void createNotificationChannels(Context c) {
		int importance = NotificationManager.IMPORTANCE_HIGH;
		NotificationManager notificationManager = c.getSystemService(NotificationManager.class);

		// Progress Notifications
		Log.d(TAG + "createNotificationChannels",
				"Creating notification channel: 'Progress Notifications'");
		NotificationChannel progressChannel = new NotificationChannel(NOTIFICATION_CH_ID_PROGRESS,
				NOTIFICATION_CH_NAME_PROGRESS, importance);
		progressChannel.setDescription(NOTIFICATION_CH_DESC_PROGRESS);
		notificationManager.createNotificationChannel(progressChannel);
	}


	public static Notification createNotification(Context context, String channel, String title,
												  String message,
												  PendingIntent pendingIntent, boolean ongoing,
												  boolean autoCancel, boolean onlyAlertOnce, int progressMax,
												  int progressCurrent, boolean indeterminate) {

		NotificationCompat.Builder notificationBuilder = new NotificationCompat.Builder(
				context, channel)
				.setSmallIcon(R.mipmap.ic_launcher_round)
				.setPriority(NotificationCompat.PRIORITY_HIGH)
				.setOngoing(ongoing)
				.setAutoCancel(autoCancel)
				.setOnlyAlertOnce(onlyAlertOnce)
				.setProgress(progressMax, progressCurrent, indeterminate);
		if (title != null)
		{
			notificationBuilder.setContentTitle(title);
		}

		if (message != null)
		{
			notificationBuilder.setContentText(message);
		}

		if (pendingIntent != null)
		{
			notificationBuilder.setContentIntent(pendingIntent);
		}

		return notificationBuilder.build();
	}

	public static boolean checkPermissions(Context context, String... permissions)
	{
		for (String permission: permissions)
		{
			if (ActivityCompat.checkSelfPermission(context, permission) !=
					PackageManager.PERMISSION_GRANTED)
			{
				return false;
			}
		}
		return true;
	}

	public static ArrayList<RecyclerUtils.ImageData> getGalleryContent() {
		ArrayList<RecyclerUtils.ImageData> images = new ArrayList<>();
		File[] galleryFiles = CAMERA_ROLL_PATH.listFiles();
		if (galleryFiles != null) {
			for (File f : galleryFiles) {
				String path = f.getAbsolutePath();
				if (f.isFile() && (path.endsWith(".jpg") || path.endsWith(".png"))) {  // To avoid adding inner directories. TODO add capability to show inner directories as well
					RecyclerUtils.ImageData img = new RecyclerUtils.ImageData();
					img.setImagePath(f.getAbsolutePath());
					images.add(img);
				}
			}
		}

		return images;
	}


	public static void processImages(Context context, @NonNull ArrayList<String> images) {
//		SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(context);
//		sp.edit().putBoolean(SP_IS_PROCESSING_KEY, true).apply();
		Intent startEngineIntent = new Intent(INTENT_START_ENGINE, null,
				context, EngineStartReceiver.class);

		startEngineIntent.putExtra(EXTRA_ENGINE_ASSETS_PATH, ASSETS_PATH.toString());
		startEngineIntent.putExtra(EXTRA_ENGINE_OUT_DIR, CAMERA_ROLL_PATH.toString());
		startEngineIntent.putStringArrayListExtra(EXTRA_ENGINE_INPUT_PICS, images);
		context.sendBroadcast(startEngineIntent);
	}
}
