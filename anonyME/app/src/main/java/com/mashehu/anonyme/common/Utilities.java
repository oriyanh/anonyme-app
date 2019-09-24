package com.mashehu.anonyme.common;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;

import com.mashehu.anonyme.R;

import static com.mashehu.anonyme.common.Constants.NOTIFICATION_CH_DESC_PROGRESS;
import static com.mashehu.anonyme.common.Constants.NOTIFICATION_CH_ID_PROGRESS;
import static com.mashehu.anonyme.common.Constants.NOTIFICATION_CH_NAME_PROGRESS;

public class Utilities {
	public static final String TAG = "anonyme.Utilities.";

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


	public static Notification createNotification(String message, Context context, String channel) {
		String appName = context.getString(R.string.app_name);
		return new NotificationCompat.Builder(context, channel)
				.setContentTitle(appName)
				.setContentText(message)
				.setSmallIcon(R.mipmap.ic_launcher)
				.setPriority(NotificationCompat.PRIORITY_HIGH)
				.build();
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
}
