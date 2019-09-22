package com.mashehu.anonyme.common;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;

import com.mashehu.anonyme.R;
import com.mashehu.anonyme.fragments.ui.ImageData;

import java.io.File;
import java.util.ArrayList;

import static android.os.Environment.DIRECTORY_DCIM;
import static android.os.Environment.DIRECTORY_PICTURES;
import static com.mashehu.anonyme.common.Constants.APP_NAME;
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
		return new NotificationCompat.Builder(context, channel)
				.setContentTitle(APP_NAME)
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

	public static ArrayList<ImageData> getGalleryContent() {
		ArrayList<ImageData> images = new ArrayList<>();
		File galleryDir = new File(Environment.getExternalStoragePublicDirectory(DIRECTORY_DCIM), "Camera");
		File[] galleryFiles = galleryDir.listFiles();
		if (galleryFiles != null) {
			for (File f : galleryDir.listFiles()) {
				if (f.isFile()) {  // To avoid adding inner directories. TODO add capability to show inner directories as well
					ImageData img = new ImageData();
					img.setImagePath(f.getAbsolutePath());
					images.add(img);
				}
			}
		}

		return images;
	}
}
