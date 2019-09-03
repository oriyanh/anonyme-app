package com.mashehu.anonyme.common;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.util.Log;

public class Utilities {
	public static final String TAG = "anonyme.Utilities";
	/**
	 * Creates notification channels for app if they don't already exist
	 */
	public static void createNotificationChannels(Context c) {
		int importance = NotificationManager.IMPORTANCE_HIGH;
		NotificationManager notificationManager = c.getSystemService(NotificationManager.class);

		// `Sending Message`
		Log.d(TAG, "Creating notification channel: 'Sending message notifications'");
//		NotificationChannel sendingChannel = new NotificationChannel(NOTIFICATION_CH_ID_SENDING_MSG,
//				NOTIFICATION_CH_NAME_SENDING_MSG, importance);
//		sendingChannel.setDescription(NOTIFICATION_CH_DESC_SENDING_MSG);
//		notificationManager.createNotificationChannel(sendingChannel);
//
//		// `Message Sent`
		Log.d(TAG, "Creating notification channel: 'Sent message notifications'");
//		NotificationChannel sentChannel = new NotificationChannel(NOTIFICATION_CH_ID_SENT_MSG,
//				NOTIFICATION_CH_NAME_SENT_MSG, importance);
//		sentChannel.setDescription(NOTIFICATION_CH_DESC_SENT_MSG);
//		notificationManager.createNotificationChannel(sentChannel);
//
//		// `Message Delivered`
		Log.d(TAG, "Creating notification channel: 'Delivered message notifications'");
//		NotificationChannel deliveredChannel = new NotificationChannel(NOTIFICATION_CH_ID_DELIVERED_MSG,
//				NOTIFICATION_CH_NAME_DELIVERED_MSG, importance);
//		deliveredChannel.setDescription(NOTIFICATION_CH_DESC_DELIVERED_MSG);
//		notificationManager.createNotificationChannel(deliveredChannel);
	}

}
