package com.mashehu.anonyme.services;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.mashehu.anonyme.common.Utilities;

import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;
import static com.mashehu.anonyme.common.Constants.INTENT_START_ENGINE;
import static com.mashehu.anonyme.common.Constants.INTENT_START_PROCESSING_IMAGES;


/**
 * A {@link BroadcastReceiver} for handling outgoing calls
 */
public class EngineStartReceiver extends BroadcastReceiver {
	final static String TAG = "anonyme.EngineStartReceiver";

	@Override
	public void onReceive(Context context, Intent intent) {
		// TODO: This method is called when the BroadcastReceiver is receiving
		// an Intent broadcast.
		Log.d(TAG, "Got intent! Action - " + intent.getAction());
		String action = intent.getAction();

		if (getResultCode() != Activity.RESULT_OK)
			return;

		if (action != null & INTENT_START_ENGINE.equals(action)) {
			Utilities.createNotificationChannels(context);
			startEngine(context, intent);
		}
		// Else, return after doing nothing since the action has nothing to do with this receiver
	}


	/**
	 *
	 * @param context App context
	 * @param intent
	 */
	private void startEngine(Context context, Intent intent) {
//		int notificationID = new Random().nextInt(); // to be used to replace notifications when escalating



		Intent startEngineIntent = new Intent(INTENT_START_PROCESSING_IMAGES, null,
				context, EngineService.class);
		startEngineIntent.putExtra(EXTRA_ENGINE_INPUT_PATH, intent.getStringArrayListExtra(EXTRA_ENGINE_INPUT_PATH));
		startEngineIntent.putExtra(EXTRA_ENGINE_OUT_DIR, intent.getStringExtra(EXTRA_ENGINE_OUT_DIR));

		Log.d(TAG, "startEngine: Starting 'Engine' service");
		context.startService(startEngineIntent);
//		JobIntentService.enqueueWork(context, EngineService.class,
//				JOB_ID_START_ENGINE_SERVICE, startEngineIntent);

//		Log.d(TAG, "OutgoingCallsReceiver: Sending SMS. UUID: " + uuid);
//		SmsManager sm = SmsManager.getDefault();
//		sm.sendTextMessage(destination, null, message,
//				PendingIntent.getBroadcast(context, 0, sentMsgIntent, 0),
//				PendingIntent.getBroadcast(context, 0, deliveredMsgIntent, 0));
	}
}
