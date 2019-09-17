package com.mashehu.anonyme.services;

import android.os.AsyncTask;
import android.util.Log;

public class MoveToGalleryAsyncTask extends AsyncTask<String, Void, String> {
	public static final String TAG = "anonyME.FinalizeProcessImageAsyncTask";

	@Override
	protected String doInBackground(String... strings) {
		Log.d(TAG, "Processing image " + strings[0]);
		return strings[0]; // todo return new path
	}
}
