package com.mashehu.anonyme.services;

import android.os.AsyncTask;
import android.util.Log;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

import static com.mashehu.anonyme.common.Constants.ENGINE_MODULE_NAME;

public class EngineAsyncTask extends AsyncTask<String, Void, String> {
	public static final String TAG = "anonyme.services.EngineAsyncTask.";
	private String in_dir;
	private String out_dir;
	ImageMover delegate;

	EngineAsyncTask(String in_dir, String out_dir) {
		this.in_dir = in_dir;
		this.out_dir = out_dir;
	}

	@Override
	protected void onPostExecute(String s) {
		super.onPostExecute(s);
		if (this.delegate != null) {
			delegate.moveToGallery(s);
		}
	}

	@Override
	protected String doInBackground(String... strings) {
		Log.d(TAG + "doInBackground", "Starting work in separate thread on " + strings[0]);
		Python py = Python.getInstance();
		PyObject res = py.getModule(ENGINE_MODULE_NAME).
				callAttr("main", this.in_dir, this.out_dir, strings[0]);
		return res.toString();
	}
}
