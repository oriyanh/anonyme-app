package com.mashehu.anonyme.services;

import android.os.AsyncTask;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

import static com.mashehu.anonyme.common.Constants.ENGINE_MODULE_NAME;

public class EngineStartAsyncTask extends AsyncTask<String, Void, String> {

	String in_dir, out_dir;
	EngineService delegate;

	EngineStartAsyncTask(String in_dir, String out_dir) {
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
		Python py = Python.getInstance();
		PyObject res = py.getModule(ENGINE_MODULE_NAME).
				callAttr("main", this.in_dir, this.out_dir, strings[0]);

		return res.toString();
	}
}
