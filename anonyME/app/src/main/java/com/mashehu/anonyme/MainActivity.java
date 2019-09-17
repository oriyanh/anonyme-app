package com.mashehu.anonyme;

import androidx.fragment.app.FragmentActivity;

import android.content.Intent;
import android.os.Bundle;
import com.mashehu.anonyme.services.EngineStartReceiver;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.CACHE_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PICS;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_NUM_IMAGES;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;
import static com.mashehu.anonyme.common.Constants.INTENT_START_ENGINE;
import static com.mashehu.anonyme.common.Constants.PERMISSIONS;
import static com.mashehu.anonyme.common.Utilities.checkPermissions;

public class MainActivity extends FragmentActivity {
	public static final String TAG = "anonyme.MainActivity.";

//	BroadcastReceiver engineStartReceiver;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		// register receiver - uncomment if necessary
//		engineStartReceiver = new EngineStartReceiver();
//		IntentFilter filter = new IntentFilter();
//		filter.addAction(INTENT_START_ENGINE);
//		registerReceiver(engineStartReceiver, filter);

		if (!checkPermissions(this, PERMISSIONS))
		{

		}


		Intent startEngineIntent = new Intent(INTENT_START_ENGINE, null,
				getApplicationContext(), EngineStartReceiver.class);

		ArrayList<String> images = new ArrayList<>();
		images.add("bill_gates_0001.png");

		startEngineIntent.putExtra(EXTRA_ENGINE_ASSETS_PATH, ASSETS_PATH);
		startEngineIntent.putExtra(EXTRA_ENGINE_OUT_DIR, CACHE_PATH);
		startEngineIntent.putExtra(EXTRA_ENGINE_NUM_IMAGES, 1);
		startEngineIntent.putExtra(EXTRA_ENGINE_INPUT_PICS, images);
		sendBroadcast(startEngineIntent);
	}
}

