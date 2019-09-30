package com.mashehu.anonyme;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.app.ActivityCompat;
import androidx.fragment.app.FragmentActivity;

import android.app.Activity;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;

import com.mashehu.anonyme.common.Utilities;
import com.mashehu.anonyme.services.EngineStartReceiver;

import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.ANONYME_PERMISSION_REQUEST_CODE;
import static com.mashehu.anonyme.common.Constants.ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.CACHE_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_ASSETS_PATH;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_INPUT_PICS;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_NUM_IMAGES;
import static com.mashehu.anonyme.common.Constants.EXTRA_ENGINE_OUT_DIR;
import static com.mashehu.anonyme.common.Constants.INTENT_START_ENGINE;
import static com.mashehu.anonyme.common.Constants.PERMISSIONS;

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

        // TODO:: Implement full permissions here, not in fragment (Possibly in app)
        if (!Utilities.checkPermissions(this, PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS,
                    ANONYME_PERMISSION_REQUEST_CODE);
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull final String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        final Activity currentActivity = this;
        boolean showRationale = false;

        // If result is for different request code,
        if (requestCode != ANONYME_PERMISSION_REQUEST_CODE) {
            return;
        }

        // If one of the required permissions was not granted, returns
        for (int i = 0; i < grantResults.length; i++) {
            if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                // Determines if dialog needs to be popped up
                showRationale |= ActivityCompat.shouldShowRequestPermissionRationale(
                        this, permissions[i]);
            }
        }

        // Show dialog in case of required rationale
        if (showRationale) {
            showPermissionRationaleDialog(currentActivity, permissions);
        }
    }

    private void showPermissionRationaleDialog(final Activity activity, final String[] permissions) {
        AlertDialog.Builder permissionRationaleDialogBuilder = new AlertDialog.Builder(this);
        permissionRationaleDialogBuilder.setTitle("All permissions are necessary")
                .setMessage("We cannot continue without these permissions")
                .setPositiveButton("Ask again", new DialogInterface.OnClickListener() {

                    // If dialog was positive asks for permission again

                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        ActivityCompat.requestPermissions(activity,
                                permissions, ANONYME_PERMISSION_REQUEST_CODE);
                    }
                })
                .setNegativeButton("Leave app", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        finish();
                    }
                }).show();
    }
}

