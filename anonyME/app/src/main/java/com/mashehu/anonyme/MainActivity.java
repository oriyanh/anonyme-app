package com.mashehu.anonyme;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.collection.ArraySet;
import androidx.core.app.ActivityCompat;
import androidx.core.content.pm.ShortcutInfoCompat;
import androidx.core.content.pm.ShortcutManagerCompat;
import androidx.core.graphics.drawable.IconCompat;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.ViewModelProviders;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;

import com.mashehu.anonyme.common.Utilities;
import com.mashehu.anonyme.fragments.AppViewModel;
import java.util.ArrayList;

import static com.mashehu.anonyme.common.Constants.*;

public class MainActivity extends FragmentActivity {
	public static final String TAG = "anonyme.MainActivity.";
	AppViewModel viewModel;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		viewModel = ViewModelProviders.of(this).get(AppViewModel.class);
		viewModel.setCurrentTab(0); // Makes sure app will start on camera capture mode

        while (!Utilities.checkPermissions(this, PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS,
                    ANONYME_PERMISSION_REQUEST_CODE);
        }

		addShareShortcut(this);
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

	private void addShareShortcut(Context context) {
		ArraySet<String> categories = new ArraySet<>();
		categories.add("com.mashehu.anonyme.sharingshortcuts.category.IMAGE_SHARE_TARGET");

		ArrayList<ShortcutInfoCompat> shareShortcuts = new ArrayList<>();

		shareShortcuts.add(new ShortcutInfoCompat.Builder(
				context, ANYNOME_SHARE_SHORTCUT_ID)
				.setShortLabel("Anonymize")
				.setLongLabel("Anonymize")
				.setLongLived()
				.setIcon(IconCompat.createWithResource(context, R.drawable.ic_launcher_foreground))
				.setIntent(new Intent(Intent.ACTION_DEFAULT))
				.setCategories(categories)
				.build());

		ShortcutManagerCompat.addDynamicShortcuts(context, shareShortcuts);
	}
}

