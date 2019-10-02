package com.mashehu.anonyme;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.collection.ArraySet;
import androidx.core.app.ActivityCompat;
import androidx.core.content.pm.ShortcutInfoCompat;
import androidx.core.content.pm.ShortcutManagerCompat;
import androidx.core.graphics.drawable.IconCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.ViewModelProviders;
import androidx.viewpager.widget.ViewPager;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;

import com.mashehu.anonyme.common.Utilities;
import com.mashehu.anonyme.fragments.AppViewModel;
import com.mashehu.anonyme.fragments.CameraCaptureFragment;
import com.mashehu.anonyme.fragments.GalleryFragment;
import com.mashehu.anonyme.fragments.ui.FragmentPagerAdapter;

import java.util.ArrayList;

import static androidx.fragment.app.FragmentStatePagerAdapter.BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT;
import static com.mashehu.anonyme.common.Constants.*;

public class MainActivity extends FragmentActivity {
    public static final String TAG = "anonyme.MainActivity.";
    AppViewModel viewModel;
    ArrayList<Fragment> fragments;
    ViewPager fragmentViewPager;
    FragmentPagerAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        viewModel = ViewModelProviders.of(this).get(AppViewModel.class);
        viewModel.setCurrentTab(1); // Makes sure app will start on camera capture mode


        fragments = new ArrayList<>();
        fragments.add(new CameraCaptureFragment());
        fragments.add(new GalleryFragment());
        fragmentViewPager = findViewById(R.id.fragmentViewPager);
//        adapter = new ViewPagerFragmentAdapter(getSupportFragmentManager(), getLifecycle(), fragments);
        adapter = new FragmentPagerAdapter(getSupportFragmentManager(),
                BEHAVIOR_RESUME_ONLY_CURRENT_FRAGMENT);
//        fragmentViewPager.setOrientation(ViewPager2.ORIENTATION_HORIZONTAL);
        fragmentViewPager.setAdapter(adapter);
        Log.d("anonyme.ContainerFragment", "Previous tab position: " + viewModel.getCurrentTab());
        if (viewModel.getCurrentTab() != -1) {
            fragmentViewPager.setCurrentItem(viewModel.getCurrentTab());
        }
        else {
            fragmentViewPager.setCurrentItem(1);
            viewModel.setCurrentTab(1);
        }

        fragmentViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
            @Override
            public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {

            }

            @Override
            public void onPageSelected(int position) {
                Log.d("anonyme.ContainerFragment", "Current tab position: " + position);
                viewModel.setCurrentTab(position);
            }

            @Override
            public void onPageScrollStateChanged(int state) {

            }
        });
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

