package com.mashehu.anonyme;
import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.drawable.Icon;
import android.os.Environment;
import android.util.Log;

import androidx.collection.ArraySet;
import androidx.core.content.pm.ShortcutInfoCompat;
import androidx.core.content.pm.ShortcutManagerCompat;
import androidx.core.graphics.drawable.IconCompat;

import com.chaquo.python.*;
import com.chaquo.python.android.AndroidPlatform;
import com.mashehu.anonyme.common.Constants;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

import static android.os.Environment.DIRECTORY_DCIM;
import static com.mashehu.anonyme.common.Constants.ANYNOME_SHARE_SHORTCUT_ID;

public class AnonyME extends Application {
    public static final String TAG = "anonyme.Application.";
    static boolean COPY_ASSETS = true;

    @Override
    public void onCreate() {
        super.onCreate();
        Constants.ASSETS_PATH = getFilesDir();
        Constants.CACHE_PATH = getCacheDir();
        Constants.CAMERA_ROLL_PATH = new File(
                Environment.getExternalStoragePublicDirectory(DIRECTORY_DCIM), "Camera");

        addShareShortcut(getApplicationContext());
        if (COPY_ASSETS) {  // Copy resources from assets dir (in APK) to local storage
            Log.d(TAG + "onCreate", "Copying assets");
            try {
                copyAssets(Constants.ASSETS_PATH.toString());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Starts python engine when app starts
        Python.start(new AndroidPlatform(this));
    }


    public void copyAssets(String dest) throws IOException{
        AssetManager assetManager = getApplicationContext().getAssets();
        String[] files = assetManager.list("");
        assert files != null;
        for (String f : files) {
            if (f.endsWith(".png") || f.endsWith(".pb")) {
                // Constants.ASSETS_PATH.toString()
                OutputStream myOutput = new FileOutputStream(dest + "/" + f);
                byte[] buffer = new byte[1024];
                int length;
                InputStream myInput = assetManager.open(f);
                while ((length = myInput.read(buffer)) > 0) {
                    myOutput.write(buffer, 0, length);
                }
                myInput.close();
                myOutput.flush();
                myOutput.close();
            }
        }
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
